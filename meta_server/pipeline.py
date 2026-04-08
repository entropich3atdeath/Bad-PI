"""
meta_server/pipeline.py

Pipelined speculative execution — the hardware analogy applied to the swarm.

Problem: Writing a new program.md via Claude LLM takes ~30 seconds.
         If workers wait for this between batches, GPUs sit idle.

Solution (stolen from CPU architecture):
  1. Branch prediction: while Batch N is still running, the meta-agent
     analyzes the *early* metrics (first 40% of progress) and predicts
     which hypothesis is winning. It starts writing program N+1 immediately.

  2. Speculative execution: workers receive and cache program N+1 before
     Batch N finishes. When Batch N ends, they load N+1 instantly → zero idle.

  3. Pipeline flush: if a late-arriving anomaly (outlier result, hypothesis
     flip) makes the prediction wrong, the pipeline issues a flush signal.
     Workers discard their cached spec and pull a corrected version.

States:
  IDLE      → no spec in progress
  DRAFTING  → LLM is writing program N+1 in background
  READY     → spec program is complete, workers can pre-cache it
  CONFIRMED → batch completed and spec was correct — promote to active
  FLUSHED   → prediction was wrong — discard, force re-pull
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


class SpecState(str, Enum):
    IDLE      = "idle"
    DRAFTING  = "drafting"
    READY     = "ready"
    CONFIRMED = "confirmed"
    FLUSHED   = "flushed"


@dataclass
class SpeculativeProgram:
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    batch_id:     str   = ""
    program_md:   str   = ""
    state:        SpecState = SpecState.IDLE
    confidence:   float = 0.5    # 0-1, based on early statistical trends
    created_at:   float = field(default_factory=time.time)
    confirmed_at: Optional[float] = None
    flush_reason: str   = ""

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class Pipeline:
    """
    Manages one speculative program at a time.
    Called by the search loop as early metrics arrive.
    """

    # Trigger speculative write once this many ticks have arrived for current batch
    SPEC_TRIGGER_TICKS     = 10
    # Flush if confidence drops below this after spec was already READY
    FLUSH_CONFIDENCE_FLOOR = 0.35
    # Max age before a READY spec is considered stale and discarded
    MAX_SPEC_AGE           = 600   # 10 minutes

    def __init__(self, program_writer_fn: Optional[Callable] = None):
        self.enabled: bool = _env_flag("BAD_PI_SPEC_EXEC_ENABLED", default=False)
        # When true, workers may use speculative program.md before normal sync.
        self.auto_deploy_enabled: bool = _env_flag("BAD_PI_SPEC_AUTO_DEPLOY", default=False)
        # Current spec must exceed this confidence to be deployable.
        self.spec_confidence_threshold: float = float(os.environ.get("BAD_PI_SPEC_CONF_THRESHOLD", "0.65"))
        # Global confidence bank must exceed this before deployment is allowed.
        self.deploy_confidence_threshold: float = float(os.environ.get("BAD_PI_DEPLOY_CONF_THRESHOLD", "0.70"))

        # Confidence bank: rises on correct predictions, drops hard on flushes.
        self.deployment_confidence: float = 0.50
        self._confirm_reward: float = 0.08
        self._flush_penalty: float = 0.20

        self.current:          Optional[SpeculativeProgram] = None
        self.flush_count:      int = 0
        self.confirmed_count:  int = 0
        self._ticks_this_batch: int = 0
        self._batch_id:        str = str(uuid.uuid4())[:8]
        self._writer:          Optional[Callable] = program_writer_fn
        self._drafting:        bool = False
        self._last_flush_token: Optional[str] = None

    # ── Public interface ──────────────────────────────────────────────────

    def on_tick(self, metrics: list[float], hypothesis_registry=None):
        """
        Called every time a tick arrives from any worker.
        Decides whether to start drafting a spec.
        """
        if not self.enabled:
            return

        self._ticks_this_batch += 1

        if self._should_start_drafting():
            confidence = self._estimate_confidence(metrics)
            asyncio.create_task(self._draft_async(confidence, hypothesis_registry))

        # Check if an existing READY spec has become stale
        if self.current and self.current.state == SpecState.READY:
            if self.current.age_seconds > self.MAX_SPEC_AGE:
                self.flush("spec stale (too old)")

    def on_batch_complete(self, early_metrics: list[float], hypothesis_registry=None):
        """
        Called when a batch of runs finishes.
        If the spec was READY and confidence is still high → CONFIRMED.
        Else → flush and write synchronously.
        """
        if not self.enabled:
            return

        if self.current and self.current.state == SpecState.READY:
            final_confidence = self._estimate_confidence(early_metrics)
            if final_confidence >= self.FLUSH_CONFIDENCE_FLOOR:
                self.current.state        = SpecState.CONFIRMED
                self.current.confirmed_at = time.time()
                self.confirmed_count     += 1
                self.deployment_confidence = min(1.0, self.deployment_confidence + self._confirm_reward)
                log.info(f"Pipeline: spec {self.current.id} CONFIRMED (conf={final_confidence:.2f})")
            else:
                self.flush(f"low confidence at batch end ({final_confidence:.2f})")

        # Start next batch
        self._batch_id          = str(uuid.uuid4())[:8]
        self._ticks_this_batch  = 0

    def flush(self, reason: str = ""):
        """Issue a pipeline flush — workers must discard cached spec."""
        flush_token = self.current.id if self.current else str(uuid.uuid4())[:8]
        if self.current:
            self.current.state        = SpecState.FLUSHED
            self.current.flush_reason = reason
        self._last_flush_token = flush_token
        self.flush_count += 1
        self.deployment_confidence = max(0.0, self.deployment_confidence - self._flush_penalty)
        self.current = None
        log.warning(f"Pipeline FLUSH: {reason} (total flushes: {self.flush_count})")

    def can_deploy_current(self) -> bool:
        if not self.enabled or not self.auto_deploy_enabled or not self.current:
            return False
        return (
            self.current.state in (SpecState.READY, SpecState.CONFIRMED)
            and self.current.age_seconds < self.MAX_SPEC_AGE
            and self.current.confidence >= self.spec_confidence_threshold
            and self.deployment_confidence >= self.deploy_confidence_threshold
        )

    def get_cached_spec(self) -> Optional[dict]:
        """Metadata + content for speculative worker preload path."""
        if not self.current:
            return None
        if not self.can_deploy_current():
            return None
        return {
            "spec_id": self.current.id,
            "program_md": self.current.program_md,
            "confidence": self.current.confidence,
            "deployment_confidence": round(self.deployment_confidence, 3),
        }

    def get_cached_program(self) -> Optional[str]:
        """
        Workers call this to pre-cache the next program.md.
        Returns the spec program_md if READY, else None.
        """
        spec = self.get_cached_spec()
        if spec:
            return spec["program_md"]
        return None

    def flush_token(self) -> Optional[str]:
        """
        Returns a flush token if workers should discard their cache.
        Workers poll this; non-None means "drop your spec and re-pull."
        """
        return self._last_flush_token

    def status(self) -> dict:
        return {
            "enabled":         self.enabled,
            "auto_deploy":     self.auto_deploy_enabled,
            "deploy_allowed":  self.can_deploy_current(),
            "deployment_confidence": round(self.deployment_confidence, 3),
            "spec_conf_threshold": self.spec_confidence_threshold,
            "deploy_conf_threshold": self.deploy_confidence_threshold,
            "batch_id":        self._batch_id,
            "ticks_this_batch": self._ticks_this_batch,
            "spec_state":      self.current.state if self.current else "idle",
            "spec_id":         self.current.id if self.current else None,
            "confidence":      self.current.confidence if self.current else None,
            "flush_count":     self.flush_count,
            "confirmed_count": self.confirmed_count,
            "last_flush_token": self._last_flush_token,
            "hit_rate":        round(
                self.confirmed_count / max(1, self.confirmed_count + self.flush_count), 3
            ),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _should_start_drafting(self) -> bool:
        if self._drafting:
            return False
        if self.current and self.current.state in (SpecState.DRAFTING, SpecState.READY):
            return False
        return self._ticks_this_batch >= self.SPEC_TRIGGER_TICKS

    def _estimate_confidence(self, metrics: list[float]) -> float:
        """
        Simple confidence estimate: how consistent are the early metrics?
        High variance → low confidence (unpredictable batch → spec may be wrong).
        Low variance → high confidence (trend is clear).
        """
        if len(metrics) < 3:
            return 0.5
        import statistics
        try:
            mean = statistics.mean(metrics)
            stdev = statistics.stdev(metrics)
            cv = stdev / abs(mean) if mean != 0 else 1.0
            confidence = max(0.1, min(0.95, 1.0 - cv))
            return round(confidence, 3)
        except Exception:
            return 0.5

    async def _draft_async(self, confidence: float, hypothesis_registry=None):
        """Background task: write the speculative program.md."""
        self._drafting = True
        spec = SpeculativeProgram(
            batch_id   = self._batch_id,
            state      = SpecState.DRAFTING,
            confidence = confidence,
        )
        self.current = spec
        log.info(f"Pipeline: drafting spec {spec.id} (conf={confidence:.2f})")

        try:
            program_md = await self._write_program(hypothesis_registry)
            spec.program_md = program_md
            spec.state      = SpecState.READY
            log.info(f"Pipeline: spec {spec.id} READY in {spec.age_seconds:.1f}s")
        except Exception as e:
            log.error(f"Pipeline: spec draft failed: {e}")
            self.flush(f"draft error: {e}")
        finally:
            self._drafting = False

    async def _write_program(self, hypothesis_registry=None) -> str:
        """
        Write the speculative program.md.
        Tries Claude API; falls back to template.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._write_sync, hypothesis_registry)

    def _write_sync(self, hypothesis_registry=None) -> str:
        if self._writer:
            try:
                return self._writer(hypothesis_registry)
            except Exception as e:
                log.debug(f"Custom writer failed: {e}")

        # Try Claude
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                hyp_summary = ""
                if hypothesis_registry:
                    hyp_summary = "\n".join(
                        f"- {h.summary()}" for h in hypothesis_registry.active[:5]
                    )
                msg = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=600,
                    messages=[{"role": "user", "content": (
                        "You are a PI writing speculative instructions for the NEXT batch of ML experiments. "
                        "Based on current beliefs:\n\n"
                        + (hyp_summary or "No hypothesis data yet.")
                        + "\n\nWrite a concise program.md (under 200 words) predicting the most promising "
                        "direction for the next batch. Label it [SPECULATIVE] at the top."
                    )}],
                )
                return msg.content[0].text
            except Exception as e:
                log.debug(f"Claude spec write failed: {e}")

        return (
            "[SPECULATIVE]\n\n"
            "Continue exploring the current best hypothesis. "
            "Concentrate on the highest-importance dimensions from the latest program.md. "
            "This instruction was pre-generated speculatively — if you receive a flush signal, "
            "discard this and re-pull from /program.md.\n"
        )


# ── Module-level singleton ────────────────────────────────────────────────────

pipeline = Pipeline()
