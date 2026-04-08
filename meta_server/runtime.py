from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from . import program_writer, store
from .belief_engine import engine as belief_engine
from .hypotheses import (
    Hypothesis,
    HypothesisRegistry,
    make_default_registry,
    make_registry_from_dimensions,
)
from .meta_log import MetaHypothesisLog
from .pipeline import pipeline
from .population_manager import PopulationManager, Population


RUNTIME_STATE_PATH = Path(
    os.environ.get(
        "META_RUNTIME_STATE_PATH",
        str(Path(__file__).parent / "runtime_state.json"),
    )
)


class RuntimeState:
    DIM_SIGNAL_THRESHOLD = 2
    CANARY_PROB = 0.12
    CANARY_EVAL_RUNS = 40
    CANARY_MIN_IMPROVEMENT = 0.001

    def __init__(self, state_path: Path = RUNTIME_STATE_PATH):
        self.state_path = state_path
        self.registry: HypothesisRegistry = HypothesisRegistry()
        self.population_manager = PopulationManager()
        self.meta_log = MetaHypothesisLog()
        self.pending_new_hypotheses: list[dict] = []
        self.pending_dimension_proposals: list[dict] = []   # LLM dim proposals when stalled
        self.dimension_signal_counts: dict[str, int] = {}
        self.canary_dimensions: dict[str, dict] = {}
        self._lock = threading.RLock()

    def initialize(self):
        with self._lock:
            self._load_locked()
            dimensions = store.get_dimensions()
            is_new_project = store.experiment_count() == 0

            # New-project bootstrap rule:
            # start with an empty/stale registry reset, then seed from schema dims.
            if is_new_project:
                self.registry = make_registry_from_dimensions(dimensions)
                self.pending_new_hypotheses = []

            # Defensive fallback for old state files with empty registries.
            if not self.registry.active and not self.registry.archived:
                self.registry = make_registry_from_dimensions(dimensions)

            # Last-resort fallback when dimensions table is empty/unavailable.
            if not self.registry.active and not self.registry.archived:
                self.registry = make_default_registry()
            self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))

    def build_belief_state(self):
        with self._lock:
            return self._build_belief_state_locked()

    def assign_worker(self, worker_id: str) -> Optional[Population]:
        with self._lock:
            if not self.population_manager.active_populations:
                self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))
            pop = self.population_manager.get_worker_population(worker_id)
            if pop and pop.active:
                return pop
            pop = self.population_manager.assign_worker(worker_id)
            self._save_locked()
            return pop

    def get_worker_population(self, worker_id: str) -> Optional[Population]:
        with self._lock:
            return self.population_manager.get_worker_population(worker_id)

    def list_dimension_proposals(self) -> list[dict]:
        with self._lock:
            return list(self.pending_dimension_proposals)

    def clear_dimension_proposals(self):
        with self._lock:
            self.pending_dimension_proposals = []
            self._save_locked()

    @staticmethod
    def _proposal_signature(proposal: dict) -> str:
        raw = str(proposal.get("name", "")).strip().lower()
        return re.sub(r"[^a-z0-9]+", "", raw)

    @staticmethod
    def _is_identifier(name: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))

    def _validate_dimension_proposal_bounds(self, proposal: dict) -> tuple[bool, str]:
        name = str(proposal.get("name", "")).strip()
        if not name or not self._is_identifier(name):
            return False, "invalid_dimension_name"
        if store.dimension_exists(name):
            return False, "duplicate_dimension_name"

        dtype = str(proposal.get("dtype", "")).strip().lower()
        if dtype not in {"float_log", "float_linear", "int", "categorical"}:
            return False, "invalid_dtype"

        if dtype == "categorical":
            cats = proposal.get("categories")
            if not isinstance(cats, list) or len(cats) < 2:
                return False, "invalid_categories"
            if len(cats) > 8:
                return False, "categories_too_large"
            return True, "ok"

        try:
            lo = float(proposal.get("range_min"))
            hi = float(proposal.get("range_max"))
        except Exception:
            return False, "invalid_range"
        if not (lo < hi):
            return False, "invalid_range_order"

        if dtype == "int":
            if (hi - lo) > 1024:
                return False, "int_range_too_wide"
            return True, "ok"

        if dtype == "float_log":
            if lo <= 0:
                return False, "float_log_requires_positive_min"
            if (hi / lo) > 1e4:
                return False, "float_log_span_too_wide"
            return True, "ok"

        # float_linear
        if (hi - lo) > 1e6:
            return False, "float_linear_span_too_wide"
        return True, "ok"

    def _adopt_dimension_locked(self, proposal: dict, total_experiments: int) -> tuple[bool, str]:
        ok, reason = self._validate_dimension_proposal_bounds(proposal)
        if not ok:
            return False, reason

        name = str(proposal["name"]).strip()
        dtype_raw = str(proposal.get("dtype", "")).strip().lower()
        if dtype_raw == "categorical":
            db_dtype = "categorical"
            min_val = None
            max_val = None
            log_scale = 0
            categories = list(proposal.get("categories") or [])
        else:
            db_dtype = "float" if dtype_raw == "float_linear" else dtype_raw
            min_val = float(proposal.get("range_min"))
            max_val = float(proposal.get("range_max"))
            log_scale = 1 if dtype_raw == "float_log" else 0
            categories = None

        inserted = store.add_dimension(
            name=name,
            dtype=db_dtype,
            min_val=min_val,
            max_val=max_val,
            log_scale=log_scale,
            categories=categories,
            importance=0.08,
            is_canary=True,
            canary_prob=self.CANARY_PROB,
        )
        if not inserted:
            return False, "insert_failed_or_duplicate"

        top1 = self._top_configs_locked(1)
        baseline = None
        if top1 and top1[0].get("delta_bpb") is not None:
            baseline = float(top1[0]["delta_bpb"])

        self.canary_dimensions[name] = {
            "adopted_at_experiments": int(total_experiments),
            "baseline_best_delta": baseline,
            "max_runs": int(self.CANARY_EVAL_RUNS),
            "min_improvement": float(self.CANARY_MIN_IMPROVEMENT),
            "proposal": proposal,
        }
        return True, "adopted_canary"

    def _evaluate_canary_dimensions_locked(self, total_experiments: int):
        if not self.canary_dimensions:
            return

        top1 = self._top_configs_locked(1)
        if not top1 or top1[0].get("delta_bpb") is None:
            return
        current_best = float(top1[0]["delta_bpb"])

        to_remove: list[str] = []
        for name, meta in list(self.canary_dimensions.items()):
            adopted_at = int(meta.get("adopted_at_experiments", total_experiments))
            max_runs = int(meta.get("max_runs", self.CANARY_EVAL_RUNS))
            if (total_experiments - adopted_at) < max_runs:
                continue

            baseline = meta.get("baseline_best_delta")
            min_improvement = float(meta.get("min_improvement", self.CANARY_MIN_IMPROVEMENT))

            if baseline is None:
                # No baseline at adoption time: keep as promoted by default.
                store.set_dimension_canary(name, is_canary=False, canary_prob=1.0)
                to_remove.append(name)
                log.info(f"Promoted canary dimension '{name}' (no baseline available at adoption).")
                continue

            improvement = float(baseline) - current_best  # lower delta_bpb is better
            if improvement < min_improvement:
                store.remove_dimension(name)
                to_remove.append(name)
                log.info(
                    f"Reverted canary dimension '{name}' after {max_runs} runs "
                    f"(improvement={improvement:+.5f} < {min_improvement:+.5f})"
                )
            else:
                store.set_dimension_canary(name, is_canary=False, canary_prob=1.0)
                to_remove.append(name)
                log.info(
                    f"Promoted canary dimension '{name}' to full search "
                    f"(improvement={improvement:+.5f} >= {min_improvement:+.5f})"
                )

        for name in to_remove:
            self.canary_dimensions.pop(name, None)

    def shape_config_for_worker(self, worker_id: str, config: dict) -> dict:
        with self._lock:
            pop = self.assign_worker(worker_id)
            shaped = dict(config)
            shaped_delta = dict(shaped.get("config_delta") or {})
            hypothesis = self.registry.get(pop.hypothesis_id) if pop else None
            best_config = self._best_config_delta_locked()

            if pop and hypothesis:
                if pop.strategy == "exploit" and best_config:
                    shaped_delta = {**best_config, **shaped_delta}
                elif pop.strategy == "falsify":
                    shaped_delta = {
                        **best_config,
                        **shaped_delta,
                    }

                if hypothesis.config_constraint:
                    shaped_delta.update(hypothesis.config_constraint)

                statement = hypothesis.statement.strip()
                # Convergence mode: override strategy to 'converge' for the validated winner
                conv_winner = self.registry.convergence_winner()
                effective_strategy = (
                    "converge" if (conv_winner and conv_winner.id == hypothesis.id)
                    else pop.strategy
                )
                prefix = f"{effective_strategy} · {pop.id}"
                shaped["note"] = f"{prefix} — {statement}" if statement else prefix
                shaped["_population_id"] = pop.id
                shaped["_population_strategy"] = effective_strategy
                shaped["_hypothesis_id"] = hypothesis.id
                shaped["_hypothesis_statement"] = hypothesis.statement
                shaped["_hypothesis_posterior"] = round(hypothesis.posterior, 4)
            else:
                shaped.setdefault("note", "default")
                shaped.setdefault("_population_id", "default")
                shaped.setdefault("_population_strategy", "investigate")
                shaped.setdefault("_hypothesis_posterior", 0.5)

            shaped["config_delta"] = shaped_delta
            return shaped

    def program_for_worker(self, worker_id: str) -> tuple[str, Optional[Population], Optional[Hypothesis]]:
        with self._lock:
            pop = self.assign_worker(worker_id)
            hypothesis = self.registry.get(pop.hypothesis_id) if pop else None
            if pop and pop.program_md:
                return pop.program_md, pop, hypothesis
            return store.latest_program_md(), pop, hypothesis

    def handle_completed_experiment(self, config_delta: dict, delta_bpb: Optional[float], total_experiments: int):
        with self._lock:
            if delta_bpb is not None:
                self.registry.ingest_experiment(config_delta, delta_bpb)

            experiments = store.recent_experiments(1000)
            dimensions = store.get_dimensions()

            # Validation-phase hypotheses: schedule missing grid/arm cells deterministically.
            validation_requests = self.registry.validation_config_requests(
                experiments=experiments,
                anchor_config=self._best_config_delta_locked(),
            )
            if validation_requests:
                store.enqueue_configs(validation_requests)

            # Validation tests update belief only when a full test completes.
            self.registry.evaluate_validation_tests(experiments)

            belief_engine.on_experiment_complete(experiments, dimensions, self.registry.active)

            # Track improvement trend for stall detection
            top1 = self._top_configs_locked(1)
            if top1 and top1[0].get("delta_bpb") is not None:
                belief_engine.record_best_delta(total_experiments, float(top1[0]["delta_bpb"]))

            # Evaluate canary dimensions for promotion/revert.
            self._evaluate_canary_dimensions_locked(total_experiments)

            eliminated = self._archive_refuted_locked()
            population_changes = self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))

            if self.meta_log.should_checkpoint(total_experiments):
                self.meta_log.write_checkpoint(
                    registry=self.registry,
                    population_manager=self.population_manager,
                    experiment_count=total_experiments,
                    active_workers=store.active_worker_count(),
                    eliminated=eliminated,
                    new_hypotheses=self.pending_new_hypotheses,
                    population_changes=population_changes,
                )
                self.pending_new_hypotheses = []

            self._save_locked()

    def generate_global_program(self, total_experiments: int) -> str:
        with self._lock:
            belief_state = self._build_belief_state_locked()

            # Annotate convergence — requires registry, not available inside engine
            winner = self.registry.convergence_winner()
            belief_state.is_converging = winner is not None

            content = program_writer.generate_program_md(belief_state, store.active_worker_count())
            store.save_program_snapshot(content, total_experiments)

            proposals = program_writer.propose_new_hypotheses(belief_state)
            proposal_payloads = [
                proposal.model_dump() if hasattr(proposal, "model_dump") else proposal.dict()
                for proposal in proposals
            ]
            decisions = self.registry.ingest_llm_proposals(proposal_payloads)
            accepted = [d["proposal"] for d in decisions if d["engine_gate"].get("accepted")]
            if accepted:
                self.pending_new_hypotheses.extend(accepted)
                self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))

            # Stall detection: ask LLM for new dimensions beyond the current search space
            if belief_state.is_stalled:
                existing_names = [d["name"] for d in store.get_dimensions()]
                new_dim_proposals = program_writer.propose_new_dimensions(belief_state, existing_names)
                if new_dim_proposals:
                    serialized = [
                        p.model_dump() if hasattr(p, "model_dump") else p.dict()
                        for p in new_dim_proposals
                    ]

                    adopted_count = 0
                    for p in serialized:
                        sig = self._proposal_signature(p)
                        if not sig:
                            p["signal_count"] = 0
                            p["adopted"] = False
                            p["adoption_reason"] = "missing_signature"
                            continue

                        count = self.dimension_signal_counts.get(sig, 0) + 1
                        self.dimension_signal_counts[sig] = count
                        p["signal_count"] = count

                        valid, reason = self._validate_dimension_proposal_bounds(p)
                        p["adoption_eligible"] = valid and (count >= self.DIM_SIGNAL_THRESHOLD)
                        if valid and count >= self.DIM_SIGNAL_THRESHOLD:
                            adopted, adopt_reason = self._adopt_dimension_locked(p, total_experiments)
                            p["adopted"] = adopted
                            p["adoption_reason"] = adopt_reason
                            if adopted:
                                adopted_count += 1
                        else:
                            p["adopted"] = False
                            p["adoption_reason"] = reason if not valid else "insufficient_repeated_signal"

                    self.pending_dimension_proposals.extend(serialized)
                    log.info(
                        f"Stall detected at n={total_experiments}: "
                        f"{len(new_dim_proposals)} new dimension proposals queued, "
                        f"{adopted_count} adopted to canary"
                    )

            self._save_locked()
            return content

    def _build_belief_state_locked(self):
        return belief_engine.build_belief_state(
            experiments=store.recent_experiments(2000),
            dimensions=store.get_dimensions(),
            hypotheses=self.registry.active,
            pipeline_hit_rate=pipeline.status().get("hit_rate", 0.0),
        )

    def _best_config_delta_locked(self) -> dict:
        top = self._top_configs_locked(1)
        if not top:
            return {}
        return dict(top[0].get("config_delta") or {})

    def _top_configs_locked(self, n: int = 5) -> list[dict]:
        normalized = []
        for exp in store.top_experiments(n):
            config_delta = exp.get("config_delta")
            normalized.append({
                "config_delta": json.loads(config_delta) if isinstance(config_delta, str) else config_delta,
                "delta_bpb": exp.get("delta_bpb"),
            })
        return normalized

    def _archive_refuted_locked(self) -> list[Hypothesis]:
        eliminated = []
        for hypothesis in list(self.registry.active):
            if hypothesis.status == "refuted" and hypothesis.n_experiments >= 12:
                eliminated.append(hypothesis)
                self.registry.archive(hypothesis.id)
        return eliminated

    def _refresh_populations_locked(self, total_workers: int) -> list[str]:
        changes = self.population_manager.sync(
            registry=self.registry,
            total_workers=total_workers,
            top_configs=self._top_configs_locked(5),
            dimensions=store.get_dimensions(),
        )
        self._save_locked()
        return changes

    def _load_locked(self):
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        if "registry" in data:
            self.registry = HypothesisRegistry.from_dict(data["registry"])
        if "population_manager" in data:
            self.population_manager = PopulationManager.from_dict(data["population_manager"])
        self.pending_new_hypotheses = list(data.get("pending_new_hypotheses", []))
        self.pending_dimension_proposals = list(data.get("pending_dimension_proposals", []))
        self.dimension_signal_counts = dict(data.get("dimension_signal_counts", {}))
        self.canary_dimensions = dict(data.get("canary_dimensions", {}))

    def _save_locked(self):
        payload = {
            "registry": self.registry.to_dict(),
            "population_manager": self.population_manager.to_dict(),
            "pending_new_hypotheses": self.pending_new_hypotheses,
            "pending_dimension_proposals": self.pending_dimension_proposals,
            "dimension_signal_counts": self.dimension_signal_counts,
            "canary_dimensions": self.canary_dimensions,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)


runtime_state = RuntimeState()