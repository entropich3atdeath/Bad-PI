"""
worker/report.py

The single function users must call from their train.py to enable early stopping.

USAGE — add these two lines to your train.py:

    from worker.report import report

    # Then inside your training loop, at each evaluation:
    report(val_bpb, progress)
    # progress = elapsed_seconds / total_budget_seconds
    # OR progress = current_step / total_steps

HOW IT WORKS:
  - Sends a compact tick to the meta-agent: {"id": run_id, "p": 0.2, "m": 1.9, "d": -0.05}
  - Meta-agent responds: {} (continue), {"action": "stop"} (kill this run),
    or {"action": "extend", "budget": 420} (extend budget)
  - If the meta-agent says stop, report() raises SystemExit(0) — a clean exit
    that the worker loop detects and treats as early termination
  - If the meta-agent is unreachable, report() silently does nothing —
    it will NEVER crash your training

WITHOUT report():
  - The meta-agent receives no intermediate signal
  - Early stopping is completely disabled for this run
  - You will see: "Early stopping disabled — no intermediate signal"

INITIALISATION:
  report() auto-initialises from environment variables set by the worker loop:
        META_AGENT_URL, META_RUN_ID, META_WORKER_ID, META_WORKER_TOKEN, META_BASELINE_BPB

  These are set automatically when you run worker/run.py.
  For standalone testing: set them manually or call report.init(...) explicitly.
"""
from __future__ import annotations

import os
import time
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── State ─────────────────────────────────────────────────────────────────────

_meta_url:    Optional[str]   = None
_run_id:      Optional[str]   = None
_worker_id:   Optional[str]   = None
_worker_token: Optional[str]  = None
_baseline:    Optional[float] = None
_initialized: bool            = False
_tick_count:  int             = 0
_last_action: Optional[str]   = None


def init(
    meta_url:    str,
    run_id:      str,
    worker_id:   str,
    worker_token: Optional[str],
    baseline_bpb: float,
) -> None:
    """
    Explicitly initialise report(). Called automatically by the worker loop.
    Only needed if you are calling train.py directly without worker/run.py.
    """
    global _meta_url, _run_id, _worker_id, _worker_token, _baseline, _initialized
    _meta_url   = meta_url.rstrip("/")
    _run_id     = run_id
    _worker_id  = worker_id
    _worker_token = worker_token
    _baseline   = baseline_bpb
    _initialized = True


def _auto_init() -> bool:
    """Try to initialise from environment variables."""
    global _initialized
    if _initialized:
        return True
    url  = os.environ.get("META_AGENT_URL")
    rid  = os.environ.get("META_RUN_ID")
    wid  = os.environ.get("META_WORKER_ID")
    token = os.environ.get("META_WORKER_TOKEN")
    base = os.environ.get("META_BASELINE_BPB")
    if url and rid and wid and base:
        try:
            init(url, rid, wid, token, float(base))
            return True
        except Exception:
            pass
    return False


# ── Main function ─────────────────────────────────────────────────────────────

def report(metric: float, progress: float) -> None:
    """
    Report current metric to the meta-agent at this progress point.

    Args:
        metric:   Current validation metric (val_bpb or equivalent). Lower is better.
        progress: Fraction of training budget elapsed. Float from 0.0 to 1.0.
                  Examples:
                    progress = step / total_steps
                    progress = elapsed_seconds / budget_seconds
                    progress = epoch / total_epochs

    Raises:
        SystemExit(0): if the meta-agent decides to kill this run early.
                       This is a clean exit — the worker loop handles it.
    """
    global _tick_count, _last_action

    if not _auto_init():
        # Silently skip — never crash training
        return

    if not _HAS_REQUESTS:
        return

    _tick_count += 1
    progress = max(0.0, min(1.0, float(progress)))
    delta    = metric - (_baseline or 0.0)

    try:
        resp = _requests.post(
            f"{_meta_url}/tick",
            json={
                "id": _run_id,
                "p":  round(progress, 3),
                "m":  round(float(metric), 6),
                "d":  round(delta, 6),
            },
            headers={"X-Worker-Token": _worker_token} if _worker_token else None,
            timeout=3,
        )
        if resp.status_code == 200:
            action = resp.json()
            _last_action = action.get("action", "")

            if _last_action == "stop":
                reason = action.get("reason", "early stopping")
                print(
                    f"\n[meta-agent] Early stop at p={progress:.0%}  |  "
                    f"metric={metric:.4f}  |  reason: {reason}"
                )
                # Store reason in env so worker/run.py can log it
                os.environ["META_LAST_STOP_REASON"] = reason
                os.environ["META_LAST_STOP_P"]      = str(round(progress, 3))
                raise SystemExit(0)

            if _last_action == "extend":
                new_budget = action.get("budget", 420)
                print(f"[meta-agent] Run extended — new budget: {new_budget}s")

    except SystemExit:
        raise   # re-raise — do not swallow the stop signal
    except Exception:
        pass    # never crash training due to network issues


# ── Diagnostics ───────────────────────────────────────────────────────────────

def stats() -> dict:
    """Return current report() statistics for debugging."""
    return {
        "initialized":    _initialized,
        "run_id":         _run_id,
        "worker_id":      _worker_id,
        "has_worker_token": bool(_worker_token),
        "tick_count":     _tick_count,
        "last_action":    _last_action,
        "meta_url":       _meta_url,
    }
