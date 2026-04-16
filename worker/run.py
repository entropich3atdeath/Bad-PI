#!/usr/bin/env python3
"""
worker/run.py

The main worker loop. Run this after setup_worker.py.
Infinite loop:
  1. Pull next config_delta from meta-agent
  2. Patch train.py
  3. Run train.py for 5 min
  4. Parse val_bpb from output
  5. Push result to meta-agent
  6. Sync program.md
  7. Restore train.py
  8. Repeat

Usage:
    python worker/run.py [--max-runs N]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from worker.client import (
    get_next_config,
    get_next_config_batch,
    submit_result,
    sync,
    get_spec_payload,
    get_flush_token,
    start_run,
    cancel_leases,
)
from worker.patcher import apply_delta, restore_backup, read_current_config

CFG_PATH = Path(__file__).parent / ".worker_config.json"
TRACKED_KEYS = [
    "LR", "BATCH_SIZE", "HIDDEN_SIZE", "N_LAYERS", "WEIGHT_DECAY", "OPTIMIZER",
    "TOTAL_WALL_CLOCK_TIME",   # ← agent-controlled budget; must exist in train.py
]

# Per-run stop reason log (worker-local, written to stop_reasons.jsonl)
STOP_LOG_PATH = Path("stop_reasons.jsonl")


def log_stop_reason(exp_id: str, reason: str, p: float, metric: float):
    """Append a stop reason to the local JSONL log for later analysis."""
    import json as _json
    entry = {"exp_id": exp_id, "reason": reason, "at_p": round(p, 2),
             "metric": round(metric, 4), "ts": time.time()}
    with STOP_LOG_PATH.open("a") as f:
        f.write(_json.dumps(entry) + "\n")


def load_config() -> dict:
    if not CFG_PATH.exists():
        print("ERROR: .worker_config.json not found. Run setup_worker.py first.")
        sys.exit(1)
    return json.loads(CFG_PATH.read_text())


def run_training(train_py: Path) -> tuple[float | None, float, str]:
    """
    Run train.py, return (val_bpb, duration_seconds, full_output).
    val_bpb is None if parsing fails or training crashes.
    """
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(train_py)],
            capture_output=True, text=True,
            cwd=train_py.parent,
            timeout=600,  # 10 min hard timeout
        )
        output = result.stdout + result.stderr
        duration = time.time() - t0

        # Parse val_bpb — handles formats like:
        #   val_bpb: 1.2345  |  val bpb 1.2345  |  "val_bpb=1.2345"
        m = re.search(r"val[_\s]bpb[:\s=]+([0-9.]+)", output, re.IGNORECASE)
        if m:
            return float(m.group(1)), duration, output
        return None, duration, output

    except subprocess.TimeoutExpired:
        return None, time.time() - t0, "TIMEOUT"
    except Exception as e:
        return None, time.time() - t0, str(e)


def _handle_spec_pipeline(
    worker_id: str,
    worker_token: Optional[str],
    cached_spec_id: Optional[str],
) -> Optional[str]:
    try:
        flush_token = get_flush_token()
        if flush_token and cached_spec_id and flush_token == cached_spec_id:
            print(f"  [spec] Flush received for spec_id={cached_spec_id}; reverting to synced program.md")
            cached_spec_id = None
            state = sync(worker_id, worker_token)
            Path("program.md").write_text(state["program_md"])
    except Exception:
        pass

    try:
        spec = get_spec_payload()
        if spec:
            cached_spec_id = spec.get("spec_id")
            Path("program.md").write_text(spec.get("program_md", ""))
            print(
                "  [spec] Using speculative program.md "
                f"(id={cached_spec_id}, conf={spec.get('confidence')}, "
                f"deploy_conf={spec.get('deployment_confidence')})"
            )
    except Exception:
        pass

    return cached_spec_id


def _state_changed(sync_state: dict, batch_meta: dict) -> bool:
    return any(
        str(sync_state.get(key) or "") != str(batch_meta.get(key) or "")
        for key in ("program_digest", "population_id", "population_strategy", "hypothesis_id")
    )


def _run_one_task(
    *,
    cfg: dict,
    task: dict,
    run_n: int,
    worker_id: str,
    worker_token: Optional[str],
    baseline: float,
    train_py: Path,
) -> dict:
    config_delta = task["config_delta"]
    exp_id = task["exp_id"]
    budget = task.get("budget_seconds", 300)

    print(f"\n{'='*60}")
    print(f"  Run #{run_n}  worker={worker_id}")
    print(f"{'='*60}")
    print(f"  Config: {json.dumps(config_delta, indent=4)}")
    print(f"  Budget: {budget}s ({budget//60}m {budget%60}s)  Note: {task.get('note', '')}")
    if task.get("population_id"):
        print(
            "  Population: "
            f"{task.get('population_id')} · {task.get('population_strategy', 'investigate')}"
        )
    if task.get("hypothesis_statement"):
        print(f"  Hypothesis: {task['hypothesis_statement']}")

    print("  → Activating run lease…")
    try:
        start_run(
            worker_id,
            worker_token,
            config_delta=config_delta,
            population_id=task.get("population_id") or "default",
            run_id=exp_id,
            budget_seconds=budget,
        )
    except Exception as e:
        print(f"  ERROR activating run: {e}")
        raise

    config_delta_with_budget = {**config_delta, "TOTAL_WALL_CLOCK_TIME": budget}
    current_config = read_current_config(train_py, TRACKED_KEYS)

    print("  → Patching train.py…")
    old_vals = apply_delta(train_py, config_delta_with_budget, backup=True)
    print(f"  Changed: {old_vals} → {config_delta_with_budget}")

    os.environ["META_RUN_ID"] = exp_id
    os.environ["META_WORKER_ID"] = worker_id
    os.environ["META_WORKER_TOKEN"] = worker_token or ""
    os.environ["META_BASELINE_BPB"] = str(baseline)
    os.environ["META_AGENT_URL"] = cfg["meta_url"]

    print("  → Running training…")
    val_bpb, duration, output = run_training(train_py)

    stop_reason = os.environ.pop("META_LAST_STOP_REASON", None)
    stop_p = float(os.environ.pop("META_LAST_STOP_P", "0") or "0")
    if stop_reason:
        print(f"  Stop reason: {stop_reason}")
        log_stop_reason(exp_id, stop_reason, stop_p, val_bpb or 999)

    try:
        import requests as _req
        tick_resp = _req.get(f"{cfg['meta_url']}/runs/active", timeout=3)
        run_info = next((r for r in tick_resp.json() if r.get("id") == exp_id), None)
        if run_info and run_info.get("progress", 0) == 0:
            print()
            print("  " + "!" * 55)
            print("  WARNING: No intermediate ticks received for this run.")
            print("  Early stopping disabled — no intermediate signal.")
            print()
            print("  Add to your train.py:")
            print("    from worker.report import report")
            print("    report(val_bpb, progress)  # at each eval step")
            print("  " + "!" * 55)
            print()
    except Exception:
        pass

    delta_bpb = (val_bpb - baseline) if val_bpb is not None else None
    if val_bpb is not None:
        symbol = "▼" if delta_bpb < 0 else "▲"
        print(f"  {symbol} val_bpb={val_bpb:.4f}  delta={delta_bpb:+.4f}  ({duration:.0f}s)")
    else:
        print(f"  ✗ Training failed or val_bpb not found  ({duration:.0f}s)")
        print(f"  Output tail: {output[-500:]}")

    restore_backup(train_py)

    print("  → Submitting result…")
    full_config = {**current_config, **config_delta}
    try:
        submit_result(
            worker_id=worker_id,
            worker_token=worker_token,
            exp_id=exp_id,
            config=full_config,
            config_delta=config_delta,
            val_bpb=val_bpb,
            delta_bpb=delta_bpb,
            duration_seconds=duration,
            error=None if val_bpb is not None else output[-500:],
        )
    except Exception as e:
        print(f"  WARNING: could not submit result: {e}")

    return task


def run_loop(
    cfg: dict,
    max_runs: int | None = None,
    use_spec_pipeline: bool = False,
    micro_batch_size: int = 3,
):
    worker_id   = cfg["worker_id"]
    worker_token = cfg.get("worker_token")
    baseline    = cfg["baseline_bpb"]
    train_py    = Path(cfg["train_py"])
    os.environ["META_AGENT_URL"] = cfg["meta_url"]

    run_n = 0
    cached_spec_id = None
    last_program_digest: str = ""
    while max_runs is None or run_n < max_runs:
        requested = max(1, micro_batch_size)
        print(f"  → Pulling next config batch (n={requested})…")
        try:
            if requested > 1:
                batch = get_next_config_batch(worker_id, worker_token, n=requested)
                tasks = batch.get("configs") or []
                batch_meta = {
                    "program_digest": batch.get("program_digest", ""),
                    "population_id": batch.get("population_id"),
                    "population_strategy": batch.get("population_strategy"),
                    "hypothesis_id": batch.get("hypothesis_id"),
                }
            else:
                task = get_next_config(worker_id, worker_token)
                tasks = [task]
                batch_meta = {
                    "program_digest": "",
                    "population_id": task.get("population_id"),
                    "population_strategy": task.get("population_strategy"),
                    "hypothesis_id": task.get("hypothesis_id"),
                }
        except Exception as e:
            print(f"  ERROR pulling config: {e}. Sleeping 30s…")
            time.sleep(30)
            continue

        if not tasks:
            print("  No configs returned in batch; retrying in 15s…")
            time.sleep(15)
            continue

        remaining_leases = [task.get("lease_id") or task["exp_id"] for task in tasks]
        for task in tasks:
            if max_runs is not None and run_n >= max_runs:
                break
            run_n += 1

            if use_spec_pipeline:
                cached_spec_id = _handle_spec_pipeline(worker_id, worker_token, cached_spec_id)

            task_lease_id = task.get("lease_id") or task["exp_id"]
            try:
                _run_one_task(
                    cfg=cfg,
                    task=task,
                    run_n=run_n,
                    worker_id=worker_id,
                    worker_token=worker_token,
                    baseline=baseline,
                    train_py=train_py,
                )
            finally:
                if task_lease_id in remaining_leases:
                    remaining_leases.remove(task_lease_id)

            print("  → Syncing program.md…")
            try:
                state = sync(worker_id, worker_token)
                incoming_program = str(state.get("program_md") or "")
                incoming_digest = str(state.get("program_digest") or "")

                if incoming_program.strip() and incoming_digest and incoming_digest != last_program_digest:
                    prog_path = Path("program.md")
                    prog_path.write_text(incoming_program)
                    last_program_digest = incoming_digest
                    print(f"  program.md updated from agent (digest={incoming_digest}).")
                elif not incoming_program.strip():
                    print("  No new program.md yet; keeping local base template.")
                else:
                    print("  program.md unchanged.")

                frozen = [d for d in state["dimensions"] if d["frozen"]]
                print(
                    f"  Swarm: {state['experiment_count']} experiments · "
                    f"{state['active_workers']} workers · {len(frozen)} frozen dims"
                )
                if state.get("population_id"):
                    print(
                        "  Assigned population: "
                        f"{state['population_id']} · {state.get('population_strategy', 'investigate')}"
                    )
                if state.get("hypothesis_statement"):
                    print(f"  Current hypothesis: {state['hypothesis_statement']}")
                if state["top_configs"]:
                    best = state["top_configs"][0]
                    print(f"  Best ever: delta={best['delta_bpb']:+.4f}")

                if remaining_leases and _state_changed(state, batch_meta):
                    print("  State changed after sync; canceling remaining leased configs…")
                    try:
                        cancel_leases(worker_id, worker_token, remaining_leases, reason="worker_state_changed")
                    except Exception as e:
                        print(f"  WARNING: could not cancel remaining leases: {e}")
                    remaining_leases = []
                    break
            except Exception as e:
                print(f"  WARNING: sync failed: {e}")

            print("  Done. Starting next run…")
            time.sleep(2)

        if remaining_leases and max_runs is not None and run_n >= max_runs:
            try:
                cancel_leases(worker_id, worker_token, remaining_leases, reason="worker_max_runs_reached")
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=None,
                   help="Stop after this many runs (default: infinite)")
    p.add_argument(
        "--use-spec-pipeline",
        action="store_true",
        help="Enable speculative program preload (/pipeline/spec_payload + /pipeline/flush_token)",
    )
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=3,
        help="Lease this many configs at once (default: 3, use 1 to disable micro-batching)",
    )
    args = p.parse_args()

    cfg = load_config()
    print(f"Starting worker '{cfg['worker_id']}' ({cfg['gpu_type']})")
    print(f"Meta-agent: {cfg['meta_url']}")
    print(f"Baseline:   {cfg['baseline_bpb']:.4f} val_bpb")
    print()
    run_loop(
        cfg,
        args.max_runs,
        use_spec_pipeline=args.use_spec_pipeline,
        micro_batch_size=max(1, args.micro_batch_size),
    )


if __name__ == "__main__":
    main()
