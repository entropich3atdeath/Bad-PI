#!/usr/bin/env python3
"""
simulate.py

Simulates 50-100 concurrent workers locally using asyncio.
No GPU needed — each worker runs a fake metric that follows a noisy trajectory.
Used to test the scheduler, early stopping, and pipeline logic end-to-end.

Usage:
    python simulate.py                   # 50 workers, 3 rounds
    python simulate.py --workers 100 --rounds 5
    python simulate.py --against-server  # run against live meta-agent HTTP server

Note:
    For real worker runs (outside this fake simulator), use the standard
    training entrypoint name: train.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent))


class TraceLogger:
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = Path(file_path).resolve() if file_path else None
        if self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text("")

    @property
    def enabled(self) -> bool:
        return self.file_path is not None

    def log(self, event_type: str, **payload: Any):
        if not self.file_path:
            return
        record = {
            "ts": round(time.time(), 6),
            "event": event_type,
            **payload,
        }
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _redact_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


# ── Fake training metric ──────────────────────────────────────────────────────

def simulate_metric(config_delta: dict, p: float, noise: float = 0.008) -> float:
    """
    Fake val_bpb trajectory for MNIST digits classifier.
    val_bpb = 1 - accuracy, so lower is better.
    Starts near 0.1 (90% accuracy), best configs reach ~0.01 (99% accuracy).
    """
    hidden   = config_delta.get("HIDDEN_SIZE", 128)
    lr       = config_delta.get("LR", 1e-3)
    n_layers = config_delta.get("N_LAYERS", 2)
    baseline = 0.10

    # Larger hidden + more layers = better (up to a point)
    arch_bonus = -0.02 * min(hidden, 256) / 256 - 0.005 * min(n_layers, 4) / 4

    # LR effect: sweet spot around 1e-3
    lr_dist    = abs(math.log10(max(lr, 1e-6)) - math.log10(1e-3))
    lr_penalty = 0.01 * lr_dist

    # Progress: metric improves over training time
    improvement = 0.07 * (1 - math.exp(-4 * p))

    noise_val = random.gauss(0, noise)
    return max(0.001, baseline + arch_bonus + lr_penalty - improvement + noise_val)


def random_config() -> dict:
    """Sample a random config from the MNIST search space."""
    return {
        "LR":           10 ** random.uniform(-4, -1),
        "HIDDEN_SIZE":  random.choice([32, 64, 128, 256, 512]),
        "N_LAYERS":     random.choice([1, 2, 3, 4, 5]),
        "BATCH_SIZE":   random.choice([16, 32, 64, 128, 256]),
        "OPTIMIZER":    random.choice(["adam", "sgd"]),
    }


# ── Local simulation (no HTTP, direct scheduler calls) ───────────────────────

async def run_local_worker(
    worker_id:  str,
    scheduler,
    run_id:     str,
    config:     dict,
    n_steps:    int = 5,   # 5 progress checkpoints
    step_delay: float = 0.02,
    trace:      Optional[TraceLogger] = None,
) -> dict:
    """Simulate one 5-minute training run as a coroutine."""
    if trace:
        trace.log("local_run_started", worker_id=worker_id, run_id=run_id, config=config)

    for step in range(1, n_steps + 1):
        p       = step / n_steps
        metric  = simulate_metric(config, p)
        delta   = metric - 0.10   # baseline = 0.10 (90% accuracy)

        action  = scheduler.update_run(run_id, p=p, metric=metric, delta=delta)
        if trace:
            trace.log(
                "local_tick",
                worker_id=worker_id,
                run_id=run_id,
                progress=round(p, 3),
                metric=round(metric, 6),
                delta=round(delta, 6),
                action=action,
            )
        await asyncio.sleep(step_delay)

        act = action.get("action", "")
        if act == "stop":
            if trace:
                trace.log("local_run_stopped", worker_id=worker_id, run_id=run_id, at_p=round(p, 3), metric=round(metric, 6), action=action)
            return {"id": run_id, "fate": "killed", "at_p": p, "metric": metric}
        if act == "extend":
            # Simulate 2 more steps for extended runs
            for extra_step in range(1, 3):
                ep      = 1.0 + extra_step * 0.1
                emetric = simulate_metric(config, min(ep, 1.2))
                scheduler.update_run(run_id, p=ep, metric=emetric, delta=emetric - 0.10)
                if trace:
                    trace.log(
                        "local_tick_extended",
                        worker_id=worker_id,
                        run_id=run_id,
                        progress=round(ep, 3),
                        metric=round(emetric, 6),
                        delta=round(emetric - 2.0, 6),
                    )
                await asyncio.sleep(step_delay)
            scheduler.complete_run(run_id)
            if trace:
                trace.log("local_run_completed", worker_id=worker_id, run_id=run_id, fate="extended", metric=round(emetric, 6))
            return {"id": run_id, "fate": "extended", "metric": emetric}

    scheduler.complete_run(run_id)
    if trace:
        trace.log("local_run_completed", worker_id=worker_id, run_id=run_id, fate="completed", metric=round(metric, 6))
    return {"id": run_id, "fate": "completed", "metric": metric}


async def simulate_local(n_workers: int = 50, n_rounds: int = 3, trace: Optional[TraceLogger] = None):
    from meta_server.scheduler import RunRegistry

    registry = RunRegistry()
    print(f"\nBad PI local simulator")
    print(f"Workers: {n_workers}   Rounds: {n_rounds}")
    print("=" * 60)

    all_results = []

    for round_n in range(1, n_rounds + 1):
        print(f"\n--- Round {round_n} ---")
        t0 = time.time()
        if trace:
            trace.log("local_round_started", round=round_n, workers=n_workers)

        tasks = []
        for i in range(n_workers):
            config    = random_config()
            run       = registry.start_run(f"worker_{i:03d}", config)
            task      = asyncio.create_task(
                run_local_worker(f"worker_{i:03d}", registry, run.id, config, trace=trace)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

        fates = defaultdict(int)
        for r in results:
            fates[r["fate"]] += 1
        all_results.extend(results)

        stats = registry.stats()
        if trace:
            trace.log("local_round_summary", round=round_n, stats=stats, fates=dict(fates), elapsed=round(elapsed, 4))
        print(f"  Completed: {fates['completed']}  Killed: {fates['killed']}  Extended: {fates['extended']}")
        print(f"  Kill rate: {stats['kill_rate']}%   Best delta: {stats['best_delta']}")
        print(f"  Elapsed:   {elapsed:.2f}s")

        # Print bucket pool percentiles
        for b_str, pool in stats["bucket_pools"].items():
            if pool:
                print(f"  Bucket p={b_str}: P25={pool.get('p25', '?'):.3f}  "
                      f"P50={pool.get('p50', '?'):.3f}  P90={pool.get('p90', '?'):.3f}  "
                      f"n={pool.get('n', 0)}")

    print("\n" + "=" * 60)
    total_killed   = sum(1 for r in all_results if r["fate"] == "killed")
    total_extended = sum(1 for r in all_results if r["fate"] == "extended")
    total          = len(all_results)
    best_metric    = min(r["metric"] for r in all_results)
    if trace:
        trace.log(
            "local_simulation_complete",
            workers=n_workers,
            rounds=n_rounds,
            total_runs=total,
            total_killed=total_killed,
            total_extended=total_extended,
            best_metric=round(best_metric, 6),
        )
    print(f"Overall: {total} runs  |  {total_killed} killed ({total_killed/total*100:.1f}%)"
          f"  |  {total_extended} extended ({total_extended/total*100:.1f}%)")
    print(f"Best metric seen: {best_metric:.4f}  (delta={best_metric - 0.10:+.4f})")


# ── HTTP simulation (against live server) ────────────────────────────────────

async def simulate_http_worker(
    worker_id:  str,
    meta_url:   str,
    session,
    n_rounds:   int = 3,
    enroll_token: str | None = None,
    trace: Optional[TraceLogger] = None,
):
    """Simulate a worker against the live HTTP meta-agent."""
    import aiohttp

    baseline_bpb = 0.10 + random.uniform(-0.02, 0.02)

    # Register
    if trace:
        trace.log(
            "http_register_request",
            worker_id=worker_id,
            meta_url=meta_url,
            payload={
                "worker_id": worker_id,
                "gpu_type": "random-choice",
                "baseline_bpb": round(baseline_bpb, 6),
                "enroll_token": "present" if enroll_token else None,
            },
        )
    async with session.post(f"{meta_url}/register", json={
        "worker_id":    worker_id,
        "gpu_type":     random.choice(["H100", "A100", "RTX3090"]),
        "baseline_bpb": baseline_bpb,
        "enroll_token": enroll_token,
    }) as r:
        if r.status != 200:
            if trace:
                trace.log("http_register_failed", worker_id=worker_id, status=r.status)
            print(f"  {worker_id} registration failed ({r.status})")
            return
        reg = await r.json()
        worker_token = reg.get("worker_token")
        if trace:
            trace.log(
                "http_register_response",
                worker_id=worker_id,
                status=r.status,
                message=reg.get("message"),
                current_program_md=reg.get("current_program_md"),
                worker_token=_redact_token(worker_token),
            )

    headers = {"X-Worker-Token": worker_token} if worker_token else {}

    for _ in range(n_rounds):
        # Get next config
        if trace:
            trace.log("http_next_config_request", worker_id=worker_id)
        async with session.get(f"{meta_url}/next_config/{worker_id}", headers=headers) as r:
            if r.status != 200:
                if trace:
                    trace.log("http_next_config_failed", worker_id=worker_id, status=r.status)
                await asyncio.sleep(2)
                continue
            task = await r.json()
            if trace:
                trace.log("http_next_config_response", worker_id=worker_id, status=r.status, response=task)

        run_id = task["exp_id"]
        config = task["config_delta"]

        # Simulate ticks at 5 progress points
        for step in range(1, 6):
            p       = step / 5
            metric  = simulate_metric(config, p)
            delta   = metric - baseline_bpb

            if trace:
                trace.log(
                    "http_tick_request",
                    worker_id=worker_id,
                    run_id=run_id,
                    payload={"id": run_id, "p": round(p, 4), "m": round(metric, 4), "d": round(delta, 4)},
                )
            async with session.post(f"{meta_url}/tick", headers=headers, json={
                "id": run_id, "p": p, "m": round(metric, 4), "d": round(delta, 4)
            }) as r:
                if r.status == 200:
                    resp = await r.json()
                    if trace:
                        trace.log("http_tick_response", worker_id=worker_id, run_id=run_id, status=r.status, response=resp)
                    if resp.get("action") == "stop":
                        break
                    if resp.get("action") == "extend":
                        # Two more ticks
                        for ep_step in range(1, 3):
                            ep      = 1.0 + ep_step * 0.1
                            em      = simulate_metric(config, 1.0) - 0.01 * ep_step
                            async with session.post(f"{meta_url}/tick", headers=headers, json={
                                "id": run_id, "p": round(ep, 2),
                                "m": round(em, 4), "d": round(em - baseline_bpb, 4)
                            }) as _:
                                if trace:
                                    trace.log(
                                        "http_tick_request_extended",
                                        worker_id=worker_id,
                                        run_id=run_id,
                                        payload={"id": run_id, "p": round(ep, 2), "m": round(em, 4), "d": round(em - baseline_bpb, 4)},
                                    )
                                pass
                        break

            await asyncio.sleep(0.05)

        # Submit final result
        result_payload = {
            "worker_id":      worker_id,
            "exp_id":         run_id,
            "config":         config,
            "config_delta":   config,
            "val_bpb":        metric,
            "delta_bpb":      delta,
            "duration_seconds": 60,
        }
        if trace:
            trace.log("http_result_request", worker_id=worker_id, run_id=run_id, payload=result_payload)
        async with session.post(f"{meta_url}/result", headers=headers, json=result_payload) as r:
            if trace:
                try:
                    result_response = await r.json()
                except Exception:
                    result_response = None
                trace.log("http_result_response", worker_id=worker_id, run_id=run_id, status=r.status, response=result_response)

        async with session.get(f"{meta_url}/sync/{worker_id}", headers=headers) as r:
            if r.status == 200:
                sync_state = await r.json()
                if trace:
                    trace.log(
                        "http_sync_response",
                        worker_id=worker_id,
                        run_id=run_id,
                        status=r.status,
                        response={
                            "experiment_count": sync_state.get("experiment_count"),
                            "active_workers":   sync_state.get("active_workers"),
                            "population_id":    sync_state.get("population_id"),
                            "population_strategy": sync_state.get("population_strategy"),
                            "hypothesis_id":    sync_state.get("hypothesis_id"),
                            "hypothesis_statement": sync_state.get("hypothesis_statement"),
                            "top_configs":      sync_state.get("top_configs"),
                            "program_md":       sync_state.get("program_md"),
                        },
                    )
            elif trace:
                trace.log("http_sync_failed", worker_id=worker_id, run_id=run_id, status=r.status)

        await asyncio.sleep(0.1)


async def simulate_against_server(
    n_workers: int,
    n_rounds: int,
    meta_url: str,
    enroll_token: str | None = None,
    trace: Optional[TraceLogger] = None,
):
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        return

    print(f"\nSimulating {n_workers} workers against {meta_url}")
    print(f"Rounds per worker: {n_rounds}")
    print("=" * 60)
    if trace:
        trace.log("http_simulation_started", workers=n_workers, rounds=n_rounds, meta_url=meta_url)

    async with aiohttp.ClientSession() as session:
        tasks = [
            simulate_http_worker(f"sim_{i:03d}", meta_url, session, n_rounds, enroll_token, trace)
            for i in range(n_workers)
        ]
        await asyncio.gather(*tasks)

    if trace:
        trace.log("http_simulation_complete", workers=n_workers, rounds=n_rounds, meta_url=meta_url)
    print("\nDone. Check leaderboard at /leaderboard")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Bad PI local simulator")
    p.add_argument("--workers", type=int, default=50)
    p.add_argument("--rounds",  type=int, default=3)
    p.add_argument("--against-server", action="store_true")
    p.add_argument("--meta-url", default="http://localhost:8000")
    p.add_argument("--enroll-token", default=None)
    p.add_argument("--trace-file", default=None,
                   help="Optional JSONL file to record simulation events and server responses")
    args = p.parse_args()

    trace = TraceLogger(args.trace_file)
    if trace.enabled:
        print(f"Writing simulation trace to {trace.file_path}")

    if args.against_server:
        asyncio.run(simulate_against_server(args.workers, args.rounds, args.meta_url, args.enroll_token, trace))
    else:
        asyncio.run(simulate_local(args.workers, args.rounds, trace))


if __name__ == "__main__":
    main()
