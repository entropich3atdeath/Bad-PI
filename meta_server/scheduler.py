"""
meta_server/scheduler.py

In-memory run lifecycle management + early stopping.
No database — all state lives in RunRegistry.

Key ideas:
  - Every run reports at 5 progress buckets (p=0.2..1.0)
  - At each bucket, the run's metric is compared against all other runs at that bucket
  - Bottom P25 → kill immediately  (action: "stop")
  - Top P90 at final bucket → extend budget (action: "extend")
  - Otherwise → continue         (action: {})

Compact wire protocol:
  Worker sends:  {"id": "run_uuid", "p": 0.2, "m": 1.9, "d": -0.05}
  Agent returns: {"action": "stop"} | {"action": "extend", "budget": 420} | {}
"""
from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

BUCKETS           = [0.2, 0.4, 0.6, 0.8, 1.0]
BUCKET_TOL        = 0.05    # p within ±0.05 of a bucket triggers comparison
KILL_PERCENTILE   = 25      # kill if metric rank < P25 at any checkpoint
EXTEND_PERCENTILE = 90      # extend if metric rank > P90 at final bucket
BASE_BUDGET       = 300     # 5 minutes in seconds
EXTEND_BY         = 120     # extend by 2 minutes
MIN_POOL_SIZE     = 5       # need at least this many runs at bucket before killing

WARMUP_RUNS       = 25      # NO early stopping until this many runs complete
                            # Rationale: need a baseline distribution to compare against.
                            # Killing runs before the pool is calibrated produces false
                            # positives — the first run always looks like an outlier.

NO_TICKS_WARNING  = (
    "\n  [meta-agent] WARNING: No intermediate ticks received for this run.\n"
    "  Early stopping is DISABLED — no intermediate signal.\n"
    "  To enable early stopping, add to your train.py:\n"
    "    from worker.report import report\n"
    "  Then call at each eval step:\n"
    "    report(val_bpb, progress)   # progress = 0.0 to 1.0\n"
)


# ── Data structures ───────────────────────────────────────────────────────────

class RunState(str, Enum):
    ACTIVE    = "active"
    STOPPED   = "stopped"     # killed by early stopping
    EXTENDED  = "extended"    # running beyond base budget
    COMPLETED = "completed"   # finished normally or after extension
    FAILED    = "failed"


@dataclass
class BucketSnapshot:
    p:         float
    metric:    float
    delta:     float    # metric - worker baseline
    timestamp: float    = field(default_factory=time.time)


@dataclass
class Run:
    id:             str
    worker_id:      str
    population_id:  str
    config_delta:   dict
    budget_seconds: float      = BASE_BUDGET
    state:          RunState   = RunState.ACTIVE
    start_time:     float      = field(default_factory=time.time)
    buckets:        list[BucketSnapshot] = field(default_factory=list)
    last_metric:    Optional[float] = None
    last_p:         float      = 0.0
    stop_reason:    str        = ""
    tick_count:     int        = 0    # how many ticks received — 0 means no report() calls

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def timed_out(self) -> bool:
        return self.elapsed > self.budget_seconds * 1.1   # 10% grace

    @property
    def has_intermediate_signal(self) -> bool:
        """False if worker never called report() — early stopping cannot operate."""
        return self.tick_count > 0

    def latest_bucket(self) -> Optional[BucketSnapshot]:
        return self.buckets[-1] if self.buckets else None


# ── Early stopping logic ──────────────────────────────────────────────────────

class EarlyStopper:
    """
    At each progress bucket, compares a run's metric against the pool of all
    runs that have reported at that bucket.

    Metric convention: lower is better (val_bpb / loss).
    A run is "good" if its metric is LOW.
    """

    def __init__(self):
        # bucket_p → list of (metric, run_id) for all runs that reached p
        self._pool: dict[float, list[tuple[float, str]]] = defaultdict(list)

    def register(self, p: float, metric: float, run_id: str):
        bucket = self._nearest_bucket(p)
        if bucket is not None:
            self._pool[bucket].append((metric, run_id))

    def evaluate(self, p: float, metric: float, run_id: str) -> str:
        """
        Returns "stop", "extend", or "" (continue).
        """
        bucket = self._nearest_bucket(p)
        if bucket is None:
            return ""

        pool = [m for m, rid in self._pool[bucket] if rid != run_id]
        if len(pool) < MIN_POOL_SIZE:
            return ""    # not enough data yet

        rank_pct = self._percentile_rank(metric, pool)

        # Lower metric = better, so low rank = bad
        if rank_pct < KILL_PERCENTILE:
            return "stop"

        # At the final bucket, extend top performers
        if bucket == BUCKETS[-1] and rank_pct >= EXTEND_PERCENTILE:
            return "extend"

        return ""

    def _nearest_bucket(self, p: float) -> Optional[float]:
        for b in BUCKETS:
            if abs(p - b) <= BUCKET_TOL:
                return b
        return None

    def _percentile_rank(self, metric: float, pool: list[float]) -> float:
        """
        Percentile rank = fraction of pool values WORSE than (higher than) metric.
        0 = worst (our metric is the highest = worst), 100 = best.
        """
        n = len(pool)
        worse = sum(1 for m in pool if m > metric)
        return (worse / n) * 100.0

    def pool_percentiles(self, p: float) -> dict:
        """Return P25/P50/P75/P90 for a given bucket (for diagnostics)."""
        bucket = self._nearest_bucket(p)
        if bucket is None or not self._pool[bucket]:
            return {}
        vals = sorted(m for m, _ in self._pool[bucket])
        n = len(vals)
        def pct(k): return vals[min(n - 1, int(k / 100 * n))]
        return {"p25": pct(25), "p50": pct(50), "p75": pct(75), "p90": pct(90), "n": n}


# ── Run registry ──────────────────────────────────────────────────────────────

class RunRegistry:
    """
    Central in-memory store for all run state.
    Thread-safety: the FastAPI event loop is single-threaded so no locks needed
    for a pure asyncio deployment. Add threading.Lock if using threads.
    """

    def __init__(self):
        self.active_runs:    dict[str, Run] = {}
        self.completed_runs: list[Run]      = []
        self.stopper                        = EarlyStopper()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start_run(
        self,
        worker_id:     str,
        config_delta:  dict,
        population_id: str = "default",
        budget:        float = BASE_BUDGET,
        run_id:        Optional[str] = None,
    ) -> Run:
        if run_id and run_id in self.active_runs:
            return self.active_runs[run_id]

        run = Run(
            id            = run_id or str(uuid.uuid4()),
            worker_id     = worker_id,
            population_id = population_id,
            config_delta  = config_delta,
            budget_seconds= budget,
        )
        self.active_runs[run.id] = run
        return run

    def update_run(self, run_id: str, p: float, metric: float, delta: float) -> dict:
        """
        Process a tick from a worker.
        Returns the action dict to send back.
        """
        run = self.active_runs.get(run_id)
        if run is None:
            return {"action": "stop", "reason": "unknown_run"}

        if run.state not in (RunState.ACTIVE, RunState.EXTENDED):
            return {"action": "stop", "reason": "run_not_active"}

        # Record snapshot
        snap = BucketSnapshot(p=p, metric=metric, delta=delta)
        run.buckets.append(snap)
        run.last_metric = metric
        run.last_p      = p
        run.tick_count += 1
        self.stopper.register(p, metric, run_id)

        # Check timed out
        if run.timed_out:
            return self._complete(run)

        # ── Warmup guard ──────────────────────────────────────────────────
        # No early stopping until WARMUP_RUNS completed runs exist.
        # Without a baseline pool, percentile comparisons are meaningless.
        if len(self.completed_runs) < WARMUP_RUNS:
            return {}   # always continue during warmup

        # Early stopping decision
        action = self.stopper.evaluate(p, metric, run_id)

        if action == "stop":
            run.state       = RunState.STOPPED
            run.stop_reason = f"killed at p={p:.2f} (below P{KILL_PERCENTILE})"
            self._archive(run)
            return {"action": "stop", "reason": run.stop_reason}

        if action == "extend":
            run.state          = RunState.EXTENDED
            run.budget_seconds += EXTEND_BY
            return {"action": "extend", "budget": int(run.budget_seconds)}

        return {}    # continue

    def stop_run(self, run_id: str, reason: str = "manual") -> bool:
        run = self.active_runs.get(run_id)
        if run is None:
            return False
        run.state       = RunState.STOPPED
        run.stop_reason = reason
        self._archive(run)
        return True

    def complete_run(self, run_id: str) -> Optional[Run]:
        run = self.active_runs.get(run_id)
        if run is None:
            return None
        return self._complete(run)

    def replace_run(
        self,
        run_id:        str,
        new_config:    dict,
        population_id: str = "default",
    ) -> Optional[Run]:
        """Stop an existing run and immediately start a replacement."""
        old = self.active_runs.get(run_id)
        if old:
            self.stop_run(run_id, reason="replaced")
        return self.start_run(old.worker_id if old else "unknown", new_config, population_id)

    def _complete(self, run: Run) -> dict:
        run.state = RunState.COMPLETED
        self._archive(run)
        return {}

    def _archive(self, run: Run):
        self.active_runs.pop(run.id, None)
        self.completed_runs.append(run)
        # Keep completed list bounded
        if len(self.completed_runs) > 2000:
            self.completed_runs = self.completed_runs[-1000:]

    # ── Queries ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        completed  = self.completed_runs
        n_active   = len(self.active_runs)
        n_done     = len(completed)
        n_killed   = sum(1 for r in completed if r.state == RunState.STOPPED)
        n_extended = sum(1 for r in completed if r.state == RunState.EXTENDED)
        deltas     = [r.latest_bucket().delta for r in completed if r.latest_bucket()]
        best       = min(deltas) if deltas else None

        return {
            "active":    n_active,
            "completed": n_done,
            "killed":    n_killed,
            "extended":  n_extended,
            "kill_rate": round(n_killed / max(n_done, 1) * 100, 1),
            "best_delta": round(best, 4) if best is not None else None,
            "bucket_pools": {
                str(b): self.stopper.pool_percentiles(b)
                for b in BUCKETS
            },
        }

    def top_runs(self, n: int = 10) -> list[Run]:
        all_runs = list(self.active_runs.values()) + self.completed_runs
        scored = [r for r in all_runs if r.last_metric is not None]
        return sorted(scored, key=lambda r: r.last_metric)[:n]


# ── Module-level singleton (used by api.py) ───────────────────────────────────

registry = RunRegistry()
