"""
meta_server/search.py

Three components:
  1. ThompsonSampler  — proposes next configs to try
  2. fANOVA           — scores dimension importance from completed experiments
  3. ASHAPromoter     — promotes good config neighborhoods, culls bad ones

All three run in the background and write to the store.
"""
from __future__ import annotations
import json
import logging
import math
import random
import time
from typing import Any

import numpy as np

from . import store

log = logging.getLogger(__name__)


# ── Dimension helpers ─────────────────────────────────────────────────────────

def _sample_value(dim: dict, rng: random.Random) -> Any:
    """Sample a single value from a dimension's current distribution."""
    if dim["frozen"]:
        return dim["frozen_value"]
    dtype = dim["dtype"]
    if dtype == "categorical":
        return rng.choice(dim["categories"])
    if dtype in ("float", "float_log"):
        lo, hi = dim["min_val"], dim["max_val"]
        if dim["log_scale"]:
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))
        return rng.uniform(lo, hi)
    if dtype == "int":
        return rng.randint(int(dim["min_val"]), int(dim["max_val"]))
    raise ValueError(f"Unknown dtype: {dtype}")


def _round_value(val: Any, dim: dict) -> Any:
    if dim["dtype"] == "int":
        return int(round(val))
    if dim["dtype"] in ("float", "float_log"):
        # Round to 4 significant figures
        if val == 0:
            return 0.0
        mag = math.floor(math.log10(abs(val)))
        return round(val, -int(mag) + 3)
    return val


# ── Thompson Sampler ──────────────────────────────────────────────────────────

class ThompsonSampler:
    """
    Per-dimension Thompson Sampling using a simple model:
      - For each (dimension, bucket), track (sum_delta, count)
      - At sample time, draw a score for each bucket from a Normal posterior,
        then pick the bucket with the best (lowest) draw.
    
    For continuous dimensions we discretise into N_BUCKETS bins.
    """
    N_BUCKETS = 8
    PRIOR_MEAN = 0.0    # assume new configs are neutral
    PRIOR_STD  = 0.02   # mild uncertainty prior

    def __init__(self):
        # {dim_name: {bucket_key: [delta_bpb values]}}
        self._data: dict[str, dict[str, list[float]]] = {}

    def ingest(self, experiments: list[dict]):
        """Update internal model from a batch of completed experiments."""
        self._data.clear()
        for exp in experiments:
            if exp["delta_bpb"] is None:
                continue
            delta = json.loads(exp["config_delta"])
            for k, v in delta.items():
                bucket = self._bucketise(k, v)
                self._data.setdefault(k, {}).setdefault(bucket, []).append(exp["delta_bpb"])

    def _bucketise(self, dim_name: str, value: Any) -> str:
        """Convert a raw value to a discrete bucket key."""
        dims = {d["name"]: d for d in store.get_dimensions()}
        dim = dims.get(dim_name)
        if dim is None or dim["dtype"] == "categorical":
            return str(value)
        lo, hi = dim["min_val"], dim["max_val"]
        if dim["log_scale"] and lo > 0:
            lo, hi = math.log(lo), math.log(hi)
            v = math.log(float(value)) if float(value) > 0 else lo
        else:
            v = float(value)
        idx = int((v - lo) / (hi - lo) * self.N_BUCKETS)
        idx = max(0, min(self.N_BUCKETS - 1, idx))
        return f"bin_{idx}"

    def _thompson_draw(self, values: list[float]) -> float:
        """Draw from Normal posterior given observed delta_bpb values."""
        n = len(values)
        if n == 0:
            return random.gauss(self.PRIOR_MEAN, self.PRIOR_STD)
        mu = sum(values) / n
        # Posterior mean shrinks toward prior with small n
        prior_weight = 2
        shrunk_mu = (mu * n + self.PRIOR_MEAN * prior_weight) / (n + prior_weight)
        posterior_std = self.PRIOR_STD / math.sqrt(n + 1)
        return random.gauss(shrunk_mu, posterior_std)

    def propose(self, dims: list[dict], rng: random.Random) -> tuple[dict, float]:
        """
        Returns (config_delta, estimated_priority).
        priority is 0-1 where 1 = looks most promising.
        """
        delta = {}
        score_sum = 0.0
        active_dims = [d for d in dims if not d["frozen"] and d["importance"] > 0.05]
        # Fallback: if every dimension is currently low-importance, still sample
        # from unfrozen dimensions so the queue never degenerates to empty config {}.
        if not active_dims:
            active_dims = [d for d in dims if not d["frozen"]]

        for dim in active_dims:
            name = dim["name"]

            # Canary dimensions are sampled on only a small share of configs.
            if bool(dim.get("is_canary")):
                p = float(dim.get("canary_prob", 0.12) or 0.12)
                if rng.random() > max(0.0, min(1.0, p)):
                    continue

            buckets = self._data.get(name, {})

            if not buckets:
                # No data yet — pure random sample
                val = _sample_value(dim, rng)
                delta[name] = _round_value(val, dim)
                continue

            # Thompson: draw a score per bucket, pick the winner
            best_bucket_draw = float("inf")
            best_bucket_key = None
            for bkey, vals in buckets.items():
                draw = self._thompson_draw(vals)
                if draw < best_bucket_draw:
                    best_bucket_draw = draw
                    best_bucket_key = bkey
            score_sum += best_bucket_draw

            # Sample a value within the winning bucket
            val = _sample_within_bucket(dim, best_bucket_key, self.N_BUCKETS, rng)
            delta[name] = _round_value(val, dim)

        # Also include frozen dims at their frozen value
        for dim in dims:
            if dim["frozen"] and dim["name"] not in delta:
                delta[dim["name"]] = dim["frozen_value"]

        # Convert score to 0-1 priority (lower delta = higher priority)
        priority = max(0.0, min(1.0, 0.5 - score_sum / max(len(active_dims), 1) * 10))
        return delta, priority


def _sample_within_bucket(dim: dict, bucket_key: str, n_buckets: int, rng: random.Random) -> Any:
    if dim["dtype"] == "categorical" or not bucket_key.startswith("bin_"):
        # Categorical: just sample from all categories (Thompson already picked bucket = value)
        if dim["dtype"] == "categorical":
            return rng.choice(dim["categories"])
        return _sample_value(dim, rng)

    idx = int(bucket_key.split("_")[1])
    lo, hi = dim["min_val"], dim["max_val"]
    is_log = bool(dim["log_scale"]) and lo > 0
    if is_log:
        lo, hi = math.log(lo), math.log(hi)
    bin_width = (hi - lo) / n_buckets
    bin_lo = lo + idx * bin_width
    bin_hi = bin_lo + bin_width
    v = rng.uniform(bin_lo, bin_hi)
    if is_log:
        v = math.exp(v)
    if dim["dtype"] == "int":
        return int(round(v))
    return v


# ── fANOVA ────────────────────────────────────────────────────────────────────

class fANOVA:
    """
    Simplified marginal fANOVA: for each dimension, measure how much of the
    total variance in delta_bpb is explained by varying that dimension alone.

    importance(d) = Var_d[ E[delta_bpb | d=v] ] / Var[delta_bpb]

    Uses at least MIN_SAMPLES experiments; runs every EVAL_EVERY experiments.
    """
    MIN_SAMPLES = 30
    EVAL_EVERY  = 50

    def run(self, experiments: list[dict], dims: list[dict]) -> dict[str, float]:
        completed = [e for e in experiments if e["delta_bpb"] is not None]
        if len(completed) < self.MIN_SAMPLES:
            return {}

        deltas = np.array([e["delta_bpb"] for e in completed])
        total_var = float(np.var(deltas))
        if total_var < 1e-10:
            return {}

        importance: dict[str, float] = {}
        for dim in dims:
            name = dim["name"]
            bucket_means = {}
            for exp in completed:
                cdelta = json.loads(exp["config_delta"])
                if name not in cdelta:
                    continue
                bucket = _bucketise_simple(name, cdelta[name], dim)
                bucket_means.setdefault(bucket, []).append(exp["delta_bpb"])

            if len(bucket_means) < 2:
                importance[name] = 0.0
                continue

            # Marginal variance = variance of bucket means (weighted by count)
            counts = [len(v) for v in bucket_means.values()]
            means  = [sum(v)/len(v) for v in bucket_means.values()]
            total_n = sum(counts)
            grand_mean = sum(c*m for c,m in zip(counts, means)) / total_n
            marginal_var = sum(c*(m-grand_mean)**2 for c,m in zip(counts, means)) / total_n
            importance[name] = min(1.0, marginal_var / total_var)

        return importance


def _bucketise_simple(name: str, value: Any, dim: dict, n: int = 6) -> str:
    if dim["dtype"] == "categorical":
        return str(value)
    lo, hi = dim["min_val"], dim["max_val"]
    if dim["log_scale"] and lo > 0 and float(value) > 0:
        lo, hi = math.log(lo), math.log(hi)
        v = math.log(float(value))
    else:
        v = float(value)
    idx = int((v - lo) / (hi - lo) * n)
    return f"b{max(0, min(n-1, idx))}"


# ── ASHA Promoter ─────────────────────────────────────────────────────────────

class ASHAPromoter:
    """
    After every RUNG_SIZE experiments, promote the top KEEP_FRACTION configs
    and add their neighborhood to the queue. Cull the rest.
    
    Since all runs are fixed at 5 min, "budget" = number of evaluations.
    """
    RUNG_SIZE      = 100   # evaluate after every N new experiments
    KEEP_FRACTION  = 0.2   # keep top 20%
    NEIGHBORHOOD_K = 3     # spawn K neighbors per promoted config

    def should_run(self, total_experiments: int) -> bool:
        return total_experiments > 0 and total_experiments % self.RUNG_SIZE == 0

    def promote(
        self,
        experiments: list[dict],
        sampler: ThompsonSampler,
        dims: list[dict],
        rng: random.Random,
    ) -> list[tuple[dict, float, str]]:
        """Returns list of (config_delta, priority, note) to enqueue."""
        completed = sorted(
            [e for e in experiments if e["delta_bpb"] is not None],
            key=lambda x: x["delta_bpb"]
        )
        if not completed:
            return []

        n_keep = max(1, int(len(completed) * self.KEEP_FRACTION))
        promoted = completed[:n_keep]
        to_enqueue = []

        for exp in promoted:
            base_delta = json.loads(exp["config_delta"])
            for _ in range(self.NEIGHBORHOOD_K):
                neighbor = _perturb(base_delta, dims, rng)
                priority = max(0.0, min(1.0, 0.7 - exp["delta_bpb"] * 5))
                note = f"neighbor of best (delta={exp['delta_bpb']:.4f})"
                to_enqueue.append((neighbor, priority, note))

        return to_enqueue


def _perturb(config_delta: dict, dims: list[dict], rng: random.Random) -> dict:
    """Slightly perturb a config: re-sample one random dimension."""
    dim_map = {d["name"]: d for d in dims if not d["frozen"]}
    result = dict(config_delta)
    if not dim_map:
        return result
    # Pick one dimension to resample
    key = rng.choice(list(dim_map.keys()))
    dim = dim_map[key]
    val = _sample_value(dim, rng)
    result[key] = _round_value(val, dim)
    return result


# ── Top-level coordinator ─────────────────────────────────────────────────────

FREEZE_THRESHOLD      = 0.03   # freeze dim if importance drops below this (raised bar)
MIN_EXPERIMENTS_FREEZE = 200   # don't freeze before this many experiments (raised bar)
QUEUE_REFILL_TARGET   = 200    # keep this many configs in the queue

rng = random.Random()
sampler = ThompsonSampler()
fanova  = fANOVA()
asha    = ASHAPromoter()


def run_search_cycle():
    """
    Called periodically (every ~60s) by the background worker.
    1. Ingest recent experiments into Thompson sampler
    2. Run fANOVA, update importance scores, maybe freeze dimensions
    3. Run ASHA promotion
    4. Refill the config queue
    """
    dims = store.get_dimensions()

    # Recovery guard: if all dimensions are frozen, unfreeze a minimal set so
    # the search can continue and workers receive non-empty config deltas.
    if dims and all(bool(d["frozen"]) for d in dims):
        rescue = sorted(dims, key=lambda d: float(d.get("importance") or 0.0), reverse=True)[:2]
        for d in rescue:
            store.unfreeze_dimension(d["name"])
        log.warning(
            "Search space collapsed (all dims frozen); unfroze %s",
            ", ".join(d["name"] for d in rescue),
        )
        dims = store.get_dimensions()
    experiments = store.recent_experiments(1000)
    total = store.experiment_count()

    # 1. Ingest into Thompson sampler
    sampler.ingest(experiments)

    # 2. fANOVA
    if total >= fanova.MIN_SAMPLES:
        importance = fanova.run(experiments, dims)
        unfrozen_names = {d["name"] for d in dims if not d["frozen"]}
        for name, score in importance.items():
            n_samples = sum(
                1 for e in experiments
                if name in json.loads(e["config_delta"])
            )
            store.update_dimension_importance(name, score, n_samples)
            # Freeze if importance is very low and we have enough data
            if (total >= MIN_EXPERIMENTS_FREEZE
                    and score < FREEZE_THRESHOLD
                    and name in unfrozen_names):
                # Safety guard: never freeze everything; keep at least two active dims.
                if len(unfrozen_names) <= 2:
                    continue
                # Find the best value for this dimension
                best_val = _best_value_for_dim(name, experiments)
                if best_val is not None:
                    log.info(f"Freezing {name}={best_val} (importance={score:.3f})")
                    store.freeze_dimension(name, best_val)
                    unfrozen_names.discard(name)

    # Reload dims after potential freezes
    dims = store.get_dimensions()

    # 3. ASHA promotion
    if asha.should_run(total):
        promoted = asha.promote(experiments, sampler, dims, rng)
        if promoted:
            store.enqueue_configs(promoted)
            log.info(f"ASHA: enqueued {len(promoted)} promoted neighbors")

    # 4. Refill queue with fresh Thompson samples
    current_depth = store.queue_depth()
    n_needed = max(0, QUEUE_REFILL_TARGET - current_depth)
    if n_needed > 0:
        new_configs = []
        for _ in range(n_needed):
            delta, priority = sampler.propose(dims, rng)
            note = "thompson sample"
            new_configs.append((delta, priority, note))
        store.enqueue_configs(new_configs)
        log.info(f"Refilled queue with {n_needed} Thompson samples")


def _best_value_for_dim(dim_name: str, experiments: list[dict]) -> Any:
    """Find the value of dim_name that produced the best mean delta_bpb."""
    buckets: dict[str, list[float]] = {}
    for exp in experiments:
        cdelta = json.loads(exp["config_delta"])
        if dim_name in cdelta and exp["delta_bpb"] is not None:
            k = str(cdelta[dim_name])
            buckets.setdefault(k, []).append(exp["delta_bpb"])
    if not buckets:
        return None
    best_key = min(buckets, key=lambda k: sum(buckets[k]) / len(buckets[k]))
    # Try to parse it back to a number
    try:
        return json.loads(best_key)
    except Exception:
        return best_key
