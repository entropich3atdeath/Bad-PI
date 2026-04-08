"""
meta_server/belief_engine.py

The single source of truth for all mathematical decisions.
Uses scipy, numpy, and statistical packages — NO LLM calls anywhere.

Architecture principle:
  BeliefEngine (math) → Decision objects → program_writer (LLM translator) → prose

The LLM never decides what to do.
It only translates what the math already decided into readable program.md text.

Algorithms used:
  - ASHA  (Asynchronous Successive Halving) for run promotion/elimination
  - fANOVA (functional ANOVA) via scipy for dimension importance
  - Beta-Binomial (conjugate Bayesian) for hypothesis belief updating
  - Thompson Sampling (numpy) for config proposal
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats


# ── Decision types ────────────────────────────────────────────────────────────

@dataclass
class Decision:
    """
    A structured mathematical decision.
    The LLM reads this and translates it — it does not produce it.
    """
    type: str                    # "kill_run" | "extend_run" | "freeze_dim" | "update_hypothesis" | "spawn_population"
    reason_code: str             # machine-readable: "asha_below_eta3" | "fanova_p005" | "beta_posterior_lt015"
    supporting_stats: dict       # raw numbers: {"percentile": 18.3, "pool_size": 42, "p_value": 0.003}
    confidence: float            # statistical confidence 0–1
    run_id: Optional[str]        = None
    dim_name: Optional[str]      = None
    hypothesis_id: Optional[str] = None
    value: Any                   = None   # e.g. frozen value for a dimension

    def readable_reason(self) -> str:
        """One-line machine-readable summary for the LLM to expand on."""
        s = self.supporting_stats
        if self.type == "kill_run":
            return (f"ASHA: run at p{s.get('p',0):.0%} ranked P{s.get('percentile',0):.0f} "
                    f"in pool of {s.get('pool_size',0)} — below eta={s.get('eta',3)} cutoff")
        if self.type == "extend_run":
            return (f"ASHA: run at p1.0 ranked P{s.get('percentile',0):.0f} "
                    f"— top {100-s.get('extend_cutoff_pct',90):.0f}% extended")
        if self.type == "freeze_dim":
            return (f"fANOVA: '{self.dim_name}' F={s.get('F',0):.2f} p={s.get('p_value',1):.4f} "
                    f"eta²={s.get('eta_squared',0):.3f} — below significance threshold")
        if self.type == "update_hypothesis":
            return (f"Beta-Binomial: '{self.hypothesis_id}' "
                    f"posterior={s.get('posterior',0.5):.2f} "
                    f"CI=[{s.get('ci_lo',0):.2f},{s.get('ci_hi',1):.2f}] "
                    f"n={s.get('n',0)}")
        return f"{self.type}: {self.reason_code}"


# ── BeliefSummary — pre-computed findings for the LLM ────────────────────────

@dataclass
class DimFinding:
    """Performance summary for one dimension — computed from binned experiments."""
    dim_name:        str
    best_range:      str    # e.g. "0.01–0.03" or "'L'"
    best_mean_delta: float
    best_n:          int
    worst_range:     str
    worst_mean_delta: float
    worst_n:         int
    gap:             float  # worst_mean - best_mean: bigger = dimension matters more


@dataclass
class BeliefSummary:
    """
    Pre-synthesised findings derived entirely from experiment data using numpy.
    This is what gets passed to the LLM — not raw F-statistics.

    The LLM receives a JSON blob matching the user's expected format:
      {
        "top_findings": ["DEPTH 12-14 performs best (delta=-0.18, n=14)", ...],
        "best_config": {"DEPTH": 12, "learning_rate": 2.1e-3, "delta_bpb": -0.21},
        "rejected": ["DEPTH < 6 underperforms (delta=+0.09)", "WINDOW_PATTERN (frozen)"],
        "promising_regions": [{"dim": "DEPTH", "range": "12-14", "mean_delta": -0.18, "n": 14}],
        "interaction_hints": ["DEPTH > 10 and lr 1e-3–3e-3 co-occur in top 10 results"]
      }
    """
    top_findings:      list[str]
    best_config:       dict          # {"dim": val, ..., "delta_bpb": float}
    rejected:          list[str]
    promising_regions: list[dict]    # [{dim, range, mean_delta, n}, ...]
    interaction_hints: list[str]
    dim_findings:      list[DimFinding]  # structured version of top_findings


def _fmt_val(v: Any) -> str:
    """Format a numeric value for display in finding strings."""
    if isinstance(v, float):
        if abs(v) < 0.001 or abs(v) >= 10_000:
            return f"{v:.2g}"
        return f"{v:.3g}"
    return str(v)


def _bucket_label(val: Any, dim: dict, n_bins: int = 5) -> str:
    """Map a value to a human-readable range label for its bin."""
    if dim["dtype"] == "categorical":
        return str(val)
    try:
        lo, hi = float(dim["min_val"]), float(dim["max_val"])
        v = float(val)
        is_log = bool(dim.get("log_scale")) and lo > 0
        if is_log and v > 0:
            lo_t, hi_t, v_t = math.log(lo), math.log(hi), math.log(v)
        else:
            lo_t, hi_t, v_t = lo, hi, v
        bin_w = (hi_t - lo_t) / n_bins
        if bin_w <= 0:
            return _fmt_val(val)
        idx = max(0, min(n_bins - 1, int((v_t - lo_t) / bin_w)))
        b_lo = lo_t + idx * bin_w
        b_hi = b_lo + bin_w
        if is_log:
            b_lo, b_hi = math.exp(b_lo), math.exp(b_hi)
        return f"{_fmt_val(b_lo)}–{_fmt_val(b_hi)}"
    except Exception:
        return str(val)


def _parse_config(exp: dict) -> dict:
    cd = exp.get("config_delta", {})
    if isinstance(cd, str):
        try:
            return json.loads(cd)
        except Exception:
            return {}
    return cd if isinstance(cd, dict) else {}


def compute_belief_summary(
    experiments: list[dict],
    dimensions:  list[dict],
    frozen_dims: dict[str, Any],
    hypotheses:  list,
    n_bins:      int = 5,
) -> BeliefSummary:
    """
    Synthesise experiment results into structured findings.
    Pure numpy — no LLM calls. Called before the LLM sees anything.

    This is the step that transforms:
      "DEPTH: F=18.4 p=0.0002 eta²=0.41"
    into:
      "DEPTH 12–14 performs best (mean delta=-0.18, n=14)"
    """
    completed = [e for e in experiments if e.get("delta_bpb") is not None]
    if not completed:
        return BeliefSummary(
            top_findings=[], best_config={}, rejected=[],
            promising_regions=[], interaction_hints=[], dim_findings=[],
        )

    # ── Best config ───────────────────────────────────────────────────────
    best_exp   = min(completed, key=lambda e: e["delta_bpb"])
    best_cfg   = _parse_config(best_exp)
    best_cfg["delta_bpb"] = round(best_exp["delta_bpb"], 4)

    # ── Per-dimension findings ─────────────────────────────────────────────
    dim_findings:      list[DimFinding] = []
    top_findings:      list[str]        = []
    rejected:          list[str]        = []
    promising_regions: list[dict]       = []

    for dim in dimensions:
        name = dim["name"]

        # ── Frozen → rejected ─────────────────────────────────────────────
        if dim.get("frozen"):
            r = frozen_dims.get(name)
            rejected.append(f"{name} (frozen at {_fmt_val(r)} — no significant effect)")
            continue

        # ── Group experiments by bin ──────────────────────────────────────
        bins: dict[str, list[float]] = {}
        for exp in completed:
            cd  = _parse_config(exp)
            val = cd.get(name)
            if val is None:
                continue
            label = _bucket_label(val, dim, n_bins)
            bins.setdefault(label, []).append(exp["delta_bpb"])

        # Need at least 2 bins with ≥2 points each
        valid = {lbl: vals for lbl, vals in bins.items() if len(vals) >= 2}
        if len(valid) < 2:
            continue

        stats_per_bin = {
            lbl: {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "n":      len(vals),
                "median": float(np.median(vals)),
            }
            for lbl, vals in valid.items()
        }

        best_lbl  = min(stats_per_bin, key=lambda k: stats_per_bin[k]["mean"])
        worst_lbl = max(stats_per_bin, key=lambda k: stats_per_bin[k]["mean"])
        best_s    = stats_per_bin[best_lbl]
        worst_s   = stats_per_bin[worst_lbl]
        gap       = worst_s["mean"] - best_s["mean"]

        # ── Build finding strings ─────────────────────────────────────────
        paren = f"mean delta={best_s['mean']:+.3f}, n={best_s['n']}"
        if dim["dtype"] == "categorical":
            finding = f"{name}='{best_lbl}' performs best ({paren})"
        else:
            finding = f"{name} {best_lbl} performs best ({paren})"
        top_findings.append(finding)

        promising_regions.append({
            "dim":        name,
            "range":      best_lbl,
            "mean_delta": round(best_s["mean"], 4),
            "std":        round(best_s["std"], 4),
            "n":          best_s["n"],
        })

        # ── Rejected regions (gap > 0.04 = meaningful) ───────────────────
        if gap > 0.04:
            if dim["dtype"] == "categorical":
                rejected.append(f"{name}='{worst_lbl}' underperforms (mean delta={worst_s['mean']:+.3f}, n={worst_s['n']})")
            else:
                rejected.append(f"{name} {worst_lbl} consistently underperforms (mean delta={worst_s['mean']:+.3f})")

        dim_findings.append(DimFinding(
            dim_name         = name,
            best_range       = best_lbl,
            best_mean_delta  = round(best_s["mean"], 4),
            best_n           = best_s["n"],
            worst_range      = worst_lbl,
            worst_mean_delta = round(worst_s["mean"], 4),
            worst_n          = worst_s["n"],
            gap              = round(gap, 4),
        ))

    # ── Sort top_findings by gap (most impactful first) ────────────────────
    dim_findings.sort(key=lambda d: -d.gap)
    top_findings = [
        (f"{d.dim_name}={d.best_range!r}" if "–" not in d.best_range else f"{d.dim_name} {d.best_range}")
        + f" performs best (mean delta={d.best_mean_delta:+.3f}, n={d.best_n})"
        for d in dim_findings
    ]

    # ── Refuted hypotheses → rejected ─────────────────────────────────────
    for h in hypotheses:
        if h.status == "refuted":
            rejected.append(f"hypothesis: \"{h.statement}\" (P={h.posterior:.2f} — refuted)")

    # ── Interaction hints ─────────────────────────────────────────────────
    interaction_hints: list[str] = []
    top10 = sorted(completed, key=lambda e: e["delta_bpb"])[:10]
    active_high_impact = [d for d in dim_findings if d.gap > 0.06][:3]

    if len(active_high_impact) >= 2 and len(top10) >= 5:
        # Find pairs that co-occur in top-10
        for i in range(len(active_high_impact)):
            for j in range(i + 1, len(active_high_impact)):
                da, db = active_high_impact[i], active_high_impact[j]
                co_count = sum(
                    1 for exp in top10
                    if (
                        _bucket_label(_parse_config(exp).get(da.dim_name, ""), active_high_impact[i].__dict__) == da.best_range
                        or _parse_config(exp).get(da.dim_name) is not None
                    ) and _parse_config(exp).get(db.dim_name) is not None
                )
                if co_count >= 4:
                    interaction_hints.append(
                        f"{da.dim_name} {da.best_range} and {db.dim_name} {db.best_range} "
                        f"co-occur in {co_count}/10 top results — possible interaction"
                    )

    return BeliefSummary(
        top_findings      = top_findings,
        best_config       = best_cfg,
        rejected          = rejected,
        promising_regions = promising_regions,
        interaction_hints = interaction_hints,
        dim_findings      = dim_findings,
    )


# ── BeliefState ───────────────────────────────────────────────────────────────

@dataclass
class BeliefState:
    """
    Complete snapshot of what the math currently knows.
    Passed to the LLM program writer for translation into prose.
    """
    experiment_count:     int
    warmup_complete:      bool
    warmup_runs_needed:   int

    # fANOVA dimension analysis
    dimension_importance: dict[str, float]
    dimension_fstats:     dict[str, dict]
    frozen_dimensions:    dict[str, Any]

    # ASHA state
    asha_eta:             int
    asha_pool_sizes:      dict[str, int]
    kill_rate:            float
    extend_rate:          float

    # Hypothesis beliefs (Beta-Binomial)
    hypotheses:           list[dict]

    # Best results
    best_delta_bpb:       float
    best_config_delta:    dict
    top_configs:          list[dict]

    # Pipeline
    pipeline_hit_rate:    float

    # Recent decisions (for the LLM to explain)
    recent_decisions:     list[Decision]

    # Pre-computed findings — what the LLM actually reads
    summary:              BeliefSummary = field(default_factory=lambda: BeliefSummary(
        top_findings=[], best_config={}, rejected=[],
        promising_regions=[], interaction_hints=[], dim_findings=[],
    ))

    # Runtime signals: engine sets is_stalled; runtime.py sets is_converging (needs registry)
    is_stalled:           bool = False   # no improvement over last 50 experiments
    is_converging:        bool = False   # a winner hypothesis has been found


# ── ASHA Scheduler ────────────────────────────────────────────────────────────

class ASHAScheduler:
    """
    Asynchronous Successive Halving Algorithm.

    Each progress bucket is a "rung". At each rung, the bottom
    (1 - 1/eta) fraction of runs are promoted to be killed.
    Top 1/eta are allowed to continue.

    With eta=3:
      - Kill 2/3 at p=0.2 (keep top 33%)
      - Kill 2/3 of survivors at p=0.4 (keep top ~11%)
      - etc.

    In practice we use a softer cutoff: kill below P(100/eta * (eta-1))
    i.e. kill bottom 66% with eta=3.
    """
    # Stochastic kill: at the worst possible rank, kill with at most this probability.
    # A run at exactly the kill threshold has 0% chance of being killed.
    # Linear ramp between the two.  Set to 1.0 for fully deterministic ASHA.
    STOCHASTIC_KILL_MAX_PROB = 0.65

    def __init__(self, eta: int = 3):
        self.eta = eta
        # rung_p → list of (metric, run_id) sorted by metric
        self._rungs: dict[float, list[tuple[float, str]]] = {}
        self._killed: set[str] = set()
        self._extended: set[str] = set()

    @property
    def kill_pct(self) -> float:
        """Fraction of runs to kill at any rung."""
        return (1 - 1 / self.eta) * 100   # e.g. 66.7% with eta=3

    @property
    def extend_pct(self) -> float:
        """Top fraction to extend at final rung."""
        return (1 / self.eta**2) * 100     # top ~11% extended with eta=3

    def register(self, p: float, metric: float, run_id: str):
        bucket = self._snap_bucket(p)
        if bucket is not None:
            rung = self._rungs.setdefault(bucket, [])
            # Remove old entry for this run if re-reporting
            rung[:] = [(m, r) for m, r in rung if r != run_id]
            rung.append((metric, run_id))

    def evaluate(self, p: float, metric: float, run_id: str, min_pool: int = 5) -> str:
        """
        Returns "stop", "extend", or "" (continue).

        Kill decision is PROBABILISTIC:
          - A run exactly at the kill threshold has 0% chance of being killed.
          - The absolute worst run in the pool has STOCHASTIC_KILL_MAX_PROB chance.
          - Linear interpolation between the two.
        This preserves some "bad" directions that might recover, while still
        aggressively pruning the consistently worst performers.
        """
        bucket = self._snap_bucket(p)
        if bucket is None:
            return ""

        rung = self._rungs.get(bucket, [])
        pool_metrics = [m for m, r in rung if r != run_id]
        if len(pool_metrics) < min_pool:
            return ""

        rank_pct = self._percentile_rank(metric, pool_metrics)
        threshold_pct = 100 - self.kill_pct   # e.g. 33.3 with eta=3

        if rank_pct < threshold_pct:
            # Severity: 0.0 at the threshold, 1.0 at absolute worst (rank=0)
            severity = (threshold_pct - rank_pct) / threshold_pct
            p_kill = self.STOCHASTIC_KILL_MAX_PROB * severity
            if random.random() < p_kill:
                self._killed.add(run_id)
                return "stop"
            # Survived the stochastic reprieve — let it continue this bucket

        if bucket == 0.8 and rank_pct >= (100 - self.extend_pct):
            # Extend after the second-to-last bucket (give them a bonus final run)
            self._extended.add(run_id)
            return "extend"

        return ""

    def make_kill_decision(
        self, p: float, metric: float, run_id: str, pool: list[float]
    ) -> Optional[Decision]:
        """Advisory kill decision that mirrors the probabilistic logic in evaluate()."""
        rank = self._percentile_rank(metric, pool)
        threshold_pct = 100 - self.kill_pct   # e.g. 33.3 with eta=3
        if rank < threshold_pct:
            severity = (threshold_pct - rank) / threshold_pct
            p_kill   = self.STOCHASTIC_KILL_MAX_PROB * severity
            if random.random() >= p_kill:
                return None   # stochastic reprieve — no advisory kill
            return Decision(
                type       = "kill_run",
                reason_code= "asha_below_cutoff_stochastic",
                run_id     = run_id,
                confidence = p_kill,
                supporting_stats = {
                    "percentile":    round(rank, 1),
                    "threshold_pct": round(threshold_pct, 1),
                    "severity":      round(severity, 3),
                    "p_kill":        round(p_kill, 3),
                    "pool_size":     len(pool),
                    "eta":           self.eta,
                    "kill_pct":      round(self.kill_pct, 1),
                    "p":             p,
                    "metric":        round(metric, 4),
                    "pool_p25":      round(float(np.percentile(pool, 25)), 4),
                    "pool_p50":      round(float(np.percentile(pool, 50)), 4),
                },
            )
        return None

    def make_extend_decision(
        self, metric: float, run_id: str, pool: list[float], extend_budget: float
    ) -> Optional[Decision]:
        rank = self._percentile_rank(metric, pool)
        cutoff = 100 - self.extend_pct
        if rank >= cutoff:
            return Decision(
                type       = "extend_run",
                reason_code= "asha_top_fraction",
                run_id     = run_id,
                confidence = rank / 100,
                value      = extend_budget,
                supporting_stats = {
                    "percentile":         round(rank, 1),
                    "extend_cutoff_pct":  round(cutoff, 1),
                    "pool_size":          len(pool),
                    "new_budget_seconds": extend_budget,
                },
            )
        return None

    def rung_stats(self) -> dict:
        out = {}
        for b, rung in self._rungs.items():
            if rung:
                metrics = [m for m, _ in rung]
                out[str(b)] = {
                    "n":   len(metrics),
                    "p25": round(float(np.percentile(metrics, 25)), 4),
                    "p50": round(float(np.percentile(metrics, 50)), 4),
                    "p90": round(float(np.percentile(metrics, 90)), 4),
                }
        return out

    @staticmethod
    def _snap_bucket(p: float) -> Optional[float]:
        for b in [0.2, 0.4, 0.6, 0.8, 1.0]:
            if abs(p - b) <= 0.05:
                return b
        return None

    @staticmethod
    def _percentile_rank(metric: float, pool: list[float]) -> float:
        """Rank of this metric vs pool. Lower metric = better → high rank."""
        if not pool:
            return 50.0
        worse = sum(1 for m in pool if m > metric)
        return worse / len(pool) * 100.0


# ── fANOVA ────────────────────────────────────────────────────────────────────

class FunctionalANOVA:
    """
    One-way ANOVA per dimension using scipy.stats.f_oneway.
    Returns:
      - F-statistic
      - p-value  (p < 0.05 = dimension is statistically significant)
      - eta² (effect size = SS_between / SS_total): how much variance it explains

    A dimension is frozen when:
      - p > FREEZE_P_THRESHOLD (not significant)  AND
      - eta² < FREEZE_ETA2_THRESHOLD (tiny effect size)  AND
      - n_experiments >= MIN_EXPERIMENTS_FOR_ANOVA
    """
    FREEZE_P_THRESHOLD    = 0.05   # not significant at 5% level (stricter — less eager to freeze)
    FREEZE_ETA2_THRESHOLD = 0.02   # explains < 2% of variance (stricter — must be truly tiny)
    MIN_EXPERIMENTS       = 80     # need more data before any freeze decision
    N_BUCKETS             = 6      # discretise continuous dims into this many groups

    def run(
        self,
        experiments: list[dict],
        dimensions: list[dict],
    ) -> dict[str, dict]:
        """
        Returns {dim_name: {"F": float, "p_value": float, "eta_squared": float, "n": int}}
        """
        results = {}
        deltas = [e["delta_bpb"] for e in experiments if e.get("delta_bpb") is not None]
        if len(deltas) < self.MIN_EXPERIMENTS:
            return {}

        for dim in dimensions:
            if dim.get("frozen"):
                continue
            name = dim["name"]

            # Group experiments by value of this dimension
            groups: dict[str, list[float]] = {}
            for exp in experiments:
                if exp.get("delta_bpb") is None:
                    continue
                cd = json.loads(exp["config_delta"]) if isinstance(exp["config_delta"], str) else exp["config_delta"]
                if name not in cd:
                    continue
                bucket = self._bucket_value(cd[name], dim)
                groups.setdefault(bucket, []).append(exp["delta_bpb"])

            valid_groups = [g for g in groups.values() if len(g) >= 3]
            if len(valid_groups) < 2:
                results[name] = {"F": 0.0, "p_value": 1.0, "eta_squared": 0.0, "n": len(deltas)}
                continue

            try:
                F, p = stats.f_oneway(*valid_groups)
                eta2 = self._eta_squared(valid_groups)
            except Exception:
                F, p, eta2 = 0.0, 1.0, 0.0

            results[name] = {
                "F":           round(float(F), 3),
                "p_value":     round(float(p), 4),
                "eta_squared": round(float(eta2), 4),
                "n":           sum(len(g) for g in valid_groups),
                "n_groups":    len(valid_groups),
            }

        return results

    def freeze_decisions(
        self,
        fanova_results: dict[str, dict],
        min_experiments: int,
    ) -> list[Decision]:
        decisions = []
        for name, r in fanova_results.items():
            if (r["n"] >= min_experiments
                    and r["p_value"] > self.FREEZE_P_THRESHOLD
                    and r["eta_squared"] < self.FREEZE_ETA2_THRESHOLD):
                decisions.append(Decision(
                    type        = "freeze_dim",
                    reason_code = "fanova_not_significant",
                    dim_name    = name,
                    confidence  = 1 - r["p_value"],
                    supporting_stats = {
                        "F":          r["F"],
                        "p_value":    r["p_value"],
                        "eta_squared": r["eta_squared"],
                        "n":          r["n"],
                        "threshold_p": self.FREEZE_P_THRESHOLD,
                        "threshold_eta2": self.FREEZE_ETA2_THRESHOLD,
                    },
                ))
        return decisions

    def _bucket_value(self, value: Any, dim: dict) -> str:
        if dim["dtype"] == "categorical":
            return str(value)
        try:
            v = float(value)
            lo, hi = float(dim["min_val"]), float(dim["max_val"])
            if dim.get("log_scale") and lo > 0:
                lo, hi, v = math.log(lo), math.log(hi), math.log(max(v, 1e-10))
            idx = int((v - lo) / (hi - lo) * self.N_BUCKETS)
            return f"b{max(0, min(self.N_BUCKETS - 1, idx))}"
        except Exception:
            return str(value)

    @staticmethod
    def _eta_squared(groups: list[list[float]]) -> float:
        all_vals = [x for g in groups for x in g]
        grand_mean = np.mean(all_vals)
        ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
        if ss_total < 1e-12:
            return 0.0
        group_means = [np.mean(g) for g in groups]
        group_ns    = [len(g) for g in groups]
        ss_between  = sum(n * (m - grand_mean) ** 2 for n, m in zip(group_ns, group_means))
        return float(ss_between / ss_total)


# ── Beta-Binomial hypothesis updater ─────────────────────────────────────────

class BetaBinomial:
    """
    Exact Bayesian update for binary hypotheses.
    Prior: Beta(alpha_0, beta_0) — defaults to Beta(2,2) (mildly uncertain, centered P=0.5)
    After k wins in n trials: posterior Beta(alpha_0+k, beta_0+(n-k))
    Posterior mean = (alpha_0+k) / (alpha_0 + beta_0 + n)
    90% credible interval via scipy.stats.beta.ppf
    """
    PRIOR_ALPHA = 2.0
    PRIOR_BETA  = 2.0
    SUPPORT_THRESHOLD = 0.60
    REFUTE_THRESHOLD  = 0.40
    EVIDENCE_CONFIDENCE = 0.90

    def posterior_stats(self, wins: int, n: int) -> dict:
        a = self.PRIOR_ALPHA + wins
        b = self.PRIOR_BETA  + (n - wins)
        dist = stats.beta(a, b)
        mean  = dist.mean()
        lo90, hi90 = dist.ppf(0.05), dist.ppf(0.95)
        support_probability = 1.0 - dist.cdf(self.SUPPORT_THRESHOLD)
        refute_probability  = dist.cdf(self.REFUTE_THRESHOLD)
        rope_probability    = dist.cdf(self.SUPPORT_THRESHOLD) - dist.cdf(self.REFUTE_THRESHOLD)
        return {
            "posterior": round(float(mean), 4),
            "ci_lo":     round(float(lo90), 4),
            "ci_hi":     round(float(hi90), 4),
            "alpha":     a,
            "beta_param": b,
            "n":         n,
            "wins":      wins,
            "support_probability": round(float(support_probability), 4),
            "refute_probability": round(float(refute_probability), 4),
            "rope_probability": round(float(rope_probability), 4),
        }

    def update_decision(self, h_id: str, wins: int, n: int, importance: float) -> Decision:
        stats_dict = self.posterior_stats(wins, n)
        p = stats_dict["posterior"]
        uncertainty = 4 * p * (1 - p)

        if stats_dict["support_probability"] >= self.EVIDENCE_CONFIDENCE and n >= 10:
            status = "supported"
        elif stats_dict["refute_probability"] >= self.EVIDENCE_CONFIDENCE and n >= 10:
            status = "refuted"
        else:
            status = "active"

        return Decision(
            type             = "update_hypothesis",
            reason_code      = f"beta_binomial_{status}",
            hypothesis_id    = h_id,
            confidence       = 1 - uncertainty,
            supporting_stats = {
                **stats_dict,
                "uncertainty":     round(uncertainty, 4),
                "importance":      importance,
                "info_value":      round(uncertainty * importance, 4),
                "status":          status,
            },
        )


# ── Thompson Sampler (numpy) ──────────────────────────────────────────────────

class ThompsonSampler:
    """
    Per-dimension Thompson Sampling.
    For each dimension bucket, maintains a Normal posterior over mean delta_bpb.
    Samples from posterior at proposal time — no LLM involved.
    """
    N_BUCKETS  = 8
    PRIOR_MEAN = 0.0
    PRIOR_STD  = 0.03
    PRIOR_N    = 2      # prior weight in observations

    def __init__(self):
        self._data: dict[str, dict[str, list[float]]] = {}   # dim → bucket → [deltas]

    def ingest(self, experiments: list[dict]):
        self._data.clear()
        for exp in experiments:
            if exp.get("delta_bpb") is None:
                continue
            cd = json.loads(exp["config_delta"]) if isinstance(exp["config_delta"], str) else exp["config_delta"]
            for k, v in cd.items():
                bucket = str(v)   # use raw value as bucket key (works for both cat and continuous)
                self._data.setdefault(k, {}).setdefault(bucket, []).append(exp["delta_bpb"])

    def propose(self, dim: dict, rng: np.random.Generator) -> Any:
        name   = dim["name"]
        bucket_data = self._data.get(name, {})

        if dim["dtype"] == "categorical":
            cats = dim["categories"]
            # Thompson draw per category
            scores = []
            for cat in cats:
                vals = bucket_data.get(str(cat), [])
                scores.append(self._draw_posterior(vals, rng))
            # Pick category with lowest drawn delta (lower = better)
            return cats[int(np.argmin(scores))]

        # Continuous: draw a bucket, then sample within it
        if not bucket_data:
            return self._random_continuous(dim, rng)

        # Draw posterior mean for each observed bucket center
        best_draw = float("inf")
        best_val  = self._random_continuous(dim, rng)
        for bucket_key, vals in bucket_data.items():
            draw = self._draw_posterior(vals, rng)
            if draw < best_draw:
                best_draw = draw
                try:
                    best_val = float(bucket_key)
                    # Small random perturbation
                    lo, hi = dim["min_val"], dim["max_val"]
                    step = (hi - lo) / (self.N_BUCKETS * 2)
                    best_val = float(np.clip(best_val + rng.normal(0, step), lo, hi))
                except ValueError:
                    best_val = self._random_continuous(dim, rng)
        return best_val

    def _draw_posterior(self, vals: list[float], rng: np.random.Generator) -> float:
        n = len(vals)
        if n == 0:
            return float(rng.normal(self.PRIOR_MEAN, self.PRIOR_STD))
        sample_mean = float(np.mean(vals))
        # Posterior mean (shrinkage toward prior)
        post_mean = (sample_mean * n + self.PRIOR_MEAN * self.PRIOR_N) / (n + self.PRIOR_N)
        post_std  = self.PRIOR_STD / math.sqrt(n + 1)
        return float(rng.normal(post_mean, post_std))

    def _random_continuous(self, dim: dict, rng: np.random.Generator) -> float:
        lo, hi = dim["min_val"], dim["max_val"]
        if dim.get("log_scale") and lo > 0:
            return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        return float(rng.uniform(lo, hi))


# ── Main Belief Engine ────────────────────────────────────────────────────────

class BeliefEngine:
    """
    Orchestrates all math. Returns Decision objects and BeliefState.
    Never calls LLM. Never writes prose. Pure computation.
    """
    WARMUP_RUNS  = 25    # no killing until this many runs are complete
    MIN_FREEZE_N = 50    # don't freeze dimensions before this many experiments

    def __init__(self):
        self.asha      = ASHAScheduler(eta=3)
        self.fanova    = FunctionalANOVA()
        self.bbayes    = BetaBinomial()
        self.thompson  = ThompsonSampler()
        self._rng      = np.random.default_rng()
        self._decisions_log: list[Decision] = []
        self._completed_count: int = 0
        self._stall_delta_log: list[tuple[int, float]] = []   # (exp_count, best_delta_bpb)

    @property
    def warmup_complete(self) -> bool:
        return self._completed_count >= self.WARMUP_RUNS

    def record_best_delta(self, experiment_count: int, delta: float):
        """Called by runtime after each experiment to track improvement trend."""
        self._stall_delta_log.append((experiment_count, delta))
        if len(self._stall_delta_log) > 200:
            self._stall_delta_log = self._stall_delta_log[-100:]

    def is_stalled(self, window: int = 50, min_improvement: float = 0.0005) -> bool:
        """
        True if best delta_bpb hasn't improved by at least min_improvement over the
        last `window` experiments.  delta_bpb is negative for real improvements,
        so improvement = oldest_delta - best_recent > 0 when making real progress.
        """
        if not self.warmup_complete or len(self._stall_delta_log) < window:
            return False
        recent = self._stall_delta_log[-window:]
        oldest_delta = recent[0][1]
        best_recent  = min(e[1] for e in recent)
        return (oldest_delta - best_recent) < min_improvement

    def on_tick(
        self,
        run_id: str,
        p: float,
        metric: float,
        delta: float,
        pool_at_bucket: list[float],
    ) -> Optional[Decision]:
        """
        Called on every worker tick. Returns a kill/extend decision or None.
        During warmup: always returns None (no early stopping).
        """
        self.asha.register(p, metric, run_id)

        if not self.warmup_complete:
            return None   # no stopping during warmup

        action = self.asha.evaluate(p, metric, run_id, min_pool=5)
        if action == "stop":
            d = self.asha.make_kill_decision(p, metric, run_id, pool_at_bucket)
            if d:
                self._log(d)
            return d
        if action == "extend":
            d = self.asha.make_extend_decision(metric, run_id, pool_at_bucket, extend_budget=420)
            if d:
                self._log(d)
            return d
        return None

    def on_experiment_complete(
        self,
        experiments: list[dict],
        dimensions: list[dict],
        hypotheses: list,
    ) -> list[Decision]:
        """
        Run after each completed experiment. Returns batch of decisions.
        """
        self._completed_count += 1
        decisions = []

        # fANOVA — run every 10 experiments once warmup is done
        if self.warmup_complete and self._completed_count % 10 == 0:
            fanova_results = self.fanova.run(experiments, dimensions)
            freeze_decisions = self.fanova.freeze_decisions(fanova_results, self.MIN_FREEZE_N)
            decisions.extend(freeze_decisions)
            for d in freeze_decisions:
                self._log(d)

        # Beta-Binomial hypothesis updates
        for h in hypotheses:
            wins = int(h.alpha - 2)   # prior is Beta(2,2)
            n    = int(h.alpha + h.beta - 4)
            if n > 0:
                d = self.bbayes.update_decision(h.id, wins, n, h.importance)
                decisions.append(d)
                self._log(d)

        return decisions

    def decide_budget(
        self,
        population_strategy: str,
        hypothesis_posterior: float = 0.5,
        queue_depth: int = 50,
    ) -> int:
        """
        Decide run duration in seconds based on population strategy and belief state.
        Returns seconds for TOTAL_WALL_CLOCK_TIME in train.py.

        Rationale:
          exploit   — we're fairly sure this direction is good, worth more compute
          falsify   — controlled experiment needs clean convergence, slightly longer
          investigate — standard, no strong prior
          moonshot  — speculative, keep short to fail fast
          warmup    — always default, don't bias the baseline pool
        """
        BASE = 300   # 5 min
        if not self.warmup_complete:
            return BASE

        budgets = {
            "exploit":     420,   # 7 min  — confirmed direction, squeeze more signal
            "falsify":     480,   # 8 min  — controlled test needs clean convergence
            "investigate": 300,   # 5 min  — standard
            "moonshot":    240,   # 4 min  — fail fast on speculative ideas
            "converge":    540,   # 9 min  — multi-worker validation of the winning config
        }
        budget = budgets.get(population_strategy, BASE)

        # Uncertainty adjustment: high uncertainty (P≈0.5) → slightly shorter runs
        # so we explore breadth first.  Near-certain hypothesis (P near 0 or 1) gets
        # the full budget for clean convergence or clean falsification.
        # uncertainty ∈ [0,1], peaks at 1.0 when P=0.5.
        uncertainty = 4.0 * hypothesis_posterior * (1.0 - hypothesis_posterior)
        # discount: −15% at max uncertainty, 0% at certainty
        budget = int(budget * (1.0 - 0.15 * uncertainty))

        # Shrink slightly if queue is deep (many workers waiting for configs)
        if queue_depth > 100:
            budget = max(240, int(budget * 0.85))

        return max(180, budget)   # floor: never shorter than 3 min

    def propose_configs(
        self,
        n: int,
        dimensions: list[dict],
        experiments: list[dict],
    ) -> list[dict]:
        """
        Thompson Sampling config proposals. Pure numpy.
        """
        self.thompson.ingest(experiments)
        configs = []
        for _ in range(n):
            delta = {}
            for dim in dimensions:
                if dim.get("frozen"):
                    delta[dim["name"]] = dim["frozen_value"]
                    continue
                val = self.thompson.propose(dim, self._rng)
                delta[dim["name"]] = self._round_val(val, dim)
            configs.append(delta)
        return configs

    def build_belief_state(
        self,
        experiments: list[dict],
        dimensions: list[dict],
        hypotheses: list,
        pipeline_hit_rate: float = 0.0,
    ) -> BeliefState:
        """
        Construct a complete BeliefState snapshot for the LLM translator.
        """
        fanova_results = (
            self.fanova.run(experiments, dimensions)
            if self.warmup_complete else {}
        )
        dim_importance = {k: v["eta_squared"] for k, v in fanova_results.items()}
        frozen = {d["name"]: d["frozen_value"] for d in dimensions if d.get("frozen")}

        sorted_hyps = sorted(
            hypotheses,
            key=lambda h: 4 * h.posterior * (1 - h.posterior) * h.importance,
            reverse=True,
        )
        hyp_dicts = [
            {
                "id": h.id, "statement": h.statement,
                "posterior": round(h.posterior, 3),
                **self.bbayes.posterior_stats(
                    int(h.alpha - 2), int(h.alpha + h.beta - 4)
                ),
                "info_value": round(4 * h.posterior * (1 - h.posterior) * h.importance, 3),
                "evidence_strength": getattr(h, "evidence_strength", "weak"),
                "status": h.status,
            }
            for h in sorted_hyps
        ]

        completed = [e for e in experiments if e.get("delta_bpb") is not None]
        best = min(completed, key=lambda e: e["delta_bpb"], default=None)
        top5 = sorted(completed, key=lambda e: e["delta_bpb"])[:5]

        asha_stats = self.asha.rung_stats()
        total = len(completed)
        killed = sum(1 for r in self.asha._killed)   # approximation

        summary = compute_belief_summary(
            experiments  = completed,
            dimensions   = dimensions,
            frozen_dims  = frozen,
            hypotheses   = sorted_hyps,
        )

        return BeliefState(
            experiment_count     = total,
            warmup_complete      = self.warmup_complete,
            warmup_runs_needed   = max(0, self.WARMUP_RUNS - total),
            dimension_importance = dim_importance,
            dimension_fstats     = fanova_results,
            frozen_dimensions    = frozen,
            asha_eta             = self.asha.eta,
            asha_pool_sizes      = {k: v["n"] for k, v in asha_stats.items()},
            kill_rate            = round(len(self.asha._killed) / max(total, 1) * 100, 1),
            extend_rate          = round(len(self.asha._extended) / max(total, 1) * 100, 1),
            hypotheses           = hyp_dicts,
            best_delta_bpb       = best["delta_bpb"] if best else 0.0,
            best_config_delta    = (
                json.loads(best["config_delta"])
                if best and isinstance(best["config_delta"], str)
                else (best["config_delta"] if best else {})
            ),
            top_configs          = [
                {"delta_bpb": e["delta_bpb"],
                 "config_delta": json.loads(e["config_delta"])
                                 if isinstance(e["config_delta"], str)
                                 else e["config_delta"]}
                for e in top5
            ],
            pipeline_hit_rate    = pipeline_hit_rate,
            recent_decisions     = list(self._decisions_log[-20:]),
            summary              = summary,
            is_stalled           = self.is_stalled(),
        )

    def _log(self, d: Decision):
        self._decisions_log.append(d)
        if len(self._decisions_log) > 500:
            self._decisions_log = self._decisions_log[-250:]

    @staticmethod
    def _round_val(val: Any, dim: dict) -> Any:
        if dim["dtype"] == "int":
            return int(round(val))
        if dim["dtype"] in ("float", "float_log"):
            if val == 0:
                return 0.0
            mag = math.floor(math.log10(abs(val)))
            return round(val, -int(mag) + 3)
        return val


# ── Module-level singleton ────────────────────────────────────────────────────

engine = BeliefEngine()
