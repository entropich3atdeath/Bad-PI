"""
meta_server/hypotheses.py

Hypothesis tracking with Bayesian (Beta-Binomial) belief updating.
Each hypothesis is a falsifiable scientific claim about what improves val_bpb.

Core idea: instead of tracking raw dimension importance scores, the PI
maintains *hypotheses* — claims like "depth > 10 is optimal" — and updates
a probability distribution over each as experiments come in.
"""
from __future__ import annotations
import difflib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from scipy import stats

PRIOR_ALPHA = 2.0   # prior "wins"  — mildly uncertain, centered at P=0.5
PRIOR_BETA  = 2.0   # prior "losses"
WIN_THRESHOLD = 0.0 # delta_bpb < 0 = improvement = win for hypothesis
SUPPORT_THRESHOLD = 0.60
REFUTE_THRESHOLD = 0.40
EVIDENCE_CONFIDENCE = 0.90
UNCERTAINTY_BAND_LO = 0.45
UNCERTAINTY_BAND_HI = 0.55
UNCERTAINTY_STREAK_TRIGGER = 20
SPRINT_COOLDOWN_EXPERIMENTS = 30
SPRINT_WINDOW_EXPERIMENTS = 12
SEMANTIC_DUPLICATE_THRESHOLD = 0.92


@dataclass
class Hypothesis:
    """
    A falsifiable claim about the search space.
    Belief is a Beta(alpha, beta) distribution.
    posterior = alpha / (alpha + beta)  ∈ [0, 1]
    """
    id: str                  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str           = ""
    type: str                = "positive"    # positive|comparative|interaction|null
    alpha: float             = PRIOR_ALPHA
    beta:  float             = PRIOR_BETA
    importance: float        = 0.5           # expected impact if true (0–1)
    config_constraint: dict  = field(default_factory=dict)  # locked values for controlled runs
    status: str              = "active"      # active|supported|refuted|complex|dissolved
    created_at: float        = field(default_factory=time.time)
    n_experiments: int       = 0
    evidence_log: list       = field(default_factory=list)
    source: str              = "default"   # "default" | "llm_proposed" — controls credibility ramp
    parent_id: Optional[str] = None
    children_ids: list[str]  = field(default_factory=list)
    linked_ids: list[str]    = field(default_factory=list)
    canonical_id: Optional[str] = None

    # Phase A: eternal uncertainty control
    uncertain_streak: int    = 0
    decision_sprints_run: int = 0
    last_sprint_at: int      = -10_000
    allocation_penalty: float = 1.0

    # Phase B: Gaussian dual-run evidence (effect-size aware)
    effect_mean: float       = 0.0
    effect_m2: float         = 0.0      # Welford accumulator
    effect_n: int            = 0
    effect_prior_sd: float   = 0.05
    effect_eps: float        = 0.002

    # ── Derived ──────────────────────────────────────────────────────────

    @property
    def posterior(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Bernoulli entropy: 1.0 at P=0.5, 0 at P=0 or P=1."""
        p = self.posterior
        return 4 * p * (1 - p)

    @property
    def llm_credibility(self) -> float:
        """
        LLM-proposed hypotheses start at 25% resource weight and ramp to 100%
        over 12 experiments as evidence accumulates and validates the claim.
        Human/default hypotheses always have full credibility from the start.
        """
        if self.source != "llm_proposed":
            return 1.0
        return min(1.0, 0.25 + 0.0625 * self.n_experiments)  # 0.25 at n=0, 1.0 at n=12

    @property
    def information_value(self) -> float:
        """Reduction in uncertainty × expected impact × LLM credibility ramp."""
        return self.uncertainty * self.importance * self.llm_credibility * self.allocation_penalty

    @property
    def effect_mu(self) -> float:
        return float(self.effect_mean)

    @property
    def effect_variance(self) -> float:
        if self.effect_n < 2:
            return self.effect_prior_sd ** 2
        return max(1e-9, self.effect_m2 / (self.effect_n - 1))

    @property
    def effect_sem(self) -> float:
        # standard error of posterior mean proxy
        n = max(1, self.effect_n)
        return math.sqrt(self.effect_variance / n)

    @property
    def gaussian_support_probability(self) -> float:
        """P(mu < -eps): effect-size-aware support probability."""
        return float(stats.norm.cdf((-self.effect_eps - self.effect_mu) / max(self.effect_sem, 1e-6)))

    @property
    def gaussian_refute_probability(self) -> float:
        """P(mu > +eps): effect-size-aware refute probability."""
        z = (self.effect_eps - self.effect_mu) / max(self.effect_sem, 1e-6)
        return float(1.0 - stats.norm.cdf(z))

    @property
    def gaussian_rope_probability(self) -> float:
        """P(-eps <= mu <= +eps)."""
        lo = (-self.effect_eps - self.effect_mu) / max(self.effect_sem, 1e-6)
        hi = (self.effect_eps - self.effect_mu) / max(self.effect_sem, 1e-6)
        return float(stats.norm.cdf(hi) - stats.norm.cdf(lo))

    @property
    def in_uncertainty_band(self) -> bool:
        p = self.posterior
        return UNCERTAINTY_BAND_LO <= p <= UNCERTAINTY_BAND_HI

    @property
    def in_decision_sprint(self) -> bool:
        if self.decision_sprints_run <= 0:
            return False
        return (self.n_experiments - self.last_sprint_at) <= SPRINT_WINDOW_EXPERIMENTS

    @property
    def credible_interval_90(self) -> tuple[float, float]:
        """
        90% exact Bayesian credible interval for the posterior.
        """
        dist = stats.beta(self.alpha, self.beta)
        lo90, hi90 = dist.ppf(0.05), dist.ppf(0.95)
        return (float(lo90), float(hi90))

    @property
    def support_probability(self) -> float:
        """Posterior probability that this hypothesis is meaningfully supported."""
        dist = stats.beta(self.alpha, self.beta)
        return float(1.0 - dist.cdf(SUPPORT_THRESHOLD))

    @property
    def refute_probability(self) -> float:
        """Posterior probability that this hypothesis is meaningfully refuted."""
        dist = stats.beta(self.alpha, self.beta)
        return float(dist.cdf(REFUTE_THRESHOLD))

    @property
    def rope_probability(self) -> float:
        """Probability mass inside the ROPE around indecision (roughly no directional evidence)."""
        dist = stats.beta(self.alpha, self.beta)
        return float(dist.cdf(SUPPORT_THRESHOLD) - dist.cdf(REFUTE_THRESHOLD))

    @property
    def evidence_strength(self) -> str:
        support = self.support_probability
        refute = self.refute_probability
        dominant = max(support, refute)
        if dominant >= 0.99:
            return "decisive"
        if dominant >= 0.95:
            return "strong"
        if dominant >= 0.90:
            return "moderate"
        return "weak"

    # ── Update ───────────────────────────────────────────────────────────

    def update(self, delta_bpb: float, config_used: Optional[dict] = None):
        """
        Bayesian update: delta_bpb < 0 (improvement) = win.
        """
        if delta_bpb < WIN_THRESHOLD:
            self.alpha += 1
            outcome = f"WIN  delta={delta_bpb:+.4f}"
        else:
            self.beta  += 1
            outcome = f"LOSS delta={delta_bpb:+.4f}"

        # Phase B: update Gaussian dual-run statistics (Welford)
        self.effect_n += 1
        d1 = delta_bpb - self.effect_mean
        self.effect_mean += d1 / self.effect_n
        d2 = delta_bpb - self.effect_mean
        self.effect_m2 += d1 * d2

        self.n_experiments += 1

        # Phase A: eternal uncertainty tracking
        if self.in_uncertainty_band:
            self.uncertain_streak += 1
        else:
            self.uncertain_streak = 0

        entry = f"[n={self.n_experiments}] {outcome}"
        if config_used:
            relevant = {k: v for k, v in config_used.items() if k in self.config_constraint or not self.config_constraint}
            entry += f"  cfg={json.dumps(relevant)}"
        self.evidence_log.append(entry)
        self._refresh_status()

        # If we became decisive again, restore allocation penalty.
        if self.support_probability >= EVIDENCE_CONFIDENCE or self.refute_probability >= EVIDENCE_CONFIDENCE:
            self.allocation_penalty = 1.0

    def maybe_trigger_decision_sprint(self) -> bool:
        """
        If uncertainty remains near 0.5 for too long, trigger a decisive sprint.
        Returns True exactly when a new sprint is triggered.
        """
        cooldown_ok = (self.n_experiments - self.last_sprint_at) >= SPRINT_COOLDOWN_EXPERIMENTS
        if self.uncertain_streak >= UNCERTAINTY_STREAK_TRIGGER and cooldown_ok:
            self.decision_sprints_run += 1
            self.last_sprint_at = self.n_experiments
            self.allocation_penalty = 0.5
            self.evidence_log.append(
                f"[n={self.n_experiments}] DECISION_SPRINT triggered "
                f"(uncertain_streak={self.uncertain_streak}, penalty={self.allocation_penalty})"
            )
            return True
        return False

    def _refresh_status(self):
        if self.n_experiments < 8:
            return
        if self.support_probability >= EVIDENCE_CONFIDENCE:
            self.status = "supported"
        elif self.refute_probability >= EVIDENCE_CONFIDENCE:
            self.status = "refuted"
        else:
            self.status = "active"

    # ── Strategy hints ────────────────────────────────────────────────────

    def needs_falsification_run(self) -> bool:
        """
        We think it's probably false but haven't run a clean controlled test.
        Triggers a dedicated Pop C style population.
        """
        return (
            0.05 < self.posterior < 0.35
            and self.n_experiments < 20
            and self.status == "active"
            and bool(self.config_constraint)
        )

    def is_concluded(self) -> bool:
        return self.n_experiments >= 12 and self.status in ("supported", "refuted")

    def summary(self) -> str:
        lo, hi = self.credible_interval_90
        return (
            f'P={self.posterior:.2f} CI=[{lo:.2f},{hi:.2f}] '
            f'support={self.support_probability:.2f} refute={self.refute_probability:.2f} '
            f'n={self.n_experiments} [{self.status}]  "{self.statement}"'
        )

    def evidence_json(self) -> dict:
        lo, hi = self.credible_interval_90
        return {
            "hypothesis_id": self.id,
            "statement": self.statement,
            "posterior": round(self.posterior, 4),
            "credible_interval_90": [round(lo, 4), round(hi, 4)],
            "support_probability": round(self.support_probability, 4),
            "refute_probability": round(self.refute_probability, 4),
            "rope_probability": round(self.rope_probability, 4),
            "evidence_strength": self.evidence_strength,
            "status": self.status,
            "n_experiments": self.n_experiments,
            "effect_mu": round(self.effect_mu, 6),
            "effect_sem": round(self.effect_sem, 6),
            "gaussian_support_probability": round(self.gaussian_support_probability, 4),
            "gaussian_refute_probability": round(self.gaussian_refute_probability, 4),
            "gaussian_rope_probability": round(self.gaussian_rope_probability, 4),
            "uncertain_streak": self.uncertain_streak,
            "decision_sprints_run": self.decision_sprints_run,
            "allocation_penalty": round(self.allocation_penalty, 3),
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id, "statement": self.statement, "type": self.type,
            "alpha": self.alpha, "beta": self.beta, "importance": self.importance,
            "config_constraint": self.config_constraint, "status": self.status,
            "created_at": self.created_at, "n_experiments": self.n_experiments,
            "evidence_log": self.evidence_log, "source": self.source,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "linked_ids": self.linked_ids,
            "canonical_id": self.canonical_id,
            "uncertain_streak": self.uncertain_streak,
            "decision_sprints_run": self.decision_sprints_run,
            "last_sprint_at": self.last_sprint_at,
            "allocation_penalty": self.allocation_penalty,
            "effect_mean": self.effect_mean,
            "effect_m2": self.effect_m2,
            "effect_n": self.effect_n,
            "effect_prior_sd": self.effect_prior_sd,
            "effect_eps": self.effect_eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Registry ──────────────────────────────────────────────────────────────────

class HypothesisRegistry:
    """
    Holds all active and archived hypotheses.
    Computes worker allocation via softmax(information_value).
    """
    SOFTMAX_TEMP    = 1.0
    MIN_WORKERS     = 3    # minimum workers per active hypothesis

    def __init__(self):
        self._active:   dict[str, Hypothesis] = {}
        self._archived: dict[str, Hypothesis] = {}

    def add(self, h: Hypothesis):
        if h.parent_id:
            parent = self.get(h.parent_id)
            if parent and h.id not in parent.children_ids:
                parent.children_ids.append(h.id)
        self._active[h.id] = h

    @staticmethod
    def _normalize_statement(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    @classmethod
    def _semantic_similarity(cls, a: str, b: str) -> float:
        a_n, b_n = cls._normalize_statement(a), cls._normalize_statement(b)
        if not a_n or not b_n:
            return 0.0
        seq = difflib.SequenceMatcher(None, a_n, b_n).ratio()
        ta, tb = cls._tokenize(a_n), cls._tokenize(b_n)
        jac = len(ta & tb) / max(1, len(ta | tb))
        return 0.6 * seq + 0.4 * jac

    def get(self, h_id: str) -> Optional[Hypothesis]:
        return self._active.get(h_id) or self._archived.get(h_id)

    def archive(self, h_id: str):
        h = self._active.pop(h_id, None)
        if h:
            self._archived[h_id] = h

    @property
    def active(self) -> list[Hypothesis]:
        return list(self._active.values())

    @property
    def archived(self) -> list[Hypothesis]:
        return list(self._archived.values())

    def convergence_winner(self) -> Optional["Hypothesis"]:
        """
        Returns the hypothesis that should absorb most workers during convergence:
        the one with support_probability >= 0.85 and n >= 15 experiments.
        Returns None when not yet in convergence territory.
        """
        candidates = [
            h for h in self.active
            if h.support_probability >= 0.85 and h.n_experiments >= 15
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda h: h.support_probability)

    def allocate_workers(self, total_workers: int) -> dict[str, int]:
        """
        Worker allocation. Two modes:

        Normal: softmax over information_value (uncertainty × importance × llm_credibility).
          LLM-proposed hypotheses start at 25% weight and ramp to full over 12 experiments.

        Convergence: when one hypothesis reaches support_probability >= 0.85 with n >= 15,
          concentrate 70% of workers on it for multi-worker validation, 30% shared among others.
        """
        hs = self.active
        if not hs:
            return {}

        # Convergence mode: one hypothesis strongly validated → spike allocation
        winner = self.convergence_winner()
        if winner and len(hs) > 1:
            others = [h for h in hs if h.id != winner.id]
            others_total = max(len(others) * self.MIN_WORKERS, int(total_workers * 0.30))
            per_other = max(self.MIN_WORKERS, others_total // len(others))
            result = {h.id: per_other for h in others}
            result[winner.id] = total_workers - sum(result.values())
            return result


        ivs = np.array([
            max(
                h.information_value,
                0.25 if h.in_decision_sprint else (0.15 if h.needs_falsification_run() else 0.0)
            )
            for h in hs
        ])
        # Softmax
        ivs_shifted = ivs / self.SOFTMAX_TEMP
        exp_ivs = np.exp(ivs_shifted - ivs_shifted.max())
        probs = exp_ivs / exp_ivs.sum()

        # Enforce minimum
        n_active = len(hs)
        available = max(0, total_workers - n_active * self.MIN_WORKERS)
        raw = probs * available + self.MIN_WORKERS

        counts = np.floor(raw).astype(int)
        remainder = total_workers - int(counts.sum())
        fractional = raw - counts
        top_idx = np.argsort(fractional)[::-1][:max(0, remainder)]
        counts[top_idx] += 1

        return {hs[i].id: int(counts[i]) for i in range(len(hs))}

    def ingest_experiment(self, config_delta: dict, delta_bpb: float):
        """
        Route an experiment result to relevant hypotheses and update beliefs.
        Simple routing: if a hypothesis has a config_constraint key present in
        config_delta, it's relevant.
        """
        for h in self.active:
            relevant = (
                not h.config_constraint                          # global hypothesis
                or any(k in config_delta for k in h.config_constraint)
            )
            if relevant and delta_bpb is not None:
                h.update(delta_bpb, config_delta)
                h.maybe_trigger_decision_sprint()

    def evaluate_llm_proposal(self, proposal: dict) -> dict:
        """
        LLM proposals are advisory only.
        Returns a structured gate decision; acceptance only adds to the registry.
        It does not force allocation or immediate pursuit.
        """
        statement = str(proposal.get("statement", "")).strip()
        normalized = " ".join(statement.lower().split())
        if not statement:
            return {
                "accepted": False,
                "reason": "missing_statement",
                "registry_add": False,
                "immediate_forced_pursuit": False,
            }

        for existing in self.active + self.archived:
            existing_norm = " ".join(existing.statement.lower().split())
            if existing_norm == normalized:
                return {
                    "accepted": False,
                    "reason": "duplicate_statement",
                    "registry_add": False,
                    "immediate_forced_pursuit": False,
                }

        # semantic near-duplicate detection
        best_match = None
        best_score = 0.0
        for existing in self.active + self.archived:
            score = self._semantic_similarity(statement, existing.statement)
            if score > best_score:
                best_score = score
                best_match = existing
        if best_match and best_score >= SEMANTIC_DUPLICATE_THRESHOLD:
            return {
                "accepted": False,
                "reason": "duplicate_semantic",
                "duplicate_of_hypothesis_id": best_match.id,
                "similarity": round(best_score, 3),
                "registry_add": False,
                "immediate_forced_pursuit": False,
            }

        importance = float(proposal.get("importance", 0.0) or 0.0)
        if importance < 0.15:
            return {
                "accepted": False,
                "reason": "importance_too_low",
                "registry_add": False,
                "immediate_forced_pursuit": False,
            }

        config_constraint = proposal.get("config_constraint") or {}
        if not isinstance(config_constraint, dict):
            return {
                "accepted": False,
                "reason": "invalid_constraint",
                "registry_add": False,
                "immediate_forced_pursuit": False,
            }

        parent_id = proposal.get("parent_id")
        if parent_id and not self.get(str(parent_id)):
            return {
                "accepted": False,
                "reason": "invalid_parent_id",
                "registry_add": False,
                "immediate_forced_pursuit": False,
            }

        return {
            "accepted": True,
            "reason": "schema_valid_and_novel",
            "registry_add": True,
            "immediate_forced_pursuit": False,
        }

    def ingest_llm_proposals(self, proposals: list[dict]) -> list[dict]:
        """Gate and optionally add LLM-suggested hypotheses to the registry."""
        decisions = []
        for proposal in proposals:
            gate = self.evaluate_llm_proposal(proposal)
            decision = {
                "proposal": proposal,
                "engine_gate": gate,
            }
            if gate["accepted"]:
                self.add(Hypothesis(
                    statement=str(proposal["statement"]).strip(),
                    type=str(proposal.get("type", "positive") or "positive"),
                    importance=float(proposal.get("importance", 0.5)),
                    config_constraint=dict(proposal.get("config_constraint") or {}),
                    source="llm_proposed",
                    parent_id=(str(proposal.get("parent_id")) if proposal.get("parent_id") else None),
                ))
            decisions.append(decision)
        return decisions

    def theory_graph(self) -> dict:
        nodes = []
        edges = []
        for h in self.active + self.archived:
            nodes.append({
                "id": h.id,
                "statement": h.statement,
                "status": h.status,
                "posterior": round(h.posterior, 4),
                "effect_mu": round(h.effect_mu, 6),
                "effect_sem": round(h.effect_sem, 6),
                "parent_id": h.parent_id,
                "children_ids": list(h.children_ids),
                "linked_ids": list(h.linked_ids),
                "canonical_id": h.canonical_id,
            })
            if h.parent_id:
                edges.append({"type": "decomposes_into", "from": h.parent_id, "to": h.id})
            for lid in h.linked_ids:
                edges.append({"type": "linked", "from": h.id, "to": lid})
        return {"nodes": nodes, "edges": edges}

    def to_dict(self) -> dict:
        return {
            "active":   {hid: h.to_dict() for hid, h in self._active.items()},
            "archived": {hid: h.to_dict() for hid, h in self._archived.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HypothesisRegistry":
        reg = cls()
        for hid, hdata in data.get("active", {}).items():
            reg._active[hid] = Hypothesis.from_dict(hdata)
        for hid, hdata in data.get("archived", {}).items():
            reg._archived[hid] = Hypothesis.from_dict(hdata)
        return reg


# ── Default hypotheses for nanochat / autoresearch ────────────────────────────

DEFAULT_HYPOTHESES = [
    dict(statement="Depth > 10 improves val_bpb",           type="positive",     importance=0.90, config_constraint={}),
    dict(statement="Learning rate × batch size interact",    type="interaction",  importance=0.75, config_constraint={}),
    dict(statement="WINDOW_PATTERN affects val_bpb",         type="comparative",  importance=0.50,
         config_constraint={"DEPTH": 8, "learning_rate": 1e-3, "TOTAL_BATCH_SIZE": 65536}),
    dict(statement="Muon LR ceiling below 8e-4",             type="positive",     importance=0.60, config_constraint={}),
    dict(statement="weight_decay matters",                   type="positive",     importance=0.45,
         config_constraint={"DEPTH": 8, "learning_rate": 1e-3}),
]


def make_default_registry() -> HypothesisRegistry:
    reg = HypothesisRegistry()
    for hdata in DEFAULT_HYPOTHESES:
        reg.add(Hypothesis(**hdata))
    return reg
