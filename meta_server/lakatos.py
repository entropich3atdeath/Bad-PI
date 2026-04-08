"""
meta_server/lakatos.py

Lakatosian research-programme layer.

This module is intentionally additive and backward-compatible:
- Existing hypothesis-level Bayesian updates remain unchanged.
- Programmes sit above hypotheses as the unit of scientific commitment.
- Hypotheses are treated as protective-belt claims linked to programmes.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


PROGRESSIVE_THRESHOLD = 0.55
DEGENERATIVE_THRESHOLD = 0.35


@dataclass
class NovelPrediction:
    id: str = field(default_factory=lambda: f"pred_{str(uuid.uuid4())[:8]}")
    statement: str = ""
    hypothesis_id: Optional[str] = None
    predicted_before_evidence: bool = True
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    confirmed: Optional[bool] = None
    resolved_at_experiment: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "hypothesis_id": self.hypothesis_id,
            "predicted_before_evidence": self.predicted_before_evidence,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "confirmed": self.confirmed,
            "resolved_at_experiment": self.resolved_at_experiment,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NovelPrediction":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ResearchProgramme:
    id: str = field(default_factory=lambda: f"prog_{str(uuid.uuid4())[:8]}")
    name: str = "Main programme"
    hard_core: list[str] = field(default_factory=list)
    protective_belt_ids: list[str] = field(default_factory=list)
    positive_heuristic: str = "Expand controlled tests around strongest effects."
    negative_heuristic: str = "Do not rewrite hard-core assumptions based on single anomalies."
    status: str = "progressive"  # progressive | degenerative | abandoned
    created_at: float = field(default_factory=time.time)
    last_assessed_at: float = field(default_factory=time.time)
    anomaly_count: int = 0
    belt_modifications: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    novel_predictions: list[NovelPrediction] = field(default_factory=list)

    @property
    def total_predictions(self) -> int:
        return len(self.novel_predictions)

    @property
    def confirmed_novel_predictions(self) -> int:
        return sum(
            1
            for p in self.novel_predictions
            if p.predicted_before_evidence and p.resolved and p.confirmed is True
        )

    @property
    def disconfirmed_novel_predictions(self) -> int:
        return sum(
            1
            for p in self.novel_predictions
            if p.predicted_before_evidence and p.resolved and p.confirmed is False
        )

    @property
    def post_hoc_explanations(self) -> int:
        return sum(1 for p in self.novel_predictions if not p.predicted_before_evidence)

    @property
    def progressiveness_ratio(self) -> float:
        denom = max(1, self.confirmed_novel_predictions + self.disconfirmed_novel_predictions)
        return float(self.confirmed_novel_predictions / denom)

    def record_prediction(
        self,
        statement: str,
        hypothesis_id: Optional[str] = None,
        predicted_before_evidence: bool = True,
    ) -> NovelPrediction:
        pred = NovelPrediction(
            statement=statement,
            hypothesis_id=hypothesis_id,
            predicted_before_evidence=predicted_before_evidence,
        )
        self.novel_predictions.append(pred)
        return pred

    def resolve_prediction(self, prediction_id: str, confirmed: bool, at_experiment: Optional[int] = None) -> bool:
        for p in self.novel_predictions:
            if p.id == prediction_id:
                p.resolved = True
                p.confirmed = bool(confirmed)
                p.resolved_at_experiment = at_experiment
                return True
        return False

    def register_anomaly(self):
        self.anomaly_count += 1

    def register_belt_modification(self):
        self.belt_modifications += 1

    def assess_status(self):
        self.last_assessed_at = time.time()
        ratio = self.progressiveness_ratio
        if ratio >= PROGRESSIVE_THRESHOLD:
            self.status = "progressive"
        elif ratio <= DEGENERATIVE_THRESHOLD and self.total_predictions >= 3:
            self.status = "degenerative"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hard_core": self.hard_core,
            "protective_belt_ids": self.protective_belt_ids,
            "positive_heuristic": self.positive_heuristic,
            "negative_heuristic": self.negative_heuristic,
            "status": self.status,
            "created_at": self.created_at,
            "last_assessed_at": self.last_assessed_at,
            "anomaly_count": self.anomaly_count,
            "belt_modifications": self.belt_modifications,
            "metadata": self.metadata,
            "novel_predictions": [p.to_dict() for p in self.novel_predictions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchProgramme":
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "novel_predictions"}
        obj = cls(**payload)
        obj.novel_predictions = [NovelPrediction.from_dict(p) for p in data.get("novel_predictions", [])]
        return obj


class ProgrammeRegistry:
    """
    Lakatosian layer tracking programme health and rivalry.
    """

    def __init__(self):
        self._active: dict[str, ResearchProgramme] = {}
        self._archived: dict[str, ResearchProgramme] = {}
        self._hypothesis_to_programme: dict[str, str] = {}
        self._pending_belt_modifications: list[dict] = []
        self._last_hypothesis_status: dict[str, str] = {}

    @property
    def active(self) -> list[ResearchProgramme]:
        return list(self._active.values())

    @property
    def archived(self) -> list[ResearchProgramme]:
        return list(self._archived.values())

    @property
    def pending_belt_modifications(self) -> list[dict]:
        return list(self._pending_belt_modifications)

    def get(self, programme_id: str) -> Optional[ResearchProgramme]:
        return self._active.get(programme_id) or self._archived.get(programme_id)

    def programme_by_name(self, name: str) -> Optional[ResearchProgramme]:
        """Case-insensitive name lookup across active programmes."""
        name_lo = name.strip().lower()
        for p in self._active.values():
            if p.name.strip().lower() == name_lo:
                return p
        return None

    def add(self, programme: ResearchProgramme):
        self._active[programme.id] = programme

    @classmethod
    def from_core_dimensions(cls, dimensions: list[dict]) -> "ProgrammeRegistry":
        """
        Seed the registry from schema dimensions tagged dim_role='core'.

        For each core categorical dimension (e.g. OPTIMIZER with values [adam, sgd])
        one ResearchProgramme is created per category value.  That value becomes the
        hard_core commitment; variable-role dimensions form the tunable protective belt.

        If no core dimensions exist the registry falls back to a single default programme
        (same behaviour as the old Popper-mode bootstrap).
        """
        reg = cls()
        core_dims = [d for d in dimensions if d.get("dim_role") == "core"]

        if not core_dims:
            reg.ensure_default_programme(dimensions)
            return reg

        variable_dims = [d for d in dimensions if d.get("dim_role") != "core"]
        var_names = [d["name"] for d in variable_dims]

        for core_dim in core_dims:
            dim_name = core_dim["name"]
            cats = core_dim.get("categories") or []

            if not cats:
                # Non-categorical core dim (e.g. int/float axis): single programme
                prog = ResearchProgramme(
                    name=f"{dim_name} programme",
                    hard_core=[f"{dim_name} is the primary research axis"],
                    positive_heuristic=(
                        f"Systematically vary {dim_name} and all supporting variable dims: "
                        + ", ".join(var_names[:5]) + "."
                    ),
                    negative_heuristic=(
                        f"Do not abandon the {dim_name} axis based on a single anomalous run."
                    ),
                    metadata={"core_dimension": dim_name},
                )
                reg.add(prog)
            else:
                for cat_val in cats:
                    prog_name = f"{dim_name}={cat_val}"
                    prog = ResearchProgramme(
                        name=prog_name,
                        hard_core=[f"{dim_name} = {cat_val}"],
                        positive_heuristic=(
                            f"Within the {cat_val} paradigm, optimise variable dims: "
                            + ", ".join(var_names[:5]) + "."
                        ),
                        negative_heuristic=(
                            f"Do not abandon {cat_val} based on isolated anomalies; "
                            f"refine the protective belt (adjust variable dims) instead."
                        ),
                        metadata={
                            "core_dimension": dim_name,
                            "core_value": str(cat_val),
                        },
                    )
                    reg.add(prog)

        return reg

    def ensure_default_programme(self, dimensions: list[dict]) -> ResearchProgramme:
        existing = next((p for p in self._active.values() if p.name == "Main programme"), None)
        if existing:
            return existing

        top_dims = [d.get("name") for d in dimensions[:3] if d.get("name")]
        core = [f"{name} is a primary driver candidate" for name in top_dims] or ["Search-space structure carries predictive signal"]
        programme = ResearchProgramme(
            name="Main programme",
            hard_core=core,
            positive_heuristic="Refine auxiliary hypotheses that increase out-of-sample performance.",
            negative_heuristic="Do not discard hard-core commitments due to isolated anomalies.",
        )
        self.add(programme)
        return programme

    def link_hypothesis(self, hypothesis_id: str, programme_id: str):
        programme = self._active.get(programme_id)
        if not programme:
            return
        if hypothesis_id not in programme.protective_belt_ids:
            programme.protective_belt_ids.append(hypothesis_id)
        self._hypothesis_to_programme[hypothesis_id] = programme_id

    def programme_for_hypothesis(self, hypothesis_id: str) -> Optional[ResearchProgramme]:
        pid = self._hypothesis_to_programme.get(hypothesis_id)
        return self._active.get(pid) if pid else None

    def sync_from_hypotheses(self, hypotheses: list[Any], dimensions: list[dict]):
        default = self.ensure_default_programme(dimensions)
        active_hids = {str(getattr(h, "id", "")) for h in hypotheses if getattr(h, "id", None)}

        # Ensure every hypothesis is linked to a programme.
        for h in hypotheses:
            hid = str(getattr(h, "id", ""))
            if not hid:
                continue
            pid = getattr(h, "programme_id", None) or self._hypothesis_to_programme.get(hid) or default.id
            self.link_hypothesis(hid, pid)
            # keep hypothesis object synchronized for compatibility
            try:
                setattr(h, "programme_id", pid)
            except Exception:
                pass

        # Remove stale links from active belt lists
        for programme in self._active.values():
            programme.protective_belt_ids = [hid for hid in programme.protective_belt_ids if hid in active_hids]

        # Remove stale hypothesis mappings
        for hid in list(self._hypothesis_to_programme.keys()):
            if hid not in active_hids:
                self._hypothesis_to_programme.pop(hid, None)

    def record_hypothesis_event(self, hypothesis: Any):
        """
        Translate hypothesis-level outcomes into Lakatosian programme events.
        """
        hid = str(getattr(hypothesis, "id", ""))
        if not hid:
            return
        programme = self.programme_for_hypothesis(hid)
        if not programme:
            return

        status = str(getattr(hypothesis, "status", "active"))
        statement = str(getattr(hypothesis, "statement", ""))

        previous_status = self._last_hypothesis_status.get(hid)
        self._last_hypothesis_status[hid] = status
        if previous_status == status:
            return

        if status == "refuted":
            programme.register_anomaly()
            self._pending_belt_modifications.append({
                "programme_id": programme.id,
                "hypothesis_id": hid,
                "type": "belt_modification",
                "reason": "auxiliary_refuted",
                "suggestion": f"Refine auxiliary claim around anomaly: {statement}",
                "created_at": time.time(),
            })
        elif status == "supported":
            if not any(p.hypothesis_id == hid for p in programme.novel_predictions):
                pred = programme.record_prediction(
                    statement=f"Novel effect expected: {statement}",
                    hypothesis_id=hid,
                    predicted_before_evidence=True,
                )
                programme.resolve_prediction(pred.id, confirmed=True)

    def classify_programmes(self):
        for programme in self._active.values():
            programme.assess_status()

    def rivalry_snapshot(self) -> list[dict]:
        rows = []
        for p in self._active.values():
            rows.append({
                "programme_id": p.id,
                "name": p.name,
                "status": p.status,
                "progressiveness_ratio": round(p.progressiveness_ratio, 4),
                "confirmed_novel_predictions": p.confirmed_novel_predictions,
                "disconfirmed_novel_predictions": p.disconfirmed_novel_predictions,
                "post_hoc_explanations": p.post_hoc_explanations,
                "anomaly_count": p.anomaly_count,
                "belt_modifications": p.belt_modifications,
                "hard_core": list(p.hard_core),
            })
        rows.sort(key=lambda x: (x["status"] != "progressive", -x["progressiveness_ratio"]))
        return rows

    def to_dict(self) -> dict:
        return {
            "active": {pid: p.to_dict() for pid, p in self._active.items()},
            "archived": {pid: p.to_dict() for pid, p in self._archived.items()},
            "hypothesis_to_programme": dict(self._hypothesis_to_programme),
            "pending_belt_modifications": list(self._pending_belt_modifications),
            "last_hypothesis_status": dict(self._last_hypothesis_status),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProgrammeRegistry":
        reg = cls()
        reg._active = {
            pid: ResearchProgramme.from_dict(pdata)
            for pid, pdata in data.get("active", {}).items()
        }
        reg._archived = {
            pid: ResearchProgramme.from_dict(pdata)
            for pid, pdata in data.get("archived", {}).items()
        }
        reg._hypothesis_to_programme = dict(data.get("hypothesis_to_programme", {}))
        reg._pending_belt_modifications = list(data.get("pending_belt_modifications", []))
        reg._last_hypothesis_status = dict(data.get("last_hypothesis_status", {}))
        return reg
