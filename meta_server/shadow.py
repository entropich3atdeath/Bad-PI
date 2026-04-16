from __future__ import annotations

import json
from typing import Any

from . import store


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _is_close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _matches_constraint(config_delta: dict, constraint: dict) -> bool:
    if not constraint:
        return True
    for k, v in constraint.items():
        if k not in config_delta:
            return False
        if not _is_close(config_delta.get(k), v):
            return False
    return True


def _classify(delta_bpb: Any, support_delta_lte: float, refute_delta_gte: float) -> str:
    if delta_bpb is None:
        return "inconclusive"
    d = float(delta_bpb)
    if d <= float(support_delta_lte):
        return "support"
    if d >= float(refute_delta_gte):
        return "refute"
    return "inconclusive"


def evaluate_experiment_for_active_hypotheses(exp: dict) -> list[dict]:
    """
    Match one completed run against all active shadow hypotheses and persist evidence.
    Returns inserted/upserted evidence rows as dicts.
    """
    config_delta = _as_dict(exp.get("config_delta"))
    active = store.list_shadow_hypotheses(active_only=True)
    created: list[dict] = []

    for h in active:
        constraint = _as_dict(h.get("config_constraint"))
        if not _matches_constraint(config_delta, constraint):
            continue

        verdict = _classify(
            exp.get("delta_bpb"),
            support_delta_lte=float(h.get("support_delta_lte", -0.001)),
            refute_delta_gte=float(h.get("refute_delta_gte", 0.001)),
        )
        reason = (
            f"matched config_constraint={json.dumps(constraint, sort_keys=True)}; "
            f"delta_bpb={exp.get('delta_bpb')}"
        )

        saved = store.upsert_shadow_evidence(
            hypothesis_id=str(h["id"]),
            exp_id=str(exp["exp_id"]),
            worker_id=str(exp.get("worker_id") or ""),
            verdict=verdict,
            delta_bpb=(float(exp["delta_bpb"]) if exp.get("delta_bpb") is not None else None),
            val_bpb=(float(exp["val_bpb"]) if exp.get("val_bpb") is not None else None),
            reason=reason,
        )
        created.append(saved)

    return created


def backfill(limit: int = 5000) -> dict:
    """
    Re-evaluate recent completed experiments for all active shadow hypotheses.
    Idempotent via (hypothesis_id, exp_id) unique constraint.
    """
    rows = store.recent_experiments(limit=max(1, int(limit or 1)))
    inserted = 0
    matched_runs = 0
    for exp in rows:
        created = evaluate_experiment_for_active_hypotheses(exp)
        if created:
            matched_runs += 1
            inserted += len(created)
    return {
        "ok": True,
        "considered_experiments": len(rows),
        "matched_runs": matched_runs,
        "evidence_rows_touched": inserted,
    }
