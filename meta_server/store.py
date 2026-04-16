"""
meta_server/store.py
All SQLite read/write operations. Thread-safe via connection-per-call pattern.
"""
from __future__ import annotations
import json
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(os.environ.get("DB_PATH", str(Path(__file__).parent / "experiments.db")))
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_PROGRAM_MD_PATH = WORKSPACE_ROOT / "program.md"

# Duplicate-trial policy (confidence vs waste)
MAX_TRIALS_PER_CONFIG = 6        # hard cap for any exact config
MAX_INFLIGHT_PER_CONFIG = 2      # prevent too many simultaneous duplicates


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript(SCHEMA_PATH.read_text())
        cols = {row[1] for row in con.execute("PRAGMA table_info(workers)").fetchall()}
        if "auth_token" not in cols:
            con.execute("ALTER TABLE workers ADD COLUMN auth_token TEXT")

        dim_cols = {row[1] for row in con.execute("PRAGMA table_info(dimensions)").fetchall()}
        if "is_canary" not in dim_cols:
            con.execute("ALTER TABLE dimensions ADD COLUMN is_canary INTEGER DEFAULT 0")
        if "canary_prob" not in dim_cols:
            con.execute("ALTER TABLE dimensions ADD COLUMN canary_prob REAL DEFAULT 1.0")


# ── Workers ───────────────────────────────────────────────────────────────────

def register_worker(worker_id: str, gpu_type: str, baseline_bpb: float, contact: Optional[str]) -> str:
    now = time.time()
    auth_token = secrets.token_urlsafe(32)
    with _conn() as con:
        con.execute("""
            INSERT INTO workers (worker_id, gpu_type, baseline_bpb, contact, auth_token, registered_at, last_seen)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(worker_id) DO UPDATE SET
                gpu_type=excluded.gpu_type,
                baseline_bpb=excluded.baseline_bpb,
                auth_token=excluded.auth_token,
                last_seen=excluded.last_seen
        """, (worker_id, gpu_type, baseline_bpb, contact, auth_token, now, now))
    return auth_token


def touch_worker(worker_id: str):
    with _conn() as con:
        con.execute("UPDATE workers SET last_seen=? WHERE worker_id=?", (time.time(), worker_id))


def get_worker(worker_id: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute("SELECT * FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
        return dict(row) if row else None


def verify_worker_token(worker_id: str, auth_token: Optional[str]) -> bool:
    if not auth_token:
        return False
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM workers WHERE worker_id=? AND auth_token=?",
            (worker_id, auth_token),
        ).fetchone()
    return row is not None


def active_worker_count(window_seconds: int = 600) -> int:
    cutoff = time.time() - window_seconds
    with _conn() as con:
        return con.execute(
            "SELECT COUNT(*) FROM workers WHERE last_seen > ?", (cutoff,)
        ).fetchone()[0]


# ── Experiments ───────────────────────────────────────────────────────────────

def save_experiment(
    exp_id: str,
    worker_id: str,
    config: dict,
    config_delta: dict,
    val_bpb: Optional[float],
    delta_bpb: Optional[float],
    duration: float,
    status: str = "completed",
    error: Optional[str] = None,
):
    now = time.time()
    with _conn() as con:
        config_json = _config_fingerprint(config)
        delta_json = _config_fingerprint(config_delta)
        con.execute("""
            INSERT OR REPLACE INTO experiments
            (exp_id, worker_id, config, config_delta, val_bpb, delta_bpb,
             duration_seconds, status, error, completed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            exp_id, worker_id,
            config_json, delta_json,
            val_bpb, delta_bpb, duration, status, error, now
        ))
        con.execute(
            "UPDATE workers SET experiment_count=experiment_count+1, last_seen=? WHERE worker_id=?",
            (now, worker_id)
        )
        con.execute(
            "UPDATE config_queue SET status='completed' WHERE exp_id=?", (exp_id,)
        )


def experiment_count() -> int:
    with _conn() as con:
        return con.execute(
            "SELECT COUNT(*) FROM experiments WHERE status='completed'"
        ).fetchone()[0]


def recent_experiments(limit: int = 500) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT e.*, w.baseline_bpb, w.gpu_type
            FROM experiments e
            JOIN workers w ON e.worker_id = w.worker_id
            WHERE e.status='completed'
            ORDER BY e.completed_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def top_experiments(n: int = 10) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM experiments
            WHERE status='completed' AND delta_bpb IS NOT NULL
            ORDER BY delta_bpb ASC
            LIMIT ?
        """, (n,)).fetchall()
        return [dict(r) for r in rows]


def get_experiment(exp_id: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM experiments WHERE exp_id=?",
            (exp_id,),
        ).fetchone()
    return dict(row) if row else None


# ── Config queue ──────────────────────────────────────────────────────────────

def _config_fingerprint(delta: dict) -> str:
    """Canonical JSON key used to identify duplicate configs across queue + experiments."""
    return json.dumps(delta, sort_keys=True, separators=(",", ":"))


def _desired_repeats(priority: float) -> int:
    """
    How many total trials we *aim* for before considering a config statistically confident.
    High-priority configs get more repeats for confidence; low-priority get fewer.
    """
    if priority >= 0.85:
        return 4
    if priority >= 0.70:
        return 3
    if priority >= 0.55:
        return 2
    return 1


def _count_completed_for_config(con: sqlite3.Connection, fingerprint: str) -> int:
    row = con.execute(
        "SELECT COUNT(*) FROM experiments WHERE status='completed' AND config_delta=?",
        (fingerprint,),
    ).fetchone()
    return int(row[0] if row else 0)


def _count_inflight_for_config(con: sqlite3.Connection, fingerprint: str) -> int:
    row = con.execute(
        """
        SELECT COUNT(*) FROM config_queue
        WHERE config_delta=? AND status IN ('pending','assigned')
        """,
        (fingerprint,),
    ).fetchone()
    return int(row[0] if row else 0)

def enqueue_configs(configs: list[tuple[dict, float, str]]):
    """configs: list of (config_delta, priority, note)"""
    with _conn() as con:
        for delta, priority, note in configs:
            fp = _config_fingerprint(delta)
            completed = _count_completed_for_config(con, fp)
            inflight = _count_inflight_for_config(con, fp)
            total_planned = completed + inflight

            desired = _desired_repeats(float(priority))
            max_trials = min(MAX_TRIALS_PER_CONFIG, desired + 2)

            # Avoid wasting resources on excessive duplicates.
            if total_planned >= max_trials:
                continue
            if inflight >= MAX_INFLIGHT_PER_CONFIG:
                continue
            # Once we've reached desired repeats, trickle additional repeats only
            # after current inflight copies finish (for controlled confidence growth).
            if total_planned >= desired and inflight >= 1:
                continue

            con.execute("""
                INSERT INTO config_queue (exp_id, config_delta, priority, note, status)
                VALUES (?,?,?,?,'pending')
            """, (str(uuid.uuid4()), fp, priority, note))


def pop_next_config(worker_id: str) -> Optional[dict]:
    """Atomically assign the highest-priority pending config to this worker."""
    now = time.time()
    # Expire old assigned configs (worker probably died)
    expire_cutoff = now - 600  # 10 min
    with _conn() as con:
        con.execute("""
            UPDATE config_queue SET status='pending', assigned_to=NULL, assigned_at=NULL
            WHERE status='assigned' AND assigned_at < ?
        """, (expire_cutoff,))
        row = con.execute("""
            SELECT exp_id, config_delta, priority, note
            FROM config_queue
            WHERE status='pending'
            ORDER BY priority DESC
            LIMIT 1
        """).fetchone()
        if row is None:
            return None
        con.execute("""
            UPDATE config_queue
            SET status='assigned', assigned_to=?, assigned_at=?
            WHERE exp_id=?
        """, (worker_id, now, row["exp_id"]))
        return {
            "exp_id": row["exp_id"],
            "config_delta": json.loads(row["config_delta"]),
            "priority": row["priority"],
            "note": row["note"],
        }


def pop_next_configs(worker_id: str, limit: int) -> list[dict]:
    """Atomically assign up to `limit` highest-priority pending configs to this worker."""
    now = time.time()
    expire_cutoff = now - 600  # 10 min
    requested = max(1, int(limit or 1))
    with _conn() as con:
        con.execute(
            """
            UPDATE config_queue SET status='pending', assigned_to=NULL, assigned_at=NULL
            WHERE status='assigned' AND assigned_at < ?
            """,
            (expire_cutoff,),
        )
        rows = con.execute(
            """
            SELECT exp_id, config_delta, priority, note
            FROM config_queue
            WHERE status='pending'
            ORDER BY priority DESC
            LIMIT ?
            """,
            (requested,),
        ).fetchall()
        if not rows:
            return []

        claimed: list[dict] = []
        for row in rows:
            con.execute(
                """
                UPDATE config_queue
                SET status='assigned', assigned_to=?, assigned_at=?
                WHERE exp_id=?
                """,
                (worker_id, now, row["exp_id"]),
            )
            claimed.append(
                {
                    "exp_id": row["exp_id"],
                    "config_delta": json.loads(row["config_delta"]),
                    "priority": row["priority"],
                    "note": row["note"],
                }
            )
        return claimed


def release_assigned_configs(worker_id: str, exp_ids: list[str]) -> list[str]:
    """Release worker-owned assigned configs back to the pending queue."""
    if not exp_ids:
        return []
    released: list[str] = []
    with _conn() as con:
        for exp_id in exp_ids:
            row = con.execute(
                """
                SELECT exp_id FROM config_queue
                WHERE exp_id=? AND status='assigned' AND assigned_to=?
                """,
                (exp_id, worker_id),
            ).fetchone()
            if row is None:
                continue
            con.execute(
                """
                UPDATE config_queue
                SET status='pending', assigned_to=NULL, assigned_at=NULL
                WHERE exp_id=?
                """,
                (exp_id,),
            )
            released.append(exp_id)
    return released


def queue_depth() -> int:
    with _conn() as con:
        return con.execute(
            "SELECT COUNT(*) FROM config_queue WHERE status='pending'"
        ).fetchone()[0]


# ── Dimensions ────────────────────────────────────────────────────────────────

def get_dimensions() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM dimensions ORDER BY importance DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d["categories"]:
                d["categories"] = json.loads(d["categories"])
            if d["frozen_value"]:
                d["frozen_value"] = json.loads(d["frozen_value"])
            d["is_canary"] = bool(d.get("is_canary", 0))
            d["canary_prob"] = float(d.get("canary_prob", 1.0) or 1.0)
            result.append(d)
        return result


def dimension_exists(name: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM dimensions WHERE lower(name)=lower(?)",
            (name,),
        ).fetchone()
    return row is not None


def add_dimension(
    *,
    name: str,
    dtype: str,
    min_val: Optional[float],
    max_val: Optional[float],
    log_scale: int,
    categories: Optional[list],
    importance: float = 0.08,
    is_canary: bool = True,
    canary_prob: float = 0.12,
) -> bool:
    if dimension_exists(name):
        return False
    with _conn() as con:
        con.execute(
            """
            INSERT INTO dimensions
            (name, dtype, min_val, max_val, log_scale, categories, frozen, frozen_value,
             importance, n_samples, updated_at, is_canary, canary_prob)
            VALUES (?,?,?,?,?,?,0,NULL,?,?,?, ?,?)
            """,
            (
                name,
                dtype,
                min_val,
                max_val,
                int(log_scale),
                json.dumps(categories) if categories is not None else None,
                float(importance),
                0,
                time.time(),
                1 if is_canary else 0,
                float(canary_prob),
            ),
        )
    return True


def set_dimension_canary(name: str, *, is_canary: bool, canary_prob: float = 1.0):
    with _conn() as con:
        con.execute(
            """
            UPDATE dimensions
            SET is_canary=?, canary_prob=?, updated_at=?
            WHERE name=?
            """,
            (1 if is_canary else 0, float(canary_prob), time.time(), name),
        )


def remove_dimension(name: str):
    with _conn() as con:
        con.execute("DELETE FROM dimensions WHERE name=?", (name,))


def update_dimension_importance(name: str, importance: float, n_samples: int):
    with _conn() as con:
        con.execute("""
            UPDATE dimensions SET importance=?, n_samples=?, updated_at=?
            WHERE name=?
        """, (importance, n_samples, time.time(), name))


def freeze_dimension(name: str, value: Any):
    with _conn() as con:
        con.execute("""
            UPDATE dimensions SET frozen=1, frozen_value=?, updated_at=?
            WHERE name=?
        """, (json.dumps(value), time.time(), name))


def unfreeze_dimension(name: str):
    with _conn() as con:
        con.execute(
            """
            UPDATE dimensions SET frozen=0, frozen_value=NULL, updated_at=?
            WHERE name=?
            """,
            (time.time(), name),
        )


# ── Shadow hypotheses / scorecards ────────────────────────────────────────────

def create_shadow_hypothesis(
    *,
    statement: str,
    config_constraint: dict,
    support_delta_lte: float = -0.001,
    refute_delta_gte: float = 0.001,
    name: Optional[str] = None,
    created_by: Optional[str] = None,
    active: bool = True,
) -> dict:
    now = time.time()
    hid = str(uuid.uuid4())[:8]
    with _conn() as con:
        con.execute(
            """
            INSERT INTO shadow_hypotheses
            (id, name, statement, config_constraint, support_delta_lte, refute_delta_gte,
             active, created_by, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                hid,
                name,
                statement,
                _config_fingerprint(config_constraint or {}),
                float(support_delta_lte),
                float(refute_delta_gte),
                1 if active else 0,
                created_by,
                now,
                now,
            ),
        )
    return get_shadow_hypothesis(hid) or {}


def get_shadow_hypothesis(hypothesis_id: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM shadow_hypotheses WHERE id=?",
            (hypothesis_id,),
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    out["config_constraint"] = json.loads(out.get("config_constraint") or "{}")
    out["active"] = bool(out.get("active", 0))
    return out


def list_shadow_hypotheses(active_only: bool = False) -> list[dict]:
    with _conn() as con:
        if active_only:
            rows = con.execute(
                """
                SELECT * FROM shadow_hypotheses
                WHERE active=1
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT * FROM shadow_hypotheses
                ORDER BY active DESC, updated_at DESC, created_at DESC
                """
            ).fetchall()

    out: list[dict] = []
    for row in rows:
        item = dict(row)
        item["config_constraint"] = json.loads(item.get("config_constraint") or "{}")
        item["active"] = bool(item.get("active", 0))
        out.append(item)
    return out


def set_shadow_hypothesis_active(hypothesis_id: str, active: bool) -> bool:
    with _conn() as con:
        con.execute(
            """
            UPDATE shadow_hypotheses
            SET active=?, updated_at=?
            WHERE id=?
            """,
            (1 if active else 0, time.time(), hypothesis_id),
        )
        row = con.execute("SELECT 1 FROM shadow_hypotheses WHERE id=?", (hypothesis_id,)).fetchone()
    return row is not None


def upsert_shadow_evidence(
    *,
    hypothesis_id: str,
    exp_id: str,
    worker_id: str,
    verdict: str,
    delta_bpb: Optional[float],
    val_bpb: Optional[float],
    reason: str,
) -> dict:
    now = time.time()
    with _conn() as con:
        con.execute(
            """
            INSERT INTO shadow_evidence
            (hypothesis_id, exp_id, worker_id, verdict, delta_bpb, val_bpb, reason, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(hypothesis_id, exp_id) DO UPDATE SET
                worker_id=excluded.worker_id,
                verdict=excluded.verdict,
                delta_bpb=excluded.delta_bpb,
                val_bpb=excluded.val_bpb,
                reason=excluded.reason,
                created_at=excluded.created_at
            """,
            (hypothesis_id, exp_id, worker_id, verdict, delta_bpb, val_bpb, reason, now),
        )
        row = con.execute(
            """
            SELECT * FROM shadow_evidence
            WHERE hypothesis_id=? AND exp_id=?
            """,
            (hypothesis_id, exp_id),
        ).fetchone()
    return dict(row) if row else {}


def shadow_scorecards() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT
                h.id,
                h.name,
                h.statement,
                h.config_constraint,
                h.support_delta_lte,
                h.refute_delta_gte,
                h.active,
                h.created_at,
                h.updated_at,
                COUNT(e.id) AS evidence_count,
                SUM(CASE WHEN e.verdict='support' THEN 1 ELSE 0 END) AS support_count,
                SUM(CASE WHEN e.verdict='refute' THEN 1 ELSE 0 END) AS refute_count,
                SUM(CASE WHEN e.verdict='inconclusive' THEN 1 ELSE 0 END) AS inconclusive_count,
                AVG(e.delta_bpb) AS mean_delta_bpb,
                MIN(e.delta_bpb) AS best_delta_bpb,
                MAX(e.delta_bpb) AS worst_delta_bpb,
                MAX(e.created_at) AS last_evidence_at
            FROM shadow_hypotheses h
            LEFT JOIN shadow_evidence e ON e.hypothesis_id = h.id
            GROUP BY h.id
            ORDER BY h.active DESC, evidence_count DESC, h.updated_at DESC
            """
        ).fetchall()

    out: list[dict] = []
    for row in rows:
        item = dict(row)
        item["config_constraint"] = json.loads(item.get("config_constraint") or "{}")
        item["active"] = bool(item.get("active", 0))
        evidence_count = int(item.get("evidence_count") or 0)
        support = int(item.get("support_count") or 0)
        refute = int(item.get("refute_count") or 0)
        inconclusive = int(item.get("inconclusive_count") or 0)
        item["support_count"] = support
        item["refute_count"] = refute
        item["inconclusive_count"] = inconclusive
        item["evidence_count"] = evidence_count
        item["support_rate"] = round((support / evidence_count), 4) if evidence_count else None
        item["refute_rate"] = round((refute / evidence_count), 4) if evidence_count else None
        item["net_score"] = round(((support - refute) / evidence_count), 4) if evidence_count else None
        out.append(item)
    return out


def shadow_evidence_for_hypothesis(hypothesis_id: str, limit: int = 200) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT *
            FROM shadow_evidence
            WHERE hypothesis_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (hypothesis_id, max(1, int(limit or 1))),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Program snapshots ─────────────────────────────────────────────────────────

def save_program_snapshot(content: str, exp_count: int):
    with _conn() as con:
        con.execute("""
            INSERT INTO program_snapshots (snapshot_id, content, experiment_count, created_at)
            VALUES (?,?,?,?)
        """, (str(uuid.uuid4()), content, exp_count, time.time()))


def has_program_snapshot() -> bool:
    with _conn() as con:
        row = con.execute("SELECT 1 FROM program_snapshots LIMIT 1").fetchone()
    return row is not None


def latest_program_md() -> str:
    with _conn() as con:
        row = con.execute("""
            SELECT content FROM program_snapshots
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()
    return row["content"] if row else _default_program_md()


def load_base_program_md() -> str:
    """
    Canonical base template for this experiment.

    Priority:
      1) META_BASE_PROGRAM_MD_PATH (user-provided)
      2) workspace-root program.md
      3) built-in fallback template
    """
    raw = os.environ.get("META_BASE_PROGRAM_MD_PATH", "").strip()
    path = Path(raw) if raw else DEFAULT_BASE_PROGRAM_MD_PATH
    if path.exists() and path.is_file():
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except Exception:
            pass
    return _default_program_md()


def _default_program_md() -> str:
    return """# Bad PI — shared research program

## Goal
Minimize val_bpb on the nanochat training task within a fixed 5-minute compute budget.

## Current strategy
Explore all dimensions broadly. Bad PI will narrow the search space
as results accumulate.

## What to modify
Only `train.py`. Bad PI will provide specific config deltas to apply.

## Reporting
After each run, report val_bpb and any observations about training stability,
loss curves, or GPU utilization anomalies.
"""
