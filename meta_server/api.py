"""
meta_server/api.py
FastAPI application — all HTTP endpoints workers talk to.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse

from pydantic import BaseModel
from shared.schemas import (
    RegisterRequest, RegisterResponse,
    ExperimentResult, NextConfig,
    ProgramSync, DimensionStatus, LeaderboardEntry,
)
from . import store, search, program_writer
from .scheduler import registry as run_registry
from .pipeline  import pipeline
from .belief_engine import engine as belief_engine
from .runtime import runtime_state


# ── Compact tick protocol schema ──────────────────────────────────────────────

class Tick(BaseModel):
    id: str          # run_id assigned when config was pulled
    p:  float        # progress 0.0–1.0 (fraction of 5-min budget elapsed)
    m:  float        # current metric (val_bpb or equivalent)
    d:  float        # delta = m - worker_baseline

log = logging.getLogger(__name__)
_last_program_write = 0
PIPELINE_BATCH_EVERY = int(os.environ.get("BAD_PI_PIPELINE_BATCH_EVERY", "20"))


def _require_enroll_token(enroll_token: Optional[str]):
    expected = os.environ.get("META_ENROLL_TOKEN")
    if expected and enroll_token != expected:
        raise HTTPException(401, "Invalid enroll token.")


def _require_worker_auth(worker_id: str, x_worker_token: Optional[str]):
    if not store.verify_worker_token(worker_id, x_worker_token):
        raise HTTPException(401, "Unauthorized worker.")


def _require_run_auth(run_id: str, x_worker_token: Optional[str]) -> str:
    run = run_registry.active_runs.get(run_id)
    if run is None:
        raise HTTPException(401, "Unknown or inactive run.")
    _require_worker_auth(run.worker_id, x_worker_token)
    return run.worker_id


# ── Background tasks ──────────────────────────────────────────────────────────

async def _search_loop():
    """Run search cycle every 60 seconds."""
    while True:
        try:
            search.run_search_cycle()
        except Exception as e:
            log.exception(f"Search cycle error: {e}")
        await asyncio.sleep(60)


async def _program_loop():
    """Regenerate program.md every WRITE_EVERY experiments."""
    global _last_program_write
    while True:
        try:
            total = store.experiment_count()
            if program_writer.should_write(total, _last_program_write):
                log.info(f"Writing new program.md at {total} experiments")
                content = runtime_state.generate_global_program(total)
                _last_program_write = total
        except Exception as e:
            log.exception(f"Program writer error: {e}")
        await asyncio.sleep(90)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    belief_engine._completed_count = store.experiment_count()
    runtime_state.initialize()
    # Seed queue with initial random configs
    search.run_search_cycle()
    asyncio.create_task(_search_loop())
    asyncio.create_task(_program_loop())
    log.info("Bad PI server started")
    yield


app = FastAPI(
    title="Bad PI",
    description="Bad PI coordination server for distributed autoresearch swarms",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Registration ──────────────────────────────────────────────────────────────

@app.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    """
    First call a worker makes. Provide your worker_id, GPU type, and
    your baseline val_bpb (from running unmodified train.py once).
    """
    _require_enroll_token(req.enroll_token)
    worker_token = store.register_worker(req.worker_id, req.gpu_type, req.baseline_bpb, req.contact)
    program_md, pop, hypothesis = runtime_state.program_for_worker(req.worker_id)
    log.info(f"Worker registered: {req.worker_id} ({req.gpu_type}) baseline={req.baseline_bpb:.4f}")
    return RegisterResponse(
        ok=True,
        message=f"Welcome {req.worker_id}! You are worker #{store.active_worker_count()}.",
        current_program_md=program_md,
        worker_token=worker_token,
    )


# ── Get next config ───────────────────────────────────────────────────────────

@app.get("/next_config/{worker_id}", response_model=NextConfig)
def next_config(worker_id: str, x_worker_token: Optional[str] = Header(default=None)):
    """
    Pull the next suggested config_delta to apply to train.py.
    Response includes budget_seconds — patch TOTAL_WALL_CLOCK_TIME with this value.
    Call this before each training run.
    """
    worker = store.get_worker(worker_id)
    if not worker:
        raise HTTPException(400, "Worker not registered. Call /register first.")
    _require_worker_auth(worker_id, x_worker_token)
    store.touch_worker(worker_id)
    config = store.pop_next_config(worker_id)
    if config is None:
        search.run_search_cycle()
        config = store.pop_next_config(worker_id)
    if config is None:
        raise HTTPException(503, "Config queue empty, retry in 10s")

    config = runtime_state.shape_config_for_worker(worker_id, config)

    # Decide budget based on population strategy and hypothesis certainty
    pop_strategy = config.get("_population_strategy", "investigate")
    pop_id = config.get("_population_id", "default")
    budget = belief_engine.decide_budget(
        population_strategy  = pop_strategy,
        hypothesis_posterior = float(config.get("_hypothesis_posterior", 0.5)),
        queue_depth          = store.queue_depth(),
    )

    # Register run lifecycle using the same exp_id workers will report/tick with.
    run_registry.start_run(
        worker_id=worker_id,
        config_delta=config["config_delta"],
        population_id=pop_id,
        budget=budget,
        run_id=config["exp_id"],
    )

    config.setdefault("note", "")
    config["note"] = f"{pop_strategy} — {budget}s budget"

    return NextConfig(
        exp_id         = config["exp_id"],
        config_delta   = config["config_delta"],
        budget_seconds = budget,
        priority       = config.get("priority", 0.5),
        note           = config["note"],
        population_id  = config.get("_population_id"),
        population_strategy = config.get("_population_strategy"),
        hypothesis_id  = config.get("_hypothesis_id"),
        hypothesis_statement = config.get("_hypothesis_statement"),
    )


# ── Submit result ─────────────────────────────────────────────────────────────

@app.post("/result")
def submit_result(result: ExperimentResult, x_worker_token: Optional[str] = Header(default=None)):
    """
    Submit the outcome of a completed (or failed) 5-minute training run.
    Always call this even if training crashed — set error field.
    """
    worker = store.get_worker(result.worker_id)
    if not worker:
        raise HTTPException(400, "Worker not registered.")
    _require_worker_auth(result.worker_id, x_worker_token)

    store.save_experiment(
        exp_id=result.exp_id,
        worker_id=result.worker_id,
        config=result.config,
        config_delta=result.config_delta,
        val_bpb=result.val_bpb,
        delta_bpb=result.delta_bpb,
        duration=result.duration_seconds,
        status="failed" if result.error else "completed",
        error=result.error,
    )
    total = store.experiment_count()

    if not result.error:
        runtime_state.handle_completed_experiment(result.config_delta, result.delta_bpb, total)
    else:
        belief_engine._completed_count = total

    log.info(
        f"Result from {result.worker_id}: delta_bpb={result.delta_bpb:+.4f} "
        f"(total={total})"
    )

    # Close any active run with the same server-issued exp_id.
    run_registry.complete_run(result.exp_id)

    # Batch-boundary speculative validation:
    # every N completed experiments, evaluate whether the speculation was right.
    if PIPELINE_BATCH_EVERY > 0 and total > 0 and (total % PIPELINE_BATCH_EVERY == 0):
        recent_metrics = [
            r.last_metric for r in run_registry.completed_runs[-PIPELINE_BATCH_EVERY:]
            if r.last_metric is not None
        ]
        pipeline.on_batch_complete(recent_metrics, runtime_state.registry)

    return {"ok": True, "total_experiments": total}


# ── Program sync ──────────────────────────────────────────────────────────────

@app.get("/sync/{worker_id}", response_model=ProgramSync)
def sync(worker_id: str, x_worker_token: Optional[str] = Header(default=None)):
    """
    Call after submitting a result to get the latest program.md and
    dimension status. Workers should update their local program.md
    from this response.
    """
    _require_worker_auth(worker_id, x_worker_token)
    program_md, pop, hypothesis = runtime_state.program_for_worker(worker_id)
    program_digest = (
        hashlib.sha256(program_md.encode("utf-8")).hexdigest()[:16]
        if program_md else ""
    )
    dims = store.get_dimensions()
    top = store.top_experiments(5)
    dim_statuses = [
        DimensionStatus(
            name=d["name"],
            frozen=bool(d["frozen"]),
            frozen_value=d["frozen_value"],
            importance=d["importance"],
            best_known_value=d["frozen_value"],
        )
        for d in dims
    ]
    top_configs = [
        {
            "config_delta": (
                e["config_delta"]
                if isinstance(e["config_delta"], dict)
                else json.loads(e["config_delta"])
            ),
            "delta_bpb": e["delta_bpb"],
        }
        for e in top
    ]
    return ProgramSync(
        program_md=program_md,
        program_digest=program_digest,
        dimensions=dim_statuses,
        top_configs=top_configs,
        experiment_count=store.experiment_count(),
        active_workers=store.active_worker_count(),
        population_id=pop.id if pop else None,
        population_strategy=pop.strategy if pop else None,
        hypothesis_id=hypothesis.id if hypothesis else None,
        hypothesis_statement=hypothesis.statement if hypothesis else None,
    )


# ── Leaderboard ───────────────────────────────────────────────────────────────

@app.get("/leaderboard", response_model=list[LeaderboardEntry])
def leaderboard():
    """Public leaderboard — best delta_bpb per worker."""
    with store._conn() as con:
        rows = con.execute("""
            SELECT
                w.worker_id, w.gpu_type, w.experiment_count,
                MIN(e.delta_bpb) AS best_delta_bpb,
                e.config_delta   AS best_config_delta
            FROM workers w
            JOIN experiments e ON e.worker_id = w.worker_id
            WHERE e.status='completed'
            GROUP BY w.worker_id
            ORDER BY best_delta_bpb ASC
            LIMIT 50
        """).fetchall()
    return [
        LeaderboardEntry(
            rank=i + 1,
            worker_id=r["worker_id"],
            gpu_type=r["gpu_type"],
            best_delta_bpb=r["best_delta_bpb"],
            best_config_delta=(
                json.loads(r["best_config_delta"])
                if isinstance(r["best_config_delta"], str)
                else r["best_config_delta"]
            ),
            experiment_count=r["experiment_count"],
        )
        for i, r in enumerate(rows)
    ]


# ── Health / stats ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "experiments": store.experiment_count(),
        "queue_depth": store.queue_depth(),
        "active_workers": store.active_worker_count(),
        "frozen_dims": sum(1 for d in store.get_dimensions() if d["frozen"]),
        "ts": time.time(),
    }


@app.get("/program.md", response_class=PlainTextResponse)
def get_program_md():
    """Download the latest program.md as plain text."""
    return store.latest_program_md()


# ── Compact tick protocol ─────────────────────────────────────────────────────

@app.post("/tick")
async def tick(t: Tick, x_worker_token: Optional[str] = Header(default=None)):
    """
    Minimal heartbeat from a running worker.
    Called at each 20% progress checkpoint.

    Request:  {"id": "run_uuid", "p": 0.2, "m": 1.9, "d": -0.05}
    Response: {"action": "stop"} | {"action": "extend", "budget": 420} | {}
    """
    _require_run_auth(t.id, x_worker_token)
    action = run_registry.update_run(t.id, p=t.p, metric=t.m, delta=t.d)

    # Feed early metrics into the pipeline for speculative execution
    active_metrics = [
        r.last_metric for r in run_registry.active_runs.values()
        if r.last_metric is not None
    ]
    pipeline.on_tick(active_metrics)

    return action


# ── Run lifecycle ─────────────────────────────────────────────────────────────

@app.post("/runs/start/{worker_id}")
def start_run(
    worker_id: str,
    config_delta: dict = None,
    population_id: str = "default",
    x_worker_token: Optional[str] = Header(default=None),
):
    """
    Explicitly start a tracked run. Returns run_id.
    (Optional: next_config already creates a run implicitly via exp_id.)
    """
    worker = store.get_worker(worker_id)
    if not worker:
        raise HTTPException(400, "Worker not registered.")
    _require_worker_auth(worker_id, x_worker_token)
    run = run_registry.start_run(
        worker_id     = worker_id,
        config_delta  = config_delta or {},
        population_id = population_id,
    )
    return {"run_id": run.id, "budget_seconds": run.budget_seconds}


@app.delete("/runs/{run_id}")
def stop_run(run_id: str, reason: str = "manual"):
    """Manually stop a run."""
    ok = run_registry.stop_run(run_id, reason)
    return {"ok": ok}


@app.get("/runs/active")
def active_runs():
    """List all currently active runs."""
    return [
        {
            "id":         r.id,
            "worker_id":  r.worker_id,
            "state":      r.state,
            "progress":   r.last_p,
            "metric":     r.last_metric,
            "elapsed":    round(r.elapsed, 1),
            "budget":     r.budget_seconds,
            "population": r.population_id,
        }
        for r in run_registry.active_runs.values()
    ]


@app.get("/runs/stats")
def run_stats():
    """Aggregate run statistics including bucket pool percentiles."""
    return run_registry.stats()


# ── Speculative pipeline ──────────────────────────────────────────────────────

@app.get("/pipeline/status")
def pipeline_status():
    """Current state of the speculative execution pipeline."""
    return pipeline.status()


@app.get("/pipeline/spec", response_class=PlainTextResponse)
def get_spec_program():
    """
    Workers can pre-cache this between runs.
    Returns speculative program.md if ready, else 204 No Content.
    """
    spec = pipeline.get_cached_program()
    if spec is None:
        from fastapi.responses import Response
        return Response(status_code=204)
    return spec


@app.get("/pipeline/spec_payload")
def get_spec_payload():
    """
    Structured speculative payload for workers:
    returns spec_id + program_md only when deployment is allowed by confidence gates.
    """
    spec = pipeline.get_cached_spec()
    if spec is None:
        from fastapi.responses import Response
        return Response(status_code=204)
    return spec


@app.get("/pipeline/flush_token")
def flush_token():
    """
    Returns a flush token if workers should discard their cached spec.
    Workers poll this periodically; non-null means 'drop your spec and re-pull.'
    """
    token = pipeline.flush_token()
    return {"flush_token": token}


@app.post("/pipeline/flush")
def pipeline_flush(reason: str = "manual"):
    """Manually flush speculative pipeline cache across workers."""
    pipeline.flush(reason)
    return {"ok": True, "reason": reason}


@app.get("/meta_log", response_class=PlainTextResponse)
def get_meta_log():
    """Latest meta-hypothesis log markdown."""
    return runtime_state.meta_log.latest_markdown()


@app.get("/theory_graph")
def theory_graph():
    """Structured parent/child/link hypothesis graph for analysis tooling."""
    return runtime_state.registry.theory_graph()


@app.get("/theory_graph/human")
def theory_graph_human(include_graph: bool = False):
    """
    Human-readable derived narrative for the theory graph.
    The JSON graph remains the authoritative source of truth.
    """
    graph = runtime_state.registry.theory_graph()
    human = program_writer.summarize_theory_graph(graph)
    payload = {
        "source_of_truth": "machine_graph_json",
        "derived_layer": human,
    }
    if include_graph:
        payload["graph"] = graph
    return payload


@app.get("/dimension_proposals")
def get_dimension_proposals():
    """
    Organizer-facing queue of LLM-suggested NEW dimensions generated when
    the search stalls. These are advisory only and must be manually reviewed.
    """
    proposals = runtime_state.list_dimension_proposals()
    return {
        "count": len(proposals),
        "proposals": proposals,
    }


@app.delete("/dimension_proposals")
def clear_dimension_proposals():
    """Clear pending dimension proposals after organizer review."""
    runtime_state.clear_dimension_proposals()
    return {"ok": True}
