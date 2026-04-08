"""
Shared schemas for meta-agent ↔ worker protocol.
Both meta_server and worker import from here.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field
import time


# ── Worker registration ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    worker_id: str                  # e.g. "alice-h100-0"
    gpu_type: str                   # "H100", "A100", "RTX3090", "M2", …
    baseline_bpb: float             # val_bpb from unmodified train.py
    contact: Optional[str] = None   # optional email/discord for coordination
    enroll_token: Optional[str] = None  # shared invite token, if server requires one


class RegisterResponse(BaseModel):
    ok: bool
    message: str
    current_program_md: str         # latest program.md to read before first run
    worker_token: str               # per-worker token for all future authenticated calls


# ── Experiment result (worker → server) ───────────────────────────────────────

class ExperimentResult(BaseModel):
    worker_id: str
    exp_id: str                     # UUID assigned when config was pulled
    config: dict[str, Any]          # full config dict that was run
    config_delta: dict[str, Any]    # only the keys that changed from base
    val_bpb: float
    delta_bpb: float                # val_bpb - worker's baseline_bpb  (negative = good)
    duration_seconds: float
    error: Optional[str] = None     # set if training crashed


# ── Next config (server → worker) ─────────────────────────────────────────────

class NextConfig(BaseModel):
    exp_id: str                     # server-assigned, must be echoed back in result
    config_delta: dict[str, Any]    # keys to patch into train.py
    budget_seconds: int = 300       # how long this run should train — agent-controlled
    priority: float                 # 0-1, informational
    note: str = ""                  # human-readable reason ("exploit pop A, 360s")
    population_id: Optional[str] = None
    population_strategy: Optional[str] = None
    hypothesis_id: Optional[str] = None
    hypothesis_statement: Optional[str] = None


# ── Dimension status (server → worker, in program.md sync) ────────────────────

class DimensionStatus(BaseModel):
    name: str
    frozen: bool
    frozen_value: Optional[Any] = None
    importance: float               # 0-1, fANOVA score
    best_known_value: Optional[Any] = None


# ── Program sync (server → worker) ────────────────────────────────────────────

class ProgramSync(BaseModel):
    program_md: str
    dimensions: list[DimensionStatus]
    top_configs: list[dict]         # top-5 (config_delta, mean_delta_bpb)
    experiment_count: int
    active_workers: int
    population_id: Optional[str] = None
    population_strategy: Optional[str] = None
    hypothesis_id: Optional[str] = None
    hypothesis_statement: Optional[str] = None
    updated_at: float = Field(default_factory=time.time)


# ── Leaderboard entry ─────────────────────────────────────────────────────────

class LeaderboardEntry(BaseModel):
    rank: int
    worker_id: str
    gpu_type: str
    best_delta_bpb: float
    best_config_delta: dict[str, Any]
    experiment_count: int
