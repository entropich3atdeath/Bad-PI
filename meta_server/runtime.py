from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from . import program_writer, store
from .belief_engine import engine as belief_engine
from .hypotheses import Hypothesis, HypothesisRegistry, make_default_registry
from .meta_log import MetaHypothesisLog
from .pipeline import pipeline
from .population_manager import PopulationManager, Population


RUNTIME_STATE_PATH = Path(
    os.environ.get(
        "META_RUNTIME_STATE_PATH",
        str(Path(__file__).parent / "runtime_state.json"),
    )
)


class RuntimeState:
    def __init__(self, state_path: Path = RUNTIME_STATE_PATH):
        self.state_path = state_path
        self.registry: HypothesisRegistry = make_default_registry()
        self.population_manager = PopulationManager()
        self.meta_log = MetaHypothesisLog()
        self.pending_new_hypotheses: list[dict] = []
        self.pending_dimension_proposals: list[dict] = []   # LLM dim proposals when stalled
        self._lock = threading.RLock()

    def initialize(self):
        with self._lock:
            self._load_locked()
            if not self.registry.active and not self.registry.archived:
                self.registry = make_default_registry()
            self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))

    def build_belief_state(self):
        with self._lock:
            return self._build_belief_state_locked()

    def assign_worker(self, worker_id: str) -> Optional[Population]:
        with self._lock:
            if not self.population_manager.active_populations:
                self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))
            pop = self.population_manager.get_worker_population(worker_id)
            if pop and pop.active:
                return pop
            pop = self.population_manager.assign_worker(worker_id)
            self._save_locked()
            return pop

    def get_worker_population(self, worker_id: str) -> Optional[Population]:
        with self._lock:
            return self.population_manager.get_worker_population(worker_id)

    def list_dimension_proposals(self) -> list[dict]:
        with self._lock:
            return list(self.pending_dimension_proposals)

    def clear_dimension_proposals(self):
        with self._lock:
            self.pending_dimension_proposals = []
            self._save_locked()

    def shape_config_for_worker(self, worker_id: str, config: dict) -> dict:
        with self._lock:
            pop = self.assign_worker(worker_id)
            shaped = dict(config)
            shaped_delta = dict(shaped.get("config_delta") or {})
            hypothesis = self.registry.get(pop.hypothesis_id) if pop else None
            best_config = self._best_config_delta_locked()

            if pop and hypothesis:
                if pop.strategy == "exploit" and best_config:
                    shaped_delta = {**best_config, **shaped_delta}
                elif pop.strategy == "falsify":
                    shaped_delta = {
                        **best_config,
                        **shaped_delta,
                    }

                if hypothesis.config_constraint:
                    shaped_delta.update(hypothesis.config_constraint)

                statement = hypothesis.statement.strip()
                # Convergence mode: override strategy to 'converge' for the validated winner
                conv_winner = self.registry.convergence_winner()
                effective_strategy = (
                    "converge" if (conv_winner and conv_winner.id == hypothesis.id)
                    else pop.strategy
                )
                prefix = f"{effective_strategy} · {pop.id}"
                shaped["note"] = f"{prefix} — {statement}" if statement else prefix
                shaped["_population_id"] = pop.id
                shaped["_population_strategy"] = effective_strategy
                shaped["_hypothesis_id"] = hypothesis.id
                shaped["_hypothesis_statement"] = hypothesis.statement
                shaped["_hypothesis_posterior"] = round(hypothesis.posterior, 4)
            else:
                shaped.setdefault("note", "default")
                shaped.setdefault("_population_id", "default")
                shaped.setdefault("_population_strategy", "investigate")
                shaped.setdefault("_hypothesis_posterior", 0.5)

            shaped["config_delta"] = shaped_delta
            return shaped

    def program_for_worker(self, worker_id: str) -> tuple[str, Optional[Population], Optional[Hypothesis]]:
        with self._lock:
            pop = self.assign_worker(worker_id)
            hypothesis = self.registry.get(pop.hypothesis_id) if pop else None
            if pop and pop.program_md:
                return pop.program_md, pop, hypothesis
            return store.latest_program_md(), pop, hypothesis

    def handle_completed_experiment(self, config_delta: dict, delta_bpb: Optional[float], total_experiments: int):
        with self._lock:
            if delta_bpb is not None:
                self.registry.ingest_experiment(config_delta, delta_bpb)

            experiments = store.recent_experiments(1000)
            dimensions = store.get_dimensions()
            belief_engine.on_experiment_complete(experiments, dimensions, self.registry.active)

            # Track improvement trend for stall detection
            top1 = self._top_configs_locked(1)
            if top1 and top1[0].get("delta_bpb") is not None:
                belief_engine.record_best_delta(total_experiments, float(top1[0]["delta_bpb"]))

            eliminated = self._archive_refuted_locked()
            population_changes = self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))

            if self.meta_log.should_checkpoint(total_experiments):
                self.meta_log.write_checkpoint(
                    registry=self.registry,
                    population_manager=self.population_manager,
                    experiment_count=total_experiments,
                    active_workers=store.active_worker_count(),
                    eliminated=eliminated,
                    new_hypotheses=self.pending_new_hypotheses,
                    population_changes=population_changes,
                )
                self.pending_new_hypotheses = []

            self._save_locked()

    def generate_global_program(self, total_experiments: int) -> str:
        with self._lock:
            belief_state = self._build_belief_state_locked()

            # Annotate convergence — requires registry, not available inside engine
            winner = self.registry.convergence_winner()
            belief_state.is_converging = winner is not None

            content = program_writer.generate_program_md(belief_state, store.active_worker_count())
            store.save_program_snapshot(content, total_experiments)

            proposals = program_writer.propose_new_hypotheses(belief_state)
            proposal_payloads = [
                proposal.model_dump() if hasattr(proposal, "model_dump") else proposal.dict()
                for proposal in proposals
            ]
            decisions = self.registry.ingest_llm_proposals(proposal_payloads)
            accepted = [d["proposal"] for d in decisions if d["engine_gate"].get("accepted")]
            if accepted:
                self.pending_new_hypotheses.extend(accepted)
                self._refresh_populations_locked(total_workers=max(store.active_worker_count(), 1))

            # Stall detection: ask LLM for new dimensions beyond the current search space
            if belief_state.is_stalled:
                existing_names = [d["name"] for d in store.get_dimensions()]
                new_dim_proposals = program_writer.propose_new_dimensions(belief_state, existing_names)
                if new_dim_proposals:
                    serialized = [
                        p.model_dump() if hasattr(p, "model_dump") else p.dict()
                        for p in new_dim_proposals
                    ]
                    self.pending_dimension_proposals.extend(serialized)
                    log.info(
                        f"Stall detected at n={total_experiments}: "
                        f"{len(new_dim_proposals)} new dimension proposals queued"
                    )

            self._save_locked()
            return content

    def _build_belief_state_locked(self):
        return belief_engine.build_belief_state(
            experiments=store.recent_experiments(2000),
            dimensions=store.get_dimensions(),
            hypotheses=self.registry.active,
            pipeline_hit_rate=pipeline.status().get("hit_rate", 0.0),
        )

    def _best_config_delta_locked(self) -> dict:
        top = self._top_configs_locked(1)
        if not top:
            return {}
        return dict(top[0].get("config_delta") or {})

    def _top_configs_locked(self, n: int = 5) -> list[dict]:
        normalized = []
        for exp in store.top_experiments(n):
            config_delta = exp.get("config_delta")
            normalized.append({
                "config_delta": json.loads(config_delta) if isinstance(config_delta, str) else config_delta,
                "delta_bpb": exp.get("delta_bpb"),
            })
        return normalized

    def _archive_refuted_locked(self) -> list[Hypothesis]:
        eliminated = []
        for hypothesis in list(self.registry.active):
            if hypothesis.status == "refuted" and hypothesis.n_experiments >= 12:
                eliminated.append(hypothesis)
                self.registry.archive(hypothesis.id)
        return eliminated

    def _refresh_populations_locked(self, total_workers: int) -> list[str]:
        changes = self.population_manager.sync(
            registry=self.registry,
            total_workers=total_workers,
            top_configs=self._top_configs_locked(5),
            dimensions=store.get_dimensions(),
        )
        self._save_locked()
        return changes

    def _load_locked(self):
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        if "registry" in data:
            self.registry = HypothesisRegistry.from_dict(data["registry"])
        if "population_manager" in data:
            self.population_manager = PopulationManager.from_dict(data["population_manager"])
        self.pending_new_hypotheses = list(data.get("pending_new_hypotheses", []))
        self.pending_dimension_proposals = list(data.get("pending_dimension_proposals", []))

    def _save_locked(self):
        payload = {
            "registry": self.registry.to_dict(),
            "population_manager": self.population_manager.to_dict(),
            "pending_new_hypotheses": self.pending_new_hypotheses,
            "pending_dimension_proposals": self.pending_dimension_proposals,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)


runtime_state = RuntimeState()