"""
meta_server/population_manager.py

Manages multiple concurrent research populations.
Each population tests a specific hypothesis with a tailored program.md.
Worker assignment is based on current allocation from HypothesisRegistry.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .hypotheses import Hypothesis, HypothesisRegistry

log = logging.getLogger(__name__)


@dataclass
class Population:
    id: str              = field(default_factory=lambda: f"pop_{str(uuid.uuid4())[:6]}")
    hypothesis_id: str   = ""
    strategy: str        = "investigate"   # exploit | investigate | falsify | moonshot
    target_workers: int  = 10
    assigned_workers: list = field(default_factory=list)
    program_md: str      = ""
    active: bool         = True
    created_at: float    = field(default_factory=time.time)

    def needs_worker(self) -> bool:
        return len(self.assigned_workers) < self.target_workers

    def excess_workers(self) -> int:
        return max(0, len(self.assigned_workers) - self.target_workers)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Population":
        return cls(**data)


class PopulationManager:
    """
    Manages the full lifecycle of populations:
      - Spawn a population when a new hypothesis is added to the registry
      - Rebalance worker targets after each allocation cycle
      - Dissolve populations whose hypotheses are concluded
      - Assign workers to populations (weighted by capacity gap)
      - Generate a dedicated program.md per population via Claude or template
    """

    def __init__(self):
        self._pops: dict[str, Population] = {}
        self._worker_pop: dict[str, str]  = {}   # worker_id → pop_id

    @property
    def active_populations(self) -> list[Population]:
        return [p for p in self._pops.values() if p.active]

    def to_dict(self) -> dict:
        return {
            "populations": {pid: pop.to_dict() for pid, pop in self._pops.items()},
            "worker_populations": dict(self._worker_pop),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PopulationManager":
        manager = cls()
        manager._pops = {
            pid: Population.from_dict(pop_data)
            for pid, pop_data in data.get("populations", {}).items()
        }
        manager._worker_pop = dict(data.get("worker_populations", {}))
        return manager

    # ── Worker assignment ─────────────────────────────────────────────────

    def assign_worker(self, worker_id: str) -> Optional[Population]:
        """
        Move worker to the population most in need of additional workers.
        Returns the assigned population.
        """
        self._remove_worker_from_current(worker_id)
        candidates = sorted(
            self.active_populations,
            key=lambda p: p.target_workers - len(p.assigned_workers),
            reverse=True,
        )
        if not candidates:
            return None
        pop = candidates[0]
        pop.assigned_workers.append(worker_id)
        self._worker_pop[worker_id] = pop.id
        return pop

    def get_worker_population(self, worker_id: str) -> Optional[Population]:
        pop_id = self._worker_pop.get(worker_id)
        return self._pops.get(pop_id) if pop_id else None

    def _remove_worker_from_current(self, worker_id: str):
        old_id = self._worker_pop.get(worker_id)
        if old_id and old_id in self._pops:
            pop = self._pops[old_id]
            if worker_id in pop.assigned_workers:
                pop.assigned_workers.remove(worker_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def sync(
        self,
        registry: HypothesisRegistry,
        total_workers: int,
        top_configs: list[dict],
        dimensions: list[dict],
    ) -> list[str]:
        """
        Reconcile populations with the current registry.
        Returns a list of human-readable change descriptions for the meta-log.
        """
        allocations = registry.allocate_workers(total_workers)
        active_h_ids = {h.id for h in registry.active}
        existing_h_ids = {p.hypothesis_id for p in self.active_populations}
        changes = []

        # Spawn populations for new hypotheses
        for h in registry.active:
            if h.id not in existing_h_ids:
                strategy = (
                    "decision_sprint" if h.in_decision_sprint else
                    ("falsify" if h.needs_falsification_run() else "investigate")
                )
                pop = Population(
                    hypothesis_id=h.id,
                    strategy=strategy,
                    target_workers=allocations.get(h.id, 5),
                )
                pop.program_md = self.generate_program_md(pop, h, top_configs, dimensions)
                self._pops[pop.id] = pop
                changes.append(f"Spawned {pop.id} ({strategy}, {pop.target_workers} workers) for \"{h.statement}\"")
                log.info(changes[-1])

        # Update existing populations
        for pop in self.active_populations:
            h = registry.get(pop.hypothesis_id)
            if not h or pop.hypothesis_id not in active_h_ids:
                pop.active = False
                freed = len(pop.assigned_workers)
                for w in list(pop.assigned_workers):
                    self._remove_worker_from_current(w)
                changes.append(f"Dissolved {pop.id} (hypothesis concluded) — {freed} workers freed")
                log.info(changes[-1])
            else:
                old_target = pop.target_workers
                pop.target_workers = allocations.get(pop.hypothesis_id, 5)
                # Update strategy
                new_strategy = (
                    "decision_sprint" if h.in_decision_sprint else
                    "falsify"  if h.needs_falsification_run()   else
                    "exploit"  if h.status == "supported"        else
                    "investigate"
                )
                if new_strategy != pop.strategy or pop.target_workers != old_target:
                    pop.strategy = new_strategy
                    pop.program_md = self.generate_program_md(pop, h, top_configs, dimensions)
                    changes.append(
                        f"Updated {pop.id}: strategy={new_strategy}, workers {old_target}→{pop.target_workers}"
                    )

        # Rebalance workers if any populations dissolved or resized
        all_assigned = {w for p in self.active_populations for w in p.assigned_workers}
        all_registered = set(self._worker_pop.keys())
        unassigned = all_registered - all_assigned
        for w in unassigned:
            self.assign_worker(w)

        return changes

    # ── Program.md generation ─────────────────────────────────────────────

    def generate_program_md(
        self,
        pop: Population,
        hypothesis: Hypothesis,
        top_configs: list[dict],
        dimensions: list[dict],
    ) -> str:
        try:
            return self._claude_program_md(pop, hypothesis, top_configs, dimensions)
        except Exception as e:
            log.debug(f"Claude program.md fallback ({e})")
            return self._template_program_md(pop, hypothesis, top_configs, dimensions)

    def _template_program_md(
        self,
        pop: Population,
        hypothesis: Hypothesis,
        top_configs: list[dict],
        dimensions: list[dict],
    ) -> str:
        top = top_configs[0] if top_configs else {}
        best_delta = top.get("delta_bpb", "unknown")
        best_cfg   = top.get("config_delta", {})

        strategy_guidance = {
            "exploit": (
                f"We are **refining the best known region**.\n\n"
                f"Current best config: `{json.dumps(best_cfg)}`  (delta_bpb = {best_delta})\n\n"
                f"Explore nearby values — small changes around the best. Do not stray far."
            ),
            "investigate": (
                f"We are **mapping the hypothesis space**.\n\n"
                f"Current best: `{json.dumps(best_cfg)}`\n\n"
                f"Explore broadly. Try a range of values for dimensions relevant to this hypothesis."
            ),
            "falsify": (
                f"**CONTROLLED FALSIFICATION EXPERIMENT**\n\n"
                f"We believe this hypothesis is probably FALSE (P = {hypothesis.posterior:.2f}). "
                f"We need clean evidence to confirm it.\n\n"
                f"**Lock all hyperparameters at their best known values:**\n"
                f"```\n{json.dumps({**best_cfg, **hypothesis.config_constraint}, indent=2)}\n```\n\n"
                f"**Only vary** the dimension(s) directly relevant to: \"{hypothesis.statement}\"\n\n"
                f"Run each variant at least 3 times to average out noise. "
                f"Report WINDOW_PATTERN (or relevant key) alongside every val_bpb."
            ),
            "decision_sprint": (
                f"**DECISION SPRINT (ETERNAL UNCERTAINTY CONTROL)**\n\n"
                f"This hypothesis has remained near indecision for too long.\n"
                f"Run controlled, decisive experiments to force a clear conclusion.\n\n"
                f"Use focused sweeps around the key dimensions in this hypothesis.\n"
                f"Repeat each arm at least 4 times and prioritize low-variance comparisons."
            ),
            "moonshot": (
                f"**MOONSHOT EXPLORATION**\n\n"
                f"Try unusual or extreme configurations. High variance is fine — "
                f"we're looking for surprises, not incremental gains."
            ),
        }

        return f"""# {pop.id} — {pop.strategy.upper()}
*Generated for hypothesis: "{hypothesis.statement}"*

## Hypothesis
**Claim:** {hypothesis.statement}
**Type:** {hypothesis.type}
**Current belief:** P = {hypothesis.posterior:.2f}  (n = {hypothesis.n_experiments} experiments so far)
**Credible interval (90%):** {hypothesis.credible_interval_90}

## Strategy
{strategy_guidance.get(pop.strategy, "Explore freely.")}

## Constraints
{f"Fix these values, do NOT change them:{chr(10)}{json.dumps(hypothesis.config_constraint, indent=2)}" if hypothesis.config_constraint else "No hard constraints — explore the full search space."}

## Reporting
After each 5-minute run, your result is submitted automatically. Include notes about
training stability, loss curve behaviour, or anything unusual.
"""

    def _claude_program_md(
        self,
        pop: Population,
        hypothesis: Hypothesis,
        top_configs: list[dict],
        dimensions: list[dict],
    ) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        top = top_configs[0] if top_configs else {}
        prompt = f"""You are the PI (principal investigator) writing instructions for
a specific research population in a distributed ML training swarm.

Population: {pop.id}  |  Strategy: {pop.strategy}
Hypothesis: "{hypothesis.statement}"
Posterior belief: P = {hypothesis.posterior:.2f}  ({hypothesis.n_experiments} experiments)
Wins / Losses: {hypothesis.alpha - 2:.0f} / {hypothesis.beta - 2:.0f}

Best config so far: {json.dumps(top.get('config_delta', {}), indent=2)}
Best delta_bpb: {top.get('delta_bpb', 'N/A')}

Locked constraints (must not change): {json.dumps(hypothesis.config_constraint, indent=2)}

Write a focused program.md (under 250 words) that tells the AI agent workers exactly:
1. What this population is trying to determine
2. What to vary and what to hold fixed
3. What a "win" looks like for this hypothesis
{'4. How to design a clean controlled test (falsification run)' if pop.strategy == 'falsify' else ''}

Be scientifically precise and actionable.
"""
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
