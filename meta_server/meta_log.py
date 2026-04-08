"""
meta_server/meta_log.py

The meta-hypothesis log: a timestamped Markdown journal of how beliefs
have changed over the course of the swarm's research.

Every time the PI re-evaluates (every CHECKPOINT_EVERY experiments), a new
checkpoint is appended recording:
  - Which hypotheses moved (and by how much)
  - Which were eliminated (and with what evidence)
  - Which new ones were generated
  - How populations changed
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .hypotheses import Hypothesis, HypothesisRegistry
from .population_manager import PopulationManager

LOG_PATH          = Path(__file__).parent / "meta_hypothesis_log.md"
CHECKPOINT_EVERY  = 100    # write a checkpoint every N experiments


@dataclass
class Checkpoint:
    number: int
    timestamp: float
    experiment_count: int
    active_workers: int

    belief_table: list[dict]         # [{statement, prior_p, current_p, delta, n, status}]
    eliminated: list[dict]           # [{statement, posterior, n, status, evidence_tail}]
    new_hypotheses: list[dict]       # [{statement, rationale, prior_p}]
    population_changes: list[str]    # human-readable change log

    def to_markdown(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.timestamp))
        lines = [
            "---",
            f"## Checkpoint {self.number}  ·  {self.experiment_count} experiments  ·  {ts}",
            f"*Active workers: {self.active_workers}*",
            "",
            "### Belief movements",
            "| Hypothesis | Prior P | Current P | Δ | n | Status |",
            "|-----------|---------|-----------|---|---|--------|",
        ]
        for b in self.belief_table:
            delta = b["current_p"] - b["prior_p"]
            arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "→")
            lines.append(
                f"| {b['statement'][:45]:45} "
                f"| {b['prior_p']:.2f} "
                f"| {b['current_p']:.2f} "
                f"| {arrow}{abs(delta):.2f} "
                f"| {b['n']:3} "
                f"| {b['status']} |"
            )

        if self.eliminated:
            lines += ["", "### Eliminated this cycle"]
            for e in self.eliminated:
                lines += [
                    f"**{e['statement']}** — **{e['status'].upper()}** "
                    f"(P={e['posterior']:.2f}, n={e['n_experiments']})",
                    f"> Evidence: {e.get('evidence_tail', 'see experiment log')}",
                    "",
                ]

        if self.new_hypotheses:
            lines += ["", "### New hypotheses generated"]
            for nh in self.new_hypotheses:
                lines += [
                    f"- **\"{nh['statement']}\"** — P={nh.get('prior_p', 0.5):.2f} (NEW)",
                    f"  *Rationale: {nh.get('rationale', '')}*",
                ]

        if self.population_changes:
            lines += ["", "### Population changes"]
            for pc in self.population_changes:
                lines.append(f"- {pc}")

        lines += [""]
        return "\n".join(lines)


class MetaHypothesisLog:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self._checkpoints: list[Checkpoint] = []
        self._prev_posteriors: dict[str, float] = {}   # hypothesis_id → last known P
        self._last_checkpoint_at: int = 0              # experiment count at last checkpoint

    def should_checkpoint(self, experiment_count: int) -> bool:
        return (experiment_count - self._last_checkpoint_at) >= CHECKPOINT_EVERY

    def write_checkpoint(
        self,
        registry: HypothesisRegistry,
        population_manager: PopulationManager,
        experiment_count: int,
        active_workers: int,
        eliminated: list[Hypothesis],
        new_hypotheses: list[dict],
        population_changes: list[str],
    ) -> Checkpoint:
        # Build belief table (active + just-eliminated)
        belief_table = []
        for h in registry.active + eliminated:
            belief_table.append({
                "statement":  h.statement,
                "prior_p":    self._prev_posteriors.get(h.id, 0.5),
                "current_p":  h.posterior,
                "n":          h.n_experiments,
                "status":     h.status,
            })

        # Update prior tracker for next checkpoint
        for h in registry.active:
            self._prev_posteriors[h.id] = h.posterior

        # Format eliminated evidence
        elim_dicts = [
            {
                "statement":    h.statement,
                "posterior":    h.posterior,
                "n_experiments": h.n_experiments,
                "status":       h.status,
                "evidence_tail": "  |  ".join(h.evidence_log[-4:]) if h.evidence_log else "",
            }
            for h in eliminated
        ]

        cp = Checkpoint(
            number=len(self._checkpoints) + 1,
            timestamp=time.time(),
            experiment_count=experiment_count,
            active_workers=active_workers,
            belief_table=belief_table,
            eliminated=elim_dicts,
            new_hypotheses=new_hypotheses,
            population_changes=population_changes,
        )
        self._checkpoints.append(cp)
        self._last_checkpoint_at = experiment_count
        self._flush()
        return cp

    def _flush(self):
        header = (
            "# Meta-hypothesis log\n"
            "*Auto-generated by the PI meta-agent — newest checkpoint first*\n\n"
        )
        body = "\n".join(cp.to_markdown() for cp in reversed(self._checkpoints))
        self.log_path.write_text(header + body, encoding="utf-8")

    def latest_markdown(self) -> str:
        return self.log_path.read_text(encoding="utf-8") if self.log_path.exists() else (
            "# Meta-hypothesis log\n\nNo checkpoints yet.\n"
        )
