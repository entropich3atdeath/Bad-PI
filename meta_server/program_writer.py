"""
meta_server/program_writer.py

LLM translation layer — receives BeliefState from belief_engine.py and
renders it into human-readable program.md and hypothesis proposals.

SCHEMA ENFORCEMENT:
  All LLM calls use Anthropic tool_use with a JSON schema.
  The LLM never returns freeform text directly.
  Output is validated as a Pydantic model before use.
  If validation fails → deterministic template fallback (no LLM retry loop).

Two schemas:
  ProgramMDOutput — structured fields for program.md sections
  HypothesisProposal — structured fields for new hypothesis generation
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .belief_engine import BeliefState

log = logging.getLogger(__name__)

WRITE_EVERY = 50


def should_write(total_experiments: int, last_written_at: int) -> bool:
    return (total_experiments - last_written_at) >= WRITE_EVERY


# ── Output schemas ─────────────────────────────────────────────────────────────

class FrozenDimEntry(BaseModel):
    name:        str   = Field(description="Dimension name, e.g. 'WINDOW_PATTERN'")
    value:       Any   = Field(description="Frozen value, e.g. 'L' or 128")
    explanation: str   = Field(description="1-sentence plain English reason why it was frozen (translate the F/p-value)")


class ActiveDimEntry(BaseModel):
    name:     str   = Field(description="Dimension name")
    guidance: str   = Field(description="1-sentence instruction: what range to explore and why")


class HypothesisEntry(BaseModel):
    statement:  str   = Field(description="The hypothesis statement")
    status:     str   = Field(description="One of: supported / refuted / uncertain")
    posterior:  float = Field(description="Current P value from Beta-Binomial")
    plain_english: str = Field(description="1-sentence plain English translation of what this posterior means for the next batch")


class ProgramMDOutput(BaseModel):
    """
    Structured program.md. Every field is required.
    Rendered to markdown by render_program_md() — not by the LLM.
    """
    phase_summary:      str   = Field(description="1-2 sentences: current research phase and most important finding")
    frozen_dims:        list[FrozenDimEntry]
    active_dims:        list[ActiveDimEntry]
    hypotheses:         list[HypothesisEntry]
    concrete_instructions: list[str] = Field(description="3-5 bullet points: what workers should do next run", min_length=2)
    warning:            Optional[str] = Field(default=None, description="Optional 1-sentence warning if something looks off (e.g. high kill rate, low pool coverage)")


class HypothesisProposal(BaseModel):
    """
    A new hypothesis proposed by the PI based on experiment patterns.
    Fed directly into HypothesisRegistry — must be well-formed.
    """
    statement:         str   = Field(description="Falsifiable claim, e.g. 'DEPTH > 12 interacts with learning_rate'")
    type:              str   = Field(description="One of: positive / comparative / interaction / null")
    importance:        float = Field(ge=0.0, le=1.0, description="Expected impact if true, 0-1")
    rationale:         str   = Field(description="1-sentence statistical rationale based on the data provided")
    config_constraint: dict  = Field(default_factory=dict, description="Values to hold fixed for a controlled test. Empty dict = no constraint.")
    parent_id:         Optional[str] = Field(default=None, description="Optional parent hypothesis id if this is a decomposition/refinement")
    phase:             str   = Field(default="exploration", description="exploration | validation")
    test_spec:         Optional[dict] = Field(default=None, description="Executable validation protocol. Required when phase=validation.")


class HypothesisProposalBatch(BaseModel):
    proposals: list[HypothesisProposal] = Field(description="1-3 new hypotheses", max_length=3)


class DimensionProposal(BaseModel):
    """
    A new search dimension proposed when the current search space has stalled.
    Structural — requires the organizer to add to schema.sql and train.py.
    The engine never applies these automatically.
    """
    name:          str             = Field(description="Python identifier, e.g. 'dropout_rate'")
    dtype:         str             = Field(description="One of: float_log / float_linear / int / categorical")
    range_min:     Optional[float] = Field(default=None, description="Min for float/int/float_log ranges")
    range_max:     Optional[float] = Field(default=None, description="Max for float/int/float_log ranges")
    categories:    Optional[list]  = Field(default=None, description="Allowed values for categorical type")
    default_value: Any             = Field(description="Sensible starting value to add to train.py")
    rationale:     str             = Field(description="1-sentence reason: what is stalling and what this might unlock")
    train_py_line: str             = Field(description="Exact top-level Python assignment, e.g. 'dropout_rate = 0.1'")


class DimensionProposalBatch(BaseModel):
    proposals: list[DimensionProposal] = Field(description="1-3 new dimensions to explore", max_length=3)


class TheoryGraphSummaryOutput(BaseModel):
    """Human-readable summary layer derived from /theory_graph JSON."""
    overview: str = Field(description="1-2 sentence plain-English summary of current theory state")
    key_points: list[str] = Field(description="3-6 concise bullets highlighting strongest supported/refuted/active relationships", min_length=1)
    next_actions: list[str] = Field(description="1-4 concrete follow-up experiment actions", min_length=1)
    caution: Optional[str] = Field(default=None, description="Optional caveat about uncertainty or sparse evidence")


# ── Tool definitions (Anthropic tool_use schema) ──────────────────────────────

PROGRAM_MD_TOOL = {
    "name":        "write_program_md",
    "description": (
        "Write structured program.md content for a research swarm. "
        "You are TRANSLATING mathematical output — do NOT make decisions. "
        "All freeze/kill/extend decisions were already made by ASHA, fANOVA, and Beta-Binomial."
    ),
    "input_schema": ProgramMDOutput.model_json_schema(),
}

HYPOTHESIS_TOOL = {
    "name":        "propose_hypotheses",
    "description": (
        "Propose 1-3 new falsifiable hypotheses based on observed statistical patterns. "
        "Each must be testable by a controlled experiment. "
        "Base proposals only on patterns visible in the data provided."
    ),
    "input_schema": HypothesisProposalBatch.model_json_schema(),
}

DIMENSION_TOOL = {
    "name":        "propose_new_dimensions",
    "description": (
        "Propose 1-3 NEW search dimensions (hyperparameters not currently in the search space) "
        "when the current search has stalled. Each must be a concrete train.py constant. "
        "Do NOT re-propose existing dimensions. "
        "Base proposals only on the stall pattern and current best results."
    ),
    "input_schema": DimensionProposalBatch.model_json_schema(),
}

THEORY_GRAPH_TOOL = {
    "name": "summarize_theory_graph",
    "description": (
        "Translate a structured hypothesis graph into concise plain English. "
        "Do not invent evidence; only summarize what appears in the graph."
    ),
    "input_schema": TheoryGraphSummaryOutput.model_json_schema(),
}


# ── Rendering (deterministic — no LLM) ────────────────────────────────────────

def render_program_md(out: ProgramMDOutput, bs: "BeliefState", active_workers: int) -> str:
    """Convert validated ProgramMDOutput → markdown string. Pure Python, no LLM."""
    phase = "WARMUP" if not bs.warmup_complete else "ACTIVE SEARCH"
    warmup_note = (
        f"\n> Early stopping DISABLED — collecting warmup data "
        f"({bs.experiment_count} / {bs.experiment_count + bs.warmup_runs_needed} runs).\n"
        if not bs.warmup_complete else ""
    )
    warning_block = f"\n> WARNING: {out.warning}\n" if out.warning else ""

    frozen_block = "\n".join(
        f"- `{e.name} = {e.value}` — {e.explanation}"
        for e in out.frozen_dims
    ) or "None yet."

    active_block = "\n".join(
        f"- `{e.name}` — {e.guidance}"
        for e in out.active_dims
    ) or "All dimensions active."

    hyp_block = "\n".join(
        f"- [{e.status.upper():9s}] P={e.posterior:.2f}  \"{e.statement}\" — {e.plain_english}"
        for e in out.hypotheses
    ) or "No hypotheses yet."

    instructions = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(out.concrete_instructions))

    return f"""# program.md — {phase}
*{bs.experiment_count} experiments · {active_workers} workers · kill rate {bs.kill_rate}% · best delta {bs.best_delta_bpb:+.4f}*
{warmup_note}{warning_block}
## Summary
{out.phase_summary}

## Frozen dimensions
{frozen_block}

## Active dimensions
{active_block}

## Hypothesis status
{hyp_block}

## Instructions for this batch
{instructions}
"""


# ── Main entry points ─────────────────────────────────────────────────────────

def generate_program_md(bs: "BeliefState", active_workers: int) -> str:
    """
    Generate program.md via schema-enforced LLM call.
    Falls back to deterministic template on any error.
    """
    try:
        out = _call_program_md_tool(bs)
        return render_program_md(out, bs, active_workers)
    except Exception as e:
        log.warning(f"program_md LLM call failed ({type(e).__name__}: {e}) — using template")
        return _template_fallback(bs, active_workers)


def propose_new_hypotheses(bs: "BeliefState") -> list[HypothesisProposal]:
    """
    Ask the LLM to propose new hypotheses based on the current BeliefState.
    Returns validated HypothesisProposal objects ready for HypothesisRegistry.
    Falls back to empty list on error — never corrupts belief state.
    """
    try:
        return _call_hypothesis_tool(bs)
    except Exception as e:
        log.warning(f"hypothesis proposal failed ({type(e).__name__}: {e}) — skipping")
        return []


def propose_new_dimensions(bs: "BeliefState", existing_dim_names: list[str]) -> list[DimensionProposal]:
    """
    Ask the LLM to propose new search dimensions when progress has stalled.
    Proposals are advisory — stored as pending_dimension_proposals for organizer review.
    Falls back to empty list on error.
    """
    try:
        return _call_dimension_tool(bs, existing_dim_names)
    except Exception as e:
        log.warning(f"dimension proposal failed ({type(e).__name__}: {e}) — skipping")
        return []


def summarize_theory_graph(graph: dict) -> dict:
    """
    Human-readable layer for theory graph JSON.
    Returns a derived narrative and metadata while keeping graph JSON authoritative.
    """
    try:
        out = _call_theory_graph_tool(graph)
        return {
            "mode": "llm",
            "derived_not_authoritative": True,
            "summary_text": _render_theory_graph_summary(out),
            "structured": out.model_dump(),
        }
    except Exception as e:
        log.warning(f"theory graph summary failed ({type(e).__name__}: {e}) — using template")
        return {
            "mode": "template",
            "derived_not_authoritative": True,
            "summary_text": _template_theory_graph_summary(graph),
            "structured": None,
        }


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _build_math_context(bs: "BeliefState") -> str:
    """
    Serialise BeliefState into prompt context.

    Structure:
      1. findings_summary — pre-computed by numpy, in the format the LLM asked for
      2. statistical_backing — raw F-stats / posteriors for reference
      3. recent_decisions — what the math already decided

    The LLM reads (1) first. It translates findings — it does not derive them.
    """
    s = bs.summary

    # ── 1. Pre-computed findings (numpy output) ────────────────────────────
    findings_json = json.dumps({
        "top_findings":      s.top_findings,
        "best_config":       s.best_config,
        "rejected":          s.rejected,
        "promising_regions": s.promising_regions,
        "interaction_hints": s.interaction_hints,
    }, indent=2)

    # ── 2. Statistical backing (for transparency) ──────────────────────────
    fanova_lines = []
    for name, r in sorted(bs.dimension_fstats.items(), key=lambda x: -x[1].get("eta_squared", 0)):
        tag = " [FROZEN]" if name in bs.frozen_dimensions else ""
        fanova_lines.append(f"  {name}: F={r['F']} p={r['p_value']} eta²={r['eta_squared']} n={r['n']}{tag}")

    hyp_lines = [
        f"  [{h['status']:9s}] P={h['posterior']:.2f} CI=[{h['ci_lo']:.2f},{h['ci_hi']:.2f}]"
        f" support={h.get('support_probability', 0):.2f}"
        f" refute={h.get('refute_probability', 0):.2f}"
        f" rope={h.get('rope_probability', 0):.2f}"
        f" g_support={h.get('gaussian_support_probability', 0):.2f}"
        f" g_refute={h.get('gaussian_refute_probability', 0):.2f}"
        f" effect_mu={h.get('effect_mu', 0):+.4f}"
        f" effect_sem={h.get('effect_sem', 0):.4f}"
        f" uncertain_streak={h.get('uncertain_streak', 0)}"
        f" sprint={h.get('in_decision_sprint', False)}"
        f" n={h['n']}  \"{h['statement']}\""
        for h in bs.hypotheses[:6]
    ]

    decision_lines = [f"  {d.type}: {d.readable_reason()}" for d in bs.recent_decisions[-6:]]

    # ── 3. Assemble ────────────────────────────────────────────────────────
    lines = [
        f"# experiment status",
        f"n={bs.experiment_count}  warmup_complete={bs.warmup_complete}"
        + (f"  ({bs.warmup_runs_needed} more needed)" if not bs.warmup_complete else ""),
        f"kill_rate={bs.kill_rate}%  extend_rate={bs.extend_rate}%"
        f"  asha_eta={bs.asha_eta}  pipeline_hit_rate={bs.pipeline_hit_rate:.0%}",
        "",
        "# findings_summary (pre-computed by numpy — translate this, do not re-derive)",
        findings_json,
        "",
        "# statistical_backing (fANOVA)",
    ] + fanova_lines + [
        "",
        "# hypothesis_posteriors (Beta-Binomial)",
    ] + hyp_lines + [
        "",
        "# recent_decisions (from belief engine)",
    ] + decision_lines

    return "\n".join(lines)


def _call_program_md_tool(bs: "BeliefState") -> ProgramMDOutput:
    import anthropic
    client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context = _build_math_context(bs)

    response = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 1024,
        tools      = [PROGRAM_MD_TOOL],
        tool_choice= {"type": "tool", "name": "write_program_md"},
        messages   = [{
            "role":    "user",
            "content": (
                "You are translating mathematical output into research instructions.\n"
                "DO NOT make decisions. Translate only what the math says.\n\n"
                "Mathematical output from belief engine:\n\n"
                f"{context}\n\n"
                "Call write_program_md with your translation."
            ),
        }],
    )

    # Extract tool_use block
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return ProgramMDOutput.model_validate(tool_block.input)


def _call_hypothesis_tool(bs: "BeliefState") -> list[HypothesisProposal]:
    import anthropic
    client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context = _build_math_context(bs)

    response = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 512,
        tools      = [HYPOTHESIS_TOOL],
        tool_choice= {"type": "tool", "name": "propose_hypotheses"},
        messages   = [{
            "role":    "user",
            "content": (
                "Based on these experiment results, propose 1-3 new hypotheses to test.\n"
                "Base proposals ONLY on patterns visible in the data below.\n"
                "If a hypothesis is mature enough, set phase=validation and include an executable test_spec.\n"
                "Supported test_spec.type values: single_factor_effect, interaction_grid.\n"
                "For single_factor_effect include {variable, values, min_runs_per_cell, decision_rule:{threshold}}.\n"
                "For interaction_grid include {variables:[a,b], grid:{a:[...],b:[...]}, min_runs_per_cell, decision_rule:{threshold}}.\n\n"
                f"{context}\n\n"
                "Call propose_hypotheses with your proposals."
            ),
        }],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    batch      = HypothesisProposalBatch.model_validate(tool_block.input)
    return batch.proposals


def _call_dimension_tool(bs: "BeliefState", existing_dim_names: list[str]) -> list[DimensionProposal]:
    import anthropic
    client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context = _build_math_context(bs)

    response = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 512,
        tools      = [DIMENSION_TOOL],
        tool_choice= {"type": "tool", "name": "propose_new_dimensions"},
        messages   = [{
            "role":    "user",
            "content": (
                "The search has stalled — no meaningful improvement over the last 50 experiments.\n"
                "Propose 1-3 NEW hyperparameters that are NOT in the current search space "
                "and might unlock further improvement.\n\n"
                f"Current search dimensions (DO NOT re-propose these): {existing_dim_names}\n\n"
                "Experiment data:\n\n"
                f"{context}\n\n"
                "Call propose_new_dimensions with your proposals."
            ),
        }],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    batch      = DimensionProposalBatch.model_validate(tool_block.input)
    return batch.proposals


def _render_theory_graph_summary(out: TheoryGraphSummaryOutput) -> str:
    bullets = "\n".join(f"- {b}" for b in out.key_points)
    actions = "\n".join(f"- {a}" for a in out.next_actions)
    caution = f"\nCaution: {out.caution}" if out.caution else ""
    return f"Overview: {out.overview}\n\nKey points:\n{bullets}\n\nNext actions:\n{actions}{caution}"


def _call_theory_graph_tool(graph: dict) -> TheoryGraphSummaryOutput:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))

    # Keep prompt bounded; preserve highest-confidence nodes first.
    nodes_sorted = sorted(nodes, key=lambda n: float(n.get("posterior", 0.0)), reverse=True)
    compact_graph = {
        "nodes": nodes_sorted[:40],
        "edges": edges[:80],
        "counts": {"nodes": len(nodes), "edges": len(edges)},
    }

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=700,
        tools=[THEORY_GRAPH_TOOL],
        tool_choice={"type": "tool", "name": "summarize_theory_graph"},
        messages=[{
            "role": "user",
            "content": (
                "Summarize this hypothesis theory graph in plain English. "
                "Do not invent data or claims.\n\n"
                f"{json.dumps(compact_graph, indent=2)}\n\n"
                "Call summarize_theory_graph with concise output."
            ),
        }],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    return TheoryGraphSummaryOutput.model_validate(tool_block.input)


def _template_theory_graph_summary(graph: dict) -> str:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    if not nodes:
        return "Overview: No hypotheses are currently available in the theory graph."

    supported = [n for n in nodes if n.get("status") == "supported"]
    refuted = [n for n in nodes if n.get("status") == "refuted"]
    active = [n for n in nodes if n.get("status") == "active"]

    top = sorted(nodes, key=lambda n: float(n.get("posterior", 0.0)), reverse=True)[:2]
    top_lines = []
    for n in top:
        top_lines.append(
            f"- \"{n.get('statement', 'unknown')}\" (status={n.get('status')}, "
            f"P={float(n.get('posterior', 0.0)):.2f}, effect_mu={float(n.get('effect_mu', 0.0)):+.4f})"
        )

    decomp = sum(1 for e in edges if e.get("type") == "decomposes_into")
    linked = sum(1 for e in edges if e.get("type") == "linked")

    return (
        f"Overview: Theory graph has {len(nodes)} hypotheses and {len(edges)} relationships. "
        f"Supported={len(supported)}, refuted={len(refuted)}, active={len(active)}.\n\n"
        f"Key points:\n{chr(10).join(top_lines) if top_lines else '- No ranked hypotheses yet.'}\n"
        f"- Structural links: decomposes_into={decomp}, linked={linked}.\n\n"
        "Next actions:\n"
        "- Allocate more runs to the strongest active child hypotheses.\n"
        "- Re-test high-posterior parent hypotheses with low-sample children.\n"
        "- Archive stale active nodes if posterior remains near 0.5 after decision sprints."
    )


# ── Deterministic fallback ─────────────────────────────────────────────────────

def _template_fallback(bs: "BeliefState", active_workers: int) -> str:
    """No LLM, no external calls. Always works."""
    phase   = "WARMUP" if not bs.warmup_complete else "ACTIVE SEARCH"
    frozen  = bs.frozen_dimensions
    active  = {k: v for k, v in bs.dimension_fstats.items() if k not in frozen}

    warmup_note = (
        f"\n> Early stopping DISABLED — warmup ({bs.experiment_count}/{bs.experiment_count + bs.warmup_runs_needed}).\n"
        if not bs.warmup_complete else ""
    )

    return f"""# program.md — {phase} (template)
*{bs.experiment_count} experiments · {active_workers} workers · kill rate {bs.kill_rate}%*
{warmup_note}
## Frozen dimensions (fANOVA: p>0.10 AND eta²<0.05)
{chr(10).join(f"- `{k} = {v}`" for k, v in frozen.items()) or "- None yet"}

## Active dimensions (by eta² descending)
{chr(10).join(f"- `{k}` (eta²={v.get('eta_squared',0):.3f})" for k, v in sorted(active.items(), key=lambda x: -x[1].get('eta_squared',0))[:5]) or "- All active"}

## Best result
delta_bpb = {bs.best_delta_bpb:+.4f}  config = {json.dumps(bs.best_config_delta)}

## Instructions
1. Apply the config_delta from /next_config to your train.py
2. Call report(val_bpb, progress) at each eval step — required for early stopping
3. Do NOT modify frozen dimensions
"""
