"""
meta_server/program_writer.py

LLM translation layer — receives BeliefState from belief_engine.py and
renders it into human-readable program.md and hypothesis proposals.

SCHEMA ENFORCEMENT:
    All LLM calls use structured JSON generation validated by Pydantic.
    Supported providers: Anthropic, OpenAI, Gemini (OpenAI-compatible endpoint).
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
import re
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .belief_engine import BeliefState

log = logging.getLogger(__name__)

WRITE_EVERY = 50
IMMUTABLE_START = "<!-- BAD_PI_IMMUTABLE_START -->"
IMMUTABLE_END = "<!-- BAD_PI_IMMUTABLE_END -->"
MUTABLE_START = "<!-- BAD_PI_MUTABLE_START -->"
MUTABLE_END = "<!-- BAD_PI_MUTABLE_END -->"
ASSIGNMENT_START = "<!-- BAD_PI_ASSIGNMENT_START -->"
ASSIGNMENT_END = "<!-- BAD_PI_ASSIGNMENT_END -->"


def should_write(total_experiments: int, last_written_at: int) -> bool:
    return (total_experiments - last_written_at) >= WRITE_EVERY


def _resolve_llm_provider() -> str:
    """
    Resolve which provider to use for structured generation.

    BAD_PI_LLM_PROVIDER: anthropic | openai | gemini | auto
    auto priority: anthropic -> openai -> gemini
    """
    pref = str(os.environ.get("BAD_PI_LLM_PROVIDER", "auto") or "auto").strip().lower()
    valid = {"anthropic", "openai", "gemini", "auto"}
    if pref not in valid:
        pref = "auto"

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    if pref == "auto":
        if has_anthropic:
            return "anthropic"
        if has_openai:
            return "openai"
        if has_gemini:
            return "gemini"
        raise RuntimeError("No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY.")

    if pref == "anthropic" and not has_anthropic:
        raise RuntimeError("BAD_PI_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
    if pref == "openai" and not has_openai:
        raise RuntimeError("BAD_PI_LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
    if pref == "gemini" and not has_gemini:
        raise RuntimeError("BAD_PI_LLM_PROVIDER=gemini but GEMINI_API_KEY/GOOGLE_API_KEY is not set")
    return pref


def _structured_call(
    *,
    prompt: str,
    model_cls: type[BaseModel],
    anthropic_tool: dict,
    max_tokens: int,
) -> BaseModel:
    """Provider-agnostic structured generation with schema validation."""
    provider = _resolve_llm_provider()

    if provider == "anthropic":
        import anthropic

        model_name = os.environ.get("BAD_PI_MODEL_ANTHROPIC", "claude-sonnet-4-20250514")
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            tools=[anthropic_tool],
            tool_choice={"type": "tool", "name": anthropic_tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return model_cls.model_validate(tool_block.input)

    # OpenAI + Gemini(openai-compatible endpoint)
    from openai import OpenAI

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        model_name = os.environ.get("BAD_PI_MODEL_GEMINI", "gemini-2.0-flash")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        model_name = os.environ.get("BAD_PI_MODEL_OPENAI", "gpt-4.1-mini")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    schema_json = json.dumps(model_cls.model_json_schema(), indent=2)
    prompt_with_schema = (
        f"{prompt}\n\n"
        "Return ONLY valid JSON matching this schema exactly.\n"
        "No markdown, no prose, no code fences.\n"
        f"JSON Schema:\n{schema_json}"
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_with_schema}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = completion.choices[0].message.content or "{}"
    return model_cls.model_validate(json.loads(raw))


def compose_program_md(base_template: Optional[str], live_update: str) -> str:
    """
    Compose final worker-facing program.md with immutable/mutable separation.

    If mutable markers exist in base_template, only replace mutable block content.
    Otherwise preserve the base template verbatim and append one live-update block.
    """
    if not base_template:
        return live_update

    base = str(base_template)
    if MUTABLE_START in base and MUTABLE_END in base:
        pre, rest = base.split(MUTABLE_START, 1)
        _, post = rest.split(MUTABLE_END, 1)
        return (
            pre
            + MUTABLE_START
            + "\n"
            + live_update.strip()
            + "\n"
            + MUTABLE_END
            + post
        )

    # Backward-compatible fallback for templates without explicit markers.
    return (
        base.rstrip()
        + "\n\n---\n"
        + "## Meta-PI live update (auto-generated)\n"
        + "\n"
        + live_update.strip()
        + "\n"
    )


def _assignment_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def render_assignment_block(assignment: dict[str, Any]) -> str:
    return (
        f"{ASSIGNMENT_START}\n"
        "```json\n"
        f"{_assignment_json(assignment)}\n"
        "```\n"
        f"{ASSIGNMENT_END}"
    )


def _focus_dims_from_statement(statement: str) -> list[str]:
    dims: list[str] = []
    text = str(statement or "").strip()
    if not text:
        return dims

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+matters\s+for\b", text, flags=re.IGNORECASE)
    if m:
        dims.append(m.group(1))

    m2 = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*[x×]\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if m2:
        dims.extend([m2.group(1), m2.group(2)])

    seen: set[str] = set()
    return [d for d in dims if not (d in seen or seen.add(d))]


def _global_strategy(bs: "BeliefState") -> str:
    if bs.is_converging:
        return "converge"
    if bs.is_stalled:
        return "decision_sprint"
    return "investigate"


def _top_active_dims(bs: "BeliefState", limit: int = 3) -> list[str]:
    ranked = sorted(
        [
            (name, float(stats.get("eta_squared", bs.dimension_importance.get(name, 0.0)) or 0.0))
            for name, stats in bs.dimension_fstats.items()
            if name not in bs.frozen_dimensions
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return [name for name, _ in ranked[:limit]]


def build_global_assignment(bs: "BeliefState") -> dict[str, Any]:
    strategy = _global_strategy(bs)
    top_hypothesis = bs.hypotheses[0] if bs.hypotheses else {}
    must_explore = _top_active_dims(bs)
    if not must_explore and top_hypothesis.get("statement"):
        must_explore = _focus_dims_from_statement(str(top_hypothesis.get("statement") or ""))

    return {
        "assignment_id": f"global_n{bs.experiment_count}",
        "guidance_version": f"program_n{bs.experiment_count}",
        "population": {
            "id": "global",
            "strategy": strategy,
        },
        "hypothesis": {
            "id": top_hypothesis.get("id"),
            "statement": top_hypothesis.get("statement"),
        },
        "budget_seconds": 300,
        "base_config_delta": dict(bs.best_config_delta or {}),
        "hard_constraints": {
            "must_hold_fixed": dict(bs.frozen_dimensions or {}),
            "must_explore": must_explore,
            "must_not_change": [
                "prepare.py",
                "evaluation harness",
                "tokenizer/data pipeline",
            ],
        },
        "soft_preferences": {
            "search_radius": "narrow" if strategy == "converge" else "broad",
            "repeat_target": 3 if strategy == "decision_sprint" else 2,
            "prefer_simple_diffs": True,
        },
        "reporting": {
            "priority_metrics": ["val_bpb", "peak_vram_mb", "status"],
            "failure_labels": ["oom", "timeout", "bug", "other"],
        },
    }


def build_population_assignment(
    pop: Any,
    hypothesis: Any,
    top_config: Optional[dict[str, Any]] = None,
    dimensions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    best_cfg = dict((top_config or {}).get("config_delta") or {})
    frozen_dims = {
        d["name"]: d["frozen_value"]
        for d in (dimensions or [])
        if bool(d.get("frozen"))
    }
    must_hold_fixed = {
        **frozen_dims,
        **dict(getattr(hypothesis, "config_constraint", {}) or {}),
    }

    test_spec = getattr(hypothesis, "test_spec", None) or {}
    must_explore: list[str] = []
    if isinstance(test_spec, dict):
        variable = test_spec.get("variable")
        if isinstance(variable, str):
            must_explore.append(variable)
        must_explore.extend(
            [v for v in test_spec.get("variables", []) if isinstance(v, str)]
        )
    if not must_explore:
        must_explore = _focus_dims_from_statement(getattr(hypothesis, "statement", ""))
    seen: set[str] = set()
    must_explore = [d for d in must_explore if not (d in seen or seen.add(d))]

    strategy = getattr(pop, "strategy", "investigate")
    search_radius = {
        "exploit": "narrow",
        "converge": "narrow",
        "falsify": "narrow",
        "validate": "narrow",
        "decision_sprint": "narrow",
        "investigate": "broad",
        "moonshot": "wide",
    }.get(strategy, "broad")
    repeat_target = {
        "validate": 3,
        "falsify": 3,
        "decision_sprint": 4,
        "exploit": 2,
        "converge": 3,
        "investigate": 2,
        "moonshot": 1,
    }.get(strategy, 2)

    assignment: dict[str, Any] = {
        "assignment_id": f"{getattr(pop, 'id', 'pop')}:{getattr(hypothesis, 'id', 'hyp')}:n{getattr(hypothesis, 'n_experiments', 0)}",
        "guidance_version": f"{getattr(pop, 'id', 'pop')}_{strategy}_{getattr(hypothesis, 'n_experiments', 0)}",
        "population": {
            "id": getattr(pop, "id", None),
            "strategy": strategy,
        },
        "hypothesis": {
            "id": getattr(hypothesis, "id", None),
            "statement": getattr(hypothesis, "statement", None),
        },
        "budget_seconds": 300,
        "base_config_delta": {
            **best_cfg,
            **dict(getattr(hypothesis, "config_constraint", {}) or {}),
        },
        "hard_constraints": {
            "must_hold_fixed": must_hold_fixed,
            "must_explore": must_explore,
            "must_not_change": [
                "prepare.py",
                "evaluation harness",
                "tokenizer/data pipeline",
            ],
        },
        "soft_preferences": {
            "search_radius": search_radius,
            "repeat_target": repeat_target,
            "prefer_simple_diffs": True,
        },
        "reporting": {
            "priority_metrics": ["val_bpb", "peak_vram_mb", "status"],
            "failure_labels": ["oom", "timeout", "bug", "other"],
        },
    }
    if test_spec:
        assignment["test_spec"] = test_spec
    return assignment


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
    test_spec:         dict = Field(description="Executable test protocol (required). Defines how to falsify the claim via controlled experiment.")


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


# ── Structured tool schemas (provider-agnostic) ───────────────────────────────

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

    assignment_block = render_assignment_block(build_global_assignment(bs))
    instructions = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(out.concrete_instructions))

    return f"""# program.md — {phase}
*{bs.experiment_count} experiments · {active_workers} workers · kill rate {bs.kill_rate}% · best delta {bs.best_delta_bpb:+.4f}*
{warmup_note}{warning_block}
## Summary
{out.phase_summary}

## Assignment
{assignment_block}

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

def generate_program_md(bs: "BeliefState", active_workers: int, base_template: Optional[str] = None) -> str:
    """
    Generate program.md via schema-enforced LLM call.
    If base_template is provided, treat it as the canonical guidance document and
    update instructions incrementally rather than rewriting from scratch.
    Falls back to deterministic template on any error.
    """
    try:
        out = _call_program_md_tool(bs, base_template=base_template)
        live = render_program_md(out, bs, active_workers)
        return compose_program_md(base_template, live)
    except Exception as e:
        log.warning(f"program_md LLM call failed ({type(e).__name__}: {e}) — using template")
        live = _template_fallback(bs, active_workers)
        return compose_program_md(base_template, live)


def propose_new_hypotheses(bs: "BeliefState", base_template: Optional[str] = None) -> list[HypothesisProposal]:
    """
    Ask the LLM to propose new hypotheses based on the current BeliefState.
    Includes the original base program.md when available so the LLM can avoid
    re-proposing search directions already covered by the charter.
    Returns validated HypothesisProposal objects ready for HypothesisRegistry.
    Falls back to empty list on error — never corrupts belief state.
    """
    try:
        return _call_hypothesis_tool(bs, base_template=base_template)
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


def propose_initial_dimensions(
    *,
    base_template: str,
    existing_dimensions: list[dict],
    schema_sql: str,
) -> list[DimensionProposal]:
    """
    One-time startup pass.

    Uses base program.md + current schema/dimensions to identify missing search axes,
    especially architecture/model-family categorical dimensions when implied by the
    experiment charter but not represented in the search space.
    """
    try:
        return _call_initial_dimension_bootstrap_tool(
            base_template=base_template,
            existing_dimensions=existing_dimensions,
            schema_sql=schema_sql,
        )
    except Exception as e:
        log.warning(f"initial dimension bootstrap failed ({type(e).__name__}: {e}) — skipping")
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


def _call_program_md_tool(bs: "BeliefState", base_template: Optional[str] = None) -> ProgramMDOutput:
    context = _build_math_context(bs)
    continuity_block = (
        "\nCanonical base template (preserve intent/continuity; update only what evidence supports):\n\n"
        f"{base_template}\n\n"
        if base_template else ""
    )

    prompt = (
        "You are translating mathematical output into research instructions.\n"
        "DO NOT make decisions. Translate only what the math says.\n\n"
        "Treat the base template as the stable experiment charter and keep continuity across updates.\n"
        "Do not discard valid standing instructions unless contradicted by evidence.\n\n"
        f"{continuity_block}"
        "Mathematical output from belief engine:\n\n"
        f"{context}\n\n"
        "Produce a JSON object for write_program_md."
    )
    out = _structured_call(
        prompt=prompt,
        model_cls=ProgramMDOutput,
        anthropic_tool=PROGRAM_MD_TOOL,
        max_tokens=1024,
    )
    return out


def _call_hypothesis_tool(bs: "BeliefState", base_template: Optional[str] = None) -> list[HypothesisProposal]:
    context = _build_math_context(bs)
    base_block = ""
    if base_template:
        base_block = (
            "Original base program.md (charter/context) — use it to avoid repeating already covered directions:\n\n"
            f"{base_template}\n\n"
        )
    prompt = (
        "Based on these experiment results, propose 1-3 new hypotheses to test.\n"
        "Base proposals ONLY on patterns visible in the data below.\n"
        "IMPORTANT: Every hypothesis MUST include a test_spec — it defines how to falsify your claim.\n"
        "phase=exploration: early-stage hypothesis, test_spec guides signal collection.\n"
        "phase=validation: mature hypothesis, test_spec is strict experimental protocol.\n"
        "Supported test_spec.type values: single_factor_effect, interaction_grid.\n"
        "For single_factor_effect include {variable, values, min_runs_per_cell, decision_rule:{threshold}}.\n"
        "For interaction_grid include {variables:[a,b], grid:{a:[...],b:[...]}, min_runs_per_cell, decision_rule:{threshold}}.\n\n"
        f"{base_block}"
        f"{context}\n\n"
        "Produce a JSON object for propose_hypotheses."
    )
    batch = _structured_call(
        prompt=prompt,
        model_cls=HypothesisProposalBatch,
        anthropic_tool=HYPOTHESIS_TOOL,
        max_tokens=512,
    )
    return batch.proposals


def _call_dimension_tool(bs: "BeliefState", existing_dim_names: list[str]) -> list[DimensionProposal]:
    context = _build_math_context(bs)
    prompt = (
        "The search has stalled — no meaningful improvement over the last 50 experiments.\n"
        "Propose 1-3 NEW hyperparameters that are NOT in the current search space "
        "and might unlock further improvement.\n\n"
        f"Current search dimensions (DO NOT re-propose these): {existing_dim_names}\n\n"
        "Experiment data:\n\n"
        f"{context}\n\n"
        "Produce a JSON object for propose_new_dimensions."
    )
    batch = _structured_call(
        prompt=prompt,
        model_cls=DimensionProposalBatch,
        anthropic_tool=DIMENSION_TOOL,
        max_tokens=512,
    )
    return batch.proposals


def _call_initial_dimension_bootstrap_tool(
    *,
    base_template: str,
    existing_dimensions: list[dict],
    schema_sql: str,
) -> list[DimensionProposal]:
    existing_summary = [
        {
            "name": d.get("name"),
            "dtype": d.get("dtype"),
            "categories": d.get("categories"),
            "min_val": d.get("min_val"),
            "max_val": d.get("max_val"),
        }
        for d in existing_dimensions
    ]

    prompt = (
        "One-time startup analysis for a distributed research worker swarm.\n"
        "Goal: detect missing search dimensions implied by the base experiment charter.\n"
        "If architecture/model family is implied but no clean dimension exists, propose a categorical dimension\n"
        "such as ARCH or MODEL_FAMILY with concrete categories (e.g. cnn, rnn, xgboost, random_forest) as appropriate.\n"
        "Only propose dimensions that can be executed by changing train.py config constants.\n"
        "Do NOT re-propose existing dimensions.\n"
        "Output 0-3 proposals.\n\n"
        "Base program.md template:\n\n"
        f"{base_template}\n\n"
        "Current dimensions (already present):\n"
        f"{json.dumps(existing_summary, indent=2)}\n\n"
        "schema.sql (for context):\n"
        f"{schema_sql}\n\n"
        "Produce a JSON object for propose_new_dimensions."
    )
    batch = _structured_call(
        prompt=prompt,
        model_cls=DimensionProposalBatch,
        anthropic_tool=DIMENSION_TOOL,
        max_tokens=700,
    )
    return batch.proposals


def _render_theory_graph_summary(out: TheoryGraphSummaryOutput) -> str:
    bullets = "\n".join(f"- {b}" for b in out.key_points)
    actions = "\n".join(f"- {a}" for a in out.next_actions)
    caution = f"\nCaution: {out.caution}" if out.caution else ""
    return f"Overview: {out.overview}\n\nKey points:\n{bullets}\n\nNext actions:\n{actions}{caution}"


def _call_theory_graph_tool(graph: dict) -> TheoryGraphSummaryOutput:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))

    # Keep prompt bounded; preserve highest-confidence nodes first.
    nodes_sorted = sorted(nodes, key=lambda n: float(n.get("posterior", 0.0)), reverse=True)
    compact_graph = {
        "nodes": nodes_sorted[:40],
        "edges": edges[:80],
        "counts": {"nodes": len(nodes), "edges": len(edges)},
    }

    prompt = (
        "Summarize this hypothesis theory graph in plain English. "
        "Do not invent data or claims.\n\n"
        f"{json.dumps(compact_graph, indent=2)}\n\n"
        "Produce a JSON object for summarize_theory_graph."
    )
    out = _structured_call(
        prompt=prompt,
        model_cls=TheoryGraphSummaryOutput,
        anthropic_tool=THEORY_GRAPH_TOOL,
        max_tokens=700,
    )
    return out


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
    assignment_block = render_assignment_block(build_global_assignment(bs))

    return f"""# program.md — {phase} (template)
*{bs.experiment_count} experiments · {active_workers} workers · kill rate {bs.kill_rate}%*
{warmup_note}
## Assignment
{assignment_block}

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
