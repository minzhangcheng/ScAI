"""Pure prompt rendering and plan-boundary normalization support."""
from __future__ import annotations

from typing import Any, List, Optional

from ...utils.formatting import as_list, coerce_json_text, truncate_for_llm

def render_pi_modifications(pi_modifications: Any, *, max_chars: int = 8_000) -> str:
    """Flatten the PI's structured per-step suggestions into prompt text.

    Plan review returns prose feedback plus a list of step-scoped suggestions.
    Only the prose used to reach the revising agent, so the concrete asks — the
    part naming what a step should actually cover — were dropped every round
    while the prose caution accumulated.

    Accepts either a flat list of suggestion dicts or the round-wrapped form
    ``[{"round": n, "modifications": [...]}]``.
    """
    if not pi_modifications:
        return ""

    def _emit(entry: Any, round_label: str) -> Optional[str]:
        if isinstance(entry, dict):
            step_id = entry.get("step_id", entry.get("step", ""))
            suggestion = str(
                entry.get("suggestion", entry.get("change", entry.get("text", "")))
            ).strip()
            if not suggestion:
                return None
            head = f"  {round_label}Step {step_id}: " if step_id != "" else f"  {round_label}"
            return head + suggestion
        text = str(entry or "").strip()
        return f"  {round_label}{text}" if text else None

    lines: List[str] = []
    for item in pi_modifications:
        if isinstance(item, dict) and isinstance(item.get("modifications"), list):
            label = f"[Round {item.get('round', '?')}] "
            for entry in item["modifications"]:
                rendered = _emit(entry, label)
                if rendered:
                    lines.append(rendered)
            continue
        rendered = _emit(item, "")
        if rendered:
            lines.append(rendered)

    if not lines:
        return ""
    return truncate_for_llm("\n".join(lines), max_chars=max_chars)


def normalize_plan_step_aliases(value: dict) -> dict:
    """Accept known structural aliases without overriding explicit fields."""
    step = dict(value)
    for alias, canonical in (
        ("step_id", "id"),
        ("output", "expected_output"),
        ("required_figures", "figure_requirements"),
    ):
        if canonical not in step and alias in step:
            step[canonical] = step[alias]
    return step


def _normalize_plan_boundary(value: Any) -> dict:
    """Normalize model-shaped planner fields before schema validation."""
    if not isinstance(value, dict):
        return {
            "steps": [],
            "parse_error": "Invalid plan schema: response must be a JSON object",
        }
    data = dict(value)
    if "parse_error" in data:
        return data
    if "overview" in data:
        data["overview"] = coerce_json_text(data.get("overview"))
    if "expected_runtime" in data:
        data["expected_runtime"] = coerce_json_text(data.get("expected_runtime"))
    if "key_decision_points" in data:
        raw_points = as_list(data.get("key_decision_points"))
        data["key_decision_points"] = [
            coerce_json_text(item) for item in raw_points if coerce_json_text(item)
        ]
    raw_steps = data.get("steps")
    if isinstance(raw_steps, dict):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        data["steps"] = []
        data["parse_error"] = "Invalid plan schema: steps must be a JSON array"
        return data
    normalized_steps = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            normalized_steps.append(raw_step)
            continue
        step = normalize_plan_step_aliases(raw_step)
        for field in (
            "title",
            "method",
            "input",
            "expected_output",
            "rationale",
            "text_only_reason",
        ):
            step[field] = coerce_json_text(step.get(field))
        step["depends_on"] = as_list(step.get("depends_on"))
        step["figure_requirements"] = as_list(step.get("figure_requirements"))
        if not isinstance(step.get("allow_text_only"), bool):
            step["allow_text_only"] = False
        normalized_steps.append(step)
    data["steps"] = normalized_steps
    return data


#: One rejected-plan policy for BOTH entry points. `generate_analysis_plan` and
#: `revise_plan_incrementally` are two real runtime paths into the same decision,
#: and they had drifted into different contracts -- one demanding every entity of
#: the superseded plan be carried forward, the other bounding the obligation to
#: the science the PI asked for.
REJECTED_PLAN_POLICY = """The following plan was rejected. Build a new approach guided by the PI's feedback in the dialogue. Replace what the feedback objects to; you do not have to discard a method the feedback did not criticise, and changing a sound method merely to look different is churn, not a response.

What must still be covered is the SCIENCE the PI asked for: the PI's tasks, the research question, and any evidence need the feedback leaves open. That is not the same as the rejected plan's inventory. An entity, gene, mechanism or method that appears only because the previous plan happened to name it, and that neither the PI's tasks nor the research context calls for, may be dropped -- say which and why. Carrying an unsupported assumption forward because a superseded plan contained it is path dependence, not thoroughness. Anything the PI's feedback explicitly asks to keep, keep.

Completed work with evidence behind it can be cited rather than mechanically redone.

When a COMPLETED step has already settled a question, say which of these the settled answer is, because only the first is a biological result:
  * observed absence -- measurable here, and not present
  * below detection -- possibly present under this assay's detection limit, depth, or filtering
  * not measured -- the required layer, covariate, modality, or annotation is absent from this dataset; this says nothing about the biology
  * not identifiable -- the data exist but the design cannot separate the effect from a confound
Never report the last three as a negative result."""


def _planner_modality_line(state) -> str:
    """Describe whether the single-cell run also includes bulk data."""
    try:
        has_bulk = bool(state.has_bulk_rnaseq())
    except AttributeError:
        return ""
    if has_bulk:
        return (
            "This run has BOTH single-cell and bulk data. Single-cell methods apply to the "
            "single-cell object only; bulk analyses run on the bulk matrices, and any "
            "cross-modal comparison must state whether the two came from the same biological "
            "samples (a consistency check) or different ones (replication).\n"
        )
    return ""
