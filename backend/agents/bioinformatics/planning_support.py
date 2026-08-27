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
        step = dict(raw_step)
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
