"""Typed normalization helpers for PI planning response boundaries."""

from typing import Any, Dict, Iterable, Optional
import uuid

from ...constants import PI_REVIEW_FALLBACK_DECISION
from ...models.schemas import AdataSourceSelection, AnalysisPlan, PlanStep
from ...utils.formatting import as_list, coerce_json_text


def normalize_adata_source_selection(
    value: Any,
) -> Optional[AdataSourceSelection]:
    """Normalize one PI-supplied source selector, or return None on omission.

    An explicit malformed selector is an operation-local schema error, not a
    silent fallback to ``current``.
    """
    if value is None:
        return None
    if isinstance(value, AdataSourceSelection):
        return value
    if isinstance(value, str):
        value = {"mode": value}
    if not isinstance(value, dict):
        raise ValueError("adata_source must be an object or null")
    return AdataSourceSelection.model_validate(value)


_PLAN_REVIEW_DECISIONS = frozenset({"approved", "revise", "reject"})
_USER_PLAN_INTENTS = frozenset({"confirm", "modify", "unclear"})
_USER_PLAN_ACTIONS = frozenset({
    "proceed_to_execution",
    "revise_with_bioinfo",
    "clarify_with_user",
})


def normalize_plan_review_result(value: Any) -> Dict[str, Any]:
    """Normalize one PI review before workflow code consumes it."""
    data = dict(value) if isinstance(value, dict) else {}
    decision = coerce_json_text(data.get("decision")).casefold()
    if decision not in _PLAN_REVIEW_DECISIONS:
        decision = PI_REVIEW_FALLBACK_DECISION
    feedback = coerce_json_text(data.get("feedback"))
    if not feedback and not data:
        feedback = "Plan needs review (fallback — PI response could not be parsed)"
    result = {
        **data,
        "decision": decision,
        "feedback": feedback,
        "modifications": as_list(data.get("modifications")),
        "modification_dispositions": as_list(
            data.get("modification_dispositions")
        ),
    }
    if "adata_source" in data:
        result["adata_source"] = normalize_adata_source_selection(
            data.get("adata_source")
        )
    return result


def select_valid_plan_seed(
    current_plan: Optional[AnalysisPlan],
    state_plan: Optional[AnalysisPlan],
    dialogue_history: Iterable[Dict[str, Any]],
    *,
    expected_stage: Optional[str] = None,
) -> Optional[AnalysisPlan]:
    """Select a non-rejected, non-empty plan seed in newest-first order."""

    def eligible(candidate: Optional[AnalysisPlan]) -> bool:
        return bool(
            candidate is not None
            and candidate.steps
            and candidate.status != "rejected"
            and (expected_stage is None or candidate.stage == expected_stage)
        )

    for candidate in (current_plan, state_plan):
        if eligible(candidate):
            return candidate

    for entry in reversed(list(dialogue_history or [])):
        if not isinstance(entry, dict) or entry.get("pi_decision") == "reject":
            continue
        payload = entry.get("plan")
        if not isinstance(payload, dict) or not payload.get("steps"):
            continue
        try:
            candidate = AnalysisPlan.model_validate(payload)
        except Exception:
            continue
        if eligible(candidate):
            return candidate
    return None


def _normalize_final_payload(value: Any) -> Dict[str, Any]:
    """Normalize scalar/mapping/list/null fields without making decisions."""
    from ..bioinformatics.planning_support import normalize_plan_step_aliases

    data = dict(value) if isinstance(value, dict) else {}
    data["overview"] = coerce_json_text(data.get("overview"))
    data["expected_runtime"] = coerce_json_text(data.get("expected_runtime"))
    data["key_decision_points"] = [
        coerce_json_text(item)
        for item in as_list(data.get("key_decision_points"))
        if coerce_json_text(item)
    ]
    raw_steps = data.get("steps")
    if isinstance(raw_steps, dict):
        raw_steps = [raw_steps]
    data["steps"] = list(raw_steps) if isinstance(raw_steps, list) else []
    normalized_steps = []
    for raw_step in data["steps"]:
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
        normalized_steps.append(step)
    data["steps"] = normalized_steps
    if "adata_source" in data:
        data["adata_source"] = normalize_adata_source_selection(
            data.get("adata_source")
        )
    return data


def build_validated_final_plan(
    parsed: Any,
    *,
    phase: str,
    stage: str,
    seed_plan: Optional[AnalysisPlan],
    termination_reason: str,
    scope_contract_version: Optional[int] = None,
    foundation_inventory_id: Optional[str] = None,
) -> AnalysisPlan:
    """Build a unilateral plan through the same host schema boundary."""
    from ..bioinformatics.planning import PlanningMixin

    normalized = _normalize_final_payload(parsed)
    normalized = PlanningMixin._prepare_plan_data(
        normalized,
        require_complete_steps=True,
    )
    if normalized.get("parse_error"):
        raise ValueError(normalized["parse_error"])
    raw_steps = normalized.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("no parseable steps")
    steps = [PlanStep.model_validate(raw_step) for raw_step in raw_steps]
    explicit_source = normalized.get("adata_source")
    source = explicit_source or (
        seed_plan.adata_source.model_copy(deep=True)
        if seed_plan is not None
        else AdataSourceSelection()
    )
    source_preparation_id = (
        seed_plan.adata_source_preparation_id
        if seed_plan is not None
        and seed_plan.adata_source_preparation_id
        and source == seed_plan.adata_source
        else None
    )
    same_source_revision = bool(
        seed_plan is not None and source == seed_plan.adata_source
    )
    run_scope_id = (
        seed_plan.bioinformatics_run_scope_id
        if same_source_revision
        else f"run_scope_{uuid.uuid4().hex}"
    )
    settlement_scope_id = (
        seed_plan.bioinformatics_settlement_scope_id
        if seed_plan is not None
        else f"settle_scope_{uuid.uuid4().hex}"
    ) or f"settle_scope_{uuid.uuid4().hex}"
    return AnalysisPlan(
        phase=phase,
        stage=stage,
        overview=coerce_json_text(normalized.get("overview")),
        steps=steps,
        key_decision_points=[
            coerce_json_text(item)
            for item in as_list(normalized.get("key_decision_points"))
            if coerce_json_text(item)
        ],
        expected_runtime=(
            coerce_json_text(normalized.get("expected_runtime"))
            or f"~{len(steps)} steps"
        ),
        status="pi_reviewing",
        revision_of=seed_plan.id if seed_plan is not None else None,
        revision_reason=f"{termination_reason} — PI finalized unilaterally",
        adata_source=source,
        adata_source_preparation_id=source_preparation_id,
        bioinformatics_run_scope_id=(
            run_scope_id or f"run_scope_{uuid.uuid4().hex}"
        ),
        bioinformatics_settlement_scope_id=settlement_scope_id,
        scope_contract_version=scope_contract_version,
        foundation_inventory_id=foundation_inventory_id,
    )


def normalize_user_plan_feedback_result(
    value: Any,
    *,
    user_input: str,
) -> Dict[str, Any]:
    """Preserve an explicit user change when the PI emits null fields."""
    data = dict(value) if isinstance(value, dict) else {}
    intent = coerce_json_text(data.get("intent")).casefold()
    action = coerce_json_text(data.get("action")).casefold()
    if action not in _USER_PLAN_ACTIONS:
        action = {
            "confirm": "proceed_to_execution",
            "unclear": "clarify_with_user",
        }.get(intent, "revise_with_bioinfo")
    if intent not in _USER_PLAN_INTENTS:
        intent = {
            "proceed_to_execution": "confirm",
            "clarify_with_user": "unclear",
        }.get(action, "modify")
    guidance = coerce_json_text(data.get("guidance_for_bioinfo"))
    clarification = coerce_json_text(data.get("clarification_question"))
    if action == "revise_with_bioinfo" and not guidance:
        guidance = coerce_json_text(user_input)
    if action == "clarify_with_user" and not clarification:
        clarification = (
            "Could you clarify whether you approve the plan or want specific changes?"
        )
    return {
        **data,
        "intent": intent,
        "action": action,
        "guidance_for_bioinfo": guidance,
        "clarification_question": clarification,
    }


def normalize_plan_deviation_result(value: Any) -> Dict[str, Any]:
    """Normalize deviation verdicts without trusting truthy string flags."""
    data = dict(value) if isinstance(value, dict) else {}
    severity = coerce_json_text(data.get("severity")).casefold()
    if severity not in {"minor", "major"}:
        severity = "minor"
    return {
        **data,
        "severity": severity,
        "reasoning": coerce_json_text(data.get("reasoning"))
        or "Unable to classify (fallback)",
        "suggested_action": coerce_json_text(data.get("suggested_action"))
        or "Proceed with best judgment",
        "needs_user_popup": data.get("needs_user_popup") is True,
    }


def normalize_supplementary_plan(value: Any, *, user_request: str) -> Dict[str, Any]:
    """Normalize supplementary task collections and structured summary text."""
    data = dict(value) if isinstance(value, dict) else {}
    return {
        **data,
        "bioinformatics_tasks": [
            text for text in (
                coerce_json_text(item)
                for item in as_list(data.get("bioinformatics_tasks"))
            ) if text
        ],
        "literature_topics": [
            text for text in (
                coerce_json_text(item)
                for item in as_list(data.get("literature_topics"))
            ) if text
        ],
        "summary": coerce_json_text(data.get("summary")) or user_request,
        **(
            {
                "adata_source": normalize_adata_source_selection(
                    data.get("adata_source")
                )
            }
            if "adata_source" in data
            else {}
        ),
    }


def normalize_data_needs(value: Any) -> Dict[str, list[str]]:
    """Normalize optional data-request collections."""
    data = dict(value) if isinstance(value, dict) else {}
    return {
        field: [
            text for text in (
                coerce_json_text(item) for item in as_list(data.get(field))
            ) if text
        ]
        for field in ("pathology", "biochemistry", "other")
    }
