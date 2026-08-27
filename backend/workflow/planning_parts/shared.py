"""Planning dialogue: PI ↔ Bioinformatics plan design, review, and user gate."""
import asyncio
import difflib
import json
import logging
import re
from datetime import datetime
from html import escape as _html_escape
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...constants import LlmFailureAction, PI_DECIDE_SIGNAL, WaitInputReason, StateSyncChangeType
from ...models.schemas import AgentType, AnalysisPhase, AnalysisPlan, AnalysisState, PlanStep, TaskStatus
from ...services.interaction_record_capture import (
    find_open_integration_record,
    find_open_pi_decide_record,
    get_or_open_direction_record,
)
from ...services.figure_contract import record_plan_figure_inheritance
from ...utils.llm_client import UpstreamExhaustedError
from ...utils.formatting import truncate_for_llm
from ..direction_coverage import (
    current_lineage_id,
    normalize_direction_key,
    uncovered_direction_requirements,
)
from ..helpers import (
    _diff_plan_steps, _get_task_text, _normalize_tasks, _DIFF_LABELS,
    ack_restore_safe_point, wait_with_skip_time,
)
from ..provenance import restore_provenance_id
from ..retry import MAX_NODE_RETRIES, _retry_backoff_delay, with_retry

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict  # runtime fallback for type annotations

logger = logging.getLogger(__name__)

MAX_PLANNING_ROUNDS = 5
MAX_SPECIES_INFERENCE_ATTEMPTS = 3
PLAN_STAGNATION_TOKEN_SIMILARITY = 0.97


def _plan_semantic_text(plan: Optional[AnalysisPlan]) -> str:
    """Canonical text used only to detect a revision with no material change."""
    if plan is None:
        return ""
    payload = {
        "overview": plan.overview,
        "steps": [
            {
                "title": step.title,
                "method": step.method,
                "input": step.input,
                "expected_output": step.expected_output,
                "figure_requirements": [
                    requirement.model_dump(mode="json")
                    for requirement in step.figure_requirements
                ],
                "allow_text_only": step.allow_text_only,
                "text_only_reason": step.text_only_reason,
                "rationale": step.rationale,
                "depends_on": sorted(step.depends_on),
            }
            for step in plan.steps
        ],
        "key_decision_points": plan.key_decision_points,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True).casefold()
    return re.sub(r"\s+", " ", text).strip()


def plans_materially_equivalent(
    previous: Optional[AnalysisPlan], current: Optional[AnalysisPlan]
) -> bool:
    """Conservatively identify a revision that only rephrases the same plan."""
    if previous is None or current is None:
        return False
    if len(previous.steps) != len(current.steps):
        return False
    before = _plan_semantic_text(previous)
    after = _plan_semantic_text(current)
    if not before or not after:
        return False
    if before == after:
        return True
    before_tokens = set(re.findall(r"[a-z0-9_]+", before))
    after_tokens = set(re.findall(r"[a-z0-9_]+", after))
    union = before_tokens | after_tokens
    token_similarity = len(before_tokens & after_tokens) / len(union) if union else 1.0
    sequence_similarity = difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).ratio()
    return (
        token_similarity >= PLAN_STAGNATION_TOKEN_SIMILARITY
        and sequence_similarity >= 0.94
    )


def _consume_user_plan_feedback(state: dict) -> str:
    value = state.get("user_plan_feedback") or ""
    state["user_plan_feedback"] = None
    return value


def resolve_round_phase(current_round: int, current_phase: str,
                        has_exploration_report: bool) -> str:
    """Decide the analysis phase when entering round 2+.

    Only advance EXPLORATION -> INNOVATION once exploration actually produced a
    report (i.e. stage1 succeeded). If exploration failed last round there is no
    report; stay in EXPLORATION so the user can retry stage 1 next round —
    honoring the guard documented in synthesis._exploration_round_succeeded."""
    if current_phase == "innovation":
        return "innovation"
    return "innovation" if has_exploration_report else "exploration"


def _selected_directions_from_decision(selection_decision):
    if isinstance(selection_decision, dict):
        selected = selection_decision.get("selected_directions", [])
        return selected if isinstance(selected, list) else []
    return selection_decision if isinstance(selection_decision, list) else []


def populate_plan_scope(plan: AnalysisPlan, state: AnalysisState) -> AnalysisPlan:
    direction = state.current_round_direction or ""
    plan.round_number = state.current_round
    plan.direction = direction
    plan.direction_key = normalize_direction_key(direction)
    lineage_id = current_lineage_id(state)
    plan.branch_id = None if lineage_id == "main" else lineage_id
    plan.created_from_snapshot_id = restore_provenance_id(state)
    return plan


def resolve_plan_review_wait(state: AnalysisState) -> int:
    if state.auto_approve_plans:
        return max(0, state.plan_review_wait_seconds)
    # Inherit round_wait_seconds semantics so plan review follows the same
    # auto-skip / countdown / manual behaviour as all other HITL gates.
    return state.round_wait_seconds


def approved_plan_matches_current_context(state: AnalysisState) -> bool:
    plan = state.current_plan
    if not plan or plan.status != "approved":
        return False
    direction = state.current_round_direction or ""
    lineage_id = current_lineage_id(state)
    branch_id = None if lineage_id == "main" else lineage_id
    return (
        plan.round_number == state.current_round
        and plan.direction_key == normalize_direction_key(direction)
        and plan.branch_id == branch_id
    )
