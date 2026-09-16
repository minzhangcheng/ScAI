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
    wait_with_skip_time,
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
        "scope_contract_version": plan.scope_contract_version,
        "foundation_inventory_id": plan.foundation_inventory_id,
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
                "analysis_scope": (
                    step.analysis_scope.model_dump(mode="json")
                    if step.analysis_scope is not None
                    else None
                ),
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


def _declined_directions(selection_decision) -> list:
    """What the gate turned down, in the shape the record stores.

    Both selectors already return this; it was thrown away. A later gate that
    re-offers the direction then has no idea it was ever weighed, and the report
    cannot tell a direction nobody considered from one that was considered and
    declined.
    """
    if not isinstance(selection_decision, dict):
        return []
    declined = []
    for entry in selection_decision.get("rejected_directions") or []:
        if not isinstance(entry, dict):
            continue
        direction = str(entry.get("direction", "")).strip()
        if not direction:
            continue
        declined.append({
            "direction": direction,
            "reason_not_selected": str(entry.get("reason_not_selected", ""))[:1000],
            "follow_up_value": str(entry.get("follow_up_value", "")),
            "dependency_note": str(entry.get("dependency_note", ""))[:500],
        })
    return declined


def _direction_budget(state) -> int:
    """The session's per-gate direction cap; 0 (the default) means no cap."""
    try:
        return max(0, int(getattr(state, "max_directions_per_gate", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _is_promoted_prerequisite(direction) -> bool:
    return bool(isinstance(direction, dict) and str(direction.get("prerequisite_for") or "").strip())


def _selected_directions_within_budget(selection_decision, state=None):
    """Split a PI gate's selection into what runs now and what waits.

    Returns ``(committed, deferred)``. With no cap configured -- the default --
    everything the PI selected is committed and nothing is deferred.

    When a cap is set, two things govern what is cut:

    * The PI's own ordering. ``selected_directions`` carries no rank, so its
      order is the ranking, and the prompt asks for it to be given in priority
      order.
    * A prerequisite that ``select_best_directions`` promoted into the selection
      is never cut, even past the cap. The closure appends promoted
      prerequisites after the directions that depend on them, so a plain slice
      removes precisely the direction that has to run first and commits the ones
      that cannot run without it.

    Deferred directions are not discarded: the gate records its full candidate
    list, so the next integration gate offers them again through
    ``unpursued_prior_candidates``.

    Applies to the PI's own decisions only. An explicit user selection is an
    instruction rather than a proposal and is installed unbudgeted.
    """
    selected = _selected_directions_from_decision(selection_decision)
    limit = _direction_budget(state)
    if limit <= 0 or len(selected) <= limit:
        return list(selected), []

    proposed = [d for d in selected if not _is_promoted_prerequisite(d)]
    promoted = [d for d in selected if _is_promoted_prerequisite(d)]

    def closure(direction) -> list:
        """A direction plus every promoted prerequisite it transitively needs."""
        chain = [direction]
        chain_texts = {str(direction.get("direction", "")).strip()}
        changed = True
        while changed:
            changed = False
            for candidate in promoted:
                text = str(candidate.get("direction", "")).strip()
                if text in chain_texts:
                    continue
                if str(candidate.get("prerequisite_for", "")).strip() in chain_texts:
                    chain.append(candidate)
                    chain_texts.add(text)
                    changed = True
        return chain

    # A promoted prerequisite is a full round too, so it counts against the cap
    # rather than riding along free. A direction whose chain does not fit is
    # deferred whole: committing it without its prerequisite is exactly the
    # inversion the closure exists to prevent.
    committed: list = []
    committed_texts: set = set()
    for direction in proposed:
        text = str(direction.get("direction", "")).strip()
        if text in committed_texts:
            continue
        chain = [d for d in closure(direction)
                 if str(d.get("direction", "")).strip() not in committed_texts]
        if len(committed) + len(chain) > limit:
            continue
        committed.extend(chain)
        committed_texts.update(str(d.get("direction", "")).strip() for d in chain)

    if not committed and proposed:
        # A cap must never turn a real selection into no selection: downstream
        # that reads as the PI choosing to stop, which is a different decision.
        committed = closure(proposed[0])
        committed_texts = {str(d.get("direction", "")).strip() for d in committed}

    deferred = [
        d for d in selected
        if str((d or {}).get("direction", "")).strip() not in committed_texts
    ]
    return committed, deferred


def note_deferred_directions(rationale: str, deferred, limit: int) -> str:
    """Record in the gate's rationale which directions the cap held back.

    Without this the record shows a shortened selection and no trace that the PI
    had asked for more, which reads as a scientific judgement it never made.
    """
    names = [
        str((d or {}).get("direction", "")).strip()
        for d in (deferred or [])
        if str((d or {}).get("direction", "")).strip()
    ]
    if not names:
        return rationale
    note = (
        f"Held back by this session's per-gate cap of {limit} direction(s); "
        f"offered again at the next integration gate: " + "; ".join(names)
    )
    return f"{rationale} {note}".strip() if rationale else note


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
