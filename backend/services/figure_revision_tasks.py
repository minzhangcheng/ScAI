"""Durable repair-task lifecycle for figures that did not clear review."""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import Iterable

from ..models.schemas import FigureRevisionTask


MAX_FIGURE_REVISION_ATTEMPTS = 2


def materialize_missing_figure_tasks(state, contract) -> list[FigureRevisionTask]:
    """Create durable PI-owned work items for missing plan deliverables."""
    created = []
    for missing in getattr(contract, "missing", None) or []:
        requirement = missing.requirement
        plan_id = str(getattr(requirement, "plan_id", "") or "") or None
        requirement_key = str(getattr(requirement, "key", "") or "") or None
        existing = [
            task for task in reversed(state.figure_revision_tasks)
            if task.status != "resolved"
            and (task.plan_id or None) == plan_id
            and (task.plan_requirement_key or None) == requirement_key
            and (task.lineage_id or "main") == (state.restore_branch_id or "main")
            and (not task.round_number or task.round_number == state.current_round)
        ]
        required = max(1, int(requirement.minimum_count))
        found = int(missing.found_count)
        # Each task owns one distinct missing output. PI omissions reserve their
        # slot as well: a fresh audit must not undo a deliberate omission.
        for _ in range(max(0, required - found - len(existing))):
            task = FigureRevisionTask(
                logical_figure_id=f"requirement:{plan_id}:{requirement_key}:{uuid.uuid4()}",
                plan_id=plan_id,
                plan_requirement_key=requirement_key,
                plan_step_id=requirement.plan_step_id,
                lineage_id=state.restore_branch_id or "main",
                round_number=state.current_round,
                required_count=required,
                found_count=found,
                reason="missing_figure",
                must_fix=[
                    f"Produce one distinct {requirement.figure_type} deliverable: "
                    f"{requirement.description} ({found}/{required} available)."
                ],
            )
            state.figure_revision_tasks.append(task)
            created.append(task)
    return created


def candidate_figures_for_task(state, task) -> list:
    """Return render-complete candidates belonging to one durable task."""
    from .report.artifacts import archival_render_available, has_frozen_publication_pair

    lineage = getattr(state, "restore_branch_id", None) or "main"
    candidates = []
    for figure in state.publication_figures:
        if (getattr(figure, "lineage_id", None) or "main") != lineage:
            continue
        if (figure.plan_id or None) != (task.plan_id or None):
            continue
        if task.round_number and figure.round_number != task.round_number:
            continue
        if str(getattr(figure, "producer_status", "published")) != "published":
            continue
        if not getattr(figure, "is_current", True) or getattr(figure, "invalidated_at", None):
            continue
        same_logical = str(getattr(figure, "logical_figure_id", "") or "") == task.logical_figure_id
        same_current = bool(task.current_figure_id) and figure.id == task.current_figure_id
        if not (same_logical or same_current):
            continue
        if not has_frozen_publication_pair(figure) or not archival_render_available(figure):
            continue
        candidates.append(figure)
    return candidates


def apply_pi_fallback_selection(
    state,
    task,
    *,
    selected_figure_id: str,
    rationale: str,
    report_note: str,
):
    """Record an explicit PI selection without erasing outstanding observations."""
    from ..models.schema_parts.enums import FigureReviewStatus

    candidates = candidate_figures_for_task(state, task)
    selected = next((item for item in candidates if item.id == selected_figure_id), None)
    if selected is None:
        return None
    note = str(report_note or rationale or "PI selected the best available candidate.").strip()
    reason = (
        "PI selected this candidate for report use after ordinary figure review "
        f"did not clear the deliverable. {note}"
    )[:4000]
    selected.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
    selected.pi_degraded_acceptance = True
    selected.pi_degraded_acceptance_reason = reason
    selected.pi_feedback = (
        f"[PI FALLBACK SELECTION] {reason}\n\n"
        f"Selection rationale: {str(rationale or '').strip()}"
    )[:12000]
    task.status = "resolved"
    task.pi_disposition = "accept_best"
    task.pi_rationale = str(rationale or "")[:4000] or None
    task.pi_report_note = note[:4000]
    task.selected_figure_id = selected.id
    task.current_figure_id = selected.id
    task.updated_at = datetime.now()
    return selected


def find_figure_revision_task(
    state, *, plan_id: str | None, plan_requirement_key: str | None,
    logical_figure_id: str | None = None,
) -> FigureRevisionTask | None:
    """Return the latest durable task for one exact plan requirement."""
    return next((
        item for item in reversed(state.figure_revision_tasks)
        if item.status in {
            "pending", "in_progress", "blocked", "exhausted_excluded",
        }
        and (item.plan_id or None) == (plan_id or None)
        and (item.plan_requirement_key or None) == (plan_requirement_key or None)
        and (not logical_figure_id or item.logical_figure_id == logical_figure_id)
        and (item.lineage_id or "main") == (state.restore_branch_id or "main")
    ), None)


def upsert_figure_revision_task(
    state,
    figure,
    *,
    reason: str,
    must_fix: Iterable[str] = (),
    suggestions: Iterable[str] = (),
) -> FigureRevisionTask:
    logical_id = str(getattr(figure, "logical_figure_id", "") or figure.id)
    requirement_key = str(
        getattr(figure, "plan_requirement_key", "") or ""
    ) or None
    plan_id = str(getattr(figure, "plan_id", "") or "") or None
    task = next((
        item for item in reversed(state.figure_revision_tasks)
        if item.status in {
            "pending", "in_progress", "blocked", "exhausted_excluded"
        }
        and (item.plan_id or None) == plan_id
        and item.logical_figure_id == logical_id
        and item.plan_requirement_key == requirement_key
        and (item.lineage_id or "main") == (getattr(figure, "lineage_id", None) or "main")
    ), None)
    now = datetime.now()
    clean_must_fix = [str(item)[:1000] for item in must_fix if str(item).strip()][:20]
    clean_suggestions = [
        str(item)[:1000] for item in suggestions if str(item).strip()
    ][:20]
    if task is None:
        task = FigureRevisionTask(
            logical_figure_id=logical_id,
            plan_id=plan_id,
            plan_requirement_key=requirement_key,
            current_figure_id=figure.id,
            source_execution_id=getattr(figure, "source_execution_id", None),
            plan_step_id=getattr(figure, "plan_step_id", None),
            lineage_id=getattr(figure, "lineage_id", None) or "main",
            round_number=int(getattr(figure, "round_number", 0) or 0),
            reason=reason,
            must_fix=clean_must_fix,
            suggestions=clean_suggestions,
            attempt_count=int(getattr(figure, "revision_count", 0) or 0),
            max_attempts=MAX_FIGURE_REVISION_ATTEMPTS,
        )
        state.figure_revision_tasks.append(task)
    else:
        task.current_figure_id = figure.id
        task.source_execution_id = getattr(figure, "source_execution_id", None)
        task.reason = reason
        task.must_fix = clean_must_fix
        task.suggestions = clean_suggestions
        task.attempt_count = max(
            task.attempt_count,
            int(getattr(figure, "revision_count", 0) or 0),
        )
        task.updated_at = now
    if reason == "attempts_exhausted" or task.attempt_count >= task.max_attempts:
        task.reason = "attempts_exhausted"
        task.status = "exhausted_excluded"
    elif task.status != "exhausted_excluded":
        task.status = "pending"
    return task


def resolve_figure_revision_tasks(state, figure) -> int:
    logical_id = str(getattr(figure, "logical_figure_id", "") or figure.id)
    requirement_key = str(
        getattr(figure, "plan_requirement_key", "") or ""
    ) or None
    plan_id = str(getattr(figure, "plan_id", "") or "") or None
    resolved = 0
    for task in state.figure_revision_tasks:
        if task.status not in {"pending", "in_progress"}:
            continue
        if task.logical_figure_id != logical_id:
            continue
        if (task.plan_id or None) != plan_id:
            continue
        if (task.lineage_id or "main") != (getattr(figure, "lineage_id", None) or "main"):
            continue
        if task.plan_requirement_key != requirement_key:
            continue
        task.status = "resolved"
        task.current_figure_id = figure.id
        task.updated_at = datetime.now()
        resolved += 1
    return resolved
