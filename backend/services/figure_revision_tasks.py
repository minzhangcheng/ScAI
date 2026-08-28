"""Durable repair-task lifecycle for figures that did not clear review."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from ..models.schemas import FigureRevisionTask


MAX_FIGURE_REVISION_ATTEMPTS = 2


def find_figure_revision_task(
    state, *, plan_id: str | None, plan_requirement_key: str | None
) -> FigureRevisionTask | None:
    """Return the latest durable task for one exact plan requirement."""
    return next((
        item for item in reversed(state.figure_revision_tasks)
        if item.status in {
            "pending", "in_progress", "blocked", "exhausted_excluded",
        }
        and (item.plan_id or None) == (plan_id or None)
        and (item.plan_requirement_key or None) == (plan_requirement_key or None)
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
        if task.plan_requirement_key != requirement_key:
            continue
        task.status = "resolved"
        task.current_figure_id = figure.id
        task.updated_at = datetime.now()
        resolved += 1
    return resolved
