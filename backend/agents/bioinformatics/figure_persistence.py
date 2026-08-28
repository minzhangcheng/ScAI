"""Publication-figure record persistence and transaction completion."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable

from .figure_preflight import FigureAttemptState


def refresh_figure_integrity_hashes(figure, *, bindings) -> None:
    """Bind a record to its full render and compact persisted preview."""
    figure.render_sha256 = bindings.image_file_sha256(figure.png_path)
    figure.preview_sha256 = bindings.preview_render_sha256(figure.png_base64)
    figure.pdf_sha256 = bindings.pdf_file_sha256(figure.pdf_path)


def clear_unpublished_figure_payload(figure) -> None:
    """Keep audit metadata while removing staged bytes and direct paths."""
    figure.png_base64 = None
    figure.png_path = ""
    figure.pdf_path = ""
    figure.render_sha256 = None
    figure.preview_sha256 = None
    figure.pdf_sha256 = None
    figure.is_published_head = False


async def persist_figure_record_boundary(agent, state, figures_dir) -> bool:
    """Durably persist the record half for canonical session workspaces."""
    try:
        from ...config import get_session_workspace

        canonical = (get_session_workspace(state.id) / "figures").resolve()
        actual = figures_dir.resolve()
    except Exception:
        return False
    if actual != canonical:
        return False
    from ...services.session_store import async_save_session

    await async_save_session(state.id, state, force=True)
    return True


async def emit_figure_record(
    agent, event_type: str, figure, *, review=None, bindings
) -> None:
    """Publish the producer's authoritative post-gate figure state."""
    review = dict(review or {})
    review["must_fix"] = bindings.normalize_text_list(
        review.get("must_fix"), limit=20
    )
    status = getattr(figure.review_status, "value", figure.review_status)
    await agent._notify(
        event_type,
        {
            "figure_id": figure.id,
            "figure_name": figure.name,
            "approved": status
            in {
                bindings.FigureReviewStatus.APPROVED.value,
                bindings.FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS.value,
            },
            "quality_score": review.get("quality_score"),
            "feedback": figure.pi_feedback or review.get("feedback", ""),
            "suggestions": review.get("suggestions", []),
            "must_fix": review.get("must_fix", []),
            "figure": {
                "id": figure.id,
                "name": figure.name,
                "description": figure.description,
                "figure_type": figure.figure_type,
                "png_path": figure.png_path,
                "pdf_path": figure.pdf_path,
                "review_status": status,
                "pi_feedback": figure.pi_feedback,
                "revision_count": figure.revision_count,
                "plan_step_id": getattr(figure, "plan_step_id", None),
                "plan_id": getattr(figure, "plan_id", None),
                "plan_requirement_key": getattr(
                    figure, "plan_requirement_key", None
                ),
                "submitted_plan_step_id": getattr(
                    figure, "submitted_plan_step_id", None
                ),
                "supervisor_binding_decision": getattr(
                    figure, "supervisor_binding_decision", None
                ),
                "supervisor_binding_reason": getattr(
                    figure, "supervisor_binding_reason", None
                ),
                "source_execution_id": getattr(
                    figure, "source_execution_id", None
                ),
                "purpose": getattr(figure, "purpose", "result") or "result",
                "logical_figure_id": getattr(
                    figure, "logical_figure_id", None
                ),
                "render_attempt_id": getattr(figure, "render_attempt_id", None),
                "producer_status": getattr(
                    figure, "producer_status", "published"
                ),
                "revision_of_id": getattr(figure, "revision_of_id", None),
                "supersedes_id": getattr(figure, "supersedes_id", None),
                "is_published_head": bool(
                    getattr(figure, "is_published_head", False)
                ),
                "round": getattr(figure, "round_number", 0) or None,
                "diagnostic": bool(getattr(figure, "diagnostic", False)),
                "layout_blocking": bool(
                    bindings.blocking_violations(figure.layout_violations)
                ),
                "created_at": figure.created_at.isoformat(),
            },
        },
    )


def upsert_publication_figure(
    agent, state, figure, *, refresh_integrity: bool = True, bindings
):
    """Update one attempt and advance its published head only when eligible."""
    if refresh_integrity:
        agent._refresh_figure_integrity_hashes(figure)
    existing_idx = next(
        (
            index
            for index, item in enumerate(state.publication_figures)
            if getattr(item, "id", None) == getattr(figure, "id", None)
        ),
        None,
    )
    if existing_idx is not None:
        state.publication_figures[existing_idx] = figure
    else:
        state.publication_figures.append(figure)
    if not bindings.is_report_eligible(figure):
        return
    refreshed_stems = {
        Path(str(figure.name)).stem,
        Path(str(figure.png_path)).stem,
    }
    state.invalidated_artifact_stems = [
        stem
        for stem in (state.invalidated_artifact_stems or [])
        if stem not in refreshed_stems
    ]
    family_key = bindings.logical_figure_key(figure)
    prior_heads = [
        item
        for item in state.publication_figures
        if item is not figure
        and bindings.logical_figure_key(item) == family_key
        and bindings.is_report_eligible(item)
    ]
    prior_head = next(
        (
            item
            for item in reversed(prior_heads)
            if getattr(item, "is_published_head", False)
        ),
        prior_heads[-1] if prior_heads else None,
    )
    if prior_head is not None and not figure.supersedes_id:
        figure.supersedes_id = prior_head.id
    for item in state.publication_figures:
        if item is not figure and bindings.logical_figure_key(item) == family_key:
            item.is_published_head = False
    figure.is_published_head = True


async def journal_pending_figure(attempt: FigureAttemptState) -> str | None:
    """Journal and persist the pending record before any external review."""
    deps = attempt.deps
    figure = attempt.figure
    try:
        attempt.pending_transaction = deps.get("register_pending_figure_attempt")(
            attempt_dir=attempt.attempt_dir,
            staged_jpg=attempt.staged_jpg_path,
            staged_pdf=attempt.staged_pdf_path,
            final_jpg=attempt.final_jpg_path,
            final_pdf=attempt.final_pdf_path,
            published_record=figure,
            session_id=attempt.state.id,
        )
    except Exception as exc:
        deps.get("_discard_render_attempt")(attempt.attempt_dir)
        return f"Could not journal publication figure attempt: {exc}"

    attempt.agent._upsert_publication_figure(
        attempt.state, figure, refresh_integrity=False
    )
    try:
        await attempt.agent._persist_figure_record_boundary(
            attempt.state, attempt.figures_dir
        )
    except Exception as exc:
        deps.get("rollback_figure_publish_transaction")(
            Path(attempt.pending_transaction["journal_path"]),
            state=attempt.state,
            producer_status="failed",
        )
        return f"Could not persist pending publication figure record: {exc}"
    return None


async def persist_terminal_rollback(attempt: FigureAttemptState) -> None:
    try:
        persisted = await attempt.agent._persist_figure_record_boundary(
            attempt.state, attempt.figures_dir
        )
        if persisted:
            attempt.deps.get("mark_figure_record_recovery_durable")(
                attempt.figures_dir,
                [str(getattr(attempt.figure, "render_attempt_id", "") or "")],
            )
    except Exception:
        attempt.deps.get("logger").exception(
            "Could not persist terminal figure rollback for %s", attempt.figure.id
        )


def abandon_render(attempt: FigureAttemptState) -> None:
    """Remove a non-terminal attempt without disturbing an old head."""
    if not attempt.render_published:
        attempt.deps.get("rollback_figure_publish_transaction")(
            Path(attempt.pending_transaction["journal_path"]),
            state=attempt.state,
            producer_status="abandoned",
        )
    else:
        attempt.agent._upsert_publication_figure(attempt.state, attempt.figure)


async def guard_attempt_await(
    attempt: FigureAttemptState, awaitable: Awaitable[Any]
) -> Any:
    """Terminalize staging when an external await aborts the producer."""
    try:
        return await awaitable
    except BaseException:
        abandon_render(attempt)
        await persist_terminal_rollback(attempt)
        raise


async def finalize_render(attempt: FigureAttemptState) -> str | None:
    """Commit the reviewed render exactly once, preserving any prior owner."""
    if attempt.render_published:
        return None
    deps = attempt.deps
    figure = attempt.figure
    lock_error = deps.get("acquire_figure_publish_locks")(
        figures_dir=attempt.figures_dir,
        final_jpg=attempt.final_jpg_path,
        published_record=figure,
    )
    if lock_error:
        deps.get("rollback_figure_publish_transaction")(
            Path(attempt.pending_transaction["journal_path"]),
            state=attempt.state,
            producer_status="failed",
        )
        await persist_terminal_rollback(attempt)
        return lock_error
    try:
        frozen_transaction = deps.get("refreeze_pending_figure_attempt")(
            attempt.pending_transaction,
            attempt_dir=attempt.attempt_dir,
            staged_jpg=attempt.jpg_path,
            staged_pdf=attempt.pdf_path,
            final_jpg=attempt.final_jpg_path,
            final_pdf=attempt.final_pdf_path,
            published_record=figure,
        )
        attempt.pending_transaction.clear()
        attempt.pending_transaction.update(frozen_transaction)
    except Exception as exc:
        freeze_error = f"Could not authenticate final reviewed figure bytes: {exc}"
        deps.get("rollback_figure_publish_transaction")(
            Path(attempt.pending_transaction["journal_path"]),
            state=attempt.state,
            producer_status="failed",
        )
        await persist_terminal_rollback(attempt)
        return freeze_error

    archival_owner = next(
        (
            item
            for item in reversed(attempt.state.publication_figures)
            if item is not figure
            and (getattr(item, "lineage_id", None) or "main")
            == attempt.active_lineage
            and str(getattr(item, "png_path", ""))
            == str(attempt.final_jpg_path)
            and deps.get("archival_render_available")(item)
        ),
        None,
    )
    figure.producer_status = "publishing"
    attempt.agent._upsert_publication_figure(attempt.state, figure)
    transaction = None
    try:
        transaction, publish_error = deps.get("begin_figure_publish_transaction")(
            attempt_dir=attempt.attempt_dir,
            staged_jpg=attempt.jpg_path,
            staged_pdf=attempt.pdf_path,
            final_jpg=attempt.final_jpg_path,
            final_pdf=attempt.final_pdf_path,
            published_record=figure,
            archived_record=archival_owner,
        )
    except BaseException:
        figure.producer_status = "abandoned"
        attempt.agent._clear_unpublished_figure_payload(figure)
        attempt.agent._upsert_publication_figure(attempt.state, figure)
        deps.get("rollback_figure_publish_transaction")(
            Path(attempt.pending_transaction["journal_path"]),
            state=attempt.state,
            producer_status="abandoned",
        )
        await persist_terminal_rollback(attempt)
        raise
    if publish_error:
        figure.producer_status = "failed"
        attempt.agent._clear_unpublished_figure_payload(figure)
        from .shared import FigureReviewStatus

        figure.review_status = FigureReviewStatus.REVISION_REQUESTED
        figure.pi_feedback = (
            f"{figure.pi_feedback}\n\n{publish_error}"
            if figure.pi_feedback
            else publish_error
        )
        attempt.agent._upsert_publication_figure(attempt.state, figure)
        await persist_terminal_rollback(attempt)
        return publish_error

    try:
        deps.get("apply_publish_record_paths")(transaction, figure, archival_owner)
        figure.producer_status = "published"
        attempt.agent._refresh_figure_integrity_hashes(figure)
        attempt.agent._upsert_publication_figure(attempt.state, figure)
        deps.get("mark_publish_record_applied")(
            transaction,
            published_record=figure,
            archived_record=archival_owner,
        )
        await attempt.agent._persist_figure_record_boundary(
            attempt.state, attempt.figures_dir
        )
        deps.get("commit_figure_publish_transaction")(transaction)
    except BaseException:
        if transaction is not None:
            deps.get("rollback_figure_publish_transaction")(
                Path(transaction["journal_path"]),
                state=attempt.state,
                producer_status="abandoned",
            )
        else:
            figure.producer_status = "abandoned"
            attempt.agent._clear_unpublished_figure_payload(figure)
            attempt.agent._upsert_publication_figure(attempt.state, figure)
        await persist_terminal_rollback(attempt)
        raise
    attempt.render_published = True
    return None
