"""Layout and PI decisions for publication figures."""
from __future__ import annotations

from .figure_persistence import finalize_render, guard_attempt_await
from .figure_preflight import FigureAttemptState
from .figure_revision import perform_revision, restore_snapshot, take_snapshot
from .shared import FigureReviewStatus
from ...services.figure_revision_tasks import (
    MAX_FIGURE_REVISION_ATTEMPTS,
    resolve_figure_revision_tasks,
    upsert_figure_revision_task,
)


def record_unresolved(attempt: FigureAttemptState, items) -> None:
    deps = attempt.deps
    figure = attempt.figure
    figure.unresolved_must_fix = deps.get("normalize_text_list")(items, limit=20)
    remaining = deps.get("blocking_violations")(figure.layout_violations)
    if remaining:
        note = (
            f"[Layout gate] {deps.get('summarize_violations')(remaining)} — text is "
            "covered and cannot be read, so this figure is held back from reports "
            "until it is redrawn."
        )
        existing = figure.pi_feedback or ""
        if note not in existing:
            figure.pi_feedback = f"{existing}\n\n{note}" if existing else note


async def _accept_unavailable_review(
    attempt: FigureAttemptState, review: dict
) -> str:
    """Finalize a valid render while recording that no review occurred."""
    figure = attempt.figure
    feedback = str(
        review.get("feedback") or "Figure review was unavailable"
    ).strip()[:500]
    observation = {
        "round": attempt.revision_count + 1,
        "candidate_id": f"{figure.id}:review:{attempt.revision_count + 1}",
        "feedback": feedback,
        "suggestions": [],
        "must_fix": [],
        "quality_score": None,
        "review_unavailable": True,
    }
    if not any(
        entry.get("review_unavailable") is True
        for entry in attempt.revision_history
    ):
        attempt.revision_history.append(observation)
    figure.review_status = FigureReviewStatus.PENDING
    figure.review_unavailable = True
    figure.pi_feedback = f"[REVIEW UNAVAILABLE] {feedback}"
    figure.revision_history = list(attempt.revision_history)
    # Reviewer opinions did not exist, so there are no reviewer must-fix items.
    # Renderer measurements remain on layout_violations and independently
    # control report eligibility.
    figure.unresolved_must_fix = []
    attempt.agent._upsert_publication_figure(attempt.state, figure)
    publish_error = await finalize_render(attempt)
    if publish_error:
        return publish_error
    await attempt.agent._emit_figure_record(
        "figure_review", figure, review=review
    )
    await attempt.agent._notify(
        "status",
        f"Figure {attempt.request.figure_name} was saved; visual review is unavailable",
    )
    return (
        f"Figure {attempt.request.figure_name} was saved. "
        f"Visual review was unavailable: {feedback}"
        + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
    )


async def _accept_approved(
    attempt: FigureAttemptState, review: dict, quality_score, blocking
) -> str:
    figure = attempt.figure
    deps = attempt.deps
    logger = deps.get("logger")
    if blocking or (attempt.revision_count >= 2 and quality_score < 8):
        figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
        logger.info(
            "[Bioinformatics] Figure %s accepted with reservations "
            "(score=%s, revisions=%s)",
            attempt.request.figure_name,
            quality_score,
            attempt.revision_count,
        )
    else:
        figure.review_status = FigureReviewStatus.APPROVED
        logger.info(
            "[Bioinformatics] Figure %s approved by PI",
            attempt.request.figure_name,
        )
    figure.revision_history = attempt.revision_history
    figure.pi_feedback = review.get("feedback", "") or None
    record_unresolved(attempt, deps.get("_violation_must_fix_items")(blocking))
    if blocking:
        upsert_figure_revision_task(
            attempt.state,
            figure,
            reason="visual_revision",
            must_fix=figure.unresolved_must_fix,
            suggestions=review.get("suggestions", []),
        )
    else:
        resolve_figure_revision_tasks(attempt.state, figure)
    publish_error = await finalize_render(attempt)
    if publish_error:
        return publish_error
    await attempt.agent._emit_figure_record("figure_approved", figure, review=review)
    await attempt.agent._notify(
        "status", f"Figure {attempt.request.figure_name} approved by PI"
    )
    layout_note = (
        f"\nUnresolved layout defects: {deps.get('summarize_violations')(blocking)}. "
        "This figure is excluded from reports until they are fixed."
        if blocking
        else ""
    )
    return (
        f"Figure {attempt.request.figure_name} has been approved by PI and saved.\n"
        f"Quality score: {quality_score}/10\n"
        f"Feedback: {review.get('feedback', 'Approved')}{layout_note}"
        + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
    )


async def _accept_reanalysis(
    attempt: FigureAttemptState,
    review: dict,
    feedback: str,
    must_fix: list[str],
    blocking,
) -> str:
    deps = attempt.deps
    figure = attempt.figure
    deps.get("logger").warning(
        "[Bioinformatics] Figure %s requires upstream reanalysis, skipping revision loop",
        attempt.request.figure_name,
    )
    await guard_attempt_await(
        attempt,
        attempt.agent._notify(
            "status",
            f"Figure {attempt.request.figure_name}: PI identified data-level issue, "
            "deferring to analysis",
        ),
    )
    figure.review_status = FigureReviewStatus.REVISION_REQUESTED
    figure.pi_feedback = f"[REQUIRES REANALYSIS] {feedback}"
    figure.revision_history = attempt.revision_history
    record_unresolved(
        attempt, must_fix + deps.get("_violation_must_fix_items")(blocking)
    )
    upsert_figure_revision_task(
        attempt.state,
        figure,
        reason="reanalysis_required",
        must_fix=figure.unresolved_must_fix,
        suggestions=review.get("suggestions", []),
    )
    publish_error = await finalize_render(attempt)
    if publish_error:
        return publish_error
    await attempt.agent._emit_figure_record("figure_revision", figure, review=review)
    layout_note = ""
    if blocking:
        layout_note = (
            "\n\nSeparately from the data issue, the rendering engine measured "
            "these layout defects. The regenerated figure must avoid them too:\n"
            + deps.get("format_violations_for_prompt")(blocking)
        )
    return (
        f"Figure {attempt.request.figure_name} has a data-level issue that cannot "
        f"be fixed by revising the plot code. PI feedback: {feedback}\n"
        "Please address this in the analysis pipeline and regenerate the figure later."
        f"{layout_note}"
        + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
    )


async def _finish_exhausted_review(attempt: FigureAttemptState) -> str:
    deps = attempt.deps
    figure = attempt.figure
    restored_best = False
    best_score = (attempt.best_snapshot or {}).get("score")
    current_score = attempt.last_review.get("quality_score")
    if not isinstance(current_score, (int, float)):
        current_score = 5
    selected_review = attempt.last_review
    selected_must_fix = attempt.last_must_fix
    if (
        isinstance(best_score, (int, float))
        and isinstance(current_score, (int, float))
        and best_score > current_score
    ):
        restored_best = restore_snapshot(attempt, attempt.best_snapshot)
        if restored_best:
            selected_review = dict(
                attempt.best_snapshot.get("review") or attempt.last_review
            )
            selected_must_fix = list(
                attempt.best_snapshot.get("must_fix") or []
            )
    remaining_blocking = deps.get("blocking_violations")(figure.layout_violations)
    record_unresolved(
        attempt,
        deps.get("normalize_text_list")(selected_must_fix, limit=20)
        + deps.get("_violation_must_fix_items")(remaining_blocking),
    )
    task = upsert_figure_revision_task(
        attempt.state,
        figure,
        reason=attempt.terminal_revision_reason,
        must_fix=figure.unresolved_must_fix,
        suggestions=selected_review.get("suggestions", []),
    )
    if attempt.terminal_revision_reason == "attempts_exhausted":
        selected_score = (
            best_score if restored_best else current_score
        )
        unresolved_summary = "; ".join(figure.unresolved_must_fix[:5])[:1000]
        reason = (
            "The bounded revision budget was exhausted. The candidate with the "
            f"highest available review score ({selected_score}/10) was retained as "
            "a candidate for the PI's explicit round-end disposition; all unresolved "
            "PI and renderer observations remain disclosed."
            + (
                f" Unresolved observations: {unresolved_summary}"
                if unresolved_summary
                else ""
            )
        )
        figure.review_status = FigureReviewStatus.REVISION_REQUESTED
        figure.pi_degraded_acceptance = False
        figure.pi_degraded_acceptance_reason = None
        selected_feedback = str(selected_review.get("feedback") or "").strip()
        figure.pi_feedback = (
            f"[AWAITING PI DISPOSITION] {reason}"
            + (
                f"\n\nReview feedback for the retained candidate: {selected_feedback}"
                if selected_feedback
                else ""
            )
        )
    else:
        figure.review_status = FigureReviewStatus.REVISION_REQUESTED
    publish_error = await finalize_render(attempt)
    if publish_error:
        return publish_error
    event_type = (
        "figure_approved"
        if figure.pi_degraded_acceptance
        else "figure_revision"
    )
    await attempt.agent._emit_figure_record(
        event_type, figure, review=selected_review
    )
    layout_note = (
        f"\nUnresolved layout defects: {deps.get('summarize_violations')(remaining_blocking)}."
        if remaining_blocking
        else ""
    )
    if figure.pi_degraded_acceptance:
        deps.get("logger").warning(
            "[Bioinformatics] Figure %s accepted with reservations after the "
            "bounded revision budget was exhausted (selected PI score=%s)",
            attempt.request.figure_name,
            selected_score,
        )
        await attempt.agent._notify(
            "status",
            f"Figure {attempt.request.figure_name} accepted with reservations "
            "after PI candidate review",
        )
        return (
            f"Figure {attempt.request.figure_name} saved after "
            f"{attempt.revision_count} revision attempts. Status: accepted with "
            "reservations; the bounded revision budget is exhausted.\n"
            f"{figure.pi_feedback}{layout_note}"
            + (
                "\nThe highest-scoring prior PI-reviewed render was restored."
                if restored_best else ""
            )
            + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
        )
    deps.get("logger").warning(
        "[Bioinformatics] Figure %s not fully approved after %s revisions",
        attempt.request.figure_name,
        attempt.revision_count,
    )
    return (
        f"Figure {attempt.request.figure_name} saved after {attempt.revision_count} "
        "revision attempts. Status: excluded from reports; the durable task "
        "awaits an explicit PI round-end disposition.\n"
        f"Last feedback: {figure.pi_feedback}{layout_note}"
        + (
            "\nThe highest-scoring prior render was restored before exclusion."
            if restored_best else ""
        )
        + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
    )


async def run_review_loop(attempt: FigureAttemptState) -> str:
    """Run PI review until approval, terminal deferral, or bounded exhaustion."""
    deps = attempt.deps
    figure = attempt.figure
    while attempt.revision_count <= MAX_FIGURE_REVISION_ATTEMPTS:
        figure.review_status = FigureReviewStatus.PENDING
        attempt.agent._upsert_publication_figure(attempt.state, figure)
        deps.get("logger").info(
            "[Bioinformatics] Submitting figure for PI review (attempt %s)",
            attempt.revision_count + 1,
        )
        await guard_attempt_await(
            attempt,
            attempt.agent._notify(
                "status", f"PI reviewing figure: {attempt.request.figure_name}..."
            ),
        )
        review = await guard_attempt_await(
            attempt, attempt.request.review_callback(figure)
        )
        if review.get("review_unavailable") is True:
            return await _accept_unavailable_review(attempt, review)
        quality_score = review.get("quality_score", 5)
        blocking = deps.get("blocking_violations")(figure.layout_violations)
        must_fix = deps.get("normalize_text_list")(
            review.get("must_fix"), limit=20
        )
        approved = bool(review.get("approved", False)) and not must_fix
        if approved and blocking:
            deps.get("logger").warning(
                "[Bioinformatics] Figure %s: reviewer approved (score=%s) but "
                "layout lint reports %s — continuing revision",
                attempt.request.figure_name,
                quality_score,
                deps.get("summarize_violations")(blocking),
            )
            approved = False
            review = dict(review)
            review["feedback"] = (
                (review.get("feedback") or "")
                + "\n\nThe reviewer approved this figure, but the rendering engine "
                "measured text drawn on top of other text. Those collisions must "
                "be resolved before the figure can be used."
            )
            review["must_fix"] = deps.get("normalize_text_list")(
                review.get("must_fix"), limit=20
            ) + deps.get("_violation_must_fix_items")(blocking)
            must_fix = review["must_fix"]
        if approved:
            return await _accept_approved(
                attempt, review, quality_score, blocking
            )

        feedback = review.get("feedback", "")
        suggestions = review.get("suggestions", [])
        review = dict(review)
        review["must_fix"] = must_fix
        attempt.last_review = review
        attempt.last_must_fix = must_fix
        if review.get("requires_reanalysis", False):
            return await _accept_reanalysis(
                attempt, review, feedback, must_fix, blocking
            )
        # Every plot-level candidate has now been reviewed by the PI. Preserve
        # the highest PI-scored render even when it still carries observations;
        # those observations remain attached if the bounded budget is exhausted.
        if (
            attempt.best_snapshot is None
            or quality_score > attempt.best_snapshot.get("score", -1)
        ):
            snapshot = take_snapshot(
                attempt,
                quality_score,
                review=review,
                must_fix=must_fix,
            )
            if snapshot is not None:
                attempt.best_snapshot = snapshot

        attempt.revision_history.append(
            {
                "round": attempt.revision_count + 1,
                "feedback": feedback,
                "suggestions": suggestions,
                "must_fix": must_fix,
                "quality_score": review.get("quality_score", None),
            }
        )
        figure.pi_feedback = feedback
        figure.revision_history = attempt.revision_history
        figure.review_status = FigureReviewStatus.REVISION_REQUESTED
        attempt.agent._upsert_publication_figure(attempt.state, figure)
        await guard_attempt_await(
            attempt,
            attempt.agent._emit_figure_record(
                "figure_revision", figure, review=review
            ),
        )
        if attempt.revision_count >= MAX_FIGURE_REVISION_ATTEMPTS:
            break
        action = await perform_revision(
            attempt,
            feedback=feedback,
            suggestions=suggestions,
            must_fix=must_fix,
            quality_score=quality_score,
        )
        if action == "break":
            break
    return await _finish_exhausted_review(attempt)
