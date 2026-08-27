"""Layout and PI decisions for publication figures."""
from __future__ import annotations

from .figure_persistence import finalize_render, guard_attempt_await
from .figure_preflight import FigureAttemptState
from .figure_revision import perform_revision, restore_snapshot, take_snapshot
from .shared import FigureReviewStatus


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


async def _accept_stalled(
    attempt: FigureAttemptState,
    review: dict,
    feedback: str,
    must_fix: list[str],
    previous_score,
    current_score,
) -> str:
    regressed = current_score < previous_score
    rollback_worth_it = (
        regressed
        and attempt.best_snapshot is not None
        and attempt.best_snapshot.get("score", -1) > current_score
    )
    rolled_back = (
        restore_snapshot(attempt, attempt.best_snapshot) if rollback_worth_it else False
    )
    attempt.deps.get("logger").warning(
        "[Bioinformatics] Figure %s: quality score %s (%s -> %s)%s, "
        "accepting current version",
        attempt.request.figure_name,
        "regressed" if regressed else "stalled",
        previous_score,
        current_score,
        ", rolled back to the best version" if rolled_back else "",
    )
    attempt.figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
    record_unresolved(attempt, must_fix)
    publish_error = await finalize_render(attempt)
    if publish_error:
        return publish_error
    await attempt.agent._emit_figure_record(
        "figure_approved", attempt.figure, review=review
    )
    await attempt.agent._notify(
        "status", f"Figure {attempt.request.figure_name} accepted (revision stalled)"
    )
    rollback_note = (
        "\nThe last revision scored lower than the previous one, so the "
        f"earlier version (score {attempt.best_snapshot.get('score')}) was restored."
        if rolled_back
        else ""
    )
    return (
        f"Figure {attempt.request.figure_name} accepted with reservations "
        f"(quality score did not improve after revision: {previous_score} -> "
        f"{current_score}).{rollback_note}\nLast feedback: {feedback}\n"
        "Please move on to the next analysis step."
        + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
    )


async def _finish_exhausted_review(attempt: FigureAttemptState) -> str:
    deps = attempt.deps
    figure = attempt.figure
    figure.review_status = FigureReviewStatus.REVISION_REQUESTED
    remaining_blocking = deps.get("blocking_violations")(figure.layout_violations)
    record_unresolved(
        attempt,
        deps.get("normalize_text_list")(attempt.last_must_fix, limit=20)
        + deps.get("_violation_must_fix_items")(remaining_blocking),
    )
    publish_error = await finalize_render(attempt)
    if publish_error:
        return publish_error
    await attempt.agent._emit_figure_record(
        "figure_revision", figure, review=attempt.last_review
    )
    deps.get("logger").warning(
        "[Bioinformatics] Figure %s not fully approved after %s revisions",
        attempt.request.figure_name,
        attempt.revision_count,
    )
    layout_note = (
        f"\nUnresolved layout defects: {deps.get('summarize_violations')(remaining_blocking)}."
        if remaining_blocking
        else ""
    )
    return (
        f"Figure {attempt.request.figure_name} saved after {attempt.revision_count} "
        "revision attempts. Status: needs further revision.\n"
        f"Last feedback: {figure.pi_feedback}{layout_note}"
        + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
    )


async def run_review_loop(attempt: FigureAttemptState) -> str:
    """Run PI review until approval, terminal deferral, stall, or exhaustion."""
    deps = attempt.deps
    figure = attempt.figure
    while attempt.revision_count <= 3:
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
        quality_score = review.get("quality_score", 5)
        blocking = deps.get("blocking_violations")(figure.layout_violations)
        budget_left = attempt.revision_count < 3
        if not blocking and (
            attempt.best_snapshot is None
            or quality_score > attempt.best_snapshot.get("score", -1)
        ):
            snapshot = take_snapshot(attempt, quality_score)
            if snapshot is not None:
                attempt.best_snapshot = snapshot

        approved = bool(review.get("approved", False))
        if approved and blocking and budget_left:
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
        if approved:
            return await _accept_approved(
                attempt, review, quality_score, blocking
            )

        feedback = review.get("feedback", "")
        suggestions = review.get("suggestions", [])
        must_fix = deps.get("normalize_text_list")(
            review.get("must_fix"), limit=20
        )
        review = dict(review)
        review["must_fix"] = must_fix
        attempt.last_review = review
        attempt.last_must_fix = must_fix
        if review.get("requires_reanalysis", False):
            return await _accept_reanalysis(
                attempt, review, feedback, must_fix, blocking
            )

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
        if attempt.revision_count >= 3:
            break
        if len(attempt.revision_history) >= 2 and not blocking:
            previous_score = attempt.revision_history[-2].get("quality_score")
            current_score = attempt.revision_history[-1].get("quality_score")
            if (
                previous_score is not None
                and current_score is not None
                and current_score <= previous_score
            ):
                return await _accept_stalled(
                    attempt,
                    review,
                    feedback,
                    must_fix,
                    previous_score,
                    current_score,
                )
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
