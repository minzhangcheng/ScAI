"""Validity and report-eligibility rules for generated analysis artifacts."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Iterable

from backend.models.schemas import FigureReviewStatus
from backend.services.figure_layout import blocking_violations


REPORTABLE_REVIEW_STATUSES = frozenset({
    FigureReviewStatus.APPROVED.value,
    FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS.value,
})


def _status_value(artifact: Any) -> str:
    status = getattr(artifact, "review_status", None)
    return str(getattr(status, "value", status) or "").lower()


def has_blocking_layout_defects(artifact: Any) -> bool:
    """Whether the renderer measured text that another artist covers.

    ``accepted_with_reservations`` is reachable both by a reviewer signing off
    and by the revision loop giving up, and neither says anything about whether
    the figure is readable. This asks the one question the status cannot: is
    some text drawn underneath something else.
    """
    return bool(blocking_violations(getattr(artifact, "layout_violations", None)))


def is_diagnostic(artifact: Any) -> bool:
    """Whether the platform rendered this artifact, rather than the analysis agent."""
    return bool(getattr(artifact, "diagnostic", False))


def is_report_eligible(artifact: Any) -> bool:
    """Return whether a reviewed artifact is current enough for final reports."""
    if not getattr(artifact, "is_current", True):
        return False
    if getattr(artifact, "invalidated_at", None):
        return False
    if has_blocking_layout_defects(artifact):
        return False
    # A platform-rendered diagnostic has no reviewer by construction, so a review
    # status it could never earn must not be what withholds it. Quality control is
    # the chapter every later claim rests on, and holding its only figure back for
    # want of a sign-off nobody was asked to give is the same mistake as letting a
    # stale stem overrule a record that says the figure is current.
    #
    # The layout gate above is deliberately still applied: whether text is covered
    # is measured by the renderer, not attested by a reviewer, so it is exactly the
    # question an unreviewed figure can still be asked.
    if is_diagnostic(artifact):
        return True
    return _status_value(artifact) in REPORTABLE_REVIEW_STATUSES


def eligible_artifacts(artifacts: Iterable[Any] | None) -> list[Any]:
    return [artifact for artifact in (artifacts or []) if is_report_eligible(artifact)]


def is_gallery_eligible(artifact: Any) -> bool:
    """Whether a reviewed artifact belongs on the browsing surface.

    Deliberately weaker than ``is_report_eligible``, and the difference is the
    point. That predicate answers "is this fit to publish", which is the right
    question for a report and the wrong one for "show me what this run made": it
    withholds a figure the review loop passed only after revision, and a figure
    the loop gave up on at the revision cap — both of which exist on disk, are
    this run's own work, and are what the operator asked to see.

    Review status and measured layout defects travel with the record as badges
    instead, so nothing is hidden and nothing is misrepresented. Only artifacts
    superseded by newer data are withheld, because those describe a matrix that
    no longer exists.
    """
    return bool(
        getattr(artifact, "is_current", True)
        and not getattr(artifact, "invalidated_at", None)
    )


def gallery_artifacts(artifacts: Iterable[Any] | None) -> list[Any]:
    return [artifact for artifact in (artifacts or []) if is_gallery_eligible(artifact)]


def artifact_stems(artifacts: Iterable[Any] | None) -> set[str]:
    stems: set[str] = set()
    for artifact in artifacts or []:
        for value in (
            getattr(artifact, "name", None),
            getattr(artifact, "png_path", None),
            getattr(artifact, "csv_path", None),
        ):
            if value:
                stems.add(os.path.splitext(os.path.basename(str(value)))[0])
    return stems


def excluded_artifact_stems(
    artifacts: Iterable[Any] | None,
    kept: Iterable[Any] | None,
    *,
    invalidated_stems: Iterable[str] | None = None,
) -> set[str]:
    """Stems an output surface must not resolve, for one set of artifacts.

    Two staleness mechanisms answer the same question and disagree.
    ``invalidate_current_artifacts`` marks the *record*: per-artifact, round-scoped,
    and carrying a reason. ``invalidated_artifact_stems`` marks the *file*: a flat,
    monotonic union of bare names. The second exists to reach artifacts that have no
    record at all -- a plain ``savefig`` left in the workspace -- so letting it
    overrule the first inverts the relationship, withholding the reviewed figure
    while the unreviewed file beside it is kept.

    Hence the rule here: a stem does not withhold an artifact whose own record still
    says it is current. Before 0.38.2 the stem set had no round bound, so a single
    representation-mutating cell in round N blacklisted every artifact from rounds
    1..N-1 for the rest of the session. That is fixed at the producer, but the sets
    already written into ``session.json`` are not, and every report surface only ever
    reads them -- so subtracting live records is also what repairs existing sessions.
    Replayed against one real 8-round session it is the difference between 10 figures
    and 5, and three of the five lost are the exploration round's own.

    Stems with no artifact record stay excluded. Nothing available here can tell a
    file that is genuinely stale from one the old producer bug named by mistake, and
    guessing in the permissive direction would put a figure of the wrong generation
    into a report.
    """
    artifacts = list(artifacts or [])
    excluded = artifact_stems(artifacts) - artifact_stems(kept)
    stale = {str(stem) for stem in (invalidated_stems or [])}
    live = artifact_stems(
        artifact for artifact in artifacts if is_gallery_eligible(artifact)
    )
    return excluded | (stale - live)


def invalidate_current_artifacts(
    artifacts: Iterable[Any] | None,
    *,
    round_number: int,
    reason: str,
) -> int:
    """Invalidate current artifacts from one round after upstream data drift."""
    invalidated = 0
    now = datetime.now()
    for artifact in artifacts or []:
        artifact_round = int(getattr(artifact, "round_number", 0) or 0)
        if artifact_round not in (0, round_number):
            continue
        if not getattr(artifact, "is_current", True):
            continue
        artifact.is_current = False
        artifact.invalidated_at = now
        artifact.invalidation_reason = reason
        invalidated += 1
    return invalidated
