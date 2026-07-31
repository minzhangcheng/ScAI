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


def is_report_eligible(artifact: Any) -> bool:
    """Return whether a reviewed artifact is current enough for final reports."""
    return bool(
        getattr(artifact, "is_current", True)
        and not getattr(artifact, "invalidated_at", None)
        and _status_value(artifact) in REPORTABLE_REVIEW_STATUSES
        and not has_blocking_layout_defects(artifact)
    )


def eligible_artifacts(artifacts: Iterable[Any] | None) -> list[Any]:
    return [artifact for artifact in (artifacts or []) if is_report_eligible(artifact)]


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
