"""Artifact selection, event payload, and invalidation owners."""
from __future__ import annotations

def compact_figure_event_payload(figure: Any, *, review_status: str | None=None, pi_feedback: str | None=None, bindings) -> dict[str, Any]:
    """Serialize a live figure event without its potentially large preview."""
    Any = getattr(bindings, 'Any')
    has_blocking_layout_defects = getattr(bindings, 'has_blocking_layout_defects')
    raw_status = review_status
    if raw_status is None:
        status = getattr(figure, "review_status", None)
        raw_status = str(getattr(status, "value", status) or "pending")
    created_at = getattr(figure, "created_at", None)
    return {
        "id": getattr(figure, "id", ""),
        "name": getattr(figure, "name", ""),
        "description": getattr(figure, "description", ""),
        "figure_type": getattr(figure, "figure_type", "other"),
        "png_path": getattr(figure, "png_path", ""),
        "pdf_path": getattr(figure, "pdf_path", ""),
        "review_status": raw_status,
        "review_unavailable": bool(
            getattr(figure, "review_unavailable", False)
        ),
        "pi_feedback": (
            pi_feedback if pi_feedback is not None
            else getattr(figure, "pi_feedback", None)
        ),
        "revision_count": int(getattr(figure, "revision_count", 0) or 0),
        "plan_step_id": getattr(figure, "plan_step_id", None),
        "plan_id": getattr(figure, "plan_id", None),
        "purpose": getattr(figure, "purpose", "result") or "result",
        "logical_figure_id": getattr(figure, "logical_figure_id", None),
        "producer_status": getattr(figure, "producer_status", "published"),
        "revision_of_id": getattr(figure, "revision_of_id", None),
        "supersedes_id": getattr(figure, "supersedes_id", None),
        "is_published_head": bool(getattr(figure, "is_published_head", False)),
        "round": int(getattr(figure, "round_number", 0) or 0) or None,
        "layout_blocking": has_blocking_layout_defects(figure),
        "diagnostic": bool(getattr(figure, "diagnostic", False)),
        "created_at": (
            created_at.isoformat() if hasattr(created_at, "isoformat")
            else str(created_at or "")
        ),
    }

def is_report_eligible(artifact: Any, *, bindings) -> bool:
    """Return whether a reviewed artifact is current enough for final reports."""
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    REPORTABLE_REVIEW_STATUSES = getattr(bindings, 'REPORTABLE_REVIEW_STATUSES')
    _artifact_value = getattr(bindings, '_artifact_value')
    _status_value = getattr(bindings, '_status_value')
    has_blocking_layout_defects = getattr(bindings, 'has_blocking_layout_defects')
    has_frozen_publication_pair = getattr(bindings, 'has_frozen_publication_pair')
    is_diagnostic = getattr(bindings, 'is_diagnostic')
    layout_was_measured = getattr(bindings, 'layout_was_measured')
    if str(_artifact_value(artifact, "producer_status", "published") or "published") != "published":
        return False
    if not _artifact_value(artifact, "is_current", True):
        return False
    if _artifact_value(artifact, "invalidated_at", None):
        return False
    is_figure = (
        "figure_type" in artifact or "png_path" in artifact
        if isinstance(artifact, Mapping)
        else hasattr(artifact, "figure_type") or hasattr(artifact, "png_path")
    )
    if not is_figure and _artifact_value(
        artifact, "plan_binding_status", None
    ) is not None:
        if str(_artifact_value(artifact, "plan_binding_status", "")) != "confirmed":
            return False
        if not str(_artifact_value(artifact, "source_execution_id", "") or ""):
            return False
    if is_figure and not has_frozen_publication_pair(artifact):
        return False
    if has_blocking_layout_defects(artifact):
        return False
    if list(_artifact_value(artifact, "unresolved_must_fix", None) or []):
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
    #
    # That is also the whole of the waiver's justification, so it cannot survive
    # the measurement failing. A diagnostic is rendered with no review callback,
    # so if its lint did not run nothing looked at the figure at all -- and the
    # figure most likely to crash the lint is the one most likely to need it.
    if is_diagnostic(artifact):
        return layout_was_measured(artifact)
    if (
        _status_value(artifact) == "pending"
        and bool(_artifact_value(artifact, "review_unavailable", False))
    ):
        return True
    return _status_value(artifact) in REPORTABLE_REVIEW_STATUSES

def logical_figure_key(artifact: Any, *, bindings) -> tuple[str, str]:
    """Return the explicit revision-family key for one artifact.

    A missing logical ID means an old record with no safe supersession evidence;
    keep it independent rather than guessing from a version-looking filename.
    """
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    lineage = str(_artifact_value(artifact, "lineage_id", None) or "main")
    logical_id = str(_artifact_value(artifact, "logical_figure_id", None) or "").strip()
    if not logical_id:
        logical_id = f"record:{_artifact_value(artifact, 'id', id(artifact))}"
    return lineage, logical_id

def published_head_artifacts(artifacts: Iterable[Any] | None, *, bindings) -> list[Any]:
    """Collapse report-eligible revisions to one archival head per logical figure.

    Pending or failed redraws are absent from the eligible input, so the previous
    reportable revision remains the head until a replacement actually qualifies.
    """
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    _artifact_value = getattr(bindings, '_artifact_value')
    is_report_eligible = getattr(bindings, 'is_report_eligible')
    logical_figure_key = getattr(bindings, 'logical_figure_key')
    eligible = [artifact for artifact in (artifacts or []) if is_report_eligible(artifact)]
    eligible_ids = {
        str(_artifact_value(artifact, "id", "")) for artifact in eligible
        if str(_artifact_value(artifact, "id", ""))
    }
    superseded_ids = {
        str(_artifact_value(artifact, "supersedes_id", ""))
        for artifact in eligible
        if str(_artifact_value(artifact, "supersedes_id", "")) in eligible_ids
    }
    best: dict[tuple[str, str], tuple[tuple[int, int, str, int], Any]] = {}
    for index, artifact in enumerate(eligible):
        created_at = _artifact_value(artifact, "created_at", None)
        created_key = (
            created_at.isoformat()
            if hasattr(created_at, "isoformat")
            else str(created_at or "")
        )
        rank = (
            0 if str(_artifact_value(artifact, "id", "")) in superseded_ids else 1,
            1 if _artifact_value(artifact, "is_published_head", False) else 0,
            created_key,
            index,
        )
        key = logical_figure_key(artifact)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, artifact)
    selected_ids = {id(entry[1]) for entry in best.values()}
    return [artifact for artifact in eligible if id(artifact) in selected_ids]

def eligible_artifacts(artifacts: Iterable[Any] | None, *, bindings) -> list[Any]:
    """Return the single reportable published head of every logical artifact."""
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    published_head_artifacts = getattr(bindings, 'published_head_artifacts')
    return published_head_artifacts(artifacts)

def eligible_artifact_by_id(artifacts: Iterable[Any] | None, artifact_id: str, *, lineage_id: str | None=None, bindings) -> Any | None:
    """Resolve an API/download ID only through the explicit eligible head set."""
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    _artifact_value = getattr(bindings, '_artifact_value')
    eligible_artifacts = getattr(bindings, 'eligible_artifacts')
    target = str(artifact_id or "")
    lineage = str(lineage_id or "main")
    return next(
        (
            artifact
            for artifact in eligible_artifacts(artifacts)
            if str(_artifact_value(artifact, "id", "")) == target
            and str(_artifact_value(artifact, "lineage_id", None) or "main")
            == lineage
        ),
        None,
    )

def is_gallery_eligible(artifact: Any, *, bindings) -> bool:
    """Whether a completed current render belongs on browsing surfaces.

    A gallery answers what the run produced, not what is fit for a report.
    Review status and layout findings travel as badges; only incomplete
    producers and figures invalidated by newer analysis data are withheld.
    """
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    return bool(
        str(
            _artifact_value(artifact, "producer_status", "published")
            or "published"
        )
        == "published"
        and _artifact_value(artifact, "is_current", True)
        and not _artifact_value(artifact, "invalidated_at", None)
    )

def gallery_artifacts(artifacts: Iterable[Any] | None, *, bindings) -> list[Any]:
    """Return every completed, current render for operator browsing."""
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    is_gallery_eligible = getattr(bindings, 'is_gallery_eligible')
    return [
        artifact
        for artifact in (artifacts or [])
        if is_gallery_eligible(artifact)
    ]

def gallery_artifact_by_id(artifacts: Iterable[Any] | None, artifact_id: str, *, lineage_id: str | None=None, bindings) -> Any | None:
    """Resolve a download ID through the browsing predicate and lineage."""
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    _artifact_value = getattr(bindings, '_artifact_value')
    gallery_artifacts = getattr(bindings, 'gallery_artifacts')
    target = str(artifact_id or "")
    lineage = str(lineage_id or "main")
    return next(
        (
            artifact
            for artifact in gallery_artifacts(artifacts)
            if str(_artifact_value(artifact, "id", "")) == target
            and str(_artifact_value(artifact, "lineage_id", None) or "main")
            == lineage
        ),
        None,
    )

def artifact_stems(artifacts: Iterable[Any] | None, *, bindings) -> set[str]:
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    os = getattr(bindings, 'os')
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

def excluded_artifact_stems(artifacts: Iterable[Any] | None, kept: Iterable[Any] | None, *, invalidated_stems: Iterable[str] | None=None, bindings) -> set[str]:
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
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    artifact_stems = getattr(bindings, 'artifact_stems')
    is_gallery_eligible = getattr(bindings, 'is_gallery_eligible')
    artifacts = list(artifacts or [])
    excluded = artifact_stems(artifacts) - artifact_stems(kept)
    stale = {str(stem) for stem in (invalidated_stems or [])}
    live = artifact_stems(
        artifact for artifact in artifacts if is_gallery_eligible(artifact)
    )
    return excluded | (stale - live)

def invalidate_current_artifacts(artifacts: Iterable[Any] | None, *, round_number: int, reason: str, lineage_id: str | None=None, bindings) -> int:
    """Invalidate current artifacts from one round after upstream data drift."""
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    datetime = getattr(bindings, 'datetime')
    invalidated = 0
    now = datetime.now()
    for artifact in artifacts or []:
        if (
            lineage_id is not None
            and hasattr(artifact, "lineage_id")
            and (getattr(artifact, "lineage_id", None) or "main")
            != (lineage_id or "main")
        ):
            continue
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
