"""Validity and report-eligibility rules for generated analysis artifacts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import stat
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from collections.abc import Mapping

from PIL import Image

from backend.models.schemas import FigureReviewStatus
from backend.services.figure_layout import blocking_violations

from . import artifact_selection, artifact_shared
from .artifact_shared import (
    ARTIFACT_REGISTRY_VERSION,
    LAYOUT_UNVERIFIED_KIND,
    MAX_ARCHIVAL_PREVIEW_ENCODED_CHARS,
    MAX_VERIFIED_ARTIFACT_BYTES,
    MAX_VERIFIED_IMAGE_PIXELS,
    MIN_ARCHIVAL_PREVIEW_HEIGHT,
    MIN_ARCHIVAL_PREVIEW_WIDTH,
    MIN_PUBLICATION_PDF_BYTES,
    MIN_RENDERED_FIGURE_BYTES,
    REPORTABLE_REVIEW_STATUSES,
)



def _status_value(artifact: Any) -> str:
    return artifact_shared._status_value(artifact, bindings=sys.modules[__name__])



def has_blocking_layout_defects(artifact: Any) -> bool:
    """Whether the renderer measured text that another artist covers.

    ``accepted_with_reservations`` is reachable both by a reviewer signing off
    and by the revision loop giving up, and neither says anything about whether
    the figure is readable. This asks the one question the status cannot: is
    some text drawn underneath something else.
    """
    return artifact_shared.has_blocking_layout_defects(artifact, bindings=sys.modules[__name__])



def is_diagnostic(artifact: Any) -> bool:
    """Whether the platform rendered this artifact, rather than the analysis agent."""
    return artifact_shared.is_diagnostic(artifact, bindings=sys.modules[__name__])



#: Kind the renderer records when its in-kernel layout lint could not run. An
#: unmeasured layout must not be stored as an empty violation list, because that
#: is exactly what a measured-clean figure carries.


def layout_was_measured(artifact: Any) -> bool:
    """Whether the renderer's layout lint actually ran on this artifact.

    ``has_blocking_layout_defects`` answers "was anything found"; a False from it
    means either "measured, nothing wrong" or "never measured", and only one of
    those is evidence.
    """
    return artifact_shared.layout_was_measured(artifact, bindings=sys.modules[__name__])



#: The blank-render floor the producer applies to every figure it accepts (see
#: ``agents/bioinformatics/figures.py``): a JPEG smaller than this is white space
#: with axes at most.


def _artifact_value(artifact: Any, key: str, default: Any = None) -> Any:
    return artifact_shared._artifact_value(artifact, key, default, bindings=sys.modules[__name__])



def _image_bytes_are_decodable(
    payload: bytes,
    *,
    min_width: int = 1,
    min_height: int = 1,
    max_pixels: int | None = None,
) -> bool:
    """Require both container verification and a complete pixel decode."""
    return artifact_shared._image_bytes_are_decodable(payload, min_width=min_width, min_height=min_height, max_pixels=max_pixels, bindings=sys.modules[__name__])



def _read_regular_file_nofollow(path: Any) -> bytes | None:
    """Read one stable regular file without following any path component.

    The returned bytes and the metadata checks come from the same open file
    description.  Callers can therefore hand the bytes to a delayed consumer
    without reopening a path that may have been replaced after validation.
    """
    return artifact_shared._read_regular_file_nofollow(path, bindings=sys.modules[__name__])



def verified_artifact_bytes(
    path: Any,
    artifact: Any = None,
    *,
    kind: str,
) -> bytes | None:
    """Return structurally valid, record-bound bytes from one no-follow FD."""
    return artifact_shared.verified_artifact_bytes(path, artifact, kind=kind, bindings=sys.modules[__name__])



def copy_verified_artifact(
    source: Any,
    destination: Any,
    artifact: Any = None,
    *,
    kind: str,
) -> bool:
    """Atomically copy the exact bytes validated from one no-follow source FD."""
    return artifact_shared.copy_verified_artifact(source, destination, artifact, kind=kind, bindings=sys.modules[__name__])



def _record_hashes(artifact: Any) -> set[str]:
    return artifact_shared._record_hashes(artifact, bindings=sys.modules[__name__])



def _bytes_match_record(payload: bytes, artifact: Any) -> bool:
    return artifact_shared._bytes_match_record(payload, artifact, bindings=sys.modules[__name__])



def image_file_sha256(
    path: Any,
    *,
    min_bytes: int = MIN_RENDERED_FIGURE_BYTES,
    min_width: int = 1,
    min_height: int = 1,
) -> str | None:
    """Return a digest only for a sufficiently large, fully decodable image."""
    return artifact_shared.image_file_sha256(path, min_bytes=min_bytes, min_width=min_width, min_height=min_height, bindings=sys.modules[__name__])



def preview_render_sha256(encoded: Any) -> str | None:
    """Digest a valid record preview; unlike the primary JPG, small is valid."""
    return artifact_shared.preview_render_sha256(encoded, bindings=sys.modules[__name__])



def publication_pdf_is_valid(path: Any) -> bool:
    """Check the minimum structural promise of a downloadable PDF render."""
    return artifact_shared.publication_pdf_is_valid(path, bindings=sys.modules[__name__])



def pdf_file_sha256(path: Any) -> str | None:
    """Return a digest only for a structurally valid publication PDF."""
    return artifact_shared.pdf_file_sha256(path, bindings=sys.modules[__name__])



def archival_pdf_path_matches_record(artifact: Any, path: Any) -> bool:
    """Validate a PDF and bind it to the exact reviewed bytes."""
    return artifact_shared.archival_pdf_path_matches_record(artifact, path, bindings=sys.modules[__name__])



def resolve_archival_pdf(artifact: Any, workspace_path: Any) -> Path | None:
    """Resolve the record PDF without trusting a stale same-name file."""
    return artifact_shared.resolve_archival_pdf(artifact, workspace_path, bindings=sys.modules[__name__])



def resolve_verified_archival_artifact(
    artifact: Any,
    workspace_path: Any,
    *,
    kind: str,
    path_authorizer: Callable[[Path], bool] | None = None,
) -> tuple[bytes, Path] | None:
    """Resolve and verify a publication artifact with one read per candidate."""
    return artifact_shared.resolve_verified_archival_artifact(artifact, workspace_path, kind=kind, path_authorizer=path_authorizer, bindings=sys.modules[__name__])



def _legacy_relative_path(record: Any, field: str, workspace_path: Any) -> Path | None:
    return artifact_shared._legacy_relative_path(record, field, workspace_path, bindings=sys.modules[__name__])



def resolve_legacy_artifact_render(record: Any, workspace_path: Any) -> Path | None:
    """Resolve one frozen legacy image by exact relative path and digest."""
    return artifact_shared.resolve_legacy_artifact_render(record, workspace_path, bindings=sys.modules[__name__])



def resolve_legacy_artifact_pdf(record: Any, workspace_path: Any) -> Path | None:
    """Resolve one optional frozen legacy PDF by exact path and digest."""
    return artifact_shared.resolve_legacy_artifact_pdf(record, workspace_path, bindings=sys.modules[__name__])



def resolve_verified_legacy_artifact(
    record: Any,
    workspace_path: Any,
    *,
    kind: str,
) -> tuple[bytes, Path] | None:
    """Resolve frozen legacy inventory bytes with one record-bound read."""
    return artifact_shared.resolve_verified_legacy_artifact(record, workspace_path, kind=kind, bindings=sys.modules[__name__])



def _decode_preview_bytes(encoded: Any) -> bytes:
    return artifact_shared._decode_preview_bytes(encoded, bindings=sys.modules[__name__])



def _preview_bytes(artifact: Any) -> bytes:
    """Decode a record-owned PNG/JPEG preview without trusting its file path."""
    return artifact_shared._preview_bytes(artifact, bindings=sys.modules[__name__])



def _artifact_created_epoch(artifact: Any) -> float | None:
    """Return the record creation time as epoch seconds when available."""
    return artifact_shared._artifact_created_epoch(artifact, bindings=sys.modules[__name__])



def _restore_artifact_mtime(path: Path, artifact: Any) -> None:
    """Keep recovered previews attributable to their original analysis round."""
    return artifact_shared._restore_artifact_mtime(path, artifact, bindings=sys.modules[__name__])



def _record_path_is_usable(
    artifact: Any, path: Any, *, allow_small_preview: bool = False
) -> bool:
    return artifact_shared._record_path_is_usable(artifact, path, allow_small_preview=allow_small_preview, bindings=sys.modules[__name__])



def has_frozen_publication_pair(artifact: Any) -> bool:
    """Whether a publication record carries immutable raster and PDF IDs."""
    return artifact_shared.has_frozen_publication_pair(artifact, bindings=sys.modules[__name__])



def archival_path_matches_record(artifact: Any, path: Any) -> bool:
    """Whether a discovered file is the record-bound archival render.

    Generic workspace discovery knows only a stem.  A later plain ``savefig``
    can overwrite that stem after review, so callers must not attach the old
    authoritative metadata merely because the new file has the same name.
    """
    return artifact_shared.archival_path_matches_record(artifact, path, bindings=sys.modules[__name__])


def _record_preview_candidates(
    artifact: Any,
    workspace_path: Any = None,
) -> list[Path]:
    return artifact_shared._record_preview_candidates(
        artifact,
        workspace_path,
        bindings=sys.modules[__name__],
    )



def archival_render_available(artifact: Any) -> bool:
    """Whether an archival report can obtain real image bytes for a record.

    Review eligibility and image availability are deliberately separate.  A
    restored record may still be approved while its old absolute path no longer
    exists.  Such a record can satisfy a figure contract only when it also
    carries a valid preview that the report can safely materialize.
    """
    return artifact_shared.archival_render_available(artifact, bindings=sys.modules[__name__])



def _write_preview_noclobber(destination: Path, payload: bytes) -> bool:
    """Create one preview without following or replacing an existing entry."""
    return artifact_shared._write_preview_noclobber(destination, payload, bindings=sys.modules[__name__])



def _materialize_archival_preview_payload(
    artifact: Any,
    workspace_path: Any,
    *,
    check_declared_path: bool,
    checked_paths: set[Path] | None = None,
) -> tuple[bytes, Path] | None:
    """Recover and return the already-verified record preview bytes and path."""
    return artifact_shared._materialize_archival_preview_payload(artifact, workspace_path, check_declared_path=check_declared_path, checked_paths=checked_paths, bindings=sys.modules[__name__])



def materialize_archival_preview(artifact: Any, workspace_path: Any) -> Path | None:
    """Recover a missing record image under ``workspace/figures``.

    The persisted path is used only for its basename; recovery never writes to
    an arbitrary absolute path from session state.
    """
    return artifact_shared.materialize_archival_preview(artifact, workspace_path, bindings=sys.modules[__name__])



def resolve_archival_render(artifact: Any, workspace_path: Any) -> Path | None:
    """Resolve a reportable image path, recovering the record preview if needed."""
    return artifact_shared.resolve_archival_render(artifact, workspace_path, bindings=sys.modules[__name__])



def render_is_intact(artifact: Any) -> bool:
    """Whether the file this record would hand over is still a real render.

    The publication wrapper saves in a ``finally``, so a revision that rendered
    nothing has already overwritten the good file before the loop can object,
    and the in-memory rollback is a no-op when no snapshot was ever taken --
    none is while a blocking layout defect is outstanding. The record then keeps
    the earlier render's base64 while the bytes beside it are blank, so the
    preview and the download are different images.

    Judged on the bytes and not on the review status, so nothing is withheld for
    merely having been revised. A path that cannot be stat'd is not judged at
    all: the record still carries its own preview, and withholding figures whose
    workspace moved is the over-filtering the browsing surface was widened to
    undo.
    """
    return artifact_shared.render_is_intact(artifact, bindings=sys.modules[__name__])



def compact_figure_event_payload(
    figure: Any,
    *,
    review_status: str | None = None,
    pi_feedback: str | None = None,
) -> dict[str, Any]:
    """Serialize a live figure event without its potentially large preview."""
    return artifact_selection.compact_figure_event_payload(figure, review_status=review_status, pi_feedback=pi_feedback, bindings=sys.modules[__name__])



def data_drift_warning(artifact: Any) -> str | None:
    """Return the recorded or compatibility warning for a data-drift artifact."""
    return artifact_selection.data_drift_warning(artifact)



def description_with_data_drift_warning(artifact: Any) -> str:
    """Return an artifact description that discloses its historical data context."""
    return artifact_selection.description_with_data_drift_warning(artifact)


def figure_reservation_notice(artifact: Any) -> dict[str, Any]:
    """Return the report-visible PI reservation disclosure for one figure."""
    return artifact_selection.figure_reservation_notice(
        artifact, bindings=sys.modules[__name__]
    )



def publication_figure_provenance_error(state: Any, figure: Any) -> str | None:
    """Revalidate figure provenance against its owning session when possible."""
    version = _artifact_value(figure, "provenance_schema_version", None)
    if state is None:
        if version is None:
            return str(
                _artifact_value(figure, "provenance_validation_reason", None) or ""
            ) or None
        reason = "Analysis state is required to revalidate schema-v2 figure provenance."
    else:
        from backend.services.figure_followup import (
            validate_publication_figure_provenance as _validate,
        )

        reason = _validate(state, figure)
    if reason:
        if isinstance(figure, dict):
            figure["provenance_validation_reason"] = str(reason)[:1000]
        else:
            try:
                figure.provenance_validation_reason = str(reason)[:1000]
            except (AttributeError, TypeError, ValueError):
                pass
    elif isinstance(figure, dict):
        figure.pop("provenance_validation_reason", None)
    elif getattr(figure, "provenance_validation_reason", None):
        try:
            figure.provenance_validation_reason = None
        except (AttributeError, TypeError, ValueError):
            pass
    return reason


def is_report_eligible(artifact: Any, *, state: Any = None) -> bool:
    """Return whether a reviewed artifact is current enough for final reports."""
    return artifact_selection.is_report_eligible(
        artifact, state=state, bindings=sys.modules[__name__]
    )



def logical_figure_key(artifact: Any) -> tuple[str, str]:
    """Return the explicit revision-family key for one artifact.

    A missing logical ID means an old record with no safe supersession evidence;
    keep it independent rather than guessing from a version-looking filename.
    """
    return artifact_selection.logical_figure_key(artifact, bindings=sys.modules[__name__])



def published_head_artifacts(
    artifacts: Iterable[Any] | None, *, state: Any = None
) -> list[Any]:
    """Collapse report-eligible revisions to one archival head per logical figure.

    Pending or failed redraws are absent from the eligible input, so the previous
    reportable revision remains the head until a replacement actually qualifies.
    """
    return artifact_selection.published_head_artifacts(
        artifacts, state=state, bindings=sys.modules[__name__]
    )



def eligible_artifacts(
    artifacts: Iterable[Any] | None, *, state: Any = None
) -> list[Any]:
    """Return the single reportable published head of every logical artifact."""
    return artifact_selection.eligible_artifacts(
        artifacts, state=state, bindings=sys.modules[__name__]
    )



def eligible_artifact_by_id(
    artifacts: Iterable[Any] | None,
    artifact_id: str,
    *,
    lineage_id: str | None = None,
    state: Any = None,
) -> Any | None:
    """Resolve an API/download ID only through the explicit eligible head set."""
    return artifact_selection.eligible_artifact_by_id(
        artifacts,
        artifact_id,
        lineage_id=lineage_id,
        state=state,
        bindings=sys.modules[__name__],
    )



def is_gallery_eligible(artifact: Any) -> bool:
    """Whether a completed current render belongs on browsing surfaces.

    A gallery answers what the run produced, not what is fit for a report.
    Review status and layout findings travel as badges; only incomplete
    producers and figures invalidated by newer analysis data are withheld.
    """
    return artifact_selection.is_gallery_eligible(artifact, bindings=sys.modules[__name__])



def gallery_artifacts(artifacts: Iterable[Any] | None) -> list[Any]:
    """Return every completed, current render for operator browsing."""
    return artifact_selection.gallery_artifacts(artifacts, bindings=sys.modules[__name__])



def gallery_artifact_by_id(
    artifacts: Iterable[Any] | None,
    artifact_id: str,
    *,
    lineage_id: str | None = None,
) -> Any | None:
    """Resolve a download ID through the browsing predicate and lineage."""
    return artifact_selection.gallery_artifact_by_id(artifacts, artifact_id, lineage_id=lineage_id, bindings=sys.modules[__name__])



def artifact_stems(artifacts: Iterable[Any] | None) -> set[str]:
    return artifact_selection.artifact_stems(artifacts, bindings=sys.modules[__name__])



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
    return artifact_selection.excluded_artifact_stems(artifacts, kept, invalidated_stems=invalidated_stems, bindings=sys.modules[__name__])



def invalidate_current_artifacts(
    artifacts: Iterable[Any] | None,
    *,
    round_number: int,
    reason: str,
    lineage_id: str | None = None,
) -> int:
    """Invalidate current artifacts from one round after upstream data drift."""
    return artifact_selection.invalidate_current_artifacts(artifacts, round_number=round_number, reason=reason, lineage_id=lineage_id, bindings=sys.modules[__name__])
