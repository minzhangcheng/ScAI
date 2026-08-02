"""Validity and report-eligibility rules for generated analysis artifacts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from collections.abc import Mapping

from PIL import Image

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


#: Kind the renderer records when its in-kernel layout lint could not run. An
#: unmeasured layout must not be stored as an empty violation list, because that
#: is exactly what a measured-clean figure carries.
LAYOUT_UNVERIFIED_KIND = "lint_skipped"


def layout_was_measured(artifact: Any) -> bool:
    """Whether the renderer's layout lint actually ran on this artifact.

    ``has_blocking_layout_defects`` answers "was anything found"; a False from it
    means either "measured, nothing wrong" or "never measured", and only one of
    those is evidence.
    """
    violations = getattr(artifact, "layout_violations", None) or []
    return not any(
        str((violation or {}).get("kind", "")) == LAYOUT_UNVERIFIED_KIND
        for violation in violations
        if isinstance(violation, dict)
    )


#: The blank-render floor the producer applies to every figure it accepts (see
#: ``agents/bioinformatics/figures.py``): a JPEG smaller than this is white space
#: with axes at most.
MIN_RENDERED_FIGURE_BYTES = 2000
MAX_ARCHIVAL_PREVIEW_ENCODED_CHARS = 24_000_000
MIN_PUBLICATION_PDF_BYTES = 512
MIN_ARCHIVAL_PREVIEW_WIDTH = 16
MIN_ARCHIVAL_PREVIEW_HEIGHT = 16


def _artifact_value(artifact: Any, key: str, default: Any = None) -> Any:
    if isinstance(artifact, Mapping):
        return artifact.get(key, default)
    return getattr(artifact, key, default)


def _image_bytes_are_decodable(
    payload: bytes, *, min_width: int = 1, min_height: int = 1
) -> bool:
    """Require both container verification and a complete pixel decode."""
    if not payload:
        return False
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        # ``verify`` checks the container but deliberately does not decode the
        # pixels. Reopen and load so truncated/corrupt image data is rejected.
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return image.width >= min_width and image.height >= min_height
    except Exception:
        return False


def _record_hashes(artifact: Any) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (
            _artifact_value(artifact, "render_sha256"),
            _artifact_value(artifact, "preview_sha256"),
        )
        if str(value or "").strip()
    }


def _bytes_match_record(payload: bytes, artifact: Any) -> bool:
    expected = _record_hashes(artifact)
    return not expected or hashlib.sha256(payload).hexdigest() in expected


def image_file_sha256(
    path: Any,
    *,
    min_bytes: int = MIN_RENDERED_FIGURE_BYTES,
    min_width: int = 1,
    min_height: int = 1,
) -> str | None:
    """Return a digest only for a sufficiently large, fully decodable image."""
    try:
        candidate = Path(str(path))
        if not candidate.is_file() or candidate.stat().st_size < min_bytes:
            return None
        payload = candidate.read_bytes()
    except OSError:
        return None
    if not _image_bytes_are_decodable(
        payload, min_width=min_width, min_height=min_height
    ):
        return None
    return hashlib.sha256(payload).hexdigest()


def preview_render_sha256(encoded: Any) -> str | None:
    """Digest a valid record preview; unlike the primary JPG, small is valid."""
    payload = _decode_preview_bytes(encoded)
    if not payload or not _image_bytes_are_decodable(
        payload,
        min_width=MIN_ARCHIVAL_PREVIEW_WIDTH,
        min_height=MIN_ARCHIVAL_PREVIEW_HEIGHT,
    ):
        return None
    return hashlib.sha256(payload).hexdigest()


def publication_pdf_is_valid(path: Any) -> bool:
    """Check the minimum structural promise of a downloadable PDF render."""
    try:
        candidate = Path(str(path))
        size = candidate.stat().st_size
        if not candidate.is_file() or size < MIN_PUBLICATION_PDF_BYTES:
            return False
        with candidate.open("rb") as handle:
            header = handle.read(8)
            handle.seek(max(0, size - 2048))
            tail = handle.read()
    except (OSError, ValueError):
        return False
    return header.startswith(b"%PDF-") and tail.rstrip().endswith(b"%%EOF")


def pdf_file_sha256(path: Any) -> str | None:
    """Return a digest only for a structurally valid publication PDF."""
    if not publication_pdf_is_valid(path):
        return None
    try:
        return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()
    except OSError:
        return None


def archival_pdf_path_matches_record(artifact: Any, path: Any) -> bool:
    """Validate a PDF and, when recorded, bind it to the reviewed bytes."""
    digest = pdf_file_sha256(path)
    if digest is None:
        return False
    expected = str(_artifact_value(artifact, "pdf_sha256") or "").strip().lower()
    return not expected or digest == expected


def resolve_archival_pdf(artifact: Any, workspace_path: Any) -> Path | None:
    """Resolve the record PDF without trusting a stale same-name file."""
    declared = str(_artifact_value(artifact, "pdf_path") or "")
    candidates: list[Path] = []
    if declared:
        candidates.append(Path(declared))
        candidates.append(
            Path(workspace_path) / "figures" / os.path.basename(declared)
        )
    for candidate in candidates:
        if archival_pdf_path_matches_record(artifact, candidate):
            return candidate
    return None


def _decode_preview_bytes(encoded: Any) -> bytes:
    encoded = str(encoded or "").strip()
    if not encoded or len(encoded) > MAX_ARCHIVAL_PREVIEW_ENCODED_CHARS:
        return b""
    if encoded.startswith("data:"):
        _header, separator, encoded = encoded.partition(",")
        if not separator:
            return b""
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return b""


def _preview_bytes(artifact: Any) -> bytes:
    """Decode a record-owned PNG/JPEG preview without trusting its file path."""
    payload = _decode_preview_bytes(_artifact_value(artifact, "png_base64"))
    if not _image_bytes_are_decodable(
        payload,
        min_width=MIN_ARCHIVAL_PREVIEW_WIDTH,
        min_height=MIN_ARCHIVAL_PREVIEW_HEIGHT,
    ):
        return b""
    if not _bytes_match_record(payload, artifact):
        return b""
    return payload


def _artifact_created_epoch(artifact: Any) -> float | None:
    """Return the record creation time as epoch seconds when available."""
    value = _artifact_value(artifact, "created_at")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None


def _restore_artifact_mtime(path: Path, artifact: Any) -> None:
    """Keep recovered previews attributable to their original analysis round."""
    created_epoch = _artifact_created_epoch(artifact)
    if created_epoch is None:
        return
    try:
        os.utime(path, (created_epoch, created_epoch))
    except OSError:
        # The image bytes are still usable if a read-only or unusual filesystem
        # refuses timestamp restoration; attribution will simply stay unresolved.
        pass


def _record_path_is_usable(
    artifact: Any, path: Any, *, allow_small_preview: bool = False
) -> bool:
    digest = image_file_sha256(path, min_bytes=1 if allow_small_preview else MIN_RENDERED_FIGURE_BYTES)
    if digest is None:
        return False
    expected = _record_hashes(artifact)
    if expected and digest not in expected:
        return False
    return True


def archival_path_matches_record(artifact: Any, path: Any) -> bool:
    """Whether a discovered file is the record-bound archival render.

    Generic workspace discovery knows only a stem.  A later plain ``savefig``
    can overwrite that stem after review, so callers must not attach the old
    authoritative metadata merely because the new file has the same name.
    """
    return _record_path_is_usable(artifact, path)


def archival_render_available(artifact: Any) -> bool:
    """Whether an archival report can obtain real image bytes for a record.

    Review eligibility and image availability are deliberately separate.  A
    restored record may still be approved while its old absolute path no longer
    exists.  Such a record can satisfy a figure contract only when it also
    carries a valid preview that the report can safely materialize.
    """
    path = _artifact_value(artifact, "png_path")
    if path and _record_path_is_usable(artifact, path):
        return True
    return bool(_preview_bytes(artifact))


def materialize_archival_preview(artifact: Any, workspace_path: Any) -> Path | None:
    """Recover a missing record image under ``workspace/figures``.

    The persisted path is used only for its basename; recovery never writes to
    an arbitrary absolute path from session state.
    """
    path = _artifact_value(artifact, "png_path")
    if path:
        candidate = Path(str(path))
        if _record_path_is_usable(artifact, candidate):
            return candidate

    payload = _preview_bytes(artifact)
    if not payload:
        return None
    extension = ".png" if payload.startswith(b"\x89PNG") else ".jpg"
    original_name = os.path.basename(str(path or ""))
    stem = Path(original_name).stem if original_name else str(
        _artifact_value(artifact, "name", "figure")
    )
    safe_stem = "".join(char if char.isalnum() or char in "-_." else "_" for char in stem).strip("._")
    destination = Path(workspace_path) / "figures" / f"{safe_stem or 'figure'}{extension}"
    # Never overwrite a different same-name image merely to recover the bytes
    # bound to an older record. Keep both and let the record resolve to its own
    # hash-qualified archival copy.
    if destination.exists() and not _record_path_is_usable(
        artifact, destination, allow_small_preview=True
    ):
        destination = destination.with_name(
            f"{destination.stem}_record_preview{destination.suffix}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    except OSError:
        return None
    if not _record_path_is_usable(artifact, destination, allow_small_preview=True):
        try:
            destination.unlink()
        except OSError:
            pass
        return None
    _restore_artifact_mtime(destination, artifact)
    return destination


def resolve_archival_render(artifact: Any, workspace_path: Any) -> Path | None:
    """Resolve a reportable image path, recovering the record preview if needed."""
    declared = str(_artifact_value(artifact, "png_path") or "")
    candidates = []
    if declared:
        candidates.append(Path(declared))
        candidates.append(Path(workspace_path) / "figures" / os.path.basename(declared))
    for index, candidate in enumerate(candidates):
        # The workspace candidate may be a deliberately materialized compact
        # record preview. The declared producer JPG retains the 2 KB floor.
        if _record_path_is_usable(
            artifact, candidate, allow_small_preview=(index > 0)
        ):
            if index > 0:
                # This is a workspace recovery path rather than the producer's
                # declared render. Repair previews materialized by older builds
                # as well as ones written during this call.
                _restore_artifact_mtime(candidate, artifact)
            return candidate
    return materialize_archival_preview(artifact, workspace_path)


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
    path = getattr(artifact, "png_path", None)
    if not path:
        return archival_render_available(artifact)
    try:
        if not Path(str(path)).exists():
            return archival_render_available(artifact)
    except OSError:
        return archival_render_available(artifact)
    return _record_path_is_usable(artifact, path)


def compact_figure_event_payload(
    figure: Any,
    *,
    review_status: str | None = None,
    pi_feedback: str | None = None,
) -> dict[str, Any]:
    """Serialize a live figure event without its potentially large preview."""
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
        "pi_feedback": (
            pi_feedback if pi_feedback is not None
            else getattr(figure, "pi_feedback", None)
        ),
        "revision_count": int(getattr(figure, "revision_count", 0) or 0),
        "plan_step_id": getattr(figure, "plan_step_id", None),
        "plan_id": getattr(figure, "plan_id", None),
        "purpose": getattr(figure, "purpose", "result") or "result",
        "round": int(getattr(figure, "round_number", 0) or 0) or None,
        "layout_blocking": has_blocking_layout_defects(figure),
        "diagnostic": bool(getattr(figure, "diagnostic", False)),
        "created_at": (
            created_at.isoformat() if hasattr(created_at, "isoformat")
            else str(created_at or "")
        ),
    }


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
    #
    # That is also the whole of the waiver's justification, so it cannot survive
    # the measurement failing. A diagnostic is rendered with no review callback,
    # so if its lint did not run nothing looked at the figure at all -- and the
    # figure most likely to crash the lint is the one most likely to need it.
    if is_diagnostic(artifact):
        return layout_was_measured(artifact)
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

    Deliberately answerable from the record alone. ``render_is_intact`` asks the
    separate question of whether the bytes on disk still match the record, and
    surfaces that serve a download call it alongside this one -- folding it in
    here would make every caller of this predicate stat the filesystem.
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
    lineage_id: str | None = None,
) -> int:
    """Invalidate current artifacts from one round after upstream data drift."""
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
