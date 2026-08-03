"""Validity and report-eligibility rules for generated analysis artifacts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import stat
import uuid
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

ARTIFACT_REGISTRY_VERSION = 1


def _status_value(artifact: Any) -> str:
    status = _artifact_value(artifact, "review_status")
    return str(getattr(status, "value", status) or "").lower()


def has_blocking_layout_defects(artifact: Any) -> bool:
    """Whether the renderer measured text that another artist covers.

    ``accepted_with_reservations`` is reachable both by a reviewer signing off
    and by the revision loop giving up, and neither says anything about whether
    the figure is readable. This asks the one question the status cannot: is
    some text drawn underneath something else.
    """
    return bool(blocking_violations(_artifact_value(artifact, "layout_violations")))


def is_diagnostic(artifact: Any) -> bool:
    """Whether the platform rendered this artifact, rather than the analysis agent."""
    return bool(_artifact_value(artifact, "diagnostic", False))


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
    violations = _artifact_value(artifact, "layout_violations", None) or []
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
MAX_VERIFIED_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_VERIFIED_IMAGE_PIXELS = 16_000_000


def _artifact_value(artifact: Any, key: str, default: Any = None) -> Any:
    if isinstance(artifact, Mapping):
        return artifact.get(key, default)
    return getattr(artifact, key, default)


def _image_bytes_are_decodable(
    payload: bytes,
    *,
    min_width: int = 1,
    min_height: int = 1,
    max_pixels: int | None = None,
) -> bool:
    """Require both container verification and a complete pixel decode."""
    if not payload:
        return False
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if max_pixels is not None and image.width * image.height > max_pixels:
                return False
            image.verify()
        # ``verify`` checks the container but deliberately does not decode the
        # pixels. Reopen and load so truncated/corrupt image data is rejected.
        with Image.open(io.BytesIO(payload)) as image:
            if max_pixels is not None and image.width * image.height > max_pixels:
                return False
            image.load()
            return image.width >= min_width and image.height >= min_height
    except Exception:
        return False


def _read_regular_file_nofollow(path: Any) -> bytes | None:
    """Read one stable regular file without following any path component.

    The returned bytes and the metadata checks come from the same open file
    description.  Callers can therefore hand the bytes to a delayed consumer
    without reopening a path that may have been replaced after validation.
    """
    try:
        absolute = Path(os.path.abspath(os.fspath(Path(str(path)))))
    except (OSError, TypeError, ValueError):
        return None
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        return None
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(absolute.anchor, directory_flags)
        for component in parts[1:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > MAX_VERIFIED_ARTIFACT_BYTES
        ):
            return None
        # A regular file read is bounded by the already-checked size plus one
        # growth sentinel. This yields one immutable bytes object instead of a
        # chunk list followed by a second full-size ``join`` allocation.
        payload = os.read(file_fd, before.st_size + 1)
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(payload) != before.st_size:
            return None
        return payload
    except (OSError, ValueError):
        return None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def verified_artifact_bytes(
    path: Any,
    artifact: Any = None,
    *,
    kind: str,
) -> bytes | None:
    """Return structurally valid, record-bound bytes from one no-follow FD."""
    payload = _read_regular_file_nofollow(path)
    if payload is None:
        return None
    digest = hashlib.sha256(payload).hexdigest()
    if kind == "image":
        if not _image_bytes_are_decodable(
            payload,
            min_width=MIN_ARCHIVAL_PREVIEW_WIDTH,
            min_height=MIN_ARCHIVAL_PREVIEW_HEIGHT,
            max_pixels=MAX_VERIFIED_IMAGE_PIXELS,
        ):
            return None
        expected = _record_hashes(artifact) if artifact is not None else set()
    elif kind == "pdf":
        if (
            len(payload) < MIN_PUBLICATION_PDF_BYTES
            or not payload.startswith(b"%PDF-")
            or not payload[-2048:].rstrip().endswith(b"%%EOF")
        ):
            return None
        expected_digest = str(
            (_artifact_value(artifact, "pdf_sha256", "") or "")
            if artifact is not None else ""
        ).strip().lower()
        expected = {expected_digest} if expected_digest else set()
    else:
        raise ValueError(f"unsupported artifact kind: {kind}")
    if artifact is not None and not expected:
        return None
    if expected and digest not in expected:
        return None
    return payload


def copy_verified_artifact(
    source: Any,
    destination: Any,
    artifact: Any = None,
    *,
    kind: str,
) -> bool:
    """Atomically copy the exact bytes validated from one no-follow source FD."""
    payload = verified_artifact_bytes(source, artifact, kind=kind)
    if payload is None:
        return False
    target = Path(str(destination))
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            os.fspath(temporary),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short artifact snapshot write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, target)
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass


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
    # A missing digest is missing provenance, not a legacy wildcard. Freezing
    # whatever happens to occupy a mutable path today would let replacement
    # bytes inherit the record's historical review decision.
    return bool(expected) and hashlib.sha256(payload).hexdigest() in expected


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
    """Validate a PDF and bind it to the exact reviewed bytes."""
    digest = pdf_file_sha256(path)
    if digest is None:
        return False
    expected = str(_artifact_value(artifact, "pdf_sha256") or "").strip().lower()
    return bool(expected) and digest == expected


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


def _legacy_relative_path(record: Any, field: str, workspace_path: Any) -> Path | None:
    relative = str(_artifact_value(record, field) or "").strip()
    if not relative:
        return None
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        return None
    root = Path(workspace_path).resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def resolve_legacy_artifact_render(record: Any, workspace_path: Any) -> Path | None:
    """Resolve one frozen legacy image by exact relative path and digest."""
    if str(_artifact_value(record, "provenance_kind", "")) != "legacy_unregistered":
        return None
    expected = str(_artifact_value(record, "render_sha256", "") or "").lower()
    candidate = _legacy_relative_path(record, "relative_path", workspace_path)
    if not expected or candidate is None:
        return None
    digest = image_file_sha256(
        candidate,
        min_bytes=1,
        min_width=MIN_ARCHIVAL_PREVIEW_WIDTH,
        min_height=MIN_ARCHIVAL_PREVIEW_HEIGHT,
    )
    return candidate if digest == expected else None


def resolve_legacy_artifact_pdf(record: Any, workspace_path: Any) -> Path | None:
    """Resolve one optional frozen legacy PDF by exact path and digest."""
    if str(_artifact_value(record, "provenance_kind", "")) != "legacy_unregistered":
        return None
    expected = str(_artifact_value(record, "pdf_sha256", "") or "").lower()
    candidate = _legacy_relative_path(record, "pdf_relative_path", workspace_path)
    if not expected or candidate is None:
        return None
    return candidate if pdf_file_sha256(candidate) == expected else None


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
    return bool(expected) and digest in expected


def has_frozen_publication_pair(artifact: Any) -> bool:
    """Whether a publication record carries immutable raster and PDF IDs."""
    render = str(_artifact_value(artifact, "render_sha256", "") or "").lower()
    pdf = str(_artifact_value(artifact, "pdf_sha256", "") or "").lower()
    return bool(
        len(render) == 64
        and len(pdf) == 64
        and all(char in "0123456789abcdef" for char in render)
        and all(char in "0123456789abcdef" for char in pdf)
    )


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
    path = _artifact_value(artifact, "png_path", None)
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


def is_report_eligible(artifact: Any) -> bool:
    """Return whether a reviewed artifact is current enough for final reports."""
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
    if is_figure and not has_frozen_publication_pair(artifact):
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


def logical_figure_key(artifact: Any) -> tuple[str, str]:
    """Return the explicit revision-family key for one artifact.

    A missing logical ID means an old record with no safe supersession evidence;
    keep it independent rather than guessing from a version-looking filename.
    """
    lineage = str(_artifact_value(artifact, "lineage_id", None) or "main")
    logical_id = str(_artifact_value(artifact, "logical_figure_id", None) or "").strip()
    if not logical_id:
        logical_id = f"record:{_artifact_value(artifact, 'id', id(artifact))}"
    return lineage, logical_id


def published_head_artifacts(artifacts: Iterable[Any] | None) -> list[Any]:
    """Collapse report-eligible revisions to one archival head per logical figure.

    Pending or failed redraws are absent from the eligible input, so the previous
    reportable revision remains the head until a replacement actually qualifies.
    """
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


def eligible_artifacts(artifacts: Iterable[Any] | None) -> list[Any]:
    """Return the single reportable published head of every logical artifact."""
    return published_head_artifacts(artifacts)


def eligible_artifact_by_id(
    artifacts: Iterable[Any] | None,
    artifact_id: str,
    *,
    lineage_id: str | None = None,
) -> Any | None:
    """Resolve an API/download ID only through the explicit eligible head set."""
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


def is_gallery_eligible(artifact: Any) -> bool:
    """Whether a completed current render belongs on browsing surfaces.

    A gallery answers what the run produced, not what is fit for a report.
    Review status and layout findings travel as badges; only incomplete
    producers and figures invalidated by newer analysis data are withheld.
    """
    return bool(
        str(
            _artifact_value(artifact, "producer_status", "published")
            or "published"
        )
        == "published"
        and _artifact_value(artifact, "is_current", True)
        and not _artifact_value(artifact, "invalidated_at", None)
    )


def gallery_artifacts(artifacts: Iterable[Any] | None) -> list[Any]:
    """Return every completed, current render for operator browsing."""
    return [
        artifact
        for artifact in (artifacts or [])
        if is_gallery_eligible(artifact)
    ]


def gallery_artifact_by_id(
    artifacts: Iterable[Any] | None,
    artifact_id: str,
    *,
    lineage_id: str | None = None,
) -> Any | None:
    """Resolve a download ID through the browsing predicate and lineage."""
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
