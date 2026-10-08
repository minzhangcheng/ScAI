"""Artifact integrity and archival materialization owners."""
from __future__ import annotations

from backend.models.schemas import FigureReviewStatus

REPORTABLE_REVIEW_STATUSES = frozenset({
    FigureReviewStatus.APPROVED.value,
    FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS.value,
})

ARTIFACT_REGISTRY_VERSION = 1

LAYOUT_UNVERIFIED_KIND = "lint_skipped"

MIN_RENDERED_FIGURE_BYTES = 2000

MAX_ARCHIVAL_PREVIEW_ENCODED_CHARS = 24_000_000

MIN_PUBLICATION_PDF_BYTES = 512

MIN_ARCHIVAL_PREVIEW_WIDTH = 16

MIN_ARCHIVAL_PREVIEW_HEIGHT = 16

MAX_VERIFIED_ARTIFACT_BYTES = 1024 * 1024 * 1024

MAX_VERIFIED_IMAGE_PIXELS = 16_000_000


def _status_value(artifact: Any, *, bindings) -> str:
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    status = _artifact_value(artifact, "review_status")
    return str(getattr(status, "value", status) or "").lower()

def has_blocking_layout_defects(artifact: Any, *, bindings) -> bool:
    """Whether the renderer measured text that another artist covers.

    ``accepted_with_reservations`` is reachable both by a reviewer signing off
    and by the revision loop giving up, and neither says anything about whether
    the figure is readable. This asks the one question the status cannot: is
    some text drawn underneath something else.
    """
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    blocking_violations = getattr(bindings, 'blocking_violations')
    return bool(blocking_violations(_artifact_value(artifact, "layout_violations")))

def is_diagnostic(artifact: Any, *, bindings) -> bool:
    """Whether the platform rendered this artifact, rather than the analysis agent."""
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    return bool(_artifact_value(artifact, "diagnostic", False))

def layout_was_measured(artifact: Any, *, bindings) -> bool:
    """Whether the renderer's layout lint actually ran on this artifact.

    ``has_blocking_layout_defects`` answers "was anything found"; a False from it
    means either "measured, nothing wrong" or "never measured", and only one of
    those is evidence.
    """
    Any = getattr(bindings, 'Any')
    LAYOUT_UNVERIFIED_KIND = getattr(bindings, 'LAYOUT_UNVERIFIED_KIND')
    _artifact_value = getattr(bindings, '_artifact_value')
    violations = _artifact_value(artifact, "layout_violations", None) or []
    return not any(
        str((violation or {}).get("kind", "")) == LAYOUT_UNVERIFIED_KIND
        for violation in violations
        if isinstance(violation, dict)
    )

def _artifact_value(artifact: Any, key: str, default: Any=None, *, bindings) -> Any:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    if isinstance(artifact, Mapping):
        return artifact.get(key, default)
    return getattr(artifact, key, default)

def _image_bytes_are_decodable(payload: bytes, *, min_width: int=1, min_height: int=1, max_pixels: int | None=None, bindings) -> bool:
    """Require both container verification and a complete pixel decode."""
    Image = getattr(bindings, 'Image')
    io = getattr(bindings, 'io')
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

def _read_regular_file_nofollow(path: Any, *, bindings) -> bytes | None:
    """Read one stable regular file without following any path component.

    The returned bytes and the metadata checks come from the same open file
    description.  Callers can therefore hand the bytes to a delayed consumer
    without reopening a path that may have been replaced after validation.
    """
    Any = getattr(bindings, 'Any')
    MAX_VERIFIED_ARTIFACT_BYTES = getattr(bindings, 'MAX_VERIFIED_ARTIFACT_BYTES')
    Path = getattr(bindings, 'Path')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
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

def verified_artifact_bytes(path: Any, artifact: Any=None, *, kind: str, bindings) -> bytes | None:
    """Return structurally valid, record-bound bytes from one no-follow FD."""
    Any = getattr(bindings, 'Any')
    MAX_VERIFIED_IMAGE_PIXELS = getattr(bindings, 'MAX_VERIFIED_IMAGE_PIXELS')
    MIN_ARCHIVAL_PREVIEW_HEIGHT = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_HEIGHT')
    MIN_ARCHIVAL_PREVIEW_WIDTH = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_WIDTH')
    MIN_PUBLICATION_PDF_BYTES = getattr(bindings, 'MIN_PUBLICATION_PDF_BYTES')
    _artifact_value = getattr(bindings, '_artifact_value')
    _image_bytes_are_decodable = getattr(bindings, '_image_bytes_are_decodable')
    _read_regular_file_nofollow = getattr(bindings, '_read_regular_file_nofollow')
    _record_hashes = getattr(bindings, '_record_hashes')
    hashlib = getattr(bindings, 'hashlib')
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
    elif kind == "table":
        expected_digest = str(
            (_artifact_value(artifact, "csv_sha256", "") or "")
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

def copy_verified_artifact(source: Any, destination: Any, artifact: Any=None, *, kind: str, bindings) -> bool:
    """Atomically copy the exact bytes validated from one no-follow source FD."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    os = getattr(bindings, 'os')
    uuid = getattr(bindings, 'uuid')
    verified_artifact_bytes = getattr(bindings, 'verified_artifact_bytes')
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

def _record_hashes(artifact: Any, *, bindings) -> set[str]:
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    return {
        str(value).strip().lower()
        for value in (
            _artifact_value(artifact, "render_sha256"),
            _artifact_value(artifact, "preview_sha256"),
        )
        if str(value or "").strip()
    }

def _bytes_match_record(payload: bytes, artifact: Any, *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    _record_hashes = getattr(bindings, '_record_hashes')
    hashlib = getattr(bindings, 'hashlib')
    expected = _record_hashes(artifact)
    # A missing digest is missing provenance, not a legacy wildcard. Freezing
    # whatever happens to occupy a mutable path today would let replacement
    # bytes inherit the record's historical review decision.
    return bool(expected) and hashlib.sha256(payload).hexdigest() in expected

def image_file_sha256(path: Any, *, min_bytes: int=MIN_RENDERED_FIGURE_BYTES, min_width: int=1, min_height: int=1, bindings) -> str | None:
    """Return a digest only for a sufficiently large, fully decodable image."""
    Any = getattr(bindings, 'Any')
    MIN_RENDERED_FIGURE_BYTES = getattr(bindings, 'MIN_RENDERED_FIGURE_BYTES')
    Path = getattr(bindings, 'Path')
    _image_bytes_are_decodable = getattr(bindings, '_image_bytes_are_decodable')
    hashlib = getattr(bindings, 'hashlib')
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

def preview_render_sha256(encoded: Any, *, bindings) -> str | None:
    """Digest a valid record preview; unlike the primary JPG, small is valid."""
    Any = getattr(bindings, 'Any')
    MIN_ARCHIVAL_PREVIEW_HEIGHT = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_HEIGHT')
    MIN_ARCHIVAL_PREVIEW_WIDTH = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_WIDTH')
    _decode_preview_bytes = getattr(bindings, '_decode_preview_bytes')
    _image_bytes_are_decodable = getattr(bindings, '_image_bytes_are_decodable')
    hashlib = getattr(bindings, 'hashlib')
    payload = _decode_preview_bytes(encoded)
    if not payload or not _image_bytes_are_decodable(
        payload,
        min_width=MIN_ARCHIVAL_PREVIEW_WIDTH,
        min_height=MIN_ARCHIVAL_PREVIEW_HEIGHT,
    ):
        return None
    return hashlib.sha256(payload).hexdigest()

def publication_pdf_is_valid(path: Any, *, bindings) -> bool:
    """Check the minimum structural promise of a downloadable PDF render."""
    Any = getattr(bindings, 'Any')
    MIN_PUBLICATION_PDF_BYTES = getattr(bindings, 'MIN_PUBLICATION_PDF_BYTES')
    Path = getattr(bindings, 'Path')
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

def pdf_file_sha256(path: Any, *, bindings) -> str | None:
    """Return a digest only for a structurally valid publication PDF."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    hashlib = getattr(bindings, 'hashlib')
    publication_pdf_is_valid = getattr(bindings, 'publication_pdf_is_valid')
    if not publication_pdf_is_valid(path):
        return None
    try:
        return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()
    except OSError:
        return None

def archival_pdf_path_matches_record(artifact: Any, path: Any, *, bindings) -> bool:
    """Validate a PDF and bind it to the exact reviewed bytes."""
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    pdf_file_sha256 = getattr(bindings, 'pdf_file_sha256')
    digest = pdf_file_sha256(path)
    if digest is None:
        return False
    expected = str(_artifact_value(artifact, "pdf_sha256") or "").strip().lower()
    return bool(expected) and digest == expected

def resolve_archival_pdf(artifact: Any, workspace_path: Any, *, bindings) -> Path | None:
    """Resolve the record PDF without trusting a stale same-name file."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    archival_pdf_path_matches_record = getattr(bindings, 'archival_pdf_path_matches_record')
    os = getattr(bindings, 'os')
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

def resolve_verified_archival_artifact(artifact: Any, workspace_path: Any, *, kind: str, path_authorizer: Callable[[Path], bool] | None=None, bindings) -> tuple[bytes, Path] | None:
    """Resolve and verify a publication artifact with one read per candidate."""
    Any = getattr(bindings, 'Any')
    Callable = getattr(bindings, 'Callable')
    MIN_RENDERED_FIGURE_BYTES = getattr(bindings, 'MIN_RENDERED_FIGURE_BYTES')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    _materialize_archival_preview_payload = getattr(bindings, '_materialize_archival_preview_payload')
    _record_preview_candidates = getattr(bindings, '_record_preview_candidates')
    _restore_artifact_mtime = getattr(bindings, '_restore_artifact_mtime')
    os = getattr(bindings, 'os')
    verified_artifact_bytes = getattr(bindings, 'verified_artifact_bytes')
    workspace = Path(workspace_path)
    if kind == "pdf":
        declared = str(_artifact_value(artifact, "pdf_path") or "")
        candidates = []
        if declared:
            candidates.extend((
                Path(declared),
                workspace / "figures" / os.path.basename(declared),
            ))
        for candidate in dict.fromkeys(candidates):
            if path_authorizer is not None and not path_authorizer(candidate):
                continue
            payload = verified_artifact_bytes(candidate, artifact, kind="pdf")
            if payload is not None:
                return payload, candidate
        return None
    if kind != "image":
        raise ValueError(f"unsupported artifact kind: {kind}")

    declared = str(_artifact_value(artifact, "png_path") or "")
    candidates = []
    if declared:
        candidates.extend((
            Path(declared),
            workspace / "figures" / os.path.basename(declared),
        ))
    candidates.extend(_record_preview_candidates(artifact, workspace))
    checked_candidates = list(dict.fromkeys(candidates))
    for index, candidate in enumerate(checked_candidates):
        if path_authorizer is not None and not path_authorizer(candidate):
            continue
        payload = verified_artifact_bytes(candidate, artifact, kind="image")
        if payload is None:
            continue
        if index == 0 and len(payload) < MIN_RENDERED_FIGURE_BYTES:
            continue
        if index > 0:
            _restore_artifact_mtime(candidate, artifact)
        return payload, candidate

    return _materialize_archival_preview_payload(
        artifact,
        workspace,
        check_declared_path=False,
        checked_paths=set(checked_candidates),
    )

def _legacy_relative_path(record: Any, field: str, workspace_path: Any, *, bindings) -> Path | None:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
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

def resolve_legacy_artifact_render(record: Any, workspace_path: Any, *, bindings) -> Path | None:
    """Resolve one frozen legacy image by exact relative path and digest."""
    Any = getattr(bindings, 'Any')
    MIN_ARCHIVAL_PREVIEW_HEIGHT = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_HEIGHT')
    MIN_ARCHIVAL_PREVIEW_WIDTH = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_WIDTH')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    _legacy_relative_path = getattr(bindings, '_legacy_relative_path')
    image_file_sha256 = getattr(bindings, 'image_file_sha256')
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

def resolve_legacy_artifact_pdf(record: Any, workspace_path: Any, *, bindings) -> Path | None:
    """Resolve one optional frozen legacy PDF by exact path and digest."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    _legacy_relative_path = getattr(bindings, '_legacy_relative_path')
    pdf_file_sha256 = getattr(bindings, 'pdf_file_sha256')
    if str(_artifact_value(record, "provenance_kind", "")) != "legacy_unregistered":
        return None
    expected = str(_artifact_value(record, "pdf_sha256", "") or "").lower()
    candidate = _legacy_relative_path(record, "pdf_relative_path", workspace_path)
    if not expected or candidate is None:
        return None
    return candidate if pdf_file_sha256(candidate) == expected else None

def resolve_verified_legacy_artifact(record: Any, workspace_path: Any, *, kind: str, bindings) -> tuple[bytes, Path] | None:
    """Resolve frozen legacy inventory bytes with one record-bound read."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _legacy_relative_path = getattr(bindings, '_legacy_relative_path')
    verified_artifact_bytes = getattr(bindings, 'verified_artifact_bytes')
    field = "pdf_relative_path" if kind == "pdf" else "relative_path"
    candidate = _legacy_relative_path(record, field, workspace_path)
    if candidate is None:
        return None
    payload = verified_artifact_bytes(candidate, record, kind=kind)
    return (payload, candidate) if payload is not None else None

def _decode_preview_bytes(encoded: Any, *, bindings) -> bytes:
    Any = getattr(bindings, 'Any')
    MAX_ARCHIVAL_PREVIEW_ENCODED_CHARS = getattr(bindings, 'MAX_ARCHIVAL_PREVIEW_ENCODED_CHARS')
    base64 = getattr(bindings, 'base64')
    binascii = getattr(bindings, 'binascii')
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

def _preview_bytes(artifact: Any, *, bindings) -> bytes:
    """Decode a record-owned PNG/JPEG preview without trusting its file path."""
    Any = getattr(bindings, 'Any')
    MIN_ARCHIVAL_PREVIEW_HEIGHT = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_HEIGHT')
    MIN_ARCHIVAL_PREVIEW_WIDTH = getattr(bindings, 'MIN_ARCHIVAL_PREVIEW_WIDTH')
    _artifact_value = getattr(bindings, '_artifact_value')
    _bytes_match_record = getattr(bindings, '_bytes_match_record')
    _decode_preview_bytes = getattr(bindings, '_decode_preview_bytes')
    _image_bytes_are_decodable = getattr(bindings, '_image_bytes_are_decodable')
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

def _artifact_created_epoch(artifact: Any, *, bindings) -> float | None:
    """Return the record creation time as epoch seconds when available."""
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    datetime = getattr(bindings, 'datetime')
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

def _restore_artifact_mtime(path: Path, artifact: Any, *, bindings) -> None:
    """Keep recovered previews attributable to their original analysis round."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _artifact_created_epoch = getattr(bindings, '_artifact_created_epoch')
    os = getattr(bindings, 'os')
    created_epoch = _artifact_created_epoch(artifact)
    if created_epoch is None:
        return
    try:
        os.utime(path, (created_epoch, created_epoch))
    except OSError:
        # The image bytes are still usable if a read-only or unusual filesystem
        # refuses timestamp restoration; attribution will simply stay unresolved.
        pass

def _record_path_is_usable(artifact: Any, path: Any, *, allow_small_preview: bool=False, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    MIN_RENDERED_FIGURE_BYTES = getattr(bindings, 'MIN_RENDERED_FIGURE_BYTES')
    _record_hashes = getattr(bindings, '_record_hashes')
    image_file_sha256 = getattr(bindings, 'image_file_sha256')
    digest = image_file_sha256(path, min_bytes=1 if allow_small_preview else MIN_RENDERED_FIGURE_BYTES)
    if digest is None:
        return False
    expected = _record_hashes(artifact)
    return bool(expected) and digest in expected

def has_frozen_publication_pair(artifact: Any, *, bindings) -> bool:
    """Whether a publication record carries immutable raster and PDF IDs."""
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    render = str(_artifact_value(artifact, "render_sha256", "") or "").lower()
    preview = str(_artifact_value(artifact, "preview_sha256", "") or "").lower()
    pdf = str(_artifact_value(artifact, "pdf_sha256", "") or "").lower()

    def valid_digest(value: str) -> bool:
        return bool(
            len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
        )

    # A damaged legacy render identity must not hide a valid record-owned
    # preview. Availability and eligibility must agree on the same fallback.
    return valid_digest(pdf) and (valid_digest(render) or valid_digest(preview))

def archival_path_matches_record(artifact: Any, path: Any, *, bindings) -> bool:
    """Whether a discovered file is the record-bound archival render.

    Generic workspace discovery knows only a stem.  A later plain ``savefig``
    can overwrite that stem after review, so callers must not attach the old
    authoritative metadata merely because the new file has the same name.
    """
    Any = getattr(bindings, 'Any')
    _record_path_is_usable = getattr(bindings, '_record_path_is_usable')
    return _record_path_is_usable(artifact, path)


def _record_preview_candidates(
    artifact: Any,
    workspace_path: Any = None,
    *,
    bindings,
) -> list[Path]:
    """Locate an already-materialized digest-bound record preview."""
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    digest = str(_artifact_value(artifact, "preview_sha256", "") or "").lower()
    declared = str(_artifact_value(artifact, "png_path", "") or "")
    if len(digest) != 64 or not all(char in "0123456789abcdef" for char in digest):
        return []

    declared_path = Path(declared) if declared else None
    stem = declared_path.stem if declared_path is not None else str(
        _artifact_value(artifact, "name", "figure") or "figure"
    )
    safe_stem = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in stem
    ).strip("._") or "figure"
    roots = []
    if declared_path is not None:
        roots.append(declared_path.parent)
    if workspace_path is not None:
        roots.append(Path(workspace_path) / "figures")
    names = [
        f"{safe_stem}_record_preview_{digest[:16]}.jpg",
        f"{safe_stem}_record_preview_{digest[:16]}.png",
        # Materialization uses the plain stem when its extension is vacant.
        f"{safe_stem}.jpg",
        f"{safe_stem}.png",
    ]
    return list(dict.fromkeys(root / name for root in roots for name in names))


def archival_render_available(artifact: Any, *, bindings) -> bool:
    """Whether an archival report can obtain real image bytes for a record.

    Review eligibility and image availability are deliberately separate.  A
    restored record may still be approved while its old absolute path no longer
    exists.  Such a record can satisfy a figure contract only when it also
    carries a valid preview that the report can safely materialize.
    """
    Any = getattr(bindings, 'Any')
    _artifact_value = getattr(bindings, '_artifact_value')
    _preview_bytes = getattr(bindings, '_preview_bytes')
    _record_preview_candidates = getattr(bindings, '_record_preview_candidates')
    _record_path_is_usable = getattr(bindings, '_record_path_is_usable')
    verified_artifact_bytes = getattr(bindings, 'verified_artifact_bytes')
    path = _artifact_value(artifact, "png_path")
    if path and _record_path_is_usable(artifact, path):
        return True
    for candidate in _record_preview_candidates(artifact):
        if verified_artifact_bytes(candidate, artifact, kind="image") is not None:
            return True
    return bool(_preview_bytes(artifact))

def _write_preview_noclobber(destination: Path, payload: bytes, *, bindings) -> bool:
    """Create one preview without following or replacing an existing entry."""
    Path = getattr(bindings, 'Path')
    os = getattr(bindings, 'os')
    descriptor: int | None = None
    created = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short archival preview write")
            view = view[written:]
        os.fsync(descriptor)
        return True
    except (FileExistsError, OSError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created and destination.exists():
            try:
                if destination.stat().st_size != len(payload):
                    destination.unlink()
            except OSError:
                pass

def _materialize_archival_preview_payload(artifact: Any, workspace_path: Any, *, check_declared_path: bool, checked_paths: set[Path] | None=None, bindings) -> tuple[bytes, Path] | None:
    """Recover and return the already-verified record preview bytes and path."""
    Any = getattr(bindings, 'Any')
    MIN_RENDERED_FIGURE_BYTES = getattr(bindings, 'MIN_RENDERED_FIGURE_BYTES')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    _preview_bytes = getattr(bindings, '_preview_bytes')
    _restore_artifact_mtime = getattr(bindings, '_restore_artifact_mtime')
    _write_preview_noclobber = getattr(bindings, '_write_preview_noclobber')
    hashlib = getattr(bindings, 'hashlib')
    os = getattr(bindings, 'os')
    verified_artifact_bytes = getattr(bindings, 'verified_artifact_bytes')
    path = _artifact_value(artifact, "png_path")
    if check_declared_path and path:
        candidate = Path(str(path))
        payload = verified_artifact_bytes(candidate, artifact, kind="image")
        if payload is not None and len(payload) >= MIN_RENDERED_FIGURE_BYTES:
            return payload, candidate

    payload = _preview_bytes(artifact)
    if not payload:
        return None
    extension = ".png" if payload.startswith(b"\x89PNG") else ".jpg"
    original_name = os.path.basename(str(path or ""))
    stem = Path(original_name).stem if original_name else str(
        _artifact_value(artifact, "name", "figure")
    )
    safe_stem = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in stem
    ).strip("._")
    destination = (
        Path(workspace_path) / "figures" / f"{safe_stem or 'figure'}{extension}"
    )

    checked_paths = checked_paths or set()
    existing = (
        None
        if destination in checked_paths
        else verified_artifact_bytes(destination, artifact, kind="image")
    )
    if existing is not None:
        _restore_artifact_mtime(destination, artifact)
        return existing, destination
    if destination.exists():
        digest = hashlib.sha256(payload).hexdigest()[:16]
        destination = destination.with_name(
            f"{destination.stem}_record_preview_{digest}{destination.suffix}"
        )
        existing = verified_artifact_bytes(destination, artifact, kind="image")
        if existing is not None:
            _restore_artifact_mtime(destination, artifact)
            return existing, destination
        if destination.exists():
            return None

    if not _write_preview_noclobber(destination, payload):
        return None
    _restore_artifact_mtime(destination, artifact)
    return payload, destination

def materialize_archival_preview(artifact: Any, workspace_path: Any, *, bindings) -> Path | None:
    """Recover a missing record image under ``workspace/figures``.

    The persisted path is used only for its basename; recovery never writes to
    an arbitrary absolute path from session state.
    """
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _materialize_archival_preview_payload = getattr(bindings, '_materialize_archival_preview_payload')
    recovered = _materialize_archival_preview_payload(
        artifact,
        workspace_path,
        check_declared_path=True,
        checked_paths=None,
    )
    return recovered[1] if recovered is not None else None

def resolve_archival_render(artifact: Any, workspace_path: Any, *, bindings) -> Path | None:
    """Resolve a reportable image path, recovering the record preview if needed."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    _record_path_is_usable = getattr(bindings, '_record_path_is_usable')
    _restore_artifact_mtime = getattr(bindings, '_restore_artifact_mtime')
    materialize_archival_preview = getattr(bindings, 'materialize_archival_preview')
    os = getattr(bindings, 'os')
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

def render_is_intact(artifact: Any, *, bindings) -> bool:
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
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _artifact_value = getattr(bindings, '_artifact_value')
    _record_path_is_usable = getattr(bindings, '_record_path_is_usable')
    archival_render_available = getattr(bindings, 'archival_render_available')
    path = _artifact_value(artifact, "png_path", None)
    if not path:
        return archival_render_available(artifact)
    try:
        if not Path(str(path)).exists():
            return archival_render_available(artifact)
    except OSError:
        return archival_render_available(artifact)
    return _record_path_is_usable(artifact, path)
