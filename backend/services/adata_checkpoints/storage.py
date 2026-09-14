"""Host-side immutable publication of AnnData checkpoint candidates."""
from __future__ import annotations

import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.utils.files import fsync_directory

from .errors import AdataStorageError
from .layout import (
    BlobIdentity,
    blob_dir,
    canonical_blob_relative_path,
    sha256_file,
    staging_dir,
    validate_sha256,
    verify_blob,
)


@dataclass(frozen=True)
class StoragePublishResult:
    identity: BlobIdentity
    reused: bool
    repaired: bool = False


def _validate_candidate(candidate: Path, *, allowed_root: Optional[Path]) -> tuple[int, int, int, int, int, int]:
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AdataStorageError("checkpoint candidate is missing") from exc
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise AdataStorageError("checkpoint candidate is outside its allowed root") from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(str(candidate), flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AdataStorageError("checkpoint candidate is not a regular file")
        if info.st_nlink != 1:
            raise AdataStorageError("checkpoint candidate has a hard-link alias")
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )
    finally:
        os.close(fd)


def publish_or_reuse(
    session_id: str,
    candidate: Path,
    *,
    byte_budget: Optional[int] = None,
    allowed_root: Optional[Path] = None,
) -> StoragePublishResult:
    """Copy a verified candidate into the content-addressed immutable store.

    The candidate is never modified.  The Host copies it into a fresh staging
    inode on the destination filesystem, hashes the copied bytes, fsyncs, and
    atomically publishes ``blobs/<sha256>.h5ad``.  An existing valid same-digest
    blob is reused; an existing invalid file at the digest path is repaired by
    this independently verified candidate.
    """
    candidate = Path(candidate)
    identity = _validate_candidate(candidate, allowed_root=allowed_root)
    candidate_size = identity[2]
    if byte_budget is not None and candidate_size > int(byte_budget):
        raise AdataStorageError(
            "checkpoint candidate exceeds the configured byte budget",
            context={"limit_bytes": int(byte_budget), "candidate_bytes": int(candidate_size)},
        )

    blobs = blob_dir(str(session_id), create=True)
    staging_root = staging_dir(str(session_id), create=True)
    # Checkpoint-specific headroom: candidate bytes plus a reserve for the
    # pending ordinary runtime save.  Compressed file size is not an RSS
    # estimate, but for disk it is the bytes the new blob will occupy.
    try:
        free = shutil.disk_usage(str(blobs)).free
    except OSError:
        free = None
    if free is not None:
        reserve = 256 * 1024 * 1024
        if free < candidate_size + reserve:
            raise AdataStorageError(
                "insufficient disk headroom for checkpoint publication",
                context={
                    "free_bytes": int(free),
                    "candidate_bytes": int(candidate_size),
                    "reserve_bytes": reserve,
                },
            )
        # The candidate itself has already consumed its serialization space.
        # When it lives on another filesystem, retain the same reserve there
        # for the ordinary runtime save that follows publication.  On one
        # filesystem the destination check above already accounts for both the
        # second copy and that reserve.
        if int(identity[0]) != int(blobs.stat().st_dev):
            try:
                candidate_free = shutil.disk_usage(str(candidate.parent)).free
            except OSError:
                candidate_free = None
            if candidate_free is not None and candidate_free < reserve:
                raise AdataStorageError(
                    "insufficient staging headroom after checkpoint serialization",
                    context={
                        "free_bytes": int(candidate_free),
                        "candidate_bytes": int(candidate_size),
                        "reserve_bytes": reserve,
                    },
                )
    staging = staging_root / f"{uuid.uuid4().hex}.h5ad"
    published = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        source_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        source_fd = os.open(str(candidate), source_flags)
        target_fd = os.open(str(staging), flags, 0o600)
        import hashlib

        digest = hashlib.sha256()
        total = 0
        try:
            while True:
                chunk = os.read(source_fd, 4 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if byte_budget is not None and total > int(byte_budget):
                    raise AdataStorageError(
                        "checkpoint candidate exceeds the configured byte budget",
                        context={
                            "limit_bytes": int(byte_budget),
                            "candidate_bytes": int(total),
                        },
                    )
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    if written <= 0:
                        raise OSError("short write while staging checkpoint candidate")
                    view = view[written:]
            os.fsync(target_fd)
        finally:
            os.close(source_fd)
            os.close(target_fd)

        candidate_digest = digest.hexdigest()
        destination = blobs / f"{candidate_digest}.h5ad"

        if destination.exists() and not destination.is_symlink():
            try:
                existing = verify_blob(
                    destination,
                    expected_sha256=candidate_digest,
                    expected_size=total,
                )
            except (OSError, ValueError):
                existing = None
            if existing is not None:
                staging.unlink(missing_ok=True)
                return StoragePublishResult(identity=existing, reused=True)

        # Publish or repair.  This is a same-filesystem atomic rename; a valid
        # same-digest blob can be replaced by byte-identical bytes, which is
        # safe because the digest is verified over the copied candidate.
        os.replace(str(staging), str(destination))
        fsync_directory(blobs)
        published = True
        final = verify_blob(
            destination,
            expected_sha256=candidate_digest,
            expected_size=total,
        )
        return StoragePublishResult(identity=final, reused=False)
    except AdataStorageError:
        raise
    except OSError as exc:
        raise AdataStorageError(
            "checkpoint candidate could not be published",
            context={"error_type": type(exc).__name__},
        ) from exc
    finally:
        if not published:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass


def remove_blob_if_unreferenced(session_id: str, sha256: str, referenced: bool) -> bool:
    """Remove one exact blob only when the caller proved it is unreferenced."""
    if referenced:
        return False
    digest = validate_sha256(sha256)
    path = blob_dir(str(session_id)) / f"{digest}.h5ad"
    if not path.is_file() or path.is_symlink():
        return False
    path.unlink()
    fsync_directory(path.parent)
    return True


def prune_unreferenced_checkpoint_blobs(session_id: str, state: Any) -> list[str]:
    """Remove only orphan blobs not referenced by any persisted catalog record.

    Maintenance must never traverse a model-authored path.  The scan is bounded
    to one validated session's ``adata-checkpoints/blobs`` directory.
    """
    import time

    root = blob_dir(str(session_id))
    if not root.is_dir():
        return []
    referenced = {
        str(getattr(record, "blob_sha256", "") or "").lower()
        for record in (getattr(state, "adata_checkpoints", None) or [])
        if str(getattr(record, "status", "")) in {"available", "unavailable"}
        and str(getattr(record, "blob_sha256", "") or "")
    }
    removed: list[str] = []
    now = time.time()
    for entry in sorted(root.iterdir()):
        try:
            info = entry.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode) or entry.is_symlink():
            continue
        name = entry.name
        if not name.endswith(".h5ad"):
            continue
        digest = name[: -len(".h5ad")].lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            continue
        if digest in referenced:
            continue
        if now - info.st_mtime < 300:
            # Grace period: a just-published blob whose catalog save is still
            # in flight must not be collected.
            continue
        try:
            entry.unlink()
        except OSError:
            continue
        removed.append(digest)
    if removed:
        fsync_directory(root)
    return removed
