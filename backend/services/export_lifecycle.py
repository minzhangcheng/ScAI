"""Generation-bound lifecycle ownership for session export archives."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from starlette.responses import (
    FileResponse,
    Headers,
    MalformedRangeHeader,
    MutableHeaders,
    PlainTextResponse,
    RangeNotSatisfiable,
)

from backend.services.session_lifecycle import (
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.state import (
    prepared_exports,
    prepared_exports_lock,
    sessions,
)
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    run_in_owned_thread,
)


logger = logging.getLogger(__name__)


EXPORT_OWNERSHIP_MARKER_SUFFIX = ".scai-export-owner"

_SECURE_COPY_CHUNK_BYTES = 1024 * 1024
_ZIP_MEMBER_OVERHEAD_BYTES = 1024
_Identity = tuple[int, int, int, int, int, int]


class SecureArchiveError(RuntimeError):
    """A secure archive operation was rejected before it could publish bytes."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


@dataclass(frozen=True)
class _SelectedArchiveMember:
    relative: Path
    parts: tuple[str, ...]
    parent_identities: tuple[_Identity, ...]
    identity: _Identity


def _secure_capabilities() -> tuple[int, int]:
    """Return descriptor flags, refusing path-resolution fallbacks."""
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if (
        not directory_flag
        or not nofollow_flag
        or os.open not in supports_dir_fd
        or os.stat not in supports_dir_fd
    ):
        raise SecureArchiveError(
            501,
            "Secure archive export requires descriptor-relative no-follow support",
        )
    common = getattr(os, "O_CLOEXEC", 0)
    return directory_flag | nofollow_flag | common, nofollow_flag | common


def _archive_identity(metadata: os.stat_result) -> _Identity:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _identity_matches(actual: _Identity, expected: tuple[int, ...]) -> bool:
    if len(expected) == 2:
        return actual[:2] == (int(expected[0]), int(expected[1]))
    return actual == tuple(int(item) for item in expected)


def _relative_parts(relative: Path) -> tuple[str, ...]:
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise SecureArchiveError(409, "Archive member path is invalid")
    return tuple(relative.parts)


def _path_identity(path: Path, *, directory: bool | None = None) -> _Identity:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SecureArchiveError(409, "Archive source changed during selection") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise SecureArchiveError(409, "Archive source changed to a symlink")
    if directory is True and not stat.S_ISDIR(metadata.st_mode):
        raise SecureArchiveError(409, "Archive parent is not a directory")
    if directory is False and not stat.S_ISREG(metadata.st_mode):
        raise SecureArchiveError(409, "Archive member is not a regular file")
    return _archive_identity(metadata)


def _selected_member(root: Path, path: Path) -> _SelectedArchiveMember:
    relative = path.relative_to(root)
    parts = _relative_parts(relative)
    parent_identities: list[_Identity] = []
    for index in range(1, len(parts)):
        parent_identities.append(
            _path_identity(root.joinpath(*parts[:index]), directory=True)
        )
    identity = _path_identity(path, directory=False)
    return _SelectedArchiveMember(
        relative=relative,
        parts=parts,
        parent_identities=tuple(parent_identities),
        identity=identity,
    )


def _iter_regular_files(root: Path, recursive: bool):
    def walk(directory: Path, prefix: tuple[str, ...]):
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise SecureArchiveError(409, "Archive source changed during selection") from exc
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                continue
            relative_parts = prefix + (entry.name,)
            entry_path = directory / entry.name
            if stat.S_ISDIR(metadata.st_mode):
                if recursive:
                    yield from walk(entry_path, relative_parts)
                continue
            if stat.S_ISREG(metadata.st_mode):
                yield entry_path

    yield from walk(root, ())


def _select_archive_members(
    root: Path,
    *,
    recursive: bool,
    include: Callable[[Path], bool] | None,
    max_files: int | None,
    max_source_bytes: int | None,
) -> tuple[list[_SelectedArchiveMember], int]:
    selected: list[_SelectedArchiveMember] = []
    total_bytes = 0
    for path in _iter_regular_files(root, recursive):
        relative = path.relative_to(root)
        if include is not None and not include(relative):
            continue
        member = _selected_member(root, path)
        selected.append(member)
        total_bytes += member.identity[3]
        if max_files is not None and len(selected) > max_files:
            raise SecureArchiveError(
                413,
                f"Archive contains more than {max_files} files",
            )
        if max_source_bytes is not None and total_bytes > max_source_bytes:
            raise SecureArchiveError(
                413,
                f"Archive source exceeds {max_source_bytes} bytes",
            )
    selected.sort(key=lambda item: item.relative.as_posix())
    return selected, total_bytes


def _open_root(root: Path, directory_flags: int) -> tuple[int, _Identity]:
    try:
        initial = _path_identity(root, directory=True)
        fd = os.open(root, os.O_RDONLY | directory_flags)
        actual = _archive_identity(os.fstat(fd))
        if not _identity_matches(actual, initial) or not stat.S_ISDIR(actual[2]):
            os.close(fd)
            raise SecureArchiveError(409, "Archive root changed during selection")
        return fd, initial
    except SecureArchiveError:
        raise
    except OSError as exc:
        raise SecureArchiveError(409, "Archive root changed during selection") from exc


def _verify_root(root: Path, root_fd: int, expected: _Identity) -> None:
    try:
        descriptor = _archive_identity(os.fstat(root_fd))
        pathname = _path_identity(root, directory=True)
    except OSError as exc:
        raise SecureArchiveError(409, "Archive root changed during streaming") from exc
    if descriptor != expected or pathname != expected:
        raise SecureArchiveError(409, "Archive root changed during streaming")


def _open_member(
    root_fd: int,
    member: _SelectedArchiveMember,
    directory_flags: int,
    file_flags: int,
) -> list[int]:
    descriptors: list[int] = []
    predecessor = root_fd
    try:
        for index, part in enumerate(member.parts[:-1]):
            expected = member.parent_identities[index]
            entry = _archive_identity(os.stat(part, dir_fd=predecessor, follow_symlinks=False))
            if entry != expected or not stat.S_ISDIR(entry[2]):
                raise SecureArchiveError(409, "Archive parent changed during selection")
            child = os.open(part, os.O_RDONLY | directory_flags, dir_fd=predecessor)
            descriptors.append(child)
            actual = _archive_identity(os.fstat(child))
            if actual != expected or actual != entry or not stat.S_ISDIR(actual[2]):
                raise SecureArchiveError(409, "Archive parent changed during open")
            predecessor = child

        leaf_name = member.parts[-1]
        entry = _archive_identity(os.stat(leaf_name, dir_fd=predecessor, follow_symlinks=False))
        if entry != member.identity or not stat.S_ISREG(entry[2]):
            raise SecureArchiveError(409, "Archive member changed during selection")
        leaf = os.open(leaf_name, os.O_RDONLY | file_flags, dir_fd=predecessor)
        descriptors.append(leaf)
        actual = _archive_identity(os.fstat(leaf))
        if actual != member.identity or actual != entry or not stat.S_ISREG(actual[2]):
            raise SecureArchiveError(409, "Archive member changed during open")
        return descriptors
    except SecureArchiveError:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise SecureArchiveError(409, "Archive member changed during open") from exc


def _verify_member(
    root_fd: int,
    descriptors: list[int],
    member: _SelectedArchiveMember,
) -> None:
    predecessor = root_fd
    try:
        for index, part in enumerate(member.parts[:-1]):
            expected = member.parent_identities[index]
            descriptor = _archive_identity(os.fstat(descriptors[index]))
            entry = _archive_identity(os.stat(part, dir_fd=predecessor, follow_symlinks=False))
            if descriptor != expected or entry != expected or descriptor != entry:
                raise SecureArchiveError(409, "Archive parent changed during streaming")
            predecessor = descriptors[index]
        leaf_descriptor = _archive_identity(os.fstat(descriptors[-1]))
        leaf_entry = _archive_identity(
            os.stat(member.parts[-1], dir_fd=predecessor, follow_symlinks=False)
        )
        if (
            leaf_descriptor != member.identity
            or leaf_entry != member.identity
            or leaf_descriptor != leaf_entry
        ):
            raise SecureArchiveError(409, "Archive member changed during streaming")
    except SecureArchiveError:
        raise
    except OSError as exc:
        raise SecureArchiveError(409, "Archive member changed during streaming") from exc


def _reserve_space(
    destination: Path,
    source_bytes: int,
    member_count: int,
    reserve_bytes: int,
) -> None:
    required = int(source_bytes) + int(member_count) * _ZIP_MEMBER_OVERHEAD_BYTES + int(reserve_bytes)
    try:
        free_bytes = int(shutil.disk_usage(destination.parent).free)
    except OSError as exc:
        raise SecureArchiveError(507, "Unable to determine archive staging space") from exc
    if free_bytes < required:
        raise SecureArchiveError(507, "Insufficient disk space for archive staging")


def _write_fd_exact(source_fd: int, destination_handle, expected_size: int) -> None:
    remaining = int(expected_size)
    while remaining:
        try:
            block = os.read(source_fd, min(_SECURE_COPY_CHUNK_BYTES, remaining))
        except OSError as exc:
            raise SecureArchiveError(409, "Archive member changed during streaming") from exc
        if not block:
            raise SecureArchiveError(409, "Archive member changed during streaming")
        destination_handle.write(block)
        remaining -= len(block)


def _validate_owned_destination(
    destination: Path,
    destination_fd: int,
) -> tuple[int, int]:
    """Bind a preallocated staging pathname to the caller-owned descriptor."""
    try:
        descriptor = os.fstat(destination_fd)
        pathname = destination.lstat()
    except OSError as exc:
        raise SecureArchiveError(409, "Archive staging path is unavailable") from exc
    if (
        not stat.S_ISREG(descriptor.st_mode)
        or descriptor.st_size != 0
        or not stat.S_ISREG(pathname.st_mode)
        or _inode_identity(descriptor) != _inode_identity(pathname)
    ):
        raise SecureArchiveError(
            409,
            "Archive staging path changed before construction",
        )
    return _inode_identity(descriptor)


def _verify_owned_destination(
    destination: Path,
    destination_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    try:
        descriptor = os.fstat(destination_fd)
        pathname = destination.lstat()
    except OSError as exc:
        raise SecureArchiveError(409, "Archive staging path changed during construction") from exc
    if (
        not stat.S_ISREG(descriptor.st_mode)
        or not stat.S_ISREG(pathname.st_mode)
        or _inode_identity(descriptor) != expected_identity
        or _inode_identity(pathname) != expected_identity
    ):
        raise SecureArchiveError(409, "Archive staging path changed during construction")


def _unlink_owned_export_paths(
    path: str | Path,
    expected_identity: tuple[int, int],
    *,
    marker_owned: bool = False,
) -> None:
    """Unlink only pathname entries that still name the owned output inode."""
    archive = Path(path)
    marker = export_ownership_marker_path(archive)
    try:
        archive_metadata = archive.lstat()
    except FileNotFoundError:
        archive_metadata = None
    if archive_metadata is not None:
        if (
            stat.S_ISREG(archive_metadata.st_mode)
            and _inode_identity(archive_metadata) == expected_identity
        ):
            archive.unlink(missing_ok=True)

    if not marker_owned:
        return
    try:
        marker_metadata = marker.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(marker_metadata.st_mode)
        and _inode_identity(marker_metadata) == expected_identity
    ):
        marker.unlink(missing_ok=True)


def build_secure_zip(
    source_root: str | Path,
    destination: str | Path,
    *,
    destination_fd: int,
    recursive: bool,
    include: Callable[[Path], bool] | None,
    max_files: int | None,
    max_source_bytes: int | None,
    reserve_bytes: int,
) -> None:
    """Build a bounded ZIP from descriptor-pinned, no-follow source files."""
    root = Path(source_root)
    target = Path(destination)
    success = False
    root_fd: int | None = None
    owned_identity: tuple[int, int] | None = None
    try:
        owned_identity = _validate_owned_destination(target, destination_fd)
        directory_flags, file_flags = _secure_capabilities()
        root_fd, root_identity = _open_root(root, directory_flags)
        selected, total_bytes = _select_archive_members(
            root,
            recursive=recursive,
            include=include,
            max_files=max_files,
            max_source_bytes=max_source_bytes,
        )
        _reserve_space(target, total_bytes, len(selected), reserve_bytes)
        _verify_root(root, root_fd, root_identity)
        _verify_owned_destination(target, destination_fd, owned_identity)
        output_fd = os.dup(destination_fd)
        try:
            os.ftruncate(output_fd, 0)
            os.lseek(output_fd, 0, os.SEEK_SET)
            with os.fdopen(output_fd, "w+b") as output:
                output_fd = -1
                with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                    for member in selected:
                        descriptors = _open_member(root_fd, member, directory_flags, file_flags)
                        try:
                            timestamp = time.localtime(member.identity[4] / 1_000_000_000)[:6]
                            if not 1980 <= timestamp[0] <= 2107:
                                timestamp = (1980, 1, 1, 0, 0, 0)
                            info = zipfile.ZipInfo(member.relative.as_posix(), timestamp)
                            info.compress_type = zipfile.ZIP_DEFLATED
                            info.external_attr = (
                                stat.S_IFREG | stat.S_IMODE(member.identity[2])
                            ) << 16
                            info.file_size = member.identity[3]
                            with archive.open(info, "w", force_zip64=True) as member_output:
                                _write_fd_exact(descriptors[-1], member_output, member.identity[3])
                            _verify_member(root_fd, descriptors, member)
                            _verify_root(root, root_fd, root_identity)
                            _verify_owned_destination(target, destination_fd, owned_identity)
                        finally:
                            for descriptor in reversed(descriptors):
                                os.close(descriptor)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output_fd >= 0:
                os.close(output_fd)
        _verify_owned_destination(target, destination_fd, owned_identity)
        success = True
    except SecureArchiveError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise SecureArchiveError(409, "Archive source changed during construction") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if not success and owned_identity is not None:
            _unlink_owned_export_paths(target, owned_identity)


def copy_secure_file(
    source_root: str | Path,
    relative_path: str | Path,
    destination: str | Path,
    *,
    destination_fd: int,
    expected_identity: tuple[int, ...],
    reserve_bytes: int,
) -> None:
    """Copy one selected regular file without reopening it by pathname."""
    root = Path(source_root)
    relative = Path(relative_path)
    target = Path(destination)
    parts = _relative_parts(relative)
    source_path = root.joinpath(*parts)
    success = False
    root_fd: int | None = None
    owned_identity: tuple[int, int] | None = None
    try:
        owned_identity = _validate_owned_destination(target, destination_fd)
        directory_flags, file_flags = _secure_capabilities()
        root_fd, root_identity = _open_root(root, directory_flags)
        member = _selected_member(root, source_path)
        if not _identity_matches(member.identity, expected_identity):
            raise SecureArchiveError(409, "Archive member changed during selection")
        _reserve_space(target, member.identity[3], 0, reserve_bytes)
        _verify_root(root, root_fd, root_identity)
        _verify_owned_destination(target, destination_fd, owned_identity)
        descriptors = _open_member(root_fd, member, directory_flags, file_flags)
        try:
            output_fd = os.dup(destination_fd)
            try:
                os.ftruncate(output_fd, 0)
                os.lseek(output_fd, 0, os.SEEK_SET)
                with os.fdopen(output_fd, "wb") as output:
                    output_fd = -1
                    _write_fd_exact(descriptors[-1], output, member.identity[3])
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                if output_fd >= 0:
                    os.close(output_fd)
            _verify_member(root_fd, descriptors, member)
            _verify_root(root, root_fd, root_identity)
            _verify_owned_destination(target, destination_fd, owned_identity)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
        success = True
    except SecureArchiveError:
        raise
    except OSError as exc:
        raise SecureArchiveError(409, "Archive source changed during copy") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if not success and owned_identity is not None:
            _unlink_owned_export_paths(target, owned_identity)


def export_ownership_marker_path(path: str | Path) -> Path:
    """Return the durable ownership sidecar for one export archive."""
    archive = Path(path)
    return archive.with_name(f"{archive.name}{EXPORT_OWNERSHIP_MARKER_SUFFIX}")


def _regular_file_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(f"export archive is not a regular file: {path}")
    return int(metadata.st_dev), int(metadata.st_ino)


def create_export_ownership_marker(
    path: str | Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> Path:
    """Persist proof that this process exclusively created *path*.

    Prefixes identify candidate export names but do not establish ownership: a
    pre-existing conflict can have the same name.  The marker binds the exact
    inode allocated by the export producer so restart cleanup never has to
    infer ownership from a filename alone.
    """
    archive = Path(path)
    archive_identity = _regular_file_identity(archive)
    if expected_identity is not None and archive_identity != tuple(expected_identity):
        raise OSError("export archive inode does not match the owned staging descriptor")
    marker = export_ownership_marker_path(archive)
    marker_created = False
    try:
        # A hard link is both the marker and the ownership proof. Keeping the
        # original inode linked prevents an unlink/recreate ABA from reusing
        # its inode while this lifecycle record remains live.
        os.link(archive, marker, follow_symlinks=False)
        marker_created = True
        if _regular_file_identity(marker) != archive_identity:
            raise OSError("export ownership hard link does not match archive")
        directory_fd = os.open(archive.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if marker_created:
            _unlink_owned_export_paths(
                archive,
                archive_identity,
                marker_owned=True,
            )
        raise
    return marker


def export_ownership_marker_matches(path: str | Path) -> bool:
    """Return whether the sidecar proves ownership of the current inode."""
    archive = Path(path)
    marker = export_ownership_marker_path(archive)
    try:
        archive_metadata = archive.lstat()
        marker_metadata = marker.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(archive_metadata.st_mode)
        and stat.S_ISREG(marker_metadata.st_mode)
        and archive_metadata.st_dev == marker_metadata.st_dev
        and archive_metadata.st_ino == marker_metadata.st_ino
    )


def unlink_export_archive_and_marker(
    path: str | Path,
    *,
    require_marker: bool = False,
    remove_marker: bool = False,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """Remove an archive only under the caller's explicit ownership proof."""
    archive = Path(path)
    marker = export_ownership_marker_path(archive)
    owned_identity = (
        tuple(int(part) for part in expected_identity)
        if expected_identity is not None
        else None
    )
    try:
        metadata = archive.lstat()
    except FileNotFoundError:
        metadata = None

    actual_identity = _inode_identity(metadata) if metadata is not None else None
    archive_is_owned = bool(
        metadata is not None
        and stat.S_ISREG(metadata.st_mode)
        and (owned_identity is None or actual_identity == owned_identity)
    )
    if (
        require_marker
        and metadata is not None
        and not export_ownership_marker_matches(archive)
    ):
        raise OSError(
            f"export ownership marker does not match current archive inode: {archive}"
        )
    if archive_is_owned:
        archive.unlink(missing_ok=True)

    if not remove_marker:
        return
    if owned_identity is None:
        marker.unlink(missing_ok=True)
        return
    try:
        marker_metadata = marker.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(marker_metadata.st_mode)
        and _inode_identity(marker_metadata) == owned_identity
    ):
        marker.unlink(missing_ok=True)


def _current_task() -> asyncio.Task:
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("export lifecycle operation requires an asyncio task")
    return task


def _record_generation(info: dict[str, Any]) -> int:
    return int(info.get("generation") or 0)


def export_record_matches_generation(
    info: dict[str, Any],
    session_id: str,
    generation: int | None,
) -> bool:
    """Return whether *info* belongs to the exact session generation.

    Generation-less records are retained for cleanup compatibility with
    archives prepared by releases before lifecycle ownership was recorded.
    """
    if str(info.get("session_id") or "") != session_id:
        return False
    recorded = _record_generation(info)
    return generation is None or recorded in (0, int(generation))


async def reserve_export_record(
    token: str,
    *,
    session_id: str,
    user_id: str,
    state: Any,
    kind: str,
) -> dict[str, Any]:
    """Publish a build owner before its first cancellable operation."""
    task = _current_task()
    generation = int(getattr(state, "lifecycle_generation", 0) or 0)
    info: dict[str, Any] = {
        "token": token,
        "path": "",
        "filename": "",
        "size": 0,
        "session_id": session_id,
        "user_id": user_id,
        "generation": generation,
        "state_ref": state,
        "kind": kind,
        "phase": "building",
        "task": task,
        "cancel_requested": False,
        "ownership_marker_required": False,
        "ownership_marker_owned": False,
        "created_at": time.time(),
    }
    async with get_lifecycle_lock(session_id):
        assert_generation_writable(session_id, state)
        current = sessions.get(session_id)
        if current is not None and current is not state:
            raise RuntimeError("session generation changed before export reservation")
        # assert_generation_writable may adopt a pre-lifecycle generation.
        info["generation"] = int(
            getattr(state, "lifecycle_generation", 0) or 0
        )
        async with prepared_exports_lock:
            if token in prepared_exports:
                raise RuntimeError("export token collision")
            prepared_exports[token] = info
    return info


def generation_bound_export_prefix(prefix: str, info: dict[str, Any]) -> str:
    """Return a restart-sweepable prefix carrying session and generation."""
    return (
        f"{prefix}{info['session_id']}.g{_record_generation(info)}."
    )


async def bind_export_path(
    info: dict[str, Any],
    path: str | Path,
    *,
    owned: bool = True,
) -> None:
    """Attach the newly allocated temp path to its exact build owner."""
    session_id = str(info["session_id"])
    state = info.get("state_ref")
    async with get_lifecycle_lock(session_id):
        if state is not None:
            assert_generation_writable(session_id, state)
        async with prepared_exports_lock:
            if prepared_exports.get(str(info["token"])) is not info:
                raise RuntimeError("export owner was retired before path binding")
            if info.get("cancel_requested"):
                raise asyncio.CancelledError()
            info["path"] = str(path)
            info["path_owned"] = bool(owned)


async def publish_export_ready(
    info: dict[str, Any],
    *,
    filename: str,
    size: int,
) -> None:
    """Atomically publish a completed archive to token redemption."""
    session_id = str(info["session_id"])
    state = info.get("state_ref")
    async with get_lifecycle_lock(session_id):
        if state is not None:
            assert_generation_writable(session_id, state)
        current = sessions.get(session_id)
        if current is not None and state is not None and current is not state:
            raise RuntimeError("session generation changed before export publication")
        async with prepared_exports_lock:
            if prepared_exports.get(str(info["token"])) is not info:
                raise RuntimeError("export owner was retired before publication")
            if info.get("cancel_requested"):
                raise asyncio.CancelledError()
            info.update({
                "filename": filename,
                "size": int(size),
                "phase": "ready",
                "task": None,
                "ready_at": time.time(),
            })


async def claim_export_stream(
    token: str,
    session_id: str,
) -> dict[str, Any] | None:
    """Consume a ready token into one tracked streaming owner."""
    task = _current_task()
    async with get_lifecycle_lock(session_id):
        async with prepared_exports_lock:
            info = prepared_exports.get(token)
            if info is None or str(info.get("session_id") or "") != session_id:
                return None
            # Missing phase is a compatibility record produced before this
            # lifecycle layer. It can still be redeemed once and then retired.
            if str(info.get("phase") or "ready") != "ready":
                return None
            state = info.get("state_ref")
            if state is not None:
                assert_generation_writable(session_id, state)
            if info.get("cancel_requested"):
                return None
            info.setdefault("token", token)
            info["phase"] = "claimed"
            info["task"] = task
            return info


async def _begin_stream(info: dict[str, Any]) -> None:
    """Revalidate a claimed token immediately before opening its file."""
    session_id = str(info["session_id"])
    current_task = _current_task()
    async with get_lifecycle_lock(session_id):
        async with prepared_exports_lock:
            if prepared_exports.get(str(info["token"])) is not info:
                raise RuntimeError("export stream owner was retired")
            if info.get("task") is not current_task:
                raise RuntimeError("export stream task ownership changed")
            state = info.get("state_ref")
            if state is not None:
                assert_generation_writable(session_id, state)
            if info.get("cancel_requested"):
                raise asyncio.CancelledError()
            info["phase"] = "streaming"


async def cleanup_export_record(
    info: dict[str, Any],
    *,
    retain_failure: bool = True,
    registry: dict[str, dict[str, Any]] | None = None,
    registry_lock: asyncio.Lock | None = None,
) -> bool:
    """Unlink one exact archive before retiring its lifecycle record."""
    target_registry = prepared_exports if registry is None else registry
    target_lock = prepared_exports_lock if registry_lock is None else registry_lock
    token = str(info.get("token") or "")
    path = str(info.get("path") or "")
    path_owned = info.get("path_owned", True) is not False
    marker_required = info.get("ownership_marker_required") is True
    marker_owned = info.get("ownership_marker_owned") is True
    try:
        if path and path_owned:
            await run_in_owned_thread(
                unlink_export_archive_and_marker,
                path,
                require_marker=marker_required,
                remove_marker=marker_owned,
            )
    except OSError as exc:
        async with target_lock:
            if target_registry.get(token) is info:
                info["phase"] = "cleanup_failed"
                info["task"] = None
                info["cleanup_error"] = str(exc)
        logger.warning("Failed to clean export archive %s: %s", path, exc)
        if not retain_failure:
            raise
        return False

    async with target_lock:
        if target_registry.get(token) is info:
            target_registry.pop(token, None)
    return True


async def abandon_export_record(info: dict[str, Any]) -> bool:
    """Cancellation-safely clean a build that never became ready."""
    cleanup = asyncio.create_task(
        cleanup_export_record(info),
        name=f"export-build-cleanup:{info.get('session_id')}:{info.get('token')}",
    )
    return await await_with_deferred_cancellation(cleanup)


async def mark_session_exports_for_deletion(
    session_id: str,
    generation: int,
) -> list[asyncio.Task]:
    """Mark exact-generation export producers and return their task handles.

    The caller owns ``get_lifecycle_lock(session_id)``. Keeping this mutation
    under that lock makes reservation and tombstoning reciprocal.
    """
    tasks: list[asyncio.Task] = []
    async with prepared_exports_lock:
        for info in prepared_exports.values():
            if not export_record_matches_generation(info, session_id, generation):
                continue
            info["cancel_requested"] = True
            task = info.get("task")
            if isinstance(task, asyncio.Task) and not task.done():
                tasks.append(task)
    return tasks


async def cleanup_session_export_records(
    session_id: str,
    generation: int,
) -> None:
    """Remove every drained archive for one tombstoned generation."""
    async with prepared_exports_lock:
        records = [
            info for info in prepared_exports.values()
            if export_record_matches_generation(info, session_id, generation)
        ]
        live = [
            info for info in records
            if isinstance(info.get("task"), asyncio.Task)
            and not info["task"].done()
        ]
    if live:
        raise RuntimeError(
            f"{len(live)} export producer(s) still live for {session_id}"
        )

    failures = []
    for info in records:
        if not await cleanup_export_record(info):
            failures.append(str(info.get("path") or info.get("token") or ""))
    if failures:
        raise OSError(
            f"failed to remove {len(failures)} export archive(s); "
            f"first path: {failures[0]}"
        )


class LifecycleFileResponse(FileResponse):
    """FileResponse whose send and unlink remain visible to session deletion."""

    def __init__(self, *args, export_info: dict[str, Any], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._export_info = export_info

    async def __call__(self, scope, receive, send) -> None:
        try:
            await _begin_stream(self._export_info)
            await super().__call__(scope, receive, send)
        finally:
            finalizer = asyncio.create_task(
                cleanup_export_record(self._export_info),
                name=(
                    "export-stream-cleanup:"
                    f"{self._export_info.get('session_id')}:"
                    f"{self._export_info.get('token')}"
                ),
            )
            await await_with_deferred_cancellation(finalizer)


class OwnedArchiveFileResponse(FileResponse):
    """Stream one already-open archive inode and retire only that inode."""

    def __init__(
        self,
        path: str | Path,
        *,
        file_fd: int,
        owned_identity: tuple[int, int],
        marker_owned: bool = False,
        **kwargs,
    ) -> None:
        self._owned_fd = int(file_fd)
        self._owned_identity = tuple(int(item) for item in owned_identity)
        self._marker_owned = bool(marker_owned)
        stat_result = os.fstat(self._owned_fd)
        super().__init__(path, stat_result=stat_result, **kwargs)

    async def _send_owned_body(
        self,
        send,
        *,
        start: int,
        end: int,
        final_more_body: bool = False,
    ) -> None:
        offset = int(start)
        limit = int(end)
        if offset >= limit:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": final_more_body,
            })
            return
        while offset < limit:
            try:
                block = os.pread(
                    self._owned_fd,
                    min(self.chunk_size, limit - offset),
                    offset,
                )
            except OSError as exc:
                raise RuntimeError("Owned archive became unreadable") from exc
            if not block:
                raise RuntimeError("Owned archive ended before the promised range")
            offset += len(block)
            await send({
                "type": "http.response.body",
                "body": block,
                "more_body": offset < limit or final_more_body,
            })

    async def _handle_owned_simple(self, send, send_header_only: bool) -> None:
        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        })
        if send_header_only:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            return
        await self._send_owned_body(send, start=0, end=self.stat_result.st_size)

    async def _handle_owned_single_range(
        self,
        send,
        start: int,
        end: int,
        file_size: int,
        send_header_only: bool,
    ) -> None:
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["content-length"] = str(end - start)
        await send({
            "type": "http.response.start",
            "status": 206,
            "headers": headers.raw,
        })
        if send_header_only:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            return
        await self._send_owned_body(send, start=start, end=end)

    async def _handle_owned_multiple_ranges(
        self,
        send,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
        boundary = os.urandom(13).hex()
        content_length, header_generator = self.generate_multipart(
            ranges,
            boundary,
            file_size,
            self.headers["content-type"],
        )
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
        headers["content-length"] = str(content_length)
        await send({
            "type": "http.response.start",
            "status": 206,
            "headers": headers.raw,
        })
        if send_header_only:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            return
        for start, end in ranges:
            await send({
                "type": "http.response.body",
                "body": header_generator(start, end),
                "more_body": True,
            })
            await self._send_owned_body(
                send,
                start=start,
                end=end,
                final_more_body=True,
            )
            await send({
                "type": "http.response.body",
                "body": b"\r\n",
                "more_body": True,
            })
        await send({
            "type": "http.response.body",
            "body": f"--{boundary}--".encode("latin-1"),
            "more_body": False,
        })

    async def __call__(self, scope, receive, send) -> None:
        try:
            scope_type = scope["type"]
            send_header_only = (
                scope_type == "http" and scope["method"].upper() == "HEAD"
            )
            if scope_type == "websocket":
                send = self._wrap_websocket_denial_send(send)

            headers = Headers(scope=scope)
            http_range = headers.get("range")
            http_if_range = headers.get("if-range")
            if http_range is None or (
                http_if_range is not None and not self._should_use_range(http_if_range)
            ):
                await self._handle_owned_simple(send, send_header_only)
            else:
                try:
                    ranges = self._parse_range_header(
                        http_range,
                        self.stat_result.st_size,
                    )
                except MalformedRangeHeader as exc:
                    await PlainTextResponse(exc.content, status_code=400)(
                        scope,
                        receive,
                        send,
                    )
                except RangeNotSatisfiable as exc:
                    await PlainTextResponse(
                        status_code=416,
                        headers={"Content-Range": f"bytes */{exc.max_size}"},
                    )(scope, receive, send)
                else:
                    if not ranges:
                        await self._handle_owned_simple(send, send_header_only)
                    elif len(ranges) == 1:
                        start, end = ranges[0]
                        await self._handle_owned_single_range(
                            send,
                            start,
                            end,
                            self.stat_result.st_size,
                            send_header_only,
                        )
                    else:
                        await self._handle_owned_multiple_ranges(
                            send,
                            ranges,
                            self.stat_result.st_size,
                            send_header_only,
                        )

            if self.background is not None:
                await self.background()
        finally:
            file_fd = self._owned_fd
            self._owned_fd = -1
            try:
                if file_fd >= 0:
                    os.close(file_fd)
            finally:
                cleanup = asyncio.create_task(
                    run_in_owned_thread(
                        _unlink_owned_export_paths,
                        self.path,
                        self._owned_identity,
                        marker_owned=self._marker_owned,
                    ),
                    name=f"owned-archive-cleanup:{self.path}",
                )
                await await_with_deferred_cancellation(cleanup)
