"""Generation-bound lifecycle ownership for session export archives."""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any

from starlette.responses import FileResponse

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


def export_ownership_marker_path(path: str | Path) -> Path:
    """Return the durable ownership sidecar for one export archive."""
    archive = Path(path)
    return archive.with_name(f"{archive.name}{EXPORT_OWNERSHIP_MARKER_SUFFIX}")


def _regular_file_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(f"export archive is not a regular file: {path}")
    return int(metadata.st_dev), int(metadata.st_ino)


def create_export_ownership_marker(path: str | Path) -> Path:
    """Persist proof that this process exclusively created *path*.

    Prefixes identify candidate export names but do not establish ownership: a
    pre-existing conflict can have the same name.  The marker binds the exact
    inode allocated by the export producer so restart cleanup never has to
    infer ownership from a filename alone.
    """
    archive = Path(path)
    _regular_file_identity(archive)
    marker = export_ownership_marker_path(archive)
    marker_created = False
    try:
        # A hard link is both the marker and the ownership proof. Keeping the
        # original inode linked prevents an unlink/recreate ABA from reusing
        # its inode while this lifecycle record remains live.
        os.link(archive, marker, follow_symlinks=False)
        marker_created = True
        if _regular_file_identity(marker) != _regular_file_identity(archive):
            raise OSError("export ownership hard link does not match archive")
        directory_fd = os.open(archive.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if marker_created:
            marker.unlink(missing_ok=True)
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
) -> None:
    """Remove an archive only under the caller's explicit ownership proof."""
    archive = Path(path)
    try:
        archive.lstat()
    except FileNotFoundError:
        if remove_marker:
            export_ownership_marker_path(archive).unlink(missing_ok=True)
        return
    if require_marker and not export_ownership_marker_matches(archive):
        raise OSError(
            f"export ownership marker does not match current archive inode: {archive}"
        )
    archive.unlink(missing_ok=True)
    if remove_marker:
        export_ownership_marker_path(archive).unlink(missing_ok=True)


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
