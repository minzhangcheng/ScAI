"""One tombstone-first deletion path shared by legacy and V1 APIs."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import weakref
from pathlib import Path

from fastapi import HTTPException

from backend.config import (
    REPORTS_DIR,
    SESSION_DIR,
    STATE_DIR,
    UPLOAD_DIR,
    WORKSPACE_DIR,
    INTEGRITY_DIR,
    get_session_dir,
    resolve_session_metadata_path,
)
from backend.services.session_lifecycle import (
    abandon_import_generation,
    adopt_persisted_generation,
    complete_tombstone_cleanup,
    get_lifecycle_lock,
    lifecycle_file_lock,
    tombstone_generation,
    read_lifecycle_record,
    list_lifecycle_records,
    SessionGoneError,
)
from backend.state import (
    _monitor_tasks,
    _run_tasks,
    chat_sandboxes,
    chat_task_lock,
    chat_tasks,
    detailed_report_tasks,
    has_live_session_task,
    initializing_sessions,
    last_save_times,
    prepared_exports,
    prepared_exports_lock,
    report_workflows,
    report_websockets,
    release_session_lock_slot,
    release_initializing_session,
    session_access_times,
    sessions,
    state_lock,
    websockets,
    workflows,
)
from backend.utils.files import legacy_session_state_paths, validate_session_id
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    consume_current_cancellation_request,
    run_in_owned_thread,
)
from backend.services.session_deletion_parts import (
    finalize as _finalize_owner,
    tasks as _tasks_owner,
)
from backend.services.export_lifecycle import (
    cleanup_session_export_records,
    mark_session_exports_for_deletion,
)


logger = logging.getLogger(__name__)
_PRODUCER_STOP_TIMEOUT_SECONDS = 10.0


from backend.services.session_deletion_parts.models import (
    SessionDeletionIdentity,
    SessionProducerStopError,
    _PendingTombstoneCleanup,
    _TombstoneCleanupLease,
    _claim_local_cleanup_lease,
    _cleanup_locks,
    _cleanup_locks_guard,
    _local_cleanup_leases,
    _local_cleanup_leases_guard,
    _local_cleanup_leases_pid,
    _release_local_cleanup_lease,
    _pending_tombstone_cleanup,
)

SessionProducerStopError.__module__ = __name__
_PendingTombstoneCleanup.__module__ = __name__
_TombstoneCleanupLease.__module__ = __name__


def _read_deletion_identity(session_id: str) -> SessionDeletionIdentity | None:
    try:
        data = json.loads(resolve_session_metadata_path(session_id).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("id", session_id) != session_id:
            return None
        owner_id = data.get("user_id")
        if not isinstance(owner_id, str) or not owner_id:
            return None
        generation = int(data.get("lifecycle_generation") or 0)
        if generation < 0:
            return None
        return SessionDeletionIdentity(session_id, owner_id, generation)
    except (OSError, ValueError, TypeError):
        return None


async def load_session_deletion_identity(
    session_id: str,
) -> SessionDeletionIdentity | None:
    """Resolve deletion ownership without loading or repairing analysis state."""
    validate_session_id(session_id)
    async with get_lifecycle_lock(session_id):
        record = read_lifecycle_record(session_id)
        if record is not None:
            owner_id = str(record.get("owner_id") or "")
            generation = int(record.get("generation") or 0)
            if record.get("status") == "deleted":
                raise SessionGoneError(session_id, owner_id, generation)
            if record.get("status") != "active" or not owner_id or generation < 1:
                return None
            return SessionDeletionIdentity(session_id, owner_id, generation)

        async with state_lock:
            cached = sessions.get(session_id)
            identity = (
                SessionDeletionIdentity(
                    session_id, cached.user_id, cached.lifecycle_generation,
                )
                if cached is not None and cached.user_id
                else None
            )
        if identity is None:
            identity = await run_in_owned_thread(_read_deletion_identity, session_id)
        if identity is not None:
            # Pin legacy metadata to a durable generation before releasing the
            # lock, so a delayed deletion cannot target a later same-ID import.
            adopt_persisted_generation(identity)
        return identity


def _get_cleanup_lock(session_id: str) -> asyncio.Lock:
    """Return a loop-local lock covering the complete tombstone state machine."""
    validate_session_id(session_id)
    loop = asyncio.get_running_loop()
    with _cleanup_locks_guard:
        loop_locks = _cleanup_locks.get(loop)
        if loop_locks is None:
            loop_locks = weakref.WeakValueDictionary()
            _cleanup_locks[loop] = loop_locks
        lock = loop_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            loop_locks[session_id] = lock
        return lock


def _try_acquire_cleanup_lease_sync(
    session_id: str,
) -> _TombstoneCleanupLease | None:
    """Try to own cleanup across processes without blocking the event loop."""
    import fcntl

    claimed, pid = _claim_local_cleanup_lease(session_id)
    if not claimed:
        return None
    path = (
        SESSION_DIR
        / ".lifecycle"
        / ".cleanup_locks"
        / f"{session_id}.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except BaseException:
        _release_local_cleanup_lease(session_id, pid)
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        _release_local_cleanup_lease(session_id, pid)
        return None
    except BaseException:
        os.close(fd)
        _release_local_cleanup_lease(session_id, pid)
        raise
    return _TombstoneCleanupLease(session_id=session_id, fd=fd, pid=pid)


async def _try_acquire_cleanup_lease(
    session_id: str,
) -> _TombstoneCleanupLease | None:
    # Keep the acquired handle visible even when caller cancellation arrives
    # while the non-blocking flock attempt is in the worker.  The owned-thread
    # helper drains and joins the worker before propagating that cancellation;
    # releasing the retained result here prevents an unreachable live lease.
    acquired: list[_TombstoneCleanupLease | None] = []

    def _acquire() -> _TombstoneCleanupLease | None:
        lease = _try_acquire_cleanup_lease_sync(session_id)
        acquired.append(lease)
        return lease

    try:
        return await run_in_owned_thread(_acquire)
    except asyncio.CancelledError:
        if acquired and acquired[0] is not None:
            acquired[0].release()
        raise


def _target_matches(
    record: dict | None,
    owner_id: str,
    generation: int,
    *,
    require_incomplete: bool = True,
) -> bool:
    if not record or record.get("status") != "deleted":
        return False
    if str(record.get("owner_id") or "") != owner_id:
        return False
    if int(record.get("generation") or 0) != int(generation or 0):
        return False
    return not require_incomplete or record.get("cleanup_complete") is not True


def _empty_cleanup_stats(*, pending: int = 0) -> dict[str, int]:
    return {
        "deleted_sessions": 0,
        "deleted_state_files": 0,
        "pending_producers": pending,
    }


async def _cancel_and_wait(tasks: list[asyncio.Task]) -> list[asyncio.Task]:
    return await _tasks_owner._cancel_and_wait(tasks, bindings=sys.modules[__name__])



def _start_cleanup_task(closer, *, name: str) -> asyncio.Task | None:
    return _tasks_owner._start_cleanup_task(closer, name=name, bindings=sys.modules[__name__])



def _cleanup_task_needs_retry(task: asyncio.Task | None) -> bool:
    """Whether a completed cleanup attempt failed to prove shutdown."""
    return _tasks_owner._cleanup_task_needs_retry(task, bindings=sys.modules[__name__])



def _refresh_cleanup_tasks(
    handles: _PendingTombstoneCleanup,
    session_id: str,
    *,
    retry_completed: bool = True,
) -> None:
    """Start a new owned attempt only after the prior one has completed.

    Long-running closers are observed, never duplicated. A completed attempt
    that raised or returned literal False is different: the retained producer
    handle exists precisely so a later tombstone sweep can call its idempotent
    close contract again.
    """
    return _tasks_owner._refresh_cleanup_tasks(handles, session_id, retry_completed=retry_completed, bindings=sys.modules[__name__])



async def _await_cleanup_task(
    task: asyncio.Task | None,
    *,
    label: str,
    session_id: str,
) -> bool:
    """Boundedly observe an owned cleanup task without cancelling it.

    `asyncio.wait_for(coro, timeout)` cancels the closer at the deadline. A
    closer that catches CancelledError can then make wait_for wait forever; a
    closer that does not catch it is still destroyed before it proves the
    producer stopped. The independently owned task remains in the tombstone
    registry. Retries observe that exact task while it is running; only a
    completed attempt that failed to prove shutdown is replaced.
    """
    return await _tasks_owner._await_cleanup_task(task, label=label, session_id=session_id, bindings=sys.modules[__name__])



async def _delete_tombstoned_files(
    session_id: str,
    owner_id: str | None = None,
    generation: int | None = None,
) -> dict[str, int]:
    """Delete one exact tombstoned tree and publish cleanup completion."""
    return await _finalize_owner._delete_tombstoned_files(session_id, owner_id, generation, bindings=sys.modules[__name__])



async def _drain_tombstone_producers(
    session_id: str,
    owner_id: str,
    generation: int,
) -> int:
    """Observe one shared set of producer handles for this tombstone."""
    return await _tasks_owner._drain_tombstone_producers(session_id, owner_id, generation, bindings=sys.modules[__name__])



async def _cleanup_tombstone_side_surfaces(
    session_id: str,
    generation: int,
) -> None:
    """Idempotently remove non-session-tree resources after tombstoning."""
    return await _tasks_owner._cleanup_tombstone_side_surfaces(session_id, generation, bindings=sys.modules[__name__])



async def _ensure_cleanup_handles(
    session_id: str,
    owner_id: str,
    generation: int,
) -> _PendingTombstoneCleanup | None:
    """Return target-bound handles carrying the cross-process cleanup lease."""
    return await _tasks_owner._ensure_cleanup_handles(session_id, owner_id, generation, bindings=sys.modules[__name__])



def _release_cleanup_handles(
    session_id: str,
    handles: _PendingTombstoneCleanup,
) -> None:
    return _tasks_owner._release_cleanup_handles(session_id, handles, bindings=sys.modules[__name__])



def _release_completed_target_handles(
    session_id: str,
    owner_id: str,
    generation: int,
) -> None:
    """Recover an exact lease orphaned after durable cleanup completion."""
    return _tasks_owner._release_completed_target_handles(session_id, owner_id, generation, bindings=sys.modules[__name__])



def cleanup_failed_import_generation(session_id: str) -> bool:
    """Synchronously retire one uncommitted import under the cleanup lease.

    Import extraction is synchronous and may fail before an async cleanup task
    can be registered. Acquiring the same cross-process lease *before* writing
    the tombstone prevents a background sweeper from observing an incomplete
    record and racing this cleanup into a later same-ID generation.
    """
    return _finalize_owner.cleanup_failed_import_generation(session_id, bindings=sys.modules[__name__])



async def _finalize_tombstone_target(
    session_id: str,
    owner_id: str,
    generation: int,
    handles: _PendingTombstoneCleanup,
) -> dict[str, int]:
    """Publish cleanup completion and release its lease as one owned task."""
    return await _finalize_owner._finalize_tombstone_target(session_id, owner_id, generation, handles, bindings=sys.modules[__name__])



async def _await_cleanup_finalizer(
    finalizer: asyncio.Task,
    *,
    session_id: str,
) -> dict[str, int]:
    """Defer caller cancellation until completion and lease release finish."""
    return await _finalize_owner._await_cleanup_finalizer(finalizer, session_id=session_id, bindings=sys.modules[__name__])



async def _cleanup_tombstone_target_locked(
    session_id: str,
    target_owner: str,
    target_generation: int,
) -> dict[str, int]:
    """Clean one captured target while the local cleanup lock is held."""
    return await _finalize_owner._cleanup_tombstone_target_locked(session_id, target_owner, target_generation, bindings=sys.modules[__name__])



async def cleanup_tombstoned_generation(
    session_id: str,
    owner_id: str | None = None,
) -> dict[str, int]:
    """Retry physical cleanup without reviving the deleted generation."""
    return await _finalize_owner.cleanup_tombstoned_generation(session_id, owner_id, bindings=sys.modules[__name__])



async def sweep_tombstoned_generations() -> dict[str, int]:
    """Best-effort physical cleanup for every stable tombstone."""
    return await _finalize_owner.sweep_tombstoned_generations(bindings=sys.modules[__name__])



async def _tombstone_detach_and_publish(
    session_id: str,
    state,
    owner_id: str,
    *,
    force_stop: bool,
    lease: _TombstoneCleanupLease,
) -> tuple[str, int]:
    """Owned linearization step from active generation to cleanup handles."""
    return await _finalize_owner._tombstone_detach_and_publish(session_id, state, owner_id, force_stop=force_stop, lease=lease, bindings=sys.modules[__name__])



async def delete_session_generation(
    session_id: str,
    state,
    owner_id: str,
    *,
    force_stop: bool = False,
) -> dict[str, int]:
    """Persist a stable tombstone, stop producers, then remove every surface.

    Interactive deletion remains conservative and rejects a running workflow.
    Account-level cascade deletion uses ``force_stop=True``: the tombstone is
    still the linearization point, so cancelled producers cannot save after it.
    """
    return await _finalize_owner.delete_session_generation(session_id, state, owner_id, force_stop=force_stop, bindings=sys.modules[__name__])
