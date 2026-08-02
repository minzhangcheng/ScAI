"""One tombstone-first deletion path shared by legacy and V1 APIs."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import weakref
from dataclasses import dataclass, field
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
)
from backend.services.session_lifecycle import (
    abandon_import_generation,
    complete_tombstone_cleanup,
    get_lifecycle_lock,
    lifecycle_file_lock,
    tombstone_generation,
    read_lifecycle_record,
    list_lifecycle_records,
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
    snapshot_restore_jobs,
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
from backend.services.export_lifecycle import (
    cleanup_session_export_records,
    mark_session_exports_for_deletion,
)


logger = logging.getLogger(__name__)
_PRODUCER_STOP_TIMEOUT_SECONDS = 10.0


class SessionProducerStopError(RuntimeError):
    """Deletion tombstoned a session but could not stop every producer."""


@dataclass
class _TombstoneCleanupLease:
    """Process-crash-safe ownership of one session's physical cleanup."""

    session_id: str
    fd: int
    pid: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        if os.getpid() != self.pid:
            # A forked child shares the inherited open-file-description. An
            # explicit LOCK_UN there would release the parent's live lease.
            os.close(self.fd)
            return
        try:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self.fd)
            finally:
                _release_local_cleanup_lease(self.session_id, self.pid)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


@dataclass
class _PendingTombstoneCleanup:
    owner_id: str = ""
    generation: int = 0
    lease: _TombstoneCleanupLease | None = None
    tasks: list[asyncio.Task] = field(default_factory=list)
    workflow_task: asyncio.Task | None = None
    report_workflow_task: asyncio.Task | None = None
    chat_task: asyncio.Task | None = None
    workflow: object | None = None
    report_workflow: object | None = None
    chat_sandbox: object | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    drain_started: bool = False


# Tombstoned generations are never made readable again. This registry retains
# only producer handles needed for a later cleanup retry in the same process;
# after a restart the producers are gone and the disk sweep can proceed.
_pending_tombstone_cleanup: dict[str, _PendingTombstoneCleanup] = {}

_cleanup_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    weakref.WeakValueDictionary[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()
_cleanup_locks_guard = threading.Lock()
_local_cleanup_leases: set[str] = set()
_local_cleanup_leases_guard = threading.Lock()
_local_cleanup_leases_pid = os.getpid()


def _claim_local_cleanup_lease(session_id: str) -> tuple[bool, int]:
    """Claim one process-local lease before relying on ``flock`` semantics."""
    global _local_cleanup_leases_pid
    pid = os.getpid()
    with _local_cleanup_leases_guard:
        # A post-fork child inherits Python memory but not the parent's logical
        # ownership. The kernel flock remains the cross-process authority.
        if _local_cleanup_leases_pid != pid:
            _local_cleanup_leases.clear()
            _local_cleanup_leases_pid = pid
        if session_id in _local_cleanup_leases:
            return False, pid
        _local_cleanup_leases.add(session_id)
        return True, pid


def _release_local_cleanup_lease(session_id: str, pid: int) -> None:
    global _local_cleanup_leases_pid
    with _local_cleanup_leases_guard:
        if _local_cleanup_leases_pid == pid == os.getpid():
            _local_cleanup_leases.discard(session_id)


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
    current = asyncio.current_task()
    pending = [task for task in tasks if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        done, still_pending = await asyncio.wait(
            pending, timeout=_PRODUCER_STOP_TIMEOUT_SECONDS,
        )
        for task in done:
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                pass
        if still_pending:
            logger.warning(
                "Session deletion timed out waiting for %d task(s)",
                len(still_pending),
            )
            return list(still_pending)
    return []


def _start_cleanup_task(closer, *, name: str) -> asyncio.Task | None:
    if closer is None:
        return None
    return asyncio.create_task(closer(), name=name)


def _cleanup_task_needs_retry(task: asyncio.Task | None) -> bool:
    """Whether a completed cleanup attempt failed to prove shutdown."""
    if task is None or not task.done():
        return False
    if task.cancelled():
        return True
    try:
        return task.result() is False
    except Exception:
        return True


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
    if retry_completed and handles.workflow is not None and _cleanup_task_needs_retry(
        handles.workflow_task
    ):
        handles.workflow_task = _start_cleanup_task(
            lambda: handles.workflow.close(save_runtime=False),
            name=f"tombstone-workflow-close:{session_id}",
        )

    if (
        retry_completed
        and handles.report_workflow is not None
        and _cleanup_task_needs_retry(handles.report_workflow_task)
    ):
        handles.report_workflow_task = _start_cleanup_task(
            lambda: handles.report_workflow.close(save_runtime=False),
            name=f"tombstone-report-workflow-close:{session_id}",
        )

    if retry_completed and handles.chat_sandbox is not None and _cleanup_task_needs_retry(
        handles.chat_task
    ):
        handles.chat_task = _start_cleanup_task(
            lambda: handles.chat_sandbox.stop(save=False),
            name=f"tombstone-chat-stop:{session_id}",
        )

    # A live chat task owns its sandbox finalizer while it is running. Once
    # cancellation has completed, independently finish a finalizer that left a
    # real kernel handle behind. Test doubles without a kernel handle retain
    # the legacy contract that task completion itself proves cleanup.
    if (
        not handles.tasks
        and handles.chat_task is None
        and handles.chat_sandbox is not None
        and getattr(handles.chat_sandbox, "km", None) is not None
    ):
        handles.chat_task = _start_cleanup_task(
            lambda: handles.chat_sandbox.stop(save=False),
            name=f"tombstone-chat-stop:{session_id}",
        )


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
    if task is None:
        return True
    if not task.done():
        done, _pending = await asyncio.wait(
            {task}, timeout=_PRODUCER_STOP_TIMEOUT_SECONDS,
        )
        if not done:
            logger.warning(
                "%s still running during tombstone cleanup for %s",
                label,
                session_id,
            )
            return False
    if task.cancelled():
        logger.error(
            "%s was cancelled before proving shutdown for tombstoned session %s",
            label,
            session_id,
        )
        return False
    try:
        result = task.result()
    except Exception:
        logger.exception(
            "%s failed for tombstoned session %s",
            label,
            session_id,
        )
        return False
    if result is False:
        logger.error(
            "%s reported incomplete cleanup for tombstoned session %s",
            label,
            session_id,
        )
        return False
    return True


async def _delete_tombstoned_files(
    session_id: str,
    owner_id: str | None = None,
    generation: int | None = None,
) -> dict[str, int]:
    """Delete one exact tombstoned tree and publish cleanup completion."""
    if owner_id is None or generation is None:
        captured = read_lifecycle_record(session_id)
        if not captured or captured.get("status") != "deleted":
            return {
                "deleted_sessions": 0,
                "deleted_state_files": 0,
            }
        owner_id = str(captured.get("owner_id") or "")
        generation = int(captured.get("generation") or 0)

    target_owner = str(owner_id)
    target_generation = int(generation)

    def _delete_from_disk() -> tuple[dict[str, int], bool]:
        deleted_session_dirs = 0
        deleted_state_files = 0
        with lifecycle_file_lock(session_id):
            lifecycle = read_lifecycle_record(session_id)
            if not _target_matches(
                lifecycle,
                target_owner,
                target_generation,
                require_incomplete=True,
            ):
                # A same-ID import may have claimed the next generation after
                # this cleanup attempt first observed the tombstone. Never let
                # a stale sweeper remove that new generation's tree.
                return ({
                    "deleted_sessions": 0,
                    "deleted_state_files": 0,
                }, False)
            session_dir = get_session_dir(session_id)
            if session_dir.is_dir():
                shutil.rmtree(session_dir)
                deleted_session_dirs += 1
            legacy_file = SESSION_DIR / f"{session_id}.json"
            if legacy_file.is_file():
                deleted_state_files += 1
            legacy_file.unlink(missing_ok=True)
            for legacy_dir in (
                UPLOAD_DIR / session_id,
                WORKSPACE_DIR / session_id,
                REPORTS_DIR / session_id,
                INTEGRITY_DIR / session_id,
            ):
                if legacy_dir.is_dir():
                    shutil.rmtree(legacy_dir)
            for state_file in legacy_session_state_paths(STATE_DIR, session_id):
                if state_file.is_file():
                    state_file.unlink()
                    deleted_state_files += 1
        return ({
            "deleted_sessions": deleted_session_dirs,
            "deleted_state_files": deleted_state_files,
        }, True)

    async with get_lifecycle_lock(session_id):
        stats, matched = await run_in_owned_thread(_delete_from_disk)
        if not matched:
            return stats

        # Same-ID import remains blocked by cleanup_complete=False while the
        # old generation's mutation-lock slot is removed. Publishing completion
        # first would let a new generation claim the ID and have this stale
        # cleanup remove its newly-created lock.
        release_session_lock_slot(session_id)
        completed = await run_in_owned_thread(
            complete_tombstone_cleanup,
            session_id,
            target_owner,
            target_generation,
        )
        if not completed:
            raise RuntimeError(
                f"tombstone target changed before cleanup completion: "
                f"{session_id} generation {target_generation}"
            )
    return stats


async def _drain_tombstone_producers(
    session_id: str,
    owner_id: str,
    generation: int,
) -> int:
    """Observe one shared set of producer handles for this tombstone."""
    handles = _pending_tombstone_cleanup.get(session_id)
    if handles is None:
        return 0
    if (
        handles.owner_id != owner_id
        or handles.generation != generation
    ):
        return 0

    async with handles.lock:
        # Another cleanup waiter may have completed and removed these handles
        # while this coroutine was queued on the same lock.
        if _pending_tombstone_cleanup.get(session_id) is not handles:
            return 0
        handles.tasks = await _cancel_and_wait(handles.tasks)
        _refresh_cleanup_tasks(
            handles,
            session_id,
            retry_completed=handles.drain_started,
        )
        handles.drain_started = True
        workflow_stopped = await _await_cleanup_task(
            handles.workflow_task,
            label="Workflow close",
            session_id=session_id,
        )
        report_workflow_stopped = await _await_cleanup_task(
            handles.report_workflow_task,
            label="Report workflow close",
            session_id=session_id,
        )
        chat_stopped = await _await_cleanup_task(
            handles.chat_task,
            label="Chat sandbox stop",
            session_id=session_id,
        )
        return (
            len(handles.tasks)
            + int(not workflow_stopped)
            + int(not report_workflow_stopped)
            + int(not chat_stopped)
        )


async def _cleanup_tombstone_side_surfaces(
    session_id: str,
    generation: int,
) -> None:
    """Idempotently remove non-session-tree resources after tombstoning."""
    from backend.utils.websocket_delivery import close_websocket

    await close_websocket("analysis", session_id, code=4001)
    await close_websocket("report", session_id, code=4001)

    from backend.routes.reports import (
        clear_detailed_report_cache,
        clear_generation_pending,
    )

    clear_detailed_report_cache(session_id)
    clear_generation_pending(session_id)

    # Export records survive until their producer/stream has drained and their
    # exact-generation archive was actually removed. Unlink failure therefore
    # keeps the tombstone incomplete and blocks same-ID reuse.
    await cleanup_session_export_records(session_id, generation)

    from backend.services.upload import cleanup_session_uploads

    # Upload rows carry only session_id today. If this fails, deleting the tree
    # and permitting same-ID reuse would let a later retry erase the new
    # generation's rows, so failure must keep the old tombstone incomplete.
    upload_cleanup = asyncio.create_task(
        cleanup_session_uploads(session_id),
        name=f"tombstone-upload-cleanup:{session_id}",
    )
    await await_with_deferred_cancellation(upload_cleanup)


async def _ensure_cleanup_handles(
    session_id: str,
    owner_id: str,
    generation: int,
) -> _PendingTombstoneCleanup | None:
    """Return target-bound handles carrying the cross-process cleanup lease."""
    existing = _pending_tombstone_cleanup.get(session_id)
    if existing is not None:
        if (
            existing.owner_id == owner_id
            and existing.generation == generation
            and existing.lease is not None
            and not existing.lease.released
        ):
            return existing
        return None

    lease = await _try_acquire_cleanup_lease(session_id)
    if lease is None:
        return None
    with lifecycle_file_lock(session_id):
        record = read_lifecycle_record(session_id)
        if not _target_matches(record, owner_id, generation):
            lease.release()
            return None
    handles = _PendingTombstoneCleanup(
        owner_id=owner_id,
        generation=generation,
        lease=lease,
    )
    _pending_tombstone_cleanup[session_id] = handles
    return handles


def _release_cleanup_handles(
    session_id: str,
    handles: _PendingTombstoneCleanup,
) -> None:
    if _pending_tombstone_cleanup.get(session_id) is handles:
        _pending_tombstone_cleanup.pop(session_id, None)
    if handles.lease is not None:
        handles.lease.release()
        handles.lease = None


def _release_completed_target_handles(
    session_id: str,
    owner_id: str,
    generation: int,
) -> None:
    """Recover an exact lease orphaned after durable cleanup completion."""
    handles = _pending_tombstone_cleanup.get(session_id)
    if handles is None:
        return
    if handles.owner_id != owner_id or handles.generation != generation:
        return
    _release_cleanup_handles(session_id, handles)


def cleanup_failed_import_generation(session_id: str) -> bool:
    """Synchronously retire one uncommitted import under the cleanup lease.

    Import extraction is synchronous and may fail before an async cleanup task
    can be registered. Acquiring the same cross-process lease *before* writing
    the tombstone prevents a background sweeper from observing an incomplete
    record and racing this cleanup into a later same-ID generation.
    """
    validate_session_id(session_id)
    lease = _try_acquire_cleanup_lease_sync(session_id)
    if lease is None:
        raise SessionProducerStopError(
            f"failed import cleanup is already in progress for {session_id}"
        )
    try:
        if not abandon_import_generation(session_id):
            return False
        record = read_lifecycle_record(session_id)
        if not record or record.get("status") != "deleted":
            raise RuntimeError(
                f"failed import {session_id} has no durable tombstone"
            )
        owner_id = str(record.get("owner_id") or "")
        generation = int(record.get("generation") or 0)

        with lifecycle_file_lock(session_id):
            current = read_lifecycle_record(session_id)
            if not _target_matches(
                current, owner_id, generation, require_incomplete=True,
            ):
                raise RuntimeError(
                    f"failed import tombstone changed before cleanup: {session_id}"
                )
            session_dir = get_session_dir(session_id)
            if session_dir.is_dir():
                shutil.rmtree(session_dir)
            if session_dir.exists():
                raise OSError(
                    f"partial import directory still exists: {session_dir}"
                )

        # No import generation is published to a workflow. The only mutable
        # in-process surfaces possible here are the cache and its persistence
        # lock, installed immediately before a commit attempt.
        sessions.pop(session_id, None)
        session_access_times.pop(session_id, None)
        last_save_times.pop(session_id, None)
        release_session_lock_slot(session_id)
        if not complete_tombstone_cleanup(
            session_id, owner_id, generation,
        ):
            raise RuntimeError(
                f"failed import tombstone changed before completion: {session_id}"
            )
        return True
    finally:
        lease.release()


async def _finalize_tombstone_target(
    session_id: str,
    owner_id: str,
    generation: int,
    handles: _PendingTombstoneCleanup,
) -> dict[str, int]:
    """Publish cleanup completion and release its lease as one owned task."""
    stats = await _delete_tombstoned_files(
        session_id, owner_id, generation,
    )
    stats["pending_producers"] = 0
    _release_cleanup_handles(session_id, handles)
    return stats


async def _await_cleanup_finalizer(
    finalizer: asyncio.Task,
    *,
    session_id: str,
) -> dict[str, int]:
    """Defer caller cancellation until completion and lease release finish."""
    try:
        return await await_with_deferred_cancellation(finalizer)
    except asyncio.CancelledError:
        logger.warning(
            "Cleanup request cancelled during terminal publication for %s; "
            "the cleanup lease was released before cancellation propagated",
            session_id,
        )
        raise


async def _cleanup_tombstone_target_locked(
    session_id: str,
    target_owner: str,
    target_generation: int,
) -> dict[str, int]:
    """Clean one captured target while the local cleanup lock is held."""
    # Claim the non-blocking process/cross-process lease before taking a
    # potentially contended lifecycle flock. `_ensure_cleanup_handles` then
    # revalidates the exact tombstone under that flock. This keeps an async
    # sweeper responsive while a synchronous failed-import cleanup owns both.
    handles = await _ensure_cleanup_handles(
        session_id, target_owner, target_generation,
    )
    if handles is None:
        current = read_lifecycle_record(session_id)
        return _empty_cleanup_stats(
            pending=int(
                _target_matches(
                    current,
                    target_owner,
                    target_generation,
                    require_incomplete=True,
                )
            )
        )

    pending = await _drain_tombstone_producers(
        session_id, target_owner, target_generation,
    )
    if pending:
        return _empty_cleanup_stats(pending=pending)

    with lifecycle_file_lock(session_id):
        current = read_lifecycle_record(session_id)
        if not _target_matches(
            current, target_owner, target_generation,
            require_incomplete=True,
        ):
            _release_cleanup_handles(session_id, handles)
            return _empty_cleanup_stats()

    await _cleanup_tombstone_side_surfaces(session_id, target_generation)
    finalizer = asyncio.create_task(
        _finalize_tombstone_target(
            session_id,
            target_owner,
            target_generation,
            handles,
        ),
        name=f"tombstone-finalize:{session_id}",
    )
    return await _await_cleanup_finalizer(finalizer, session_id=session_id)


async def cleanup_tombstoned_generation(
    session_id: str,
    owner_id: str | None = None,
) -> dict[str, int]:
    """Retry physical cleanup without reviving the deleted generation."""
    validate_session_id(session_id)
    # Capture the exact target before waiting on the local single-flight lock.
    # A delayed generation-g waiter must never retarget itself to generation
    # g+1 merely because that newer generation was also deleted meanwhile.
    record = read_lifecycle_record(session_id)
    if not record or record.get("status") != "deleted":
        raise ValueError(f"session {session_id} is not tombstoned")
    target_owner = str(record.get("owner_id") or "")
    target_generation = int(record.get("generation") or 0)
    if owner_id is not None and target_owner != owner_id:
        raise PermissionError("session owner does not match lifecycle owner")
    if record.get("cleanup_complete") is True:
        async with _get_cleanup_lock(session_id):
            _release_completed_target_handles(
                session_id, target_owner, target_generation,
            )
        return _empty_cleanup_stats()

    async with _get_cleanup_lock(session_id):
        return await _cleanup_tombstone_target_locked(
            session_id, target_owner, target_generation,
        )


async def sweep_tombstoned_generations() -> dict[str, int]:
    """Best-effort physical cleanup for every stable tombstone."""
    completed = 0
    pending = 0
    failed = 0
    for record in list_lifecycle_records(status="deleted"):
        session_id = str(record.get("session_id") or "")
        if not session_id:
            continue
        try:
            stats = await cleanup_tombstoned_generation(session_id)
            if int(stats.get("pending_producers") or 0):
                pending += 1
            else:
                completed += 1
        except Exception:
            failed += 1
            logger.exception(
                "Background tombstone cleanup failed for session %s",
                session_id,
            )
    return {"completed": completed, "pending": pending, "failed": failed}


async def _tombstone_detach_and_publish(
    session_id: str,
    state,
    owner_id: str,
    *,
    force_stop: bool,
    lease: _TombstoneCleanupLease,
) -> tuple[str, int]:
    """Owned linearization step from active generation to cleanup handles."""
    tasks_to_cancel: list[asyncio.Task] = []
    workflow = None
    report_workflow = None
    chat_sandbox = None
    chat_task_owns_sandbox = False

    async with get_lifecycle_lock(session_id):
        existing_record = read_lifecycle_record(session_id)
        if existing_record and existing_record.get("status") == "deleted":
            if str(existing_record.get("owner_id") or "") != owner_id:
                raise PermissionError("session owner does not match lifecycle owner")
            raise SessionProducerStopError(
                f"session {session_id} deletion is already owned by another cleaner"
            )

        async with state_lock:
            if (
                session_id in workflows
                or session_id in report_workflows
                or session_id in initializing_sessions
                or has_live_session_task(session_id)
            ) and not force_stop:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot delete a running session. Stop the analysis "
                        "first."
                    ),
                )

        tombstone = tombstone_generation(
            session_id,
            owner_id,
            int(getattr(state, "lifecycle_generation", 0) or 0),
        )
        target_generation = int(tombstone.get("generation") or 0)

        from backend.services.workflow_runner import _report_tasks

        async with state_lock:
            workflow = workflows.pop(session_id, None)
            report_workflow = report_workflows.pop(session_id, None)
            if report_workflow is workflow:
                report_workflow = None
            for owned_workflow in (workflow, report_workflow):
                if owned_workflow is not None:
                    setattr(owned_workflow, "_restore_superseded", True)
            for registry in (
                _run_tasks,
                _report_tasks,
                _monitor_tasks,
                detailed_report_tasks,
            ):
                task = registry.pop(session_id, None)
                if isinstance(task, asyncio.Task):
                    tasks_to_cancel.append(task)
            chat_owner_task = chat_tasks.pop(session_id, None)
            if isinstance(chat_owner_task, asyncio.Task):
                chat_task_owns_sandbox = not chat_owner_task.done()
                tasks_to_cancel.append(chat_owner_task)
            release_initializing_session(session_id)
            sessions.pop(session_id, None)
            websockets.pop(session_id, None)
            report_websockets.pop(session_id, None)
            session_access_times.pop(session_id, None)
            last_save_times.pop(session_id, None)

        async with chat_task_lock:
            chat_sandbox = chat_sandboxes.pop(session_id, None)

        for transaction_id, handle in list(snapshot_restore_jobs.items()):
            restore_record = getattr(handle, "record", None)
            if getattr(restore_record, "session_id", None) != session_id:
                continue
            cancel_event = getattr(handle, "cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()
            task = getattr(handle, "task", None)
            if isinstance(task, asyncio.Task):
                tasks_to_cancel.append(task)
            snapshot_restore_jobs.pop(transaction_id, None)

        tasks_to_cancel.extend(
            await mark_session_exports_for_deletion(
                session_id,
                target_generation,
            )
        )

        cleanup_handles = _PendingTombstoneCleanup(
            owner_id=owner_id,
            generation=target_generation,
            lease=lease,
            tasks=list(tasks_to_cancel),
            workflow_task=_start_cleanup_task(
                (lambda: workflow.close(save_runtime=False))
                if workflow is not None
                else None,
                name=f"tombstone-workflow-close:{session_id}",
            ),
            report_workflow_task=_start_cleanup_task(
                (lambda: report_workflow.close(save_runtime=False))
                if report_workflow is not None
                else None,
                name=f"tombstone-report-workflow-close:{session_id}",
            ),
            chat_task=_start_cleanup_task(
                (lambda: chat_sandbox.stop(save=False))
                if chat_sandbox is not None and not chat_task_owns_sandbox
                else None,
                name=f"tombstone-chat-stop:{session_id}",
            ),
            workflow=workflow,
            report_workflow=report_workflow,
            chat_sandbox=chat_sandbox,
        )
        if session_id in _pending_tombstone_cleanup:
            raise RuntimeError(
                f"cleanup handles already exist for active session {session_id}"
            )
        _pending_tombstone_cleanup[session_id] = cleanup_handles
        return owner_id, target_generation


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
    validate_session_id(session_id)
    newly_tombstoned = False
    async with _get_cleanup_lock(session_id):
        existing = read_lifecycle_record(session_id)
        if existing and existing.get("status") == "deleted":
            target_owner = str(existing.get("owner_id") or "")
            target_generation = int(existing.get("generation") or 0)
            if target_owner != owner_id:
                raise PermissionError("session owner does not match lifecycle owner")
            if existing.get("cleanup_complete") is True:
                return _empty_cleanup_stats()
        else:
            lease = await _try_acquire_cleanup_lease(session_id)
            if lease is None:
                raise SessionProducerStopError(
                    f"session {session_id} deletion is already in progress"
                )
            publication = asyncio.create_task(
                _tombstone_detach_and_publish(
                    session_id,
                    state,
                    owner_id,
                    force_stop=force_stop,
                    lease=lease,
                ),
                name=f"tombstone-publish:{session_id}",
            )
            cancellation: asyncio.CancelledError | None = None
            try:
                while True:
                    try:
                        target_owner, target_generation = await asyncio.shield(
                            publication
                        )
                        break
                    except asyncio.CancelledError as exc:
                        consume_current_cancellation_request()
                        if publication.done():
                            target_owner, target_generation = publication.result()
                            cancellation = cancellation or exc
                            break
                        cancellation = cancellation or exc
                        logger.warning(
                            "Deletion request cancelled while publishing "
                            "tombstone handles for %s; deferring cancellation",
                            session_id,
                        )
                newly_tombstoned = True
            except BaseException:
                if _pending_tombstone_cleanup.get(session_id) is None:
                    lease.release()
                raise
            if cancellation is not None:
                # The durable tombstone and producer handles now exist. A later
                # request or background sweep can safely finish cleanup.
                raise cancellation

        stats = await _cleanup_tombstone_target_locked(
            session_id, target_owner, target_generation,
        )

    pending_producers = int(stats.get("pending_producers") or 0)
    if newly_tombstoned and pending_producers:
        raise SessionProducerStopError(
            f"session {session_id} remains tombstoned; "
            f"{pending_producers} producer(s) did not stop before timeout"
        )
    return stats
