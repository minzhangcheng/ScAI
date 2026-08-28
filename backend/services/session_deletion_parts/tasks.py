"""Cleanup task ownership for session deletion."""
from __future__ import annotations

async def _cancel_and_wait(tasks: list[_binding_asyncio.Task], *, bindings) -> list[_binding_asyncio.Task]:
    _binding__PRODUCER_STOP_TIMEOUT_SECONDS = getattr(bindings, "_PRODUCER_STOP_TIMEOUT_SECONDS")
    _binding_asyncio = getattr(bindings, "asyncio")
    _binding_logger = getattr(bindings, "logger")
    current = _binding_asyncio.current_task()
    pending = [task for task in tasks if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        done, still_pending = await _binding_asyncio.wait(
            pending, timeout=_binding__PRODUCER_STOP_TIMEOUT_SECONDS,
        )
        for task in done:
            try:
                task.result()
            except (_binding_asyncio.CancelledError, Exception):
                pass
        if still_pending:
            _binding_logger.warning(
                "Session deletion timed out waiting for %d task(s)",
                len(still_pending),
            )
            return list(still_pending)
    return []

def _start_cleanup_task(closer, *, name: str, bindings) -> _binding_asyncio.Task | None:
    _binding_asyncio = getattr(bindings, "asyncio")
    if closer is None:
        return None
    return _binding_asyncio.create_task(closer(), name=name)

def _cleanup_task_needs_retry(task: _binding_asyncio.Task | None, *, bindings) -> bool:
    """Whether a completed cleanup attempt failed to prove shutdown."""
    _binding_asyncio = getattr(bindings, "asyncio")
    if task is None or not task.done():
        return False
    if task.cancelled():
        return True
    try:
        return task.result() is False
    except Exception:
        return True

def _refresh_cleanup_tasks(handles: _binding__PendingTombstoneCleanup, session_id: str, *, retry_completed: bool=True, bindings) -> None:
    """Start a new owned attempt only after the prior one has completed.

    Long-running closers are observed, never duplicated. A completed attempt
    that raised or returned literal False is different: the retained producer
    handle exists precisely so a later tombstone sweep can call its idempotent
    close contract again.
    """
    _binding__PendingTombstoneCleanup = getattr(bindings, "_PendingTombstoneCleanup")
    _binding__cleanup_task_needs_retry = getattr(bindings, "_cleanup_task_needs_retry")
    _binding__start_cleanup_task = getattr(bindings, "_start_cleanup_task")
    if retry_completed and handles.workflow is not None and _binding__cleanup_task_needs_retry(
        handles.workflow_task
    ):
        handles.workflow_task = _binding__start_cleanup_task(
            lambda: handles.workflow.close(save_runtime=False),
            name=f"tombstone-workflow-close:{session_id}",
        )

    if (
        retry_completed
        and handles.report_workflow is not None
        and _binding__cleanup_task_needs_retry(handles.report_workflow_task)
    ):
        handles.report_workflow_task = _binding__start_cleanup_task(
            lambda: handles.report_workflow.close(save_runtime=False),
            name=f"tombstone-report-workflow-close:{session_id}",
        )

    if retry_completed and handles.chat_sandbox is not None and _binding__cleanup_task_needs_retry(
        handles.chat_task
    ):
        handles.chat_task = _binding__start_cleanup_task(
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
        handles.chat_task = _binding__start_cleanup_task(
            lambda: handles.chat_sandbox.stop(save=False),
            name=f"tombstone-chat-stop:{session_id}",
        )

async def _await_cleanup_task(task: _binding_asyncio.Task | None, *, label: str, session_id: str, bindings) -> bool:
    """Boundedly observe an owned cleanup task without cancelling it.

    `asyncio.wait_for(coro, timeout)` cancels the closer at the deadline. A
    closer that catches CancelledError can then make wait_for wait forever; a
    closer that does not catch it is still destroyed before it proves the
    producer stopped. The independently owned task remains in the tombstone
    registry. Retries observe that exact task while it is running; only a
    completed attempt that failed to prove shutdown is replaced.
    """
    _binding__PRODUCER_STOP_TIMEOUT_SECONDS = getattr(bindings, "_PRODUCER_STOP_TIMEOUT_SECONDS")
    _binding_asyncio = getattr(bindings, "asyncio")
    _binding_logger = getattr(bindings, "logger")
    if task is None:
        return True
    if not task.done():
        done, _pending = await _binding_asyncio.wait(
            {task}, timeout=_binding__PRODUCER_STOP_TIMEOUT_SECONDS,
        )
        if not done:
            _binding_logger.warning(
                "%s still running during tombstone cleanup for %s",
                label,
                session_id,
            )
            return False
    if task.cancelled():
        _binding_logger.error(
            "%s was cancelled before proving shutdown for tombstoned session %s",
            label,
            session_id,
        )
        return False
    try:
        result = task.result()
    except Exception:
        _binding_logger.exception(
            "%s failed for tombstoned session %s",
            label,
            session_id,
        )
        return False
    if result is False:
        _binding_logger.error(
            "%s reported incomplete cleanup for tombstoned session %s",
            label,
            session_id,
        )
        return False
    return True

async def _drain_tombstone_producers(session_id: str, owner_id: str, generation: int, *, bindings) -> int:
    """Observe one shared set of producer handles for this tombstone."""
    _binding__await_cleanup_task = getattr(bindings, "_await_cleanup_task")
    _binding__cancel_and_wait = getattr(bindings, "_cancel_and_wait")
    _binding__pending_tombstone_cleanup = getattr(bindings, "_pending_tombstone_cleanup")
    _binding__refresh_cleanup_tasks = getattr(bindings, "_refresh_cleanup_tasks")
    handles = _binding__pending_tombstone_cleanup.get(session_id)
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
        if _binding__pending_tombstone_cleanup.get(session_id) is not handles:
            return 0
        handles.tasks = await _binding__cancel_and_wait(handles.tasks)
        _binding__refresh_cleanup_tasks(
            handles,
            session_id,
            retry_completed=handles.drain_started,
        )
        handles.drain_started = True
        workflow_stopped = await _binding__await_cleanup_task(
            handles.workflow_task,
            label="Workflow close",
            session_id=session_id,
        )
        report_workflow_stopped = await _binding__await_cleanup_task(
            handles.report_workflow_task,
            label="Report workflow close",
            session_id=session_id,
        )
        chat_stopped = await _binding__await_cleanup_task(
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

async def _cleanup_tombstone_side_surfaces(session_id: str, generation: int, *, bindings) -> None:
    """Idempotently remove non-session-tree resources after tombstoning."""
    _binding_asyncio = getattr(bindings, "asyncio")
    _binding_await_with_deferred_cancellation = getattr(bindings, "await_with_deferred_cancellation")
    _binding_cleanup_session_export_records = getattr(bindings, "cleanup_session_export_records")
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
    await _binding_cleanup_session_export_records(session_id, generation)

    from backend.services.upload import cleanup_session_uploads

    # Upload rows carry only session_id today. If this fails, deleting the tree
    # and permitting same-ID reuse would let a later retry erase the new
    # generation's rows, so failure must keep the old tombstone incomplete.
    upload_cleanup = _binding_asyncio.create_task(
        cleanup_session_uploads(session_id),
        name=f"tombstone-upload-cleanup:{session_id}",
    )
    await _binding_await_with_deferred_cancellation(upload_cleanup)

async def _ensure_cleanup_handles(session_id: str, owner_id: str, generation: int, *, bindings) -> _binding__PendingTombstoneCleanup | None:
    """Return target-bound handles carrying the cross-process cleanup lease."""
    _binding__PendingTombstoneCleanup = getattr(bindings, "_PendingTombstoneCleanup")
    _binding__pending_tombstone_cleanup = getattr(bindings, "_pending_tombstone_cleanup")
    _binding__target_matches = getattr(bindings, "_target_matches")
    _binding__try_acquire_cleanup_lease = getattr(bindings, "_try_acquire_cleanup_lease")
    _binding_lifecycle_file_lock = getattr(bindings, "lifecycle_file_lock")
    _binding_read_lifecycle_record = getattr(bindings, "read_lifecycle_record")
    existing = _binding__pending_tombstone_cleanup.get(session_id)
    if existing is not None:
        if (
            existing.owner_id == owner_id
            and existing.generation == generation
            and existing.lease is not None
            and not existing.lease.released
        ):
            return existing
        return None

    lease = await _binding__try_acquire_cleanup_lease(session_id)
    if lease is None:
        return None
    with _binding_lifecycle_file_lock(session_id):
        record = _binding_read_lifecycle_record(session_id)
        if not _binding__target_matches(record, owner_id, generation):
            lease.release()
            return None
    handles = _binding__PendingTombstoneCleanup(
        owner_id=owner_id,
        generation=generation,
        lease=lease,
    )
    _binding__pending_tombstone_cleanup[session_id] = handles
    return handles

def _release_cleanup_handles(session_id: str, handles: _binding__PendingTombstoneCleanup, *, bindings) -> None:
    _binding__PendingTombstoneCleanup = getattr(bindings, "_PendingTombstoneCleanup")
    _binding__pending_tombstone_cleanup = getattr(bindings, "_pending_tombstone_cleanup")
    if _binding__pending_tombstone_cleanup.get(session_id) is handles:
        _binding__pending_tombstone_cleanup.pop(session_id, None)
    if handles.lease is not None:
        handles.lease.release()
        handles.lease = None

def _release_completed_target_handles(session_id: str, owner_id: str, generation: int, *, bindings) -> None:
    """Recover an exact lease orphaned after durable cleanup completion."""
    _binding__pending_tombstone_cleanup = getattr(bindings, "_pending_tombstone_cleanup")
    _binding__release_cleanup_handles = getattr(bindings, "_release_cleanup_handles")
    handles = _binding__pending_tombstone_cleanup.get(session_id)
    if handles is None:
        return
    if handles.owner_id != owner_id or handles.generation != generation:
        return
    _binding__release_cleanup_handles(session_id, handles)
