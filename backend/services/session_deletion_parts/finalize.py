"""Tombstone finalization for session deletion."""
from __future__ import annotations

async def _delete_tombstoned_files(session_id: str, owner_id: str | None=None, generation: int | None=None, *, bindings) -> dict[str, int]:
    """Delete one exact tombstoned tree and publish cleanup completion."""
    _binding_INTEGRITY_DIR = getattr(bindings, "INTEGRITY_DIR")
    _binding_REPORTS_DIR = getattr(bindings, "REPORTS_DIR")
    _binding_SESSION_DIR = getattr(bindings, "SESSION_DIR")
    _binding_STATE_DIR = getattr(bindings, "STATE_DIR")
    _binding_UPLOAD_DIR = getattr(bindings, "UPLOAD_DIR")
    _binding_WORKSPACE_DIR = getattr(bindings, "WORKSPACE_DIR")
    _binding__target_matches = getattr(bindings, "_target_matches")
    _binding_complete_tombstone_cleanup = getattr(bindings, "complete_tombstone_cleanup")
    _binding_get_lifecycle_lock = getattr(bindings, "get_lifecycle_lock")
    _binding_get_session_dir = getattr(bindings, "get_session_dir")
    _binding_legacy_session_state_paths = getattr(bindings, "legacy_session_state_paths")
    _binding_lifecycle_file_lock = getattr(bindings, "lifecycle_file_lock")
    _binding_read_lifecycle_record = getattr(bindings, "read_lifecycle_record")
    _binding_release_session_lock_slot = getattr(bindings, "release_session_lock_slot")
    _binding_run_in_owned_thread = getattr(bindings, "run_in_owned_thread")
    _binding_shutil = getattr(bindings, "shutil")
    if owner_id is None or generation is None:
        captured = _binding_read_lifecycle_record(session_id)
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
        with _binding_lifecycle_file_lock(session_id):
            lifecycle = _binding_read_lifecycle_record(session_id)
            if not _binding__target_matches(
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
            session_dir = _binding_get_session_dir(session_id)
            if session_dir.is_dir():
                _binding_shutil.rmtree(session_dir)
                deleted_session_dirs += 1
            legacy_file = _binding_SESSION_DIR / f"{session_id}.json"
            if legacy_file.is_file():
                deleted_state_files += 1
            legacy_file.unlink(missing_ok=True)
            for legacy_dir in (
                _binding_UPLOAD_DIR / session_id,
                _binding_WORKSPACE_DIR / session_id,
                _binding_REPORTS_DIR / session_id,
                _binding_INTEGRITY_DIR / session_id,
            ):
                if legacy_dir.is_dir():
                    _binding_shutil.rmtree(legacy_dir)
            for state_file in _binding_legacy_session_state_paths(_binding_STATE_DIR, session_id):
                if state_file.is_file():
                    state_file.unlink()
                    deleted_state_files += 1
        return ({
            "deleted_sessions": deleted_session_dirs,
            "deleted_state_files": deleted_state_files,
        }, True)

    async with _binding_get_lifecycle_lock(session_id):
        stats, matched = await _binding_run_in_owned_thread(_delete_from_disk)
        if not matched:
            return stats

        # Same-ID import remains blocked by cleanup_complete=False while the
        # old generation's mutation-lock slot is removed. Publishing completion
        # first would let a new generation claim the ID and have this stale
        # cleanup remove its newly-created lock.
        _binding_release_session_lock_slot(session_id)
        completed = await _binding_run_in_owned_thread(
            _binding_complete_tombstone_cleanup,
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

def cleanup_failed_import_generation(session_id: str, *, bindings) -> bool:
    """Synchronously retire one uncommitted import under the cleanup lease.

    Import extraction is synchronous and may fail before an async cleanup task
    can be registered. Acquiring the same cross-process lease *before* writing
    the tombstone prevents a background sweeper from observing an incomplete
    record and racing this cleanup into a later same-ID generation.
    """
    _binding_INTEGRITY_DIR = getattr(bindings, "INTEGRITY_DIR")
    _binding_SessionProducerStopError = getattr(bindings, "SessionProducerStopError")
    _binding__target_matches = getattr(bindings, "_target_matches")
    _binding__try_acquire_cleanup_lease_sync = getattr(bindings, "_try_acquire_cleanup_lease_sync")
    _binding_abandon_import_generation = getattr(bindings, "abandon_import_generation")
    _binding_complete_tombstone_cleanup = getattr(bindings, "complete_tombstone_cleanup")
    _binding_get_session_dir = getattr(bindings, "get_session_dir")
    _binding_last_save_times = getattr(bindings, "last_save_times")
    _binding_lifecycle_file_lock = getattr(bindings, "lifecycle_file_lock")
    _binding_read_lifecycle_record = getattr(bindings, "read_lifecycle_record")
    _binding_release_session_lock_slot = getattr(bindings, "release_session_lock_slot")
    _binding_session_access_times = getattr(bindings, "session_access_times")
    _binding_sessions = getattr(bindings, "sessions")
    _binding_shutil = getattr(bindings, "shutil")
    _binding_validate_session_id = getattr(bindings, "validate_session_id")
    _binding_validate_session_id(session_id)
    lease = _binding__try_acquire_cleanup_lease_sync(session_id)
    if lease is None:
        raise _binding_SessionProducerStopError(
            f"failed import cleanup is already in progress for {session_id}"
        )
    try:
        record = _binding_read_lifecycle_record(session_id)
        if record is None or record.get("status") == "importing":
            if not _binding_abandon_import_generation(session_id):
                return False
            record = _binding_read_lifecycle_record(session_id)
        elif not (
            record.get("status") == "deleted"
            and record.get("import_failed") is True
            and record.get("cleanup_complete") is not True
        ):
            return False
        if not record or record.get("status") != "deleted":
            raise RuntimeError(
                f"failed import {session_id} has no durable tombstone"
            )
        owner_id = str(record.get("owner_id") or "")
        generation = int(record.get("generation") or 0)

        with _binding_lifecycle_file_lock(session_id):
            current = _binding_read_lifecycle_record(session_id)
            if not _binding__target_matches(
                current, owner_id, generation, require_incomplete=True,
            ):
                raise RuntimeError(
                    f"failed import tombstone changed before cleanup: {session_id}"
                )
            session_dir = _binding_get_session_dir(session_id)
            if session_dir.is_dir():
                _binding_shutil.rmtree(session_dir)
            if session_dir.exists():
                raise OSError(
                    f"partial import directory still exists: {session_dir}"
                )
            integrity_dir = _binding_INTEGRITY_DIR / session_id
            if integrity_dir.is_dir():
                _binding_shutil.rmtree(integrity_dir)
            if integrity_dir.exists():
                raise OSError(
                    f"partial import integrity data still exists: {integrity_dir}"
                )

        # No import generation is published to a workflow. The only mutable
        # in-process surfaces possible here are the cache and its persistence
        # lock, installed immediately before a commit attempt.
        _binding_sessions.pop(session_id, None)
        _binding_session_access_times.pop(session_id, None)
        _binding_last_save_times.pop(session_id, None)
        _binding_release_session_lock_slot(session_id)
        if not _binding_complete_tombstone_cleanup(
            session_id, owner_id, generation,
        ):
            raise RuntimeError(
                f"failed import tombstone changed before completion: {session_id}"
            )
        return True
    finally:
        lease.release()

async def _finalize_tombstone_target(session_id: str, owner_id: str, generation: int, handles: _binding__PendingTombstoneCleanup, *, bindings) -> dict[str, int]:
    """Publish cleanup completion and release its lease as one owned task."""
    _binding__PendingTombstoneCleanup = getattr(bindings, "_PendingTombstoneCleanup")
    _binding__delete_tombstoned_files = getattr(bindings, "_delete_tombstoned_files")
    _binding__release_cleanup_handles = getattr(bindings, "_release_cleanup_handles")
    stats = await _binding__delete_tombstoned_files(
        session_id, owner_id, generation,
    )
    stats["pending_producers"] = 0
    _binding__release_cleanup_handles(session_id, handles)
    return stats

async def _await_cleanup_finalizer(finalizer: _binding_asyncio.Task, *, session_id: str, bindings) -> dict[str, int]:
    """Defer caller cancellation until completion and lease release finish."""
    _binding_asyncio = getattr(bindings, "asyncio")
    _binding_await_with_deferred_cancellation = getattr(bindings, "await_with_deferred_cancellation")
    _binding_logger = getattr(bindings, "logger")
    try:
        return await _binding_await_with_deferred_cancellation(finalizer)
    except _binding_asyncio.CancelledError:
        _binding_logger.warning(
            "Cleanup request cancelled during terminal publication for %s; "
            "the cleanup lease was released before cancellation propagated",
            session_id,
        )
        raise

async def _cleanup_tombstone_target_locked(session_id: str, target_owner: str, target_generation: int, *, bindings) -> dict[str, int]:
    """Clean one captured target while the local cleanup lock is held."""
    _binding__await_cleanup_finalizer = getattr(bindings, "_await_cleanup_finalizer")
    _binding__cleanup_tombstone_side_surfaces = getattr(bindings, "_cleanup_tombstone_side_surfaces")
    _binding__drain_tombstone_producers = getattr(bindings, "_drain_tombstone_producers")
    _binding__empty_cleanup_stats = getattr(bindings, "_empty_cleanup_stats")
    _binding__ensure_cleanup_handles = getattr(bindings, "_ensure_cleanup_handles")
    _binding__finalize_tombstone_target = getattr(bindings, "_finalize_tombstone_target")
    _binding__release_cleanup_handles = getattr(bindings, "_release_cleanup_handles")
    _binding__target_matches = getattr(bindings, "_target_matches")
    _binding_asyncio = getattr(bindings, "asyncio")
    _binding_lifecycle_file_lock = getattr(bindings, "lifecycle_file_lock")
    _binding_read_lifecycle_record = getattr(bindings, "read_lifecycle_record")
    # Claim the non-blocking process/cross-process lease before taking a
    # potentially contended lifecycle flock. `_ensure_cleanup_handles` then
    # revalidates the exact tombstone under that flock. This keeps an async
    # sweeper responsive while a synchronous failed-import cleanup owns both.
    handles = await _binding__ensure_cleanup_handles(
        session_id, target_owner, target_generation,
    )
    if handles is None:
        current = _binding_read_lifecycle_record(session_id)
        return _binding__empty_cleanup_stats(
            pending=int(
                _binding__target_matches(
                    current,
                    target_owner,
                    target_generation,
                    require_incomplete=True,
                )
            )
        )

    pending = await _binding__drain_tombstone_producers(
        session_id, target_owner, target_generation,
    )
    if pending:
        return _binding__empty_cleanup_stats(pending=pending)

    with _binding_lifecycle_file_lock(session_id):
        current = _binding_read_lifecycle_record(session_id)
        if not _binding__target_matches(
            current, target_owner, target_generation,
            require_incomplete=True,
        ):
            _binding__release_cleanup_handles(session_id, handles)
            return _binding__empty_cleanup_stats()

    await _binding__cleanup_tombstone_side_surfaces(session_id, target_generation)
    finalizer = _binding_asyncio.create_task(
        _binding__finalize_tombstone_target(
            session_id,
            target_owner,
            target_generation,
            handles,
        ),
        name=f"tombstone-finalize:{session_id}",
    )
    return await _binding__await_cleanup_finalizer(finalizer, session_id=session_id)

async def cleanup_tombstoned_generation(session_id: str, owner_id: str | None=None, *, bindings) -> dict[str, int]:
    """Retry physical cleanup without reviving the deleted generation."""
    _binding__cleanup_tombstone_target_locked = getattr(bindings, "_cleanup_tombstone_target_locked")
    _binding__empty_cleanup_stats = getattr(bindings, "_empty_cleanup_stats")
    _binding__get_cleanup_lock = getattr(bindings, "_get_cleanup_lock")
    _binding__release_completed_target_handles = getattr(bindings, "_release_completed_target_handles")
    _binding_read_lifecycle_record = getattr(bindings, "read_lifecycle_record")
    _binding_validate_session_id = getattr(bindings, "validate_session_id")
    _binding_validate_session_id(session_id)
    # Capture the exact target before waiting on the local single-flight lock.
    # A delayed generation-g waiter must never retarget itself to generation
    # g+1 merely because that newer generation was also deleted meanwhile.
    record = _binding_read_lifecycle_record(session_id)
    if not record or record.get("status") != "deleted":
        raise ValueError(f"session {session_id} is not tombstoned")
    target_owner = str(record.get("owner_id") or "")
    target_generation = int(record.get("generation") or 0)
    if owner_id is not None and target_owner != owner_id:
        raise PermissionError("session owner does not match lifecycle owner")
    if record.get("cleanup_complete") is True:
        async with _binding__get_cleanup_lock(session_id):
            _binding__release_completed_target_handles(
                session_id, target_owner, target_generation,
            )
        return _binding__empty_cleanup_stats()

    async with _binding__get_cleanup_lock(session_id):
        return await _binding__cleanup_tombstone_target_locked(
            session_id, target_owner, target_generation,
        )

async def sweep_tombstoned_generations(*, bindings) -> dict[str, int]:
    """Best-effort physical cleanup for every stable tombstone."""
    _binding_cleanup_tombstoned_generation = getattr(bindings, "cleanup_tombstoned_generation")
    _binding_list_lifecycle_records = getattr(bindings, "list_lifecycle_records")
    _binding_logger = getattr(bindings, "logger")
    completed = 0
    pending = 0
    failed = 0
    for record in _binding_list_lifecycle_records(status="deleted"):
        session_id = str(record.get("session_id") or "")
        if not session_id:
            continue
        try:
            stats = await _binding_cleanup_tombstoned_generation(session_id)
            if int(stats.get("pending_producers") or 0):
                pending += 1
            else:
                completed += 1
        except Exception:
            failed += 1
            _binding_logger.exception(
                "Background tombstone cleanup failed for session %s",
                session_id,
            )
    return {"completed": completed, "pending": pending, "failed": failed}

async def _tombstone_detach_and_publish(session_id: str, state, owner_id: str, *, force_stop: bool, lease: _binding__TombstoneCleanupLease, bindings) -> tuple[str, int]:
    """Owned linearization step from active generation to cleanup handles."""
    _binding_HTTPException = getattr(bindings, "HTTPException")
    _binding_SessionProducerStopError = getattr(bindings, "SessionProducerStopError")
    _binding__PendingTombstoneCleanup = getattr(bindings, "_PendingTombstoneCleanup")
    _binding__TombstoneCleanupLease = getattr(bindings, "_TombstoneCleanupLease")
    _binding__monitor_tasks = getattr(bindings, "_monitor_tasks")
    _binding__pending_tombstone_cleanup = getattr(bindings, "_pending_tombstone_cleanup")
    _binding__run_tasks = getattr(bindings, "_run_tasks")
    _binding__start_cleanup_task = getattr(bindings, "_start_cleanup_task")
    _binding_asyncio = getattr(bindings, "asyncio")
    _binding_chat_sandboxes = getattr(bindings, "chat_sandboxes")
    _binding_chat_task_lock = getattr(bindings, "chat_task_lock")
    _binding_chat_tasks = getattr(bindings, "chat_tasks")
    _binding_detailed_report_tasks = getattr(bindings, "detailed_report_tasks")
    _binding_get_lifecycle_lock = getattr(bindings, "get_lifecycle_lock")
    _binding_has_live_session_task = getattr(bindings, "has_live_session_task")
    _binding_initializing_sessions = getattr(bindings, "initializing_sessions")
    _binding_last_save_times = getattr(bindings, "last_save_times")
    _binding_mark_session_exports_for_deletion = getattr(bindings, "mark_session_exports_for_deletion")
    _binding_read_lifecycle_record = getattr(bindings, "read_lifecycle_record")
    _binding_release_initializing_session = getattr(bindings, "release_initializing_session")
    _binding_report_websockets = getattr(bindings, "report_websockets")
    _binding_report_workflows = getattr(bindings, "report_workflows")
    _binding_session_access_times = getattr(bindings, "session_access_times")
    _binding_sessions = getattr(bindings, "sessions")
    _binding_snapshot_restore_jobs = getattr(bindings, "snapshot_restore_jobs")
    _binding_state_lock = getattr(bindings, "state_lock")
    _binding_tombstone_generation = getattr(bindings, "tombstone_generation")
    _binding_websockets = getattr(bindings, "websockets")
    _binding_workflows = getattr(bindings, "workflows")
    tasks_to_cancel: list[_binding_asyncio.Task] = []
    workflow = None
    report_workflow = None
    chat_sandbox = None
    chat_task_owns_sandbox = False

    async with _binding_get_lifecycle_lock(session_id):
        existing_record = _binding_read_lifecycle_record(session_id)
        if existing_record and existing_record.get("status") == "deleted":
            if str(existing_record.get("owner_id") or "") != owner_id:
                raise PermissionError("session owner does not match lifecycle owner")
            raise _binding_SessionProducerStopError(
                f"session {session_id} deletion is already owned by another cleaner"
            )

        async with _binding_state_lock:
            if (
                session_id in _binding_workflows
                or session_id in _binding_report_workflows
                or session_id in _binding_initializing_sessions
                or _binding_has_live_session_task(session_id)
            ) and not force_stop:
                raise _binding_HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot delete a running session. Stop the analysis "
                        "first."
                    ),
                )

        tombstone = _binding_tombstone_generation(
            session_id,
            owner_id,
            int(getattr(state, "lifecycle_generation", 0) or 0),
        )
        target_generation = int(tombstone.get("generation") or 0)

        from backend.services.workflow_runner import _report_tasks

        async with _binding_state_lock:
            workflow = _binding_workflows.pop(session_id, None)
            report_workflow = _binding_report_workflows.pop(session_id, None)
            if report_workflow is workflow:
                report_workflow = None
            for owned_workflow in (workflow, report_workflow):
                if owned_workflow is not None:
                    setattr(owned_workflow, "_restore_superseded", True)
            for registry in (
                _binding__run_tasks,
                _report_tasks,
                _binding__monitor_tasks,
                _binding_detailed_report_tasks,
            ):
                task = registry.pop(session_id, None)
                if isinstance(task, _binding_asyncio.Task):
                    tasks_to_cancel.append(task)
            chat_owner_task = _binding_chat_tasks.pop(session_id, None)
            if isinstance(chat_owner_task, _binding_asyncio.Task):
                chat_task_owns_sandbox = not chat_owner_task.done()
                tasks_to_cancel.append(chat_owner_task)
            _binding_release_initializing_session(session_id)
            _binding_sessions.pop(session_id, None)
            _binding_websockets.pop(session_id, None)
            _binding_report_websockets.pop(session_id, None)
            _binding_session_access_times.pop(session_id, None)
            _binding_last_save_times.pop(session_id, None)

        async with _binding_chat_task_lock:
            chat_sandbox = _binding_chat_sandboxes.pop(session_id, None)

        for transaction_id, handle in list(_binding_snapshot_restore_jobs.items()):
            restore_record = getattr(handle, "record", None)
            if getattr(restore_record, "session_id", None) != session_id:
                continue
            cancel_event = getattr(handle, "cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()
            task = getattr(handle, "task", None)
            if isinstance(task, _binding_asyncio.Task):
                tasks_to_cancel.append(task)
            _binding_snapshot_restore_jobs.pop(transaction_id, None)

        tasks_to_cancel.extend(
            await _binding_mark_session_exports_for_deletion(
                session_id,
                target_generation,
            )
        )

        cleanup_handles = _binding__PendingTombstoneCleanup(
            owner_id=owner_id,
            generation=target_generation,
            lease=lease,
            tasks=list(tasks_to_cancel),
            workflow_task=_binding__start_cleanup_task(
                (lambda: workflow.close(save_runtime=False))
                if workflow is not None
                else None,
                name=f"tombstone-workflow-close:{session_id}",
            ),
            report_workflow_task=_binding__start_cleanup_task(
                (lambda: report_workflow.close(save_runtime=False))
                if report_workflow is not None
                else None,
                name=f"tombstone-report-workflow-close:{session_id}",
            ),
            chat_task=_binding__start_cleanup_task(
                (lambda: chat_sandbox.stop(save=False))
                if chat_sandbox is not None and not chat_task_owns_sandbox
                else None,
                name=f"tombstone-chat-stop:{session_id}",
            ),
            workflow=workflow,
            report_workflow=report_workflow,
            chat_sandbox=chat_sandbox,
        )
        if session_id in _binding__pending_tombstone_cleanup:
            raise RuntimeError(
                f"cleanup handles already exist for active session {session_id}"
            )
        _binding__pending_tombstone_cleanup[session_id] = cleanup_handles
        return owner_id, target_generation

async def delete_session_generation(session_id: str, state, owner_id: str, *, force_stop: bool=False, bindings) -> dict[str, int]:
    """Persist a stable tombstone, stop producers, then remove every surface.

    Interactive deletion remains conservative and rejects a running workflow.
    Account-level cascade deletion uses ``force_stop=True``: the tombstone is
    still the linearization point, so cancelled producers cannot save after it.
    """
    _binding_SessionProducerStopError = getattr(bindings, "SessionProducerStopError")
    _binding__cleanup_tombstone_target_locked = getattr(bindings, "_cleanup_tombstone_target_locked")
    _binding__empty_cleanup_stats = getattr(bindings, "_empty_cleanup_stats")
    _binding__get_cleanup_lock = getattr(bindings, "_get_cleanup_lock")
    _binding__pending_tombstone_cleanup = getattr(bindings, "_pending_tombstone_cleanup")
    _binding__tombstone_detach_and_publish = getattr(bindings, "_tombstone_detach_and_publish")
    _binding__try_acquire_cleanup_lease = getattr(bindings, "_try_acquire_cleanup_lease")
    _binding_asyncio = getattr(bindings, "asyncio")
    _binding_consume_current_cancellation_request = getattr(bindings, "consume_current_cancellation_request")
    _binding_logger = getattr(bindings, "logger")
    _binding_read_lifecycle_record = getattr(bindings, "read_lifecycle_record")
    _binding_validate_session_id = getattr(bindings, "validate_session_id")
    _binding_validate_session_id(session_id)
    newly_tombstoned = False
    async with _binding__get_cleanup_lock(session_id):
        existing = _binding_read_lifecycle_record(session_id)
        if existing and existing.get("status") == "deleted":
            target_owner = str(existing.get("owner_id") or "")
            target_generation = int(existing.get("generation") or 0)
            if target_owner != owner_id:
                raise PermissionError("session owner does not match lifecycle owner")
            if existing.get("cleanup_complete") is True:
                return _binding__empty_cleanup_stats()
        else:
            lease = await _binding__try_acquire_cleanup_lease(session_id)
            if lease is None:
                raise _binding_SessionProducerStopError(
                    f"session {session_id} deletion is already in progress"
                )
            publication = _binding_asyncio.create_task(
                _binding__tombstone_detach_and_publish(
                    session_id,
                    state,
                    owner_id,
                    force_stop=force_stop,
                    lease=lease,
                ),
                name=f"tombstone-publish:{session_id}",
            )
            cancellation: _binding_asyncio.CancelledError | None = None
            try:
                while True:
                    try:
                        target_owner, target_generation = await _binding_asyncio.shield(
                            publication
                        )
                        break
                    except _binding_asyncio.CancelledError as exc:
                        _binding_consume_current_cancellation_request()
                        if publication.done():
                            target_owner, target_generation = publication.result()
                            cancellation = cancellation or exc
                            break
                        cancellation = cancellation or exc
                        _binding_logger.warning(
                            "Deletion request cancelled while publishing "
                            "tombstone handles for %s; deferring cancellation",
                            session_id,
                        )
                newly_tombstoned = True
            except BaseException:
                if _binding__pending_tombstone_cleanup.get(session_id) is None:
                    lease.release()
                raise
            if cancellation is not None:
                # The durable tombstone and producer handles now exist. A later
                # request or background sweep can safely finish cleanup.
                raise cancellation

        stats = await _binding__cleanup_tombstone_target_locked(
            session_id, target_owner, target_generation,
        )

    pending_producers = int(stats.get("pending_producers") or 0)
    if newly_tombstoned and pending_producers:
        raise _binding_SessionProducerStopError(
            f"session {session_id} remains tombstoned; "
            f"{pending_producers} producer(s) did not stop before timeout"
        )
    return stats
