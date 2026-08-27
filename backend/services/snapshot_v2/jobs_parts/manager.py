"""Restore-job manager execution owners."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from backend.services.snapshot_v2.errors import SnapshotTransactionError
from backend.services.snapshot_v2.models import (
    SnapshotRestoreRequest,
    SnapshotTransactionRecord,
)
from backend.services.snapshot_v2.preflight import RestorePreflightResult
from backend.services.snapshot_v2.safe_point import SafePointRequest

logger = logging.getLogger(__name__)


async def enqueue(
    manager,
    session_id: str,
    preflight: RestorePreflightResult,
    request: SnapshotRestoreRequest,
    *,
    timeout_seconds: int = 300,
    retry_of_transaction_id: str | None = None,
    bindings,
) -> SnapshotTransactionRecord:
    """Create and start a background restore job.

    Returns a ``SnapshotTransactionRecord`` with status ``"queued"``.
    The caller should return this immediately (HTTP 202) while the
    background task progresses through the state machine.
    """
    from backend.state import snapshot_restore_jobs

    tx_id = bindings._new_transaction_id()
    record = SnapshotTransactionRecord(
        transaction_id=tx_id,
        session_id=session_id,
        snapshot_id=request.snapshot_id,
        status="queued",
        retry_of_transaction_id=retry_of_transaction_id,
    )

    owner_token = f"restore:{tx_id}"
    from backend.services.session_lifecycle import (
        SessionGenerationError,
        SessionGoneError,
        establish_snapshot_operation,
    )
    from backend.state import detailed_report_tasks, state_lock

    # Detailed-report generation has no cooperative snapshot boundary and
    # commonly outlives the restore timeout.  Fail immediately instead of
    # publishing a restore job that can only sit in waiting_safe_point.  The
    # shared reservation barrier makes this check atomic with every workflow
    # producer publication.  Reservation itself is non-blocking: waiting for
    # a foreign hold while retaining the lifecycle lock would deadlock a
    # capture that needs the same lock to persist before releasing its hold.
    async def _validate_report_owner() -> None:
        async with state_lock:
            detailed_task = detailed_report_tasks.get(session_id)
            if detailed_task is not None and not detailed_task.done():
                raise SnapshotTransactionError(
                    "SESSION_BUSY",
                    "Detailed report generation is active for this session",
                    details={"session_id": session_id},
                )

    handle: RestoreJobHandle | None = None

    def _publish_restore_job(
        reservation: SafePointRequest,
        lifecycle_generation: int,
    ) -> None:
        """Publish the complete owner handle before deletion can proceed."""
        nonlocal handle, record
        handle = bindings.RestoreJobHandle(
            record=record,
            task=None,
            cancel_event=asyncio.Event(),
            paths=manager.paths,
            owner_token=owner_token,
            safe_point_request=reservation,
            lifecycle_generation=lifecycle_generation,
        )
        if not bindings._persist_handle(handle):
            raise SessionGenerationError(
                "restore generation changed before job publication"
            )
        snapshot_restore_jobs[tx_id] = handle
        handle.task = asyncio.create_task(
            manager._run_job(handle, preflight, request, timeout_seconds)
        )
        handle.task.add_done_callback(lambda _: manager._on_task_done(tx_id))

    try:
        reservation = await establish_snapshot_operation(
            session_id,
            lambda: manager._safe_point.request_safe_point(
                session_id, owner_token,
            ),
            validate=_validate_report_owner,
            publish=_publish_restore_job,
        )
        if reservation is None or reservation is False or handle is None:
            raise SnapshotTransactionError(
                "SAFE_POINT_BUSY",
                "Another snapshot or restore operation owns the session "
                "safe point",
                details={"session_id": session_id},
            )
    except BaseException as exc:
        if handle is not None:
            if handle.task is not None and not handle.task.done():
                handle.task.cancel()
            handle.record = handle.record.model_copy(
                update={
                    "status": "failed",
                    "failure": {
                        "code": "RESTORE_JOB_START_FAILED",
                        "message": str(exc) or type(exc).__name__,
                    },
                }
            )
            try:
                bindings._persist_handle(handle)
            except Exception:
                logger.exception(
                    "[snapshot-v2] failed to persist restore job start "
                    "failure %s",
                    tx_id,
                )
        # Keep a published handle visible until its reservation is gone.
        # Deletion therefore cannot miss a safe-point owner between these
        # two operations.
        await manager._safe_point.cancel_safe_point_request(
            session_id, owner_token,
        )
        if handle is not None and snapshot_restore_jobs.get(tx_id) is handle:
            snapshot_restore_jobs.pop(tx_id, None)
        if isinstance(exc, (SessionGoneError, SessionGenerationError)):
            raise SnapshotTransactionError(
                "SESSION_GONE",
                "Session was deleted or replaced before restore publication",
                details={"session_id": session_id},
            ) from exc
        raise

    logger.info(
        "[snapshot-v2] Restore job %s enqueued for session %s (snapshot %s)",
        tx_id, session_id, request.snapshot_id,
    )
    return record


async def _run_job(
    manager,
    handle: Any,
    preflight: RestorePreflightResult,
    request: SnapshotRestoreRequest,
    timeout_seconds: int,
    *,
    bindings,
) -> None:
    """Background coroutine — safe-point wait then service orchestration."""
    record = handle.record
    session_id = record.session_id
    tx_id = record.transaction_id
    owner_token = handle.owner_token or f"restore:{tx_id}"
    if not handle.owner_token:
        handle.owner_token = owner_token
    stop_event = handle.cancel_event
    t0 = time.monotonic()
    from backend.services.session_lifecycle import (
        SessionGenerationError,
        SessionGoneError,
        assert_session_generation_active,
    )

    try:
        assert_session_generation_active(
            session_id,
            expected_generation=handle.lifecycle_generation,
        )
        # ── queued → waiting_safe_point ──────────────────────
        record = manager._update(handle, "waiting_safe_point")
        reservation = handle.safe_point_request
        if reservation is None:
            raise SnapshotTransactionError(
                "SAFE_POINT_REQUEST_MISSING",
                "Restore job started without an owned safe-point request",
                details={"transaction_id": tx_id},
            )
        # This runner proves quiescence immediately and then rechecks it.
        # A proof expires the moment the activity it observed ends: every
        # other acknowledger in the system — the graph boundaries, the HITL
        # gate hooks, the sandbox step loop — needs a LIVE run task to reach
        # it. So when the chat delegation / report generation / initializing
        # workflow that made the session non-quiescent finishes without ever
        # passing an acknowledgement site (the chat and report paths contain
        # none at all), nothing is left to hand the reservation over. The
        # single `wait_for` below then waited on an event that no longer had
        # a setter: the restore burned its full timeout and reported
        # `timed_out` on a session that had been idle for minutes, while its
        # pending request made every other snapshot operation answer
        # SAFE_POINT_BUSY for the whole window.
        #
        # Re-prove quiescence in slices instead. `_acknowledge_if_quiescent`
        # does the entire probe under `state_lock` and the acknowledgement is
        # generation-checked, so a run that registered in the meantime makes
        # the probe decline, and a workflow that acknowledged first leaves no
        # pending request for the ack to match.
        deadline = time.monotonic() + timeout_seconds
        try:
            await manager._acknowledge_if_quiescent(
                session_id,
                reservation,
                expected_lifecycle_generation=handle.lifecycle_generation,
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                try:
                    await asyncio.wait_for(
                        manager._race(reservation.event, stop_event),
                        timeout=min(remaining, bindings.QUIESCENCE_RECHECK_SECONDS),
                    )
                    break
                except asyncio.TimeoutError:
                    if await manager._acknowledge_if_quiescent(
                        session_id,
                        reservation,
                        expected_lifecycle_generation=(
                            handle.lifecycle_generation
                        ),
                    ):
                        break
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            record = manager._update(handle, "timed_out")
            await manager._safe_point.cancel_safe_point_request(
                session_id, owner_token
            )
            logger.info(
                "[snapshot-v2] Restore job %s timed out after %.1fs",
                tx_id, elapsed,
            )
            return

        if stop_event.is_set():
            record = manager._update(handle, "cancelled")
            await manager._safe_point.cancel_safe_point_request(
                session_id, owner_token,
            )
            return

        # Test barriers: allow freeze before orchestration (legacy hooks).
        await manager._yield_for_barrier("staging")
        if stop_event.is_set():
            record = manager._update(handle, "cancelled")
            await manager._safe_point.release(session_id, owner_token)
            return

        # ── Sole orchestration: SnapshotService.restore_snapshot ──
        service = manager._get_or_create_service()
        # Mutation journal uses restore-apply-{suffix}; job status stays
        # on restore-job-{suffix} so the two envelopes never share a file.
        apply_tx_id = bindings.restore_apply_transaction_id(tx_id)

        def _on_status(rec: SnapshotTransactionRecord) -> None:
            try:
                assert_session_generation_active(
                    session_id,
                    expected_generation=handle.lifecycle_generation,
                )
            except (SessionGoneError, SessionGenerationError):
                return
            # Map orchestration status onto the in-memory handle only.
            # Do NOT persist intermediate statuses — the restore coordinator
            # owns the restore_apply envelope during stage/apply.
            current = handle.record.status
            if current in ("cancelled", "failed", "timed_out", "committed"):
                if rec.status != current:
                    return
            # Keep job transaction_id stable for status polls / cancel.
            mapped = rec.model_copy(update={
                "transaction_id": tx_id,
                "retry_of_transaction_id": (
                    handle.record.retry_of_transaction_id
                ),
            })
            handle.record = mapped
            if manager._state_hook is not None:
                asyncio.ensure_future(manager._state_hook(mapped))
            asyncio.ensure_future(manager._emit_state_sync(
                mapped,
                0.0,
                lifecycle_generation=handle.lifecycle_generation,
            ))

        assert_session_generation_active(
            session_id,
            expected_generation=handle.lifecycle_generation,
        )
        result = await service.restore_snapshot(
            request,
            apply_tx_id,
            stop_event,
            job_transaction_id=tx_id,
            status_callback=_on_status,
            expected_lifecycle_generation=(
                handle.lifecycle_generation
            ),
        )

        # Final status on the in-memory handle (job id, not apply id).
        assert_session_generation_active(
            session_id,
            expected_generation=handle.lifecycle_generation,
        )
        handle.record = result.model_copy(update={
            "transaction_id": tx_id,
            "retry_of_transaction_id": handle.record.retry_of_transaction_id,
        })
        # Persist restore_job envelope; never overwrites restore_apply.
        #
        # Every terminal outcome is written, recovery-required failures
        # included. Skipping those left the journal on the last status
        # ``_update`` wrote (waiting_safe_point) while the job was in fact
        # over, and nothing ever reconciles it: recovery removes only
        # *terminal* restore_job journals, so the stale one survives every
        # session load and keeps compute_protected_snapshot_ids reporting
        # the target as restore_target - the snapshot is undeletable and
        # un-reclaimable for good, and status polls after a restart read
        # waiting_safe_point forever. What still needs recovery is the
        # restore_apply journal (left APPLYING) plus the session's
        # recovery_blocked marker; those protect the half-applied session,
        # not a job-status record that lies about the job's state.
        if bindings._persist_handle(handle):
            if manager._state_hook is not None:
                asyncio.ensure_future(manager._state_hook(handle.record))
            asyncio.ensure_future(manager._emit_state_sync(
                handle.record,
                0.0,
                lifecycle_generation=handle.lifecycle_generation,
            ))

        await manager._safe_point.release(session_id, owner_token)

        elapsed = time.monotonic() - t0
        logger.info(
            "[snapshot-v2] Restore job %s finished status=%s in %.1fs",
            tx_id, handle.record.status, elapsed,
        )

    except asyncio.CancelledError:
        if handle.record.status not in ("cancelled", "committed", "failed", "timed_out"):
            manager._update(handle, "cancelled")
        raise
    except (SessionGoneError, SessionGenerationError) as exc:
        logger.info(
            "[snapshot-v2] Restore job %s stopped after lifecycle "
            "generation replacement: %s",
            tx_id,
            exc,
        )
        manager._update(
            handle,
            "failed",
            failure={
                "code": "SESSION_GENERATION_STALE",
                "message": str(exc),
            },
        )
    except Exception as exc:
        logger.exception("[snapshot-v2] Restore job %s failed: %s", tx_id, exc)
        manager._update(handle, "failed", failure={"message": str(exc)})
    finally:
        # The hold must not outlive the job on ANY path. Both handlers above
        # persist a journal record before releasing, and that write can fail
        # on its own (ENOSPC, EDQUOT, read-only mount) — which used to skip
        # the release entirely and strand the safe point in the process-wide
        # manager, so every later capture and restore for the session failed
        # SAFE_POINT_BUSY until a restart. Same structural gap that was
        # closed in SnapshotService._capture_impl.
        #
        # cancel_safe_point_request is idempotent: it withdraws a pending
        # request and releases an acquired hold under matching ownership,
        # and returns False without side effects when there is nothing to
        # release — so calling it unconditionally is safe on the paths that
        # already released.
        #
        # shield keeps the release itself from being cancelled partway; a
        # failure here is logged rather than raised so it can never mask the
        # exception that is already unwinding.
        try:
            await asyncio.shield(
                manager._safe_point.cancel_safe_point_request(
                    session_id, owner_token,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[snapshot-v2] Restore job %s: failed to release the safe "
                "point; the session may be left SAFE_POINT_BUSY", tx_id,
            )


def _update(
    manager,
    handle: Any,
    status: str,
    *,
    bindings,
    **kw: Any,
) -> SnapshotTransactionRecord:
    """Atomically update the record status, persist, then emit events.

    Persist-before-WebSocket: journal is written before any WS progress.
    """
    # Do not overwrite a terminal status set by concurrent cancel
    current = handle.record.status
    if current in ("cancelled", "failed", "timed_out", "committed") and status != current:
        return handle.record

    record = handle.record.model_copy(update={"status": status, **kw})
    handle.record = record
    # Persist first. A stale generation must not publish state or events
    # into a replacement session with the same ID.
    if not bindings._persist_handle(handle):
        return record

    # Notify test hook
    if manager._state_hook is not None:
        asyncio.ensure_future(manager._state_hook(record))

    # Emit WebSocket state sync after persist
    asyncio.ensure_future(manager._emit_state_sync(
        record,
        0.0,
        lifecycle_generation=handle.lifecycle_generation,
    ))
    return record


async def _emit_state_sync(
    manager,
    record: SnapshotTransactionRecord,
    elapsed: float = 0.0,
    *,
    lifecycle_generation: int | None = None,
    bindings,
) -> None:
    from backend.services.session_lifecycle import (
        SessionGenerationError,
        SessionGoneError,
        assert_session_generation_active,
    )
    from backend.utils.websocket_delivery import send_analysis_json

    try:
        assert_session_generation_active(
            record.session_id,
            expected_generation=lifecycle_generation,
        )
    except (SessionGoneError, SessionGenerationError):
        return

    payload = record.model_dump(mode="json")
    payload["elapsed_seconds"] = round(elapsed, 1)
    payload["active_sandbox_step"] = 0
    payload["timeout_seconds"] = 300

    await send_analysis_json(record.session_id, {
        "type": "state_sync",
        "change_type": "snapshot_transaction",
        "snapshot_transaction": payload,
    })
