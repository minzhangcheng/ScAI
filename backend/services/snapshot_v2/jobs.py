"""Cancellable background restore jobs for Snapshot V2.

Provides ``RestoreJobManager`` which manages long-running restore transactions
with support for cancellation, safe-point coordination, timeout, and WebSocket
progress events.

Jobs are registered in the process-wide ``snapshot_restore_jobs`` dict so that
cancel/status requests from independent HTTP handlers share the same handle.

Job state machine::

    queued → waiting_safe_point → staging → staged → applying → committed
                                                                    ↓
                                      ┌────── ← ─ ─ ─ ─ ─ ─ ─ ─ ───┤
                                      ↓                             ↓
                                  cancelled                     failed

Timeout branches from ``waiting_safe_point`` → ``timed_out``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.snapshot_v2.errors import SnapshotTransactionError
from backend.services.snapshot_v2.models import SnapshotRestoreRequest, SnapshotTransactionRecord
from backend.services.snapshot_v2.preflight import RestorePreflightResult
from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_transaction_id() -> str:
    """Job status journal id (distinct from restore_apply mutation journal)."""
    return f"restore-job-{uuid.uuid4().hex[:16]}"


def restore_apply_transaction_id(job_tx_id: str) -> str:
    """Map restore_job id → restore_apply id so both can coexist on disk."""
    if job_tx_id.startswith("restore-job-"):
        return "restore-apply-" + job_tx_id[len("restore-job-"):]
    if job_tx_id.startswith("restore-apply-"):
        return job_tx_id
    # Legacy / tests may still use restore-* for both; keep as-is only when
    # a single id is intentional (tests). Prefer distinct ids in production.
    return job_tx_id


@dataclass
class RestoreJobHandle:
    """Process-wide handle for a restore job.

    Terminal jobs remain queryable: only ``task`` is cleared on cleanup so
    ``record`` (and the on-disk journal) stay available for status polling.
    """

    record: SnapshotTransactionRecord
    task: asyncio.Task | None
    cancel_event: asyncio.Event
    paths: Any = None  # SnapshotV2Paths-like; journals directory for load_transaction


def get_restore_job(transaction_id: str) -> RestoreJobHandle | None:
    """Return the process-wide handle for *transaction_id*, or ``None``."""
    from backend.state import snapshot_restore_jobs

    return snapshot_restore_jobs.get(transaction_id)


def load_transaction(
    transaction_id: str,
    *,
    paths: Any = None,
) -> SnapshotTransactionRecord | None:
    """Load the latest transaction record from the shared handle or journal.

    Prefers the in-memory handle when present. When no handle exists, falls
    back to a verified ``restore_job`` envelope. Terminal jobs keep the handle
    (with ``task=None``) so this remains queryable after task cleanup.
    """
    handle = get_restore_job(transaction_id)
    if handle is not None:
        return handle.record
    if paths is None:
        return None
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.errors import SnapshotIntegrityError
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    store = JournalStore(paths)
    try:
        env = store.try_read(transaction_id)
    except RecoveryFailure:
        # Unsigned flat SnapshotTransactionRecord JSON (legacy/tests):
        # accept exact transaction-record shape only.
        journals = getattr(paths, "journals", None)
        if journals is None:
            return None
        journal_path = Path(journals) / f"{transaction_id}.json"
        if not journal_path.is_file():
            return None
        try:
            return SnapshotTransactionRecord.model_validate_json(
                journal_path.read_text()
            )
        except Exception:
            return None
    except SnapshotIntegrityError:
        return None
    if env is None:
        return None
    if env.kind == "restore_job":
        payload = env.payload or {}
        raw = payload.get("record")
        if isinstance(raw, dict):
            try:
                return SnapshotTransactionRecord.model_validate(raw)
            except Exception:
                pass
        try:
            return SnapshotTransactionRecord(
                transaction_id=env.transaction_id,
                session_id=env.session_id,
                snapshot_id=str(payload.get("snapshot_id") or ""),
                status=env.status,  # type: ignore[arg-type]
                recovery_mode=str(payload.get("recovery_mode") or "rollback"),  # type: ignore[arg-type]
                failure=payload.get("failure"),
            )
        except Exception:
            logger.debug(
                "[snapshot-v2] load_transaction: failed to parse restore_job %s",
                transaction_id,
                exc_info=True,
            )
            return None
    return None


async def cancel_restore_job(
    transaction_id: str,
    *,
    session_id: str | None = None,
) -> None:
    """Cancel a restore job via the process-wide registry (cross-request).

    When *session_id* is provided, the handle must belong to that session;
    missing or mismatched jobs raise ``SnapshotTransactionError`` (HTTP 404).
    """
    handle = get_restore_job(transaction_id)
    if handle is None:
        raise SnapshotTransactionError(
            "TRANSACTION_NOT_FOUND",
            f"Restore transaction not found: {transaction_id}",
        )

    record = handle.record
    if session_id is not None and record.session_id != session_id:
        raise SnapshotTransactionError(
            "TRANSACTION_NOT_FOUND",
            f"Restore transaction not found for session: {transaction_id}",
        )

    if record.status in ("committed", "failed", "timed_out", "cancelled"):
        logger.debug(
            "[snapshot-v2] cancel_restore_job: %s already terminal (%s)",
            transaction_id, record.status,
        )
        return

    # Persist cancelled before signalling so concurrent status polls see it
    updated = record.model_copy(update={"status": "cancelled"})
    handle.record = updated
    _persist_handle(handle)

    # Release safe point if pending or acquired
    get_snapshot_safe_point_manager().cancel_safe_point_request(record.session_id)

    handle.cancel_event.set()
    if handle.task is not None and not handle.task.done():
        handle.task.cancel()


def _persist_handle(handle: RestoreJobHandle) -> None:
    """Write restore_job envelope for the handle's current record (before WS).

    Never overwrites a ``restore_apply`` mutation journal. Job status uses
    ``restore-job-{id}`` transaction IDs; the coordinator uses
    ``restore-apply-{id}`` (or the same id only when tests force it).
    """
    if handle.paths is None:
        return
    journals = getattr(handle.paths, "journals", None)
    if journals is None:
        return
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope
    from backend.services.snapshot_v2.errors import SnapshotIntegrityError
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    store = JournalStore(handle.paths)
    tx_id = handle.record.transaction_id
    # Refuse to clobber an existing restore_apply envelope at the same path.
    try:
        existing = store.try_read(tx_id)
        if existing is not None and existing.kind == "restore_apply":
            logger.warning(
                "[snapshot-v2] refuse restore_job persist over restore_apply %s",
                tx_id,
            )
            return
    except (SnapshotIntegrityError, RecoveryFailure):
        # Integrity failure on existing file — do not overwrite
        logger.warning(
            "[snapshot-v2] refuse restore_job persist over unreadable journal %s",
            tx_id,
        )
        return

    record = handle.record
    env = make_envelope(
        kind="restore_job",
        transaction_id=tx_id,
        session_id=record.session_id,
        status=record.status,
        payload={
            "snapshot_id": record.snapshot_id,
            "recovery_mode": record.recovery_mode,
            "failure": record.failure,
            "record": record.model_dump(mode="json"),
        },
    )
    store.write(env)




def collect_active_restore_protected_ids(
    session_id: str,
    *,
    paths: Any = None,
) -> dict[str, set[str]]:
    """Return restore-related protections for *session_id*.

    Keys: restore_target, pre_restore.
    Combines the in-process job registry with on-disk restore journals.
    """
    out: dict[str, set[str]] = {"restore_target": set(), "pre_restore": set()}
    terminal = {"committed", "failed", "cancelled", "timed_out"}

    try:
        from backend.state import snapshot_restore_jobs
        for handle in list(snapshot_restore_jobs.values()):
            record = getattr(handle, "record", None)
            if record is None:
                continue
            if getattr(record, "session_id", None) != session_id:
                continue
            if getattr(record, "status", None) in terminal:
                continue
            sid = getattr(record, "snapshot_id", None)
            if sid:
                out["restore_target"].add(str(sid))
    except Exception:
        pass

    if paths is not None:
        from backend.services.snapshot_v2.journal_store import JournalStore
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.recovery import RecoveryFailure

        store = JournalStore(paths)
        journals = getattr(paths, "journals", None)
        if journals is not None:
            jdir = Path(journals)
            if jdir.exists():
                for entry in jdir.iterdir():
                    if not entry.is_file() or entry.suffix != ".json":
                        continue
                    try:
                        env = store.read(entry.stem)
                    except (SnapshotIntegrityError, RecoveryFailure, Exception):
                        continue
                    if env.session_id not in (None, "", session_id):
                        continue
                    if env.kind == "restore_apply":
                        payload = env.payload or {}
                        st = str(env.status or "").lower()
                        if st in ("cancelled", "timed_out"):
                            continue
                        if st == "committed" and not payload.get("pre_restore_ref"):
                            continue
                        target = payload.get("target_snapshot_id")
                        if target:
                            out["restore_target"].add(str(target))
                        pre = payload.get("pre_restore_ref")
                        if pre:
                            out["pre_restore"].add(str(pre))
                    elif env.kind == "restore_job":
                        st = str(env.status or "").lower()
                        if st in ("committed", "failed", "cancelled", "timed_out"):
                            continue
                        sid = (env.payload or {}).get("snapshot_id")
                        if sid:
                            out["restore_target"].add(str(sid))
    return out


class RestoreJobManager:
    """Manages cancellable background restore jobs.

    Each job is an ``asyncio.Task`` that progresses through the restore
    state machine.  Jobs are stored in the global ``snapshot_restore_jobs``
    dict (from ``backend.state``) and persisted as JSON journals under the
    session's ``journals/`` directory.

    The job runner never invents lifecycle states independently after
    safe-point acknowledgement: it delegates to
    ``SnapshotService.restore_snapshot`` as the sole orchestration method.
    """

    # Order of states for the restore lifecycle
    STATE_ORDER = (
        "waiting_safe_point",
        "staging",
        "staged",
        "applying",
        "committed",
    )

    def __init__(
        self,
        paths: Any,
        session_id: str,
        *,
        snapshot_service: Any = None,
    ) -> None:
        self.paths = paths
        self.session_id = session_id
        self._state_hook = None
        self._safe_point = get_snapshot_safe_point_manager()
        self._barrier_events: dict[str, asyncio.Event] = {}
        self._snapshot_service = snapshot_service
        self._pending_request: SnapshotRestoreRequest | None = None

    # ── Public API ───────────────────────────────────────────────────

    async def enqueue(
        self,
        session_id: str,
        preflight: RestorePreflightResult,
        request: SnapshotRestoreRequest,
        *,
        timeout_seconds: int = 300,
    ) -> SnapshotTransactionRecord:
        """Create and start a background restore job.

        Returns a ``SnapshotTransactionRecord`` with status ``"queued"``.
        The caller should return this immediately (HTTP 202) while the
        background task progresses through the state machine.
        """
        from backend.state import snapshot_restore_jobs

        tx_id = _new_transaction_id()
        record = SnapshotTransactionRecord(
            transaction_id=tx_id,
            session_id=session_id,
            snapshot_id=request.snapshot_id,
            status="queued",
        )

        cancel_event = asyncio.Event()
        handle = RestoreJobHandle(
            record=record,
            task=None,
            cancel_event=cancel_event,
            paths=self.paths,
        )
        # Persist before registering so status polls never see a missing journal
        _persist_handle(handle)
        snapshot_restore_jobs[tx_id] = handle

        task = asyncio.create_task(
            self._run_job(handle, preflight, request, timeout_seconds),
        )
        handle.task = task
        task.add_done_callback(lambda _: self._on_task_done(tx_id))

        logger.info(
            "[snapshot-v2] Restore job %s enqueued for session %s (snapshot %s)",
            tx_id, session_id, request.snapshot_id,
        )
        return record

    async def cancel(self, transaction_id: str) -> None:
        """Cancel a running restore job (delegates to process-wide cancel)."""
        await cancel_restore_job(transaction_id, session_id=self.session_id)

    def get(self, transaction_id: str) -> SnapshotTransactionRecord | None:
        """Return the current record for *transaction_id*, or ``None``."""
        handle = get_restore_job(transaction_id)
        if handle is not None:
            return handle.record
        return None

    # ── Internal job runner ──────────────────────────────────────────

    async def _yield_for_barrier(self, checkpoint: str) -> None:
        """Yield control at a checkpoint and wait if a barrier event is set.

        If ``_barrier_events[checkpoint]`` exists and is *not* set, this
        waits on it — allowing tests to freeze the job at a specific state.
        The event is then consumed (removed) so subsequent calls pass through.
        """
        event = self._barrier_events.pop(checkpoint, None)
        if event is not None and not event.is_set():
            await event.wait()

    def _get_or_create_service(self) -> Any:
        if self._snapshot_service is not None:
            return self._snapshot_service
        from backend.services.snapshot_v2.service import SnapshotService
        return SnapshotService(self.session_id, self.paths)

    async def _run_job(
        self,
        handle: RestoreJobHandle,
        preflight: RestorePreflightResult,
        request: SnapshotRestoreRequest,
        timeout_seconds: int,
    ) -> None:
        """Background coroutine — safe-point wait then service orchestration."""
        record = handle.record
        session_id = record.session_id
        tx_id = record.transaction_id
        stop_event = handle.cancel_event
        t0 = time.monotonic()

        try:
            # ── queued → waiting_safe_point ──────────────────────
            record = self._update(handle, "waiting_safe_point")
            self._safe_point.request_safe_point(session_id)

            # Wait for safe-point acknowledgment, cancellation, or timeout
            ack_event = self._safe_point.get_pending_event(session_id)
            if ack_event is not None:
                try:
                    await asyncio.wait_for(
                        self._race(ack_event, stop_event),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - t0
                    record = self._update(handle, "timed_out")
                    self._safe_point.cancel_safe_point_request(session_id)
                    logger.info(
                        "[snapshot-v2] Restore job %s timed out after %.1fs",
                        tx_id, elapsed,
                    )
                    return

            if stop_event.is_set():
                record = self._update(handle, "cancelled")
                self._safe_point.cancel_safe_point_request(session_id)
                return

            # Test barriers: allow freeze before orchestration (legacy hooks).
            await self._yield_for_barrier("staging")
            if stop_event.is_set():
                record = self._update(handle, "cancelled")
                self._safe_point.release(session_id)
                return

            # ── Sole orchestration: SnapshotService.restore_snapshot ──
            service = self._get_or_create_service()
            # Mutation journal uses restore-apply-{suffix}; job status stays
            # on restore-job-{suffix} so the two envelopes never share a file.
            apply_tx_id = restore_apply_transaction_id(tx_id)

            def _on_status(rec: SnapshotTransactionRecord) -> None:
                # Map orchestration status onto the in-memory handle only.
                # Do NOT persist intermediate statuses — the restore coordinator
                # owns the restore_apply envelope during stage/apply.
                current = handle.record.status
                if current in ("cancelled", "failed", "timed_out", "committed"):
                    if rec.status != current:
                        return
                # Keep job transaction_id stable for status polls / cancel.
                mapped = rec.model_copy(update={"transaction_id": tx_id})
                handle.record = mapped
                if self._state_hook is not None:
                    asyncio.ensure_future(self._state_hook(mapped))
                asyncio.ensure_future(self._emit_state_sync(mapped, 0.0))

            result = await service.restore_snapshot(
                request,
                apply_tx_id,
                stop_event,
                status_callback=_on_status,
            )

            # Final status on the in-memory handle (job id, not apply id).
            handle.record = result.model_copy(update={"transaction_id": tx_id})
            # Persist restore_job envelope; never overwrites restore_apply.
            failure = result.failure or {}
            recovery_required = bool(failure.get("recovery_required"))
            if result.status in ("committed", "cancelled", "timed_out") or (
                result.status == "failed" and not recovery_required
            ):
                _persist_handle(handle)
            if self._state_hook is not None:
                asyncio.ensure_future(self._state_hook(result))
            asyncio.ensure_future(self._emit_state_sync(result, 0.0))

            self._safe_point.release(session_id)

            elapsed = time.monotonic() - t0
            logger.info(
                "[snapshot-v2] Restore job %s finished status=%s in %.1fs",
                tx_id, handle.record.status, elapsed,
            )

        except asyncio.CancelledError:
            if handle.record.status not in ("cancelled", "committed", "failed", "timed_out"):
                self._update(handle, "cancelled")
            self._safe_point.cancel_safe_point_request(session_id)
            raise
        except Exception as exc:
            logger.exception("[snapshot-v2] Restore job %s failed: %s", tx_id, exc)
            self._update(handle, "failed", failure={"message": str(exc)})
            self._safe_point.cancel_safe_point_request(session_id)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    async def _race(*events: asyncio.Event) -> None:
        """Wait for any of the given events to be set."""
        tasks = [asyncio.create_task(e.wait()) for e in events]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

    def _update(
        self,
        handle: RestoreJobHandle,
        status: str,
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
        # Persist first
        _persist_handle(handle)

        # Notify test hook
        if self._state_hook is not None:
            asyncio.ensure_future(self._state_hook(record))

        # Emit WebSocket state sync after persist
        asyncio.ensure_future(self._emit_state_sync(record, 0.0))
        return record

    async def _emit_state_sync(
        self,
        record: SnapshotTransactionRecord,
        elapsed: float = 0.0,
    ) -> None:
        from backend.state import websockets

        ws = websockets.get(record.session_id)
        if ws is None:
            return
        try:
            payload = record.model_dump(mode="json")
            payload["elapsed_seconds"] = round(elapsed, 1)
            payload["active_sandbox_step"] = 0
            payload["timeout_seconds"] = 300

            await ws.send_json({
                "type": "state_sync",
                "change_type": "snapshot_transaction",
                "snapshot_transaction": payload,
            })
        except Exception:
            logger.debug(
                "[%s] Failed to emit snapshot_transaction state sync",
                record.session_id,
            )

    def _on_task_done(self, tx_id: str) -> None:
        """Release only the task reference; keep handle for journal queries."""
        handle = get_restore_job(tx_id)
        if handle is not None:
            handle.task = None
        # Also clear legacy snapshot_jobs entry if present
        from backend.state import snapshot_jobs

        snapshot_jobs.pop(tx_id, None)
