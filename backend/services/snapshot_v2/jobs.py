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
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.snapshot_v2.errors import SnapshotTransactionError
from backend.services.snapshot_v2.models import SnapshotRestoreRequest, SnapshotTransactionRecord
from backend.services.snapshot_v2.preflight import RestorePreflightResult
from backend.services.snapshot_v2.safe_point import (
    SafePointRequest,
    get_snapshot_safe_point_manager,
)

logger = logging.getLogger(__name__)

from .jobs_parts import manager as _manager
from .jobs_parts import registry as _registry

SAFE_POINT_RESERVATION_TIMEOUT_SECONDS = 5.0

# How often a waiting restore job re-proves that the session went quiescent.
# Quiescence is a property of the session over time, not of the instant the
# restore was filed — see the poll in ``_run_job``.
QUIESCENCE_RECHECK_SECONDS = 2.0


def _now_iso() -> str:
    return _registry.now_iso()


def _new_transaction_id() -> str:
    """Job status journal id (distinct from restore_apply mutation journal)."""
    return _registry.new_transaction_id()


def restore_apply_transaction_id(job_tx_id: str) -> str:
    """Map restore_job id → restore_apply id so both can coexist on disk."""
    return _registry.restore_apply_transaction_id(job_tx_id)


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
    owner_token: str = ""  # Safe-point requester token for this job
    safe_point_request: SafePointRequest | None = None
    lifecycle_generation: int = 0


def get_restore_job(transaction_id: str) -> RestoreJobHandle | None:
    """Return the process-wide handle for *transaction_id*, or ``None``."""
    return _registry.get_restore_job(transaction_id)


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
    return _registry.load_transaction(
        transaction_id,
        paths=paths,
        bindings=sys.modules[__name__],
    )


async def cancel_restore_job(
    transaction_id: str,
    *,
    session_id: str | None = None,
) -> None:
    """Cancel a restore job via the process-wide registry (cross-request).

    When *session_id* is provided, the handle must belong to that session;
    missing or mismatched jobs raise ``SnapshotTransactionError`` (HTTP 404).
    """
    await _registry.cancel_restore_job(
        transaction_id,
        session_id=session_id,
        bindings=sys.modules[__name__],
    )


def _persist_handle(handle: RestoreJobHandle) -> bool:
    """Write restore_job envelope for the handle's current record (before WS).

    Never overwrites a ``restore_apply`` mutation journal. Job status uses
    ``restore-job-{id}`` transaction IDs; the coordinator uses
    ``restore-apply-{id}`` (or the same id only when tests force it).

    Returns ``False`` when the job belongs to a deleted or replaced lifecycle
    generation. In that case no path is created and no event may be emitted.
    """
    return _registry.persist_handle(handle)




def collect_active_restore_protected_ids(
    session_id: str,
    *,
    paths: Any = None,
) -> dict[str, set[str]]:
    """Return restore-related protections for *session_id*.

    Keys: restore_target, pre_restore.
    Combines the in-process job registry with on-disk restore journals.
    """
    return _registry.collect_active_restore_protected_ids(
        session_id,
        paths=paths,
    )


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

    async def _acknowledge_if_quiescent(
        self,
        session_id: str,
        reservation: SafePointRequest,
        *,
        expected_lifecycle_generation: int | None = None,
    ) -> bool:
        """Acknowledge a restore request when no workflow can reach a boundary.

        Workflow creation uses ``state_lock`` too, so the idle proof and the
        acknowledgement are atomic with respect to registering a new workflow.
        Sessions with a live workflow or run task keep the normal cooperative
        boundary handshake.
        """
        from backend.state import (
            chat_sandboxes,
            has_live_session_task,
            initializing_sessions,
            sessions,
            state_lock,
            workflows,
        )
        from backend.services.session_lifecycle import (
            SessionGenerationError,
            SessionGoneError,
            assert_session_generation_active,
        )

        async with state_lock:
            try:
                assert_session_generation_active(
                    session_id,
                    expected_generation=expected_lifecycle_generation,
                )
            except (SessionGoneError, SessionGenerationError):
                return False
            workflow = workflows.get(session_id)
            state = sessions.get(session_id)
            if (
                state is None
                or session_id in initializing_sessions
                or session_id in chat_sandboxes
                or has_live_session_task(session_id)
            ):
                return False

            status = getattr(state, "status", None)
            status_value = getattr(status, "value", status)
            parked_at_gate = str(status_value or "").lower().startswith("waiting")

            # A workflow OBJECT outlives its run: once a HITL gate returns, the
            # object stays registered in `workflows` while nothing executes. The
            # cooperative handshake needs a *running* graph to reach a boundary
            # and acknowledge — a session parked at a wait gate never will, and
            # its only acknowledgement site is this one. Blocking on the mere
            # presence of the object therefore left a restore requested at a
            # gate with nowhere to be acknowledged: it waited out the full
            # timeout and failed, on a session that was completely idle.
            #
            # Every executing path is already excluded above (run/report/chat
            # tasks, initialization, chat sandboxes), so reaching here with a
            # workflow object means the object is idle. Still require the gate
            # status: anything else with a live workflow keeps the normal
            # cooperative handshake and fails closed.
            if workflow is not None and not parked_at_gate:
                return False

            # Idle is not the same as kernel-less. A session parked at a gate
            # keeps its workflow object, and that object keeps its sandbox: the
            # WebSocket runner drops the run task before `workflow.close()`
            # stops the kernel, so the namespace can still be live right here.
            # `_resolve_kernel_present` trusts this descriptor over the sandbox
            # it is handed, so a blanket False would make the pre_restore
            # capture believe there is no runtime to save and commit an
            # empty-complete manifest over a kernel full of state.
            kernel_present = (
                getattr(getattr(workflow, "sandbox", None), "km", None) is not None
            )

            descriptor = {
                "kernel_present": kernel_present,
                "current_round": getattr(state, "current_round", 0) or 0,
                "inactive": True,
                "status": str(status_value or "idle"),
            }
            if status_value in ("completed", "stopped"):
                descriptor["continuation_kind"] = "terminal"

            acknowledged = await self._safe_point.acknowledge_safe_point_request(
                session_id,
                reservation.generation,
                descriptor,
                role="quiescent_coordinator",
            )

        if acknowledged:
            logger.info(
                "[snapshot-v2] Restore safe point acknowledged for quiescent "
                "session %s (generation %s)",
                session_id,
                reservation.generation,
            )
        return acknowledged

    async def enqueue(
        self,
        session_id: str,
        preflight: RestorePreflightResult,
        request: SnapshotRestoreRequest,
        *,
        timeout_seconds: int = 300,
        retry_of_transaction_id: str | None = None,
    ) -> SnapshotTransactionRecord:
        """Create and start a background restore job."""
        return await _manager.enqueue(
            self,
            session_id,
            preflight,
            request,
            timeout_seconds=timeout_seconds,
            retry_of_transaction_id=retry_of_transaction_id,
            bindings=sys.modules[__name__],
        )


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
        await _manager._run_job(
            self,
            handle,
            preflight,
            request,
            timeout_seconds,
            bindings=sys.modules[__name__],
        )


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
        """Atomically update, persist, and publish a job record."""
        return _manager._update(
            self,
            handle,
            status,
            bindings=sys.modules[__name__],
            **kw,
        )


    async def _emit_state_sync(
        self,
        record: SnapshotTransactionRecord,
        elapsed: float = 0.0,
        *,
        lifecycle_generation: int | None = None,
    ) -> None:
        await _manager._emit_state_sync(
            self,
            record,
            elapsed,
            lifecycle_generation=lifecycle_generation,
            bindings=sys.modules[__name__],
        )


    def _on_task_done(self, tx_id: str) -> None:
        """Release only the task reference; keep handle for journal queries."""
        handle = get_restore_job(tx_id)
        if handle is not None:
            handle.task = None
        # Also clear legacy snapshot_jobs entry if present
        from backend.state import snapshot_jobs

        snapshot_jobs.pop(tx_id, None)
