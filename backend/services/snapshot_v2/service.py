"""Snapshot V2 capture service — atomic capture transaction.

Lock order (fixed):
  session_lock → safe-point acknowledgement → await_pending_saves → sandbox._exec_lock

Never acquire session_lock while holding _exec_lock.
"""

from __future__ import annotations

import asyncio
import copy
import contextlib
import hashlib
import json
import logging
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config._config import default_config, get_session_states
from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
from backend.services.snapshot_v2.continuation import ContinuationRegistry
from backend.services.snapshot_v2.errors import (
    RequiredSnapshotBlocked,
    SnapshotCaptureError,
    SnapshotDeleteConflict,
    SnapshotStorageError,
    SnapshotV2Error,
)
from backend.services.snapshot_v2.models import (
    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    RuntimeGenerationManifest,
    RuntimeManifest,
    SnapshotCaptureRequest,
    SnapshotFailureRecord,
    SnapshotManifestV2,
    SnapshotRecordV2,
    SnapshotRestoreRequest,
    SnapshotTransactionRecord,
    WorkspaceEntry,
    empty_complete_runtime,
    runtime_manifest_from_generation,
    validate_system_commit,
)
from backend.services.snapshot_v2.object_store import (
    ObjectPutResult,
    ObjectStore,
    ManifestStore,
    canonical_json,
)
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.runtime import RuntimeCapture
from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker
from backend.services.snapshot_v2.safe_point import (
    SafePointManager,
    get_snapshot_safe_point_manager,
)
from backend.services.snapshot_v2.workspace import WorkspaceCapture
from backend.state import get_session_lock
from backend.utils.asyncio_tasks import run_in_owned_thread

logger = logging.getLogger(__name__)

# Re-exported so `backend.services.snapshot_v2.service.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .service_parts.shared import (  # noqa: F401
    asyncio,
    copy,
    contextlib,
    hashlib,
    json,
    logging,
    shutil,
    uuid,
    datetime,
    timezone,
    Path,
    Any,
    default_config,
    get_session_states,
    checkpoint_sqlite_path,
    ContinuationRegistry,
    RequiredSnapshotBlocked,
    SnapshotCaptureError,
    SnapshotDeleteConflict,
    SnapshotStorageError,
    SnapshotV2Error,
    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    RuntimeGenerationManifest,
    RuntimeManifest,
    SnapshotCaptureRequest,
    SnapshotFailureRecord,
    SnapshotManifestV2,
    SnapshotRecordV2,
    SnapshotRestoreRequest,
    SnapshotTransactionRecord,
    WorkspaceEntry,
    empty_complete_runtime,
    runtime_manifest_from_generation,
    validate_system_commit,
    ObjectPutResult,
    ObjectStore,
    ManifestStore,
    canonical_json,
    SnapshotV2Paths,
    RuntimeCapture,
    RuntimeDependencyTracker,
    SafePointManager,
    get_snapshot_safe_point_manager,
    WorkspaceCapture,
    get_session_lock,
    _now_iso,
    _new_snapshot_id,
    _as_name_set,
    _null_async_cm,
    _workflow_state_for_snapshot,
    _has_trusted_workflow_snapshot_context,
)
from .service_parts import capture_flow as _capture_flow
from .service_parts import restore_flow as _restore_flow
from .service_parts import (  # noqa: F401
    RestoreOpsMixin,
    QueriesMixin,
    CaptureStepsMixin,
    CaptureSupportMixin,
    SessionStateMixin,
)
class SnapshotService(
    RestoreOpsMixin,
    QueriesMixin,
    CaptureStepsMixin,
    CaptureSupportMixin,
    SessionStateMixin,
):
    """Atomic snapshot capture and restore orchestration service."""

    def __init__(
        self,
        session_id: str,
        paths: SnapshotV2Paths,
        *,
        session_quota_bytes: int | None = None,
        pre_restore_quota_grace_bytes: int | None = None,
        free_space_reserve_bytes: int | None = None,
        restore_coordinator: Any = None,
        pre_restore_capture: Any = None,
        audit_bundle: Any = None,
        continuation_dispatcher: Any = None,
        safe_point_manager: SafePointManager | None = None,
        transaction_repository: Any = None,
        session_persistence: Any = None,
    ) -> None:
        self.session_id = session_id
        self.paths = paths
        self.object_store = ObjectStore(paths)
        self.manifest_store = ManifestStore(paths)
        self.workspace_capture = WorkspaceCapture()
        self.runtime_capture = RuntimeCapture(self.object_store, paths)
        self.continuation_registry = ContinuationRegistry(self.object_store)
        # Process-wide coordinator keyed by session_id (not instance-local).
        self.safe_point = safe_point_manager or get_snapshot_safe_point_manager()
        self.session_quota_bytes = (
            session_quota_bytes
            if session_quota_bytes is not None
            else 0
        )
        self.pre_restore_quota_grace_bytes = (
            pre_restore_quota_grace_bytes
            if pre_restore_quota_grace_bytes is not None
            else default_config.snapshot_v2_pre_restore_quota_grace_bytes
        )
        self.free_space_reserve_bytes = (
            free_space_reserve_bytes
            if free_space_reserve_bytes is not None
            else 0
        )

        # Restore orchestration dependencies (keyword-only, production defaults).
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.services.snapshot_v2.pre_restore import AuditBundle, PreRestoreCapture
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        self.restore_coordinator = restore_coordinator or RestoreCoordinator()
        self.pre_restore_capture = pre_restore_capture or PreRestoreCapture(
            self, self.object_store, self.manifest_store,
        )
        self.audit_bundle = audit_bundle or AuditBundle(self)
        self.continuation_dispatcher = (
            continuation_dispatcher or ContinuationDispatcher(self.object_store)
        )
        self.transaction_repository = transaction_repository
        self.session_persistence = session_persistence


    # ── Public API ───────────────────────────────────────────────────

    async def capture(
        self,
        request: SnapshotCaptureRequest,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
        transaction: str | None = None,
        workflow_state: dict[str, Any] | None = None,
        analysis_state: Any = None,
        safe_point_owner_token: str | None = None,
    ) -> SnapshotRecordV2:
        """Execute the capture transaction under the per-session lock."""
        session_id = request.session_id
        lock = get_session_lock(session_id)
        async with lock:
            return await self._capture_locked(
                request,
                sandbox=sandbox,
                workspace_root=workspace_root,
                transaction=transaction,
                workflow_state=workflow_state,
                analysis_state=analysis_state,
                safe_point_owner_token=safe_point_owner_token,
            )

    async def restore_snapshot(
        self,
        request: SnapshotRestoreRequest,
        transaction_id: str,
        cancel_event: asyncio.Event | None = None,
        *,
        job_transaction_id: str | None = None,
        status_callback: Any = None,
        expected_lifecycle_generation: int | None = None,
    ) -> SnapshotTransactionRecord:
        """Sole orchestration method for restore jobs.

        Sequence under the session lock after safe-point acknowledgement
        (caller must already have obtained the safe point):

          preflight → pre_restore|forward-only audit → stage → cancel check
          → apply → validate checkpoint → activate dispatcher → persist
          → commit journal → emit committed → cleanup

        ``committed`` is only returned after live side effects are verified.
        """
        return await _restore_flow.restore_snapshot(
            self,
            request,
            transaction_id,
            cancel_event,
            job_transaction_id=job_transaction_id,
            status_callback=status_callback,
            expected_lifecycle_generation=expected_lifecycle_generation,
            bindings=sys.modules[__name__],
        )





    def _has_live_run(self) -> bool:
        """True when a task that can observe the superseded flag is executing.

        The flag is only ever read from inside a running graph invocation
        (``await_restore_safe_point`` at node entry, the bioinformatics step
        loop) or a report/chat task built on one.
        """
        try:
            from backend.state import has_live_session_task

            return has_live_session_task(self.session_id)
        except Exception:  # pragma: no cover - defensive, bookkeeping only
            # Unknown means "assume observable": marking a flag nobody reads is
            # recoverable, failing to mark a live run is not.
            return True

    async def _retire_superseded_workflow(self, workflow: Any) -> None:
        """Retire the pre-restore run so apply() can replace the checkpoint DB.

        Three things, in this order:

        1. Flag the run superseded. It is frozen in ``await_restore_safe_point``;
           when the safe point releases it sees the flag and unwinds via
           ``WorkflowSupersededError`` without touching session state (the
           restore persists the authoritative state itself). Continuing is never
           right — the run's graph position predates the restore by definition.
        2. Close the checkpointer connection. REQUIRED: the DB is in WAL mode, so
           an open connection means live ``-wal``/``-shm`` that survive apply()'s
           replacement of the main file and are replayed over it afterwards.
        3. Evict it, so the next run builds a fresh workflow and connection
           against the restored file.

        Deliberately does NOT call ``workflow.close()``: that stops the sandbox,
        which ends in ``save_session()`` + a notebook save — writing the very
        artifacts apply() is about to install.

        Failure to close the checkpoint connection aborts the restore before
        apply.  That handle is live mutation authority, not bookkeeping.
        """
        from backend.state import report_workflows, workflows

        if workflow is None:
            return
        # Session-scoped: the eviction below makes the workflow object
        # unreachable, so every other site frozen on this safe point (the
        # bioinformatics step loop in particular) has to learn about it here.
        #
        # Only when something can actually observe and clear it. The flag's
        # whole lifecycle is "set here, cleared by the run that unwinds via
        # WorkflowSupersededError" — every clear_superseded() call site sits in
        # such a handler. But a workflow OBJECT outlives its run, and
        # `_acknowledge_if_quiescent` deliberately admits restores on a session
        # that has one with nothing executing. Marking there left the flag set
        # for the life of the process, where a later, unrelated run reaching any
        # node entry would find it and unwind mid-round — and `workflow.close()`
        # would drop the kernel save.
        if self._has_live_run():
            self.safe_point.mark_superseded(self.session_id)
        else:
            # Nothing is frozen on the safe point, so nothing would ever clear
            # the flag. Drop any stale one instead of adding to it.
            self.safe_point.clear_superseded(self.session_id)
        try:
            workflow._restore_superseded = True
        except Exception:  # pragma: no cover - exotic workflow stand-ins
            pass

        conn = getattr(workflow, "_db_conn", None)
        if conn is not None:
            try:
                await conn.close()
            except Exception as exc:
                from backend.services.snapshot_v2.errors import (
                    SnapshotTransactionError,
                )

                raise SnapshotTransactionError(
                    "CHECKPOINT_RETIRE_FAILED",
                    (
                        "Could not close the pre-restore checkpoint connection; "
                        "restore apply was refused to protect the checkpoint WAL"
                    ),
                    details={"session_id": self.session_id},
                ) from exc
            # After a confirmed close, core.py must rebuild every checkpoint
            # object against the restored file rather than reuse this handle.
            for attr in ("_db_conn", "checkpointer", "compiled_graph"):
                try:
                    setattr(workflow, attr, None)
                except Exception:
                    pass
        try:
            # Only evict the object we actually superseded — a workflow rebuilt
            # concurrently is already reading the restored checkpoint.
            evicted = False
            for registry in (workflows, report_workflows):
                if registry.get(self.session_id) is workflow:
                    registry.pop(self.session_id, None)
                    evicted = True
            if evicted:
                logger.info(
                    "[%s] Retired the pre-restore workflow; the next run rebuilds "
                    "it against the restored checkpoint",
                    self.session_id,
                )
        except Exception:
            logger.exception(
                "[%s] Failed to evict superseded workflow", self.session_id,
            )

    async def capture_required_snapshot(
        self,
        state: Any,
        request: SnapshotCaptureRequest,
        sandbox: Any = None,
        workflow_state: dict[str, Any] | None = None,
        *,
        workspace_root: str | None = None,
        safe_point_owner_token: str | None = None,
    ) -> SnapshotRecordV2:
        """Capture a required system snapshot and apply result to *state*.

        On success: mark runtime committed, rebuild ``state.snapshots`` from
        verified manifests, clear failure gate.

        On failure: persist WAITING_INPUT / snapshot_failure (no-Skip actions)
        and raise ``RequiredSnapshotBlocked`` so forward mutation stops.
        """
        from backend.models.schemas import TaskStatus
        from backend.services.snapshot_v2.errors import SNAPSHOT_FAILURE_ACTIONS

        if workspace_root is None:
            try:
                from backend.config._config import get_session_workspace
                workspace_root = str(get_session_workspace(request.session_id))
            except Exception:
                workspace_root = None

        # Idempotency: return already-committed manifest for same key.
        existing = self._find_committed_by_idempotency(request)
        if existing is not None:
            record = SnapshotRecordV2(
                snapshot_id=existing.snapshot_id,
                manifest=existing,
                failure=None,
                committed=True,
                session_reconciliation_pending=False,
            )
            state.snapshot_failure = None
            state.snapshot_transaction = {
                "snapshot_id": record.snapshot_id,
                "status": "completed",
                "completed_at": _now_iso(),
                "idempotent_replay": True,
            }
            if state.wait_input_reason == "snapshot_failure":
                state.wait_input_reason = None
            try:
                state.snapshots = self.rebuild_snapshots_from_manifests()
                state.last_snapshot_id = record.snapshot_id
            except Exception as exc:
                logger.warning(
                    "[%s] rebuild snapshots after idempotent capture failed: %s",
                    self.session_id, exc,
                )
            return record

        record = await self.capture(
            request,
            sandbox=sandbox,
            workspace_root=workspace_root,
            workflow_state=workflow_state,
            analysis_state=state,
            safe_point_owner_token=safe_point_owner_token,
        )
        if record.failure is not None:
            failure_payload = record.failure.model_dump(mode="json")
            failure_payload["allowed_actions"] = list(SNAPSHOT_FAILURE_ACTIONS)
            phase = (
                getattr(request, "phase", None)
                or getattr(request, "reason", None)
            )
            details = dict(record.failure.details or {})
            if phase and not details.get("phase"):
                details["phase"] = phase
            failure_payload["details"] = details
            blocked_error = RequiredSnapshotBlocked(
                record.failure.code,
                record.failure.message,
                details=details,
                snapshot_id=record.snapshot_id,
                phase=phase,
            )
            previous_status = getattr(state, "status", None)
            previous_status_value = (
                getattr(previous_status, "value", str(previous_status))
                if previous_status is not None
                else "running"
            )
            state.snapshot_failure = failure_payload
            state.snapshot_transaction = {
                "snapshot_id": record.snapshot_id,
                "status": "failed",
                "failed_at": _now_iso(),
            }
            state.wait_input_reason = "snapshot_failure"
            # Fail closed: block forward progress
            try:
                state.status = TaskStatus.WAITING_INPUT
            except Exception:
                pass
            # Type known workflow/post-report producers before the first event
            # is emitted. Otherwise the frontend can briefly (or permanently
            # on catch paths without a second sync) receive a stale base-action
            # gate that hides the safe Skip action.
            from backend.services.snapshot_v2.failure_gate import (
                ensure_workflow_boundary_retry_context,
            )

            ensure_workflow_boundary_retry_context(
                state,
                blocked_error,
                previous_status=previous_status_value,
                phase=phase,
            )
            failure_payload = dict(state.snapshot_failure or failure_payload)
            await self._send_state_sync({
                "change_type": "snapshot_failure",
                "version": getattr(state, "state_version", 0),
                "status": getattr(state, "status", None),
                "snapshot_failure": failure_payload,
                "snapshot_transaction": state.snapshot_transaction,
            })
            raise blocked_error

        state.snapshot_failure = None
        state.snapshot_transaction = {
            "snapshot_id": record.snapshot_id,
            "status": "completed",
            "completed_at": _now_iso(),
        }
        if state.wait_input_reason == "snapshot_failure":
            state.wait_input_reason = None
        try:
            state.snapshots = self.rebuild_snapshots_from_manifests()
            state.last_snapshot_id = record.snapshot_id
        except Exception as exc:
            logger.warning(
                "[%s] rebuild snapshots after capture failed: %s",
                self.session_id, exc,
            )

        # `_capture_impl` treats manifest publication as the commit point and
        # deliberately does not raise when the post-publish session save fails —
        # a retry with a new idempotency key could duplicate the snapshot. It
        # signals the condition with `session_reconciliation_pending` instead,
        # and until now nothing anywhere read that flag: the escape hatch the
        # design relies on had no other end. The state written just before that
        # failed save carries `saved_runtime_revision`/`runtime_revision`, so
        # losing it makes the next pre_restore capture re-serialize a kernel
        # that had not changed. Reconcile here, forced past the save throttle,
        # and surface the condition on the transaction record if it still fails
        # so a later load or resync can act on it.
        if record.session_reconciliation_pending:
            try:
                from backend.services.session_store import (
                    async_save_session,
                    update_session_cache,
                )

                await update_session_cache(self.session_id, state)
                await async_save_session(self.session_id, state, force=True)
            except Exception as exc:
                logger.warning(
                    "[%s] post-capture session reconciliation failed: %s",
                    self.session_id, exc,
                )
                transaction = dict(state.snapshot_transaction or {})
                transaction["session_reconciliation_pending"] = True
                state.snapshot_transaction = transaction
        return record







    async def delete_leaf(self, snapshot_id: str) -> Any:
        """Delete a leaf snapshot under the session lock; reconcile listings."""
        self._guard_active_safe_point()
        lock = get_session_lock(self.session_id)
        async with lock:
            result = self.delete_snapshot(snapshot_id, branch=False)
            await self._reconcile_analysis_state_snapshots()
            return result

    async def delete_branch(self, snapshot_id: str) -> Any:
        """Delete a snapshot and descendants under the session lock."""
        self._guard_active_safe_point()
        lock = get_session_lock(self.session_id)
        async with lock:
            result = self.delete_snapshot(snapshot_id, branch=True)
            await self._reconcile_analysis_state_snapshots()
            return result

    async def export_snapshots(
        self,
        *,
        include_runtime: bool = True,
        snapshot_ids: list[str] | None = None,
        staging_dir: Any = None,
        output_path: Any = None,
        output_preallocated: bool = False,
    ) -> Path:
        """Export verified V2 snapshots under the session lock.

        *staging_dir* should be the directory the caller will keep the archive
        in, so the archive is built on that filesystem and publishing it is a
        rename rather than a cross-device copy.
        """
        from backend.services.snapshot_v2.exporter import SnapshotExporter

        lock = get_session_lock(self.session_id)
        async with lock:
            exporter = SnapshotExporter(self.session_id, self.paths)
            return await run_in_owned_thread(
                exporter.export,
                self.session_id,
                include_runtime=include_runtime,
                snapshot_ids=snapshot_ids,
                staging_dir=Path(staging_dir) if staging_dir is not None else None,
                output_path=Path(output_path) if output_path is not None else None,
                output_preallocated=output_preallocated,
            )


    async def import_snapshots(
        self,
        archive_path: str | Path,
        *,
        trust_runtime: bool = False,
    ) -> Any:
        """Import a V2 export archive under the session lock; reconcile listings."""
        self._guard_active_safe_point()
        from backend.services.snapshot_v2.importer import SnapshotImporter

        from backend.services.snapshot_v2.object_store import SnapshotImportLimits

        lock = get_session_lock(self.session_id)
        async with lock:
            # Import safety limits still bound archive shape and metadata
            # materialization. Disk quota and reserve defaults are unbounded.
            importer = SnapshotImporter(
                self.session_id,
                self.paths,
                limits=SnapshotImportLimits.from_config(default_config),
            )
            result = await run_in_owned_thread(
                importer.import_bundle,
                str(archive_path),
                trust_runtime=trust_runtime,
            )
            if getattr(result, "success", False):
                await self._reconcile_analysis_state_snapshots()
            return result







    # ── Internal capture implementation ──────────────────────────────

    async def _capture_impl(
        self,
        request: SnapshotCaptureRequest,
        snapshot_id: str,
        tx: str,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
        workflow_state: dict[str, Any] | None = None,
        analysis_state: Any = None,
        skip_safe_point_lifecycle: bool = False,
        persist_session: bool = True,
    ) -> tuple[SnapshotManifestV2, bool]:
        return await _capture_flow.capture_impl(
            self,
            request,
            snapshot_id,
            tx,
            sandbox=sandbox,
            workspace_root=workspace_root,
            workflow_state=workflow_state,
            analysis_state=analysis_state,
            skip_safe_point_lifecycle=skip_safe_point_lifecycle,
            persist_session=persist_session,
            bindings=sys.modules[__name__],
        )

    # ── Content capture helpers ──────────────────────────────────────







    # ── Step helpers ─────────────────────────────────────────────────










    async def _capture_runtime(
        self,
        sandbox: Any,
        workflow_state: dict[str, Any] | None = None,
        analysis_state: Any = None,
        *,
        exec_lock_held: bool = False,
    ) -> RuntimeManifest:
        """Capture required runtime from *sandbox*.

        Raises ``SnapshotCaptureError`` on typed capture failures. Does **not**
        convert those failures into empty complete (fail closed).

        ``exec_lock_held``: caller already holds ``sandbox._exec_lock``.
        """
        declared: set[str] = set()
        sandbox_state: dict[str, Any] = {}
        if analysis_state is None:
            try:
                analysis_state = await self._get_session_state(self.session_id)
            except Exception:
                analysis_state = None
        if analysis_state is not None:
            raw_ss = getattr(analysis_state, "sandbox_state", None) or {}
            if isinstance(raw_ss, dict):
                sandbox_state = raw_ss

        tracker = self._rehydrate_tracker(sandbox, sandbox_state)
        if tracker is not None:
            declared |= set(tracker.required_outputs)

        for name in sandbox_state.get("runtime_dependencies_v2") or []:
            if name:
                declared.add(str(name))

        if workflow_state:
            declared |= _as_name_set(workflow_state.get("runtime_dependencies_v2"))
            declared |= _as_name_set(workflow_state.get("persistent_outputs"))

        checkpoint_deps = self._collect_checkpoint_dependencies(
            workflow_state, sandbox_state, analysis_state,
        )

        kernel_present = self._resolve_kernel_present(sandbox, workflow_state)

        result = await self.runtime_capture.capture(
            sandbox,
            workflow_state=workflow_state,
            declared_outputs=declared or None,
            checkpoint_dependencies=checkpoint_deps or None,
            kernel_present=kernel_present,
            exec_lock_held=exec_lock_held,
        )
        # RuntimeCapture.capture returns RuntimeManifest directly
        return result









    # ── Failure helpers ──────────────────────────────────────────────
