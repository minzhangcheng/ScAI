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
            else default_config.snapshot_v2_session_quota_bytes
        )
        self.pre_restore_quota_grace_bytes = (
            pre_restore_quota_grace_bytes
            if pre_restore_quota_grace_bytes is not None
            else default_config.snapshot_v2_pre_restore_quota_grace_bytes
        )
        self.free_space_reserve_bytes = (
            free_space_reserve_bytes
            if free_space_reserve_bytes is not None
            else default_config.snapshot_v2_free_space_reserve_bytes
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
    ) -> SnapshotTransactionRecord:
        """Sole orchestration method for restore jobs.

        Sequence under the session lock after safe-point acknowledgement
        (caller must already have obtained the safe point):

          preflight → pre_restore|forward-only audit → stage → cancel check
          → apply → validate checkpoint → activate dispatcher → persist
          → commit journal → emit committed → cleanup

        ``committed`` is only returned after live side effects are verified.
        """
        from backend.config._config import (
            get_session_dir,
            get_session_states,
            get_session_workspace,
        )
        from backend.services.snapshot_v2.errors import (
            RequiredSnapshotBlocked,
            SnapshotCaptureError,
            SnapshotIntegrityError,
            SnapshotTransactionError,
            SnapshotV2Error,
        )
        from backend.services.snapshot_v2.pre_restore import RestoreCaptureContext
        from backend.services.snapshot_v2.preflight import RestorePreflight
        from backend.state import sessions, workflows

        def _status(status: str, **kw: Any) -> SnapshotTransactionRecord:
            rec = SnapshotTransactionRecord(
                transaction_id=transaction_id,
                session_id=self.session_id,
                snapshot_id=request.snapshot_id,
                status=status,  # type: ignore[arg-type]
                **kw,
            )
            if status_callback is not None:
                status_callback(rec)
            return rec

        def _cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        if _cancelled():
            return _status("cancelled")

        # Authoritative preflight (recheck before mutation).
        preflight = RestorePreflight(self.paths).run(request.snapshot_id, request)
        if preflight.status != "ready":
            return _status(
                "failed",
                failure={
                    "code": "PREFLIGHT_FAILED",
                    "message": str(
                        preflight.compatibility_errors or preflight.status
                    ),
                },
            )

        # Capsule kind must be known before live mutation.
        capsule_kind = (preflight.capsule or {}).get("kind")
        if capsule_kind not in ("graph_node", "semantic_gate", "terminal", "idle"):
            return _status(
                "failed",
                failure={
                    "code": "UNKNOWN_CONTINUATION",
                    "message": f"Unknown continuation kind: {capsule_kind!r}",
                },
            )

        if _cancelled():
            return _status("cancelled")

        lock = get_session_lock(self.session_id)
        applied = False
        staged = None
        recovery_mode = "rollback"
        pre_restore_ref = ""

        async with lock:
            if _cancelled():
                return _status("cancelled")

            analysis_state = sessions.get(self.session_id)
            workflow = workflows.get(self.session_id)
            sandbox = getattr(workflow, "sandbox", None) if workflow else None
            workflow_state = _workflow_state_for_snapshot(workflow)
            live_sp = self.safe_point.get_descriptor(self.session_id) or {}
            if not _has_trusted_workflow_snapshot_context(
                workflow, workflow_state, live_sp,
            ):
                return _status(
                    "failed",
                    failure={
                        "code": "LIVE_WORKFLOW_STATE_UNAVAILABLE",
                        "message": (
                            "Active workflow has no observable snapshot state "
                            "or trusted safe-point descriptor"
                        ),
                    },
                )
            if live_sp:
                workflow_state = {**workflow_state, **live_sp}

            try:
                workspace_root = str(get_session_workspace(self.session_id))
            except Exception:
                workspace_root = None
            try:
                session_dir = str(get_session_dir(self.session_id))
            except Exception:
                session_dir = str(self.paths.root.parent)
            try:
                states_root = str(get_session_states(self.session_id))
            except Exception:
                states_root = str(Path(session_dir) / "states")

            context = RestoreCaptureContext(
                session_id=self.session_id,
                paths=self.paths,
                sandbox=sandbox,
                workspace_root=workspace_root,
                workflow_state=workflow_state,
                analysis_state=analysis_state,
                prev_manifest=self._resolve_previous_manifest(analysis_state),
                live_safe_point=live_sp,
                # The apply transaction is an internal journal identifier.
                # Retry lineage and every client-facing contract use the
                # public restore-job identifier when the manager supplies it.
                restore_transaction_id=job_transaction_id or transaction_id,
                restore_request=request,
            )

            # pre_restore or confirmed forward-only audit
            if request.current_state_choice == "discard_current":
                # Audit must succeed before forward-only APPLYING.
                try:
                    audit_ref = await self.audit_bundle.capture_locked(
                        context,
                        failures=[{
                            "code": "DISCARD_CURRENT",
                            "message": "User confirmed discard of current state",
                        }],
                    )
                except Exception as exc:
                    return _status(
                        "failed",
                        failure={
                            "code": "AUDIT_FAILED",
                            "message": str(exc),
                        },
                    )
                # Verify audit objects are committed.
                try:
                    self.manifest_store.load(audit_ref.snapshot_id)
                    if audit_ref.manifest.analysis_state_ref is not None:
                        self.object_store.verify(audit_ref.manifest.analysis_state_ref)
                    if audit_ref.manifest.notebook_ref is not None:
                        self.object_store.verify(audit_ref.manifest.notebook_ref)
                    for we in audit_ref.manifest.workspace or []:
                        self.object_store.verify(
                            ObjectRef(
                                sha256=we.object_hash,
                                size=we.size,
                                media_type="application/octet-stream",
                            )
                        )
                except Exception as exc:
                    return _status(
                        "failed",
                        failure={
                            "code": "AUDIT_VERIFY_FAILED",
                            "message": str(exc),
                        },
                    )
                recovery_mode = "forward_only"
                pre_restore_ref = ""
            else:
                # Every way the pre-restore capture can fail reports the same
                # stable code. The distinction that matters to the client is not
                # which serializer gave up but that the *safety* snapshot could
                # not be taken — and that nothing has been mutated yet, so
                # re-posting the restore with current_state_choice=
                # 'discard_current' is a legitimate way forward. The client has
                # always branched on PRE_RESTORE_UNAVAILABLE to offer exactly
                # that; no producer ever emitted it, so the discard path (and
                # the whole audit-bundle branch behind it) was dead code and a
                # user whose kernel could not be captured could only Retry into
                # the same failure. The underlying code is preserved in details.
                try:
                    pre_rec = await self.pre_restore_capture.capture_locked(context)
                except RequiredSnapshotBlocked as exc:
                    details = dict(exc.details or {})
                    details["cause"] = exc.code
                    return _status(
                        "failed",
                        failure={
                            "code": "PRE_RESTORE_UNAVAILABLE",
                            "message": str(exc),
                            "details": details,
                        },
                    )
                except Exception as exc:
                    return _status(
                        "failed",
                        failure={
                            "code": "PRE_RESTORE_UNAVAILABLE",
                            "message": str(exc),
                            "details": {"cause": "PRE_RESTORE_FAILED"},
                        },
                    )
                if pre_rec.failure is not None:
                    return _status(
                        "failed",
                        failure={
                            "code": "PRE_RESTORE_UNAVAILABLE",
                            "message": pre_rec.failure.message,
                            "details": {"cause": pre_rec.failure.code},
                        },
                    )
                recovery_mode = "rollback"
                pre_restore_ref = pre_rec.snapshot_id

            if _cancelled():
                return _status("cancelled")

            _status("staging", recovery_mode=recovery_mode)

            branch_id = f"restore-{transaction_id}"
            try:
                staged = await self.restore_coordinator.stage(
                    request.snapshot_id,
                    self.session_id,
                    self.paths,
                    recovery_mode=recovery_mode,
                    pre_restore_ref=pre_restore_ref,
                    branch_id=branch_id,
                    continuation_kind=str(capsule_kind),
                    transaction_id=transaction_id,
                )
            except Exception as exc:
                return _status(
                    "failed",
                    recovery_mode=recovery_mode,
                    failure={"code": "STAGE_FAILED", "message": str(exc)},
                )

            _status("staged", recovery_mode=recovery_mode)

            # Recheck cancellation BEFORE APPLYING.
            if _cancelled():
                try:
                    await self.restore_coordinator.cleanup(transaction_id, self.paths)
                except Exception:
                    pass
                return _status("cancelled", recovery_mode=recovery_mode)

            _status("applying", recovery_mode=recovery_mode)

            # A terminal/idle capsule resumes nothing, so the kernel that
            # predates the restore must not survive as this session's runtime.
            # Tear it down HERE, before apply.
            #
            # sandbox.stop() is not a bare shutdown: it ends with save_session()
            # + a notebook save, which write the very live pkl and notebook that
            # apply() installs and _verify_live_restore then hashes against the
            # target manifest. Running it after apply meant the session's own
            # teardown overwrote what the restore had just installed, and every
            # terminal-capsule restore died LIVE_VERIFY_FAILED -> RECOVERY_BLOCKED.
            # Run it before apply and that save is harmless: apply overwrites it
            # moments later, and if apply fails nothing is lost — pre_restore
            # captured the same kernel and rollback re-installs its residual.
            # Retire the pre-restore run BEFORE apply replaces the checkpoint DB.
            #
            # This is a correctness requirement, not tidiness. LangGraph opens
            # states/checkpoints.sqlite in WAL mode, so it is three files, and
            # apply() replaces only the main one. Leave the live connection open
            # and the stale -wal/-shm survive: a rebuilt connection replays them
            # and reads the PRE-restore graph position, and when the stale
            # connection finally closes it checkpoints those frames into the
            # restored main file — destroying the restored bytes for good. The
            # restore still reports committed, because _verify_live_restore
            # hashes the main file while it does briefly hold the staged bytes.
            # Closing first removes -wal/-shm, and only then is the os.replace
            # visible to everyone afterwards.
            await self._retire_superseded_workflow(workflow)

            if capsule_kind in ("terminal", "idle"):
                if workflow is not None and getattr(workflow, "sandbox", None):
                    try:
                        stop = getattr(workflow.sandbox, "stop", None)
                        if stop is not None:
                            maybe = stop()
                            if asyncio.iscoroutine(maybe):
                                await maybe
                    except Exception as exc:
                        logger.warning(
                            "[%s] terminal/idle kernel teardown failed: %s",
                            self.session_id, exc,
                        )
                    workflow.sandbox = None

            try:
                applied_result = await self.restore_coordinator.apply(
                    staged,
                    self.paths,
                    session_dir=session_dir,
                    workspace_root=workspace_root or str(Path(session_dir) / "workspace"),
                    states_root=states_root,
                    runtime_choice=request.runtime_choice,
                )
                applied = True
            except Exception as exc:
                # After APPLYING, cancellation/failure → recovery path (not pretend clean).
                return _status(
                    "failed",
                    recovery_mode=recovery_mode,
                    failure=self._partial_apply_failure(
                        "APPLY_FAILED", str(exc),
                    ),
                )

            if _cancelled():
                # Cancel after APPLYING → recovery, not cancelled-clean.
                return _status(
                    "failed",
                    recovery_mode=recovery_mode,
                    failure=self._partial_apply_failure(
                        "CANCELLED_AFTER_APPLYING",
                        "Cancelled after apply; recovery required",
                    ),
                )

            # Validate staged checkpoint when present.
            if staged.staged_checkpoint:
                ckpt_path = checkpoint_sqlite_path(states_root)
                if not ckpt_path.is_file() or ckpt_path.stat().st_size <= 0:
                    return _status(
                        "failed",
                        recovery_mode=recovery_mode,
                        failure=self._partial_apply_failure(
                            "CHECKPOINT_INVALID",
                            "Applied checkpoint missing or empty",
                        ),
                    )

            # Load applied analysis state into memory + activate dispatcher.
            target_manifest = self.manifest_store.load(request.snapshot_id)
            capsule = target_manifest.continuation

            restored_state = analysis_state
            from backend.models.schemas import AnalysisState as _AS
            disk_state = Path(session_dir) / "session.json"
            if disk_state.is_file():
                try:
                    restored_state = _AS.model_validate_json(disk_state.read_bytes())
                except Exception as exc:
                    return _status(
                        "failed",
                        recovery_mode=recovery_mode,
                        failure=self._partial_apply_failure(
                            "STATE_LOAD_FAILED", str(exc),
                        ),
                    )

            if restored_state is None:
                return _status(
                    "failed",
                    recovery_mode=recovery_mode,
                    failure=self._partial_apply_failure(
                        "MISSING_STATE", "No analysis state after apply",
                    ),
                )

            from backend.utils.secret_redaction import preserve_runtime_secrets
            preserve_runtime_secrets(analysis_state, restored_state)

            # Exact continuation only — no fresh-run fallback.
            try:
                staged_ckpt = self.continuation_dispatcher.stage_checkpoint(
                    capsule,
                    workflow_state,
                    restored_state,
                    snapshot_id=request.snapshot_id,
                    transaction_id=transaction_id,
                )
            except SnapshotCaptureError as exc:
                return _status(
                    "failed",
                    recovery_mode=recovery_mode,
                    failure=self._partial_apply_failure(exc.code, str(exc)),
                )

            # NOTE: the terminal/idle kernel teardown that used to live here has
            # moved to just before apply(). apply() is the sole authority on live
            # runtime in every branch — it installs the target's pkl/h5ad set for
            # a complete runtime, quarantines for empty-complete and for
            # non-complete+reset, and deliberately keeps the live one for
            # non-complete+keep — so anything touching runtime files after it ran
            # was overwriting a decision apply had already made correctly.

            # semantic_gate: rebuild exact waiting payload via activate
            # graph_node: seed exact next-node via activate
            # terminal: do not instantiate/run a workflow
            # idle: load content without starting analysis
            self.continuation_dispatcher.activate(restored_state, staged_ckpt)

            if getattr(applied_result, "runtime_restore_status", None):
                restored_state.runtime_restore_status = applied_result.runtime_restore_status
            if getattr(applied_result, "runtime_warning", None):
                restored_state.runtime_restore_warning = applied_result.runtime_warning

            # Persist state and branch. Journal remains APPLYING until live verify.
            try:
                await self._persist_restored_state(restored_state)
            except Exception as exc:
                failure = {
                    "code": "PERSIST_FAILED",
                    "message": str(exc),
                    "recovery_required": True,
                }
                self._block_session_after_partial_apply(failure)
                return _status(
                    "failed", recovery_mode=recovery_mode, failure=failure,
                )

            # Journal is still APPLYING here. Any failure remains recoverable.
            try:
                verification_proof = await self._verify_live_restore(
                    target_manifest=target_manifest,
                    restored_state=restored_state,
                    staged_checkpoint=staged_ckpt,
                    workspace_root=Path(workspace_root or session_dir),
                    session_dir=Path(session_dir),
                    states_root=Path(states_root),
                    runtime_choice=request.runtime_choice,
                )
            except SnapshotTransactionError as exc:
                failure = {
                    "code": exc.code,
                    "message": str(exc),
                    "recovery_required": True,
                }
                self._block_session_after_partial_apply(failure)
                return _status(
                    "failed", recovery_mode=recovery_mode, failure=failure,
                )

            # Only after live verification: HMAC-signed COMMITTED with proof.
            try:
                await self.restore_coordinator.commit(
                    transaction_id,
                    self.paths,
                    verification_proof=verification_proof,
                )
            except Exception as exc:
                failure = {
                    "code": "COMMIT_FAILED",
                    "message": str(exc),
                    "recovery_required": True,
                }
                self._block_session_after_partial_apply(failure)
                return _status(
                    "failed", recovery_mode=recovery_mode, failure=failure,
                )

            # Cleanup staging (journal may remain for COMMITTED recovery cleanup).
            try:
                await self.restore_coordinator.cleanup(transaction_id, self.paths)
            except Exception as exc:
                logger.warning(
                    "[%s] restore cleanup failed for %s: %s",
                    self.session_id, transaction_id, exc,
                )

            return _status("committed", recovery_mode=recovery_mode)





    def _has_live_run(self) -> bool:
        """True when a task that can observe the superseded flag is executing.

        The flag is only ever read from inside a running graph invocation
        (``await_restore_safe_point`` at node entry, the bioinformatics step
        loop) or a report/chat task built on one.
        """
        try:
            from backend.services.workflow_runner import _report_tasks
            from backend.state import _run_tasks, chat_tasks

            for registry in (_run_tasks, _report_tasks, chat_tasks):
                task = registry.get(self.session_id)
                if task is not None and not task.done():
                    return True
        except Exception:  # pragma: no cover - defensive, bookkeeping only
            # Unknown means "assume observable": marking a flag nobody reads is
            # recoverable, failing to mark a live run is not.
            return True
        return False

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

        Best-effort: a restore must not fail over bookkeeping.
        """
        from backend.state import workflows

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
            except Exception:
                logger.warning(
                    "[%s] Could not close the pre-restore checkpoint connection; "
                    "stale -wal may shadow the restored checkpoint",
                    self.session_id,
                    exc_info=True,
                )
            # Cleared regardless: core.py rebuilds the checkpointer when
            # _db_conn is None, and a half-closed handle must not be reused.
            for attr in ("_db_conn", "checkpointer", "compiled_graph"):
                try:
                    setattr(workflow, attr, None)
                except Exception:
                    pass
        try:
            # Only evict the object we actually superseded — a workflow rebuilt
            # concurrently is already reading the restored checkpoint.
            if workflows.get(self.session_id) is workflow:
                workflows.pop(self.session_id, None)
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
            return await asyncio.to_thread(
                exporter.export,
                self.session_id,
                include_runtime=include_runtime,
                snapshot_ids=snapshot_ids,
                staging_dir=Path(staging_dir) if staging_dir is not None else None,
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
            # `SnapshotImportLimits.from_config` maps the whole
            # `snapshot_v2_import_*` family plus the session quota and free-space
            # reserve out of SystemConfig, but this — the only production
            # construction of the importer — passed no limits at all, so it used
            # the hard-coded defaults and an operator who lowered any of those
            # settings on a shared installation had no effect on the one code
            # path they exist for.
            importer = SnapshotImporter(
                self.session_id,
                self.paths,
                limits=SnapshotImportLimits.from_config(default_config),
            )
            result = await asyncio.to_thread(
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
        existing_object_hashes_before_capture: set[str] = set()
        object_baseline_captured = False
        manifest_published = False
        # Step 1 — validate state
        self._validate_state(request)

        # Step 2 — lock is already held by capture() / _capture_locked()

        # Step 3 — create safe point (stores descriptor for kernel_present etc.)
        # Restore path already holds the cooperative safe point; do not
        # create/release it or pre_restore capture would unpause the workflow.
        if not skip_safe_point_lifecycle:
            await self._create_safe_point(workflow_state or {})

        # Released either at Step 11 (success) or by the finally below. The flag
        # keeps the success path's ordering — release before returning the
        # manifest — without letting the finally double-release.
        safe_point_released = False

        try:
            # Consistency barrier: await pending saves before freezing execution.
            if sandbox is not None and hasattr(sandbox, "await_pending_saves"):
                await sandbox.await_pending_saves()

            # Snapshot object hashes present before any put in this capture.
            # Used for quota accounting of refs that only expose ObjectRef
            # (workspace/runtime/continuation) — never use register_existing_ref
            # return value as a proxy for "already existed before capture".
            objects_dir = Path(str(self.paths.objects))
            if objects_dir.is_dir():
                existing_object_hashes_before_capture = {
                    path.name for path in objects_dir.iterdir() if path.is_file()
                }
            else:
                existing_object_hashes_before_capture = set()
            object_baseline_captured = True

            estimated_addition = await self._estimate_capture_addition(
                request, workspace_root, analysis_state, sandbox=sandbox,
            )
            current_physical = self._physical_object_store_bytes()
            phase = str(getattr(request.phase, "value", request.phase))
            allow_over_quota_baseline = phase == "pre_restore"
            quota_ceiling = (
                current_physical + self.pre_restore_quota_grace_bytes
                if (
                    allow_over_quota_baseline
                    and current_physical > self.session_quota_bytes
                )
                else self.session_quota_bytes
            )
            if current_physical + estimated_addition > quota_ceiling:
                raise SnapshotStorageError(
                    "snapshot_quota_exceeded",
                    "Conservative pre-write estimate would use "
                    f"{current_physical + estimated_addition} bytes, exceeding "
                    f"capture ceiling {quota_ceiling} bytes "
                    f"(session quota {self.session_quota_bytes})",
                )

            # Acquire exec lock (never acquire session_lock while holding this).
            exec_cm = (
                sandbox._exec_lock
                if sandbox is not None and hasattr(sandbox, "_exec_lock")
                else _null_async_cm()
            )

            # Freeze revision under lock; mark after publish uses this value.
            frozen_runtime_revision: int | None = None
            if sandbox is not None:
                frozen_runtime_revision = int(
                    getattr(sandbox, "runtime_revision", 0) or 0
                )

            async with exec_cm:
                # physical_added_bytes from ObjectPutResult.created where available,
                # else from pre-capture object hash set for bare ObjectRefs.

                # Freeze AnalysisState, notebook, checkpoint while execution excluded.
                analysis_state_result = await self._capture_analysis_state(
                    request, analysis_state,
                )
                notebook_result = await self._capture_notebook(request)
                checkpoint_result = await self._capture_checkpoint(request)

                # Re-freeze under lock in case revision moved before acquire.
                if sandbox is not None:
                    frozen_runtime_revision = int(
                        getattr(sandbox, "runtime_revision", 0) or 0
                    )

                # Runtime capture (may execute kernel code under exec lock).
                runtime: RuntimeManifest | None = request.runtime
                runtime_snapshot: RuntimeGenerationManifest | None = None
                if runtime is None and request.runtime_snapshot_capture is not None:
                    runtime_snapshot = request.runtime_snapshot_capture
                    runtime = runtime_manifest_from_generation(
                        runtime_snapshot, status="complete",
                    )
                elif runtime is None and sandbox is not None:
                    # SnapshotCaptureError propagates — never swallow.
                    runtime = await self._capture_runtime(
                        sandbox,
                        workflow_state=workflow_state,
                        analysis_state=analysis_state,
                        exec_lock_held=True,
                    )
                if runtime is None and sandbox is None:
                    # Only an eligibility-trusted descriptor (cooperative ack
                    # or explicit producer acquire) may authorize empty-
                    # complete. Raw workflow_state seeded via
                    # _create_safe_point is untrusted metadata only.
                    descriptor = self.safe_point.get_eligibility_descriptor(
                        self.session_id,
                    )
                    if not self._resolve_empty_runtime_eligibility(
                        request.phase or request.reason or "manual",
                        sandbox,
                        descriptor,
                    ):
                        raise SnapshotCaptureError(
                            "LIVE_SANDBOX_REQUIRED",
                            "A live or unproven session cannot commit "
                            "empty-complete runtime",
                        )
                    runtime = empty_complete_runtime()
                elif runtime is None:
                    # sandbox present but no runtime produced — fail closed
                    # rather than invent empty-complete for a live session.
                    raise SnapshotCaptureError(
                        "LIVE_SANDBOX_REQUIRED",
                        "A live or unproven session cannot commit "
                        "empty-complete runtime",
                    )
                runtime = validate_system_commit(runtime)

                # Workspace scope (filesystem; still under exec lock for consistency).
                workspace_entries: list[WorkspaceEntry] = []
                if workspace_root:
                    workspace_entries, _ = await self._capture_workspace(
                        str(workspace_root),
                    )

            # Mutable sources frozen — exec lock released.

            analysis_state_ref = analysis_state_result.ref
            notebook_ref = notebook_result.ref
            checkpoint_ref = checkpoint_result.ref if checkpoint_result is not None else None

            continuation = request.continuation or request.capsule
            if continuation is None:
                raise SnapshotCaptureError(
                    "MISSING_CONTINUATION", "continuation capsule is required",
                )

            all_objects: dict[str, ObjectRef] = {}
            physical_added = 0
            # Refs proven by put_file/put_bytes/put_stream in this transaction
            # (or request-supplied external refs fully verified once).
            transaction_evidence: dict[str, ObjectRef] = {}

            def _record_transaction_evidence(ref: ObjectRef) -> None:
                transaction_evidence[ref.sha256] = ref

            def _track_ref_with_result(
                ref: ObjectRef,
                result: ObjectPutResult | None = None,
            ) -> None:
                """Declare *ref* on the manifest and count quota if newly created.

                Prefer ``ObjectPutResult.created`` when available. For bare
                ObjectRefs (workspace/runtime/continuation), count size only
                when the hash was absent from the object store before this
                capture transaction started. Never use
                ``register_existing_ref()`` return value for quota decisions.
                """
                nonlocal physical_added
                if ref.sha256 in all_objects:
                    return
                all_objects[ref.sha256] = ref
                if result is not None:
                    if result.created:
                        physical_added += ref.size
                elif ref.sha256 not in existing_object_hashes_before_capture:
                    physical_added += ref.size

            def _add_ref_with_result(put_result: ObjectPutResult) -> ObjectRef:
                """Add an ObjectPutResult to the manifest's objects dict.

                Returns the ObjectRef. The ``created`` flag controls
                whether the size counts toward physical_added_bytes.

                These objects were put before the runtime capture's awaits and
                a capture writes no journal, so nothing marks them reachable in
                between: a concurrent ensure_session_loaded() runs the
                post-recovery GC, whose mark() sees only committed manifests
                and journals. An ObjectPutResult therefore proves the object
                existed *then*, not that it still exists - re-prove it here,
                after the last suspension point, exactly as the workspace and
                runtime refs below do, so the capture fails closed instead of
                publishing a manifest with dangling refs.
                """
                ref = put_result.ref
                if (
                    self.object_store._get_evidence(ref.sha256) is None
                    and not self.object_store.register_existing_ref(ref)
                ):
                    raise SnapshotCaptureError(
                        "MISSING_TRANSACTION_EVIDENCE",
                        f"Object {ref.sha256} was removed from the store "
                        f"before this capture could publish",
                    )
                _record_transaction_evidence(ref)
                _track_ref_with_result(ref, put_result)
                return ref

            _add_ref_with_result(analysis_state_result)
            _add_ref_with_result(notebook_result)
            if checkpoint_result is not None:
                _add_ref_with_result(checkpoint_result)
            for we in workspace_entries:
                ref = ObjectRef(
                    sha256=we.object_hash,
                    size=we.size,
                    media_type="application/octet-stream",
                )
                _track_ref_with_result(ref)
            for ref in runtime.adata_refs.values():
                _track_ref_with_result(ref)
            if runtime.residual_ref is not None:
                _track_ref_with_result(runtime.residual_ref)
            if runtime_snapshot is not None:
                for ref in (runtime_snapshot.captured_adata_refs or {}).values():
                    _track_ref_with_result(ref)
                if runtime_snapshot.captured_residual_ref is not None:
                    _track_ref_with_result(runtime_snapshot.captured_residual_ref)

            # Continuation payload/workflow-state refs must be declared in objects
            # so restore stage can verify them as first-class content.
            for attr in ("workflow_state_ref", "payload_ref"):
                cref = getattr(continuation, attr, None)
                if cref is not None:
                    _track_ref_with_result(cref)

            logical_size = sum(r.size for r in all_objects.values())

            manifest = SnapshotManifestV2(
                format_version=FORMAT_VERSION,
                session_id=request.session_id,
                snapshot_id=snapshot_id,
                name=request.name or snapshot_id,
                description=request.description or "",
                phase=request.phase or request.reason or "manual",
                created_at=_now_iso(),
                round_number=request.round_number,
                parent_id=request.parent_id,
                branch_id=request.branch_id or "main",
                branch_label=request.branch_label or "main",
                source_restore_transaction_id=request.source_restore_transaction_id,
                analysis_state_ref=analysis_state_ref,
                notebook_ref=notebook_ref,
                checkpoint_ref=checkpoint_ref,
                runtime=runtime,
                workspace=workspace_entries,
                continuation=continuation,
                logical_size_bytes=logical_size,
                physical_added_bytes=physical_added,
                root_hash="",
                objects=all_objects,
                idempotency_key=getattr(request, "idempotency_key", None),
            )

            # Record workspace/runtime/continuation refs proven during this
            # transaction's put_* calls (evidence already on the object store).
            for we in workspace_entries:
                ref = ObjectRef(
                    sha256=we.object_hash,
                    size=we.size,
                    media_type="application/octet-stream",
                )
                if ref.sha256 not in transaction_evidence:
                    # put_file during workspace capture registered store evidence
                    if self.object_store._get_evidence(ref.sha256) is not None:
                        _record_transaction_evidence(ref)
                    elif self.object_store.register_existing_ref(ref):
                        _record_transaction_evidence(ref)
            for ref in runtime.adata_refs.values():
                if ref.sha256 not in transaction_evidence:
                    if self.object_store._get_evidence(ref.sha256) is not None:
                        _record_transaction_evidence(ref)
                    elif self.object_store.register_existing_ref(ref):
                        _record_transaction_evidence(ref)
            if runtime.residual_ref is not None:
                rref = runtime.residual_ref
                if rref.sha256 not in transaction_evidence:
                    if self.object_store._get_evidence(rref.sha256) is not None:
                        _record_transaction_evidence(rref)
                    elif self.object_store.register_existing_ref(rref):
                        _record_transaction_evidence(rref)
            for attr in ("workflow_state_ref", "payload_ref"):
                cref = getattr(continuation, attr, None)
                if cref is not None and cref.sha256 not in transaction_evidence:
                    if self.object_store._get_evidence(cref.sha256) is not None:
                        _record_transaction_evidence(cref)
                    elif self.object_store.register_existing_ref(cref):
                        _record_transaction_evidence(cref)

            # Fully verify only request-supplied external refs (not put in this txn).
            external_refs: list[ObjectRef] = []
            if request.analysis_state_ref is not None:
                external_refs.append(request.analysis_state_ref)
            if request.notebook_ref is not None:
                external_refs.append(request.notebook_ref)
            if request.checkpoint_ref is not None:
                external_refs.append(request.checkpoint_ref)
            if request.runtime is not None:
                for ref in (request.runtime.adata_refs or {}).values():
                    external_refs.append(ref)
                if request.runtime.residual_ref is not None:
                    external_refs.append(request.runtime.residual_ref)
            for eref in external_refs:
                if eref.sha256 in transaction_evidence:
                    continue
                self.object_store.verify(eref)
                _record_transaction_evidence(eref)

            # Every other manifest ref must already have transaction evidence.
            for sha, ref in all_objects.items():
                if sha not in transaction_evidence:
                    raise SnapshotCaptureError(
                        "MISSING_TRANSACTION_EVIDENCE",
                        f"Object {sha} lacks put/verify evidence in this capture",
                    )

            # Step 7 — compute root hash
            await self._compute_root_hash(manifest)

            # Step 8 — check quota (uses physical_added + free space)
            await self._check_quota(
                manifest,
                quota_baseline_bytes=current_physical,
                allow_over_quota_baseline=allow_over_quota_baseline,
            )

            # Step 9 — publish manifest
            staging_path = self.manifest_store.write_staging(manifest, prefix=f"{tx}_")
            published = self.manifest_store.publish_staging(staging_path)
            manifest_published = True

            # Re-read verification
            reloaded = self.manifest_store.load(published.snapshot_id)
            if reloaded.root_hash != published.root_hash:
                raise SnapshotCaptureError(
                    "MANIFEST_REREAD_MISMATCH",
                    "Published manifest failed re-read verification",
                )

            # Mark runtime committed after successful publish
            if sandbox is not None:
                self._mark_runtime_committed(
                    sandbox, runtime_revision=frozen_runtime_revision,
                )
                # Persist saved_runtime_revision onto analysis state
                try:
                    state = analysis_state or await self._get_session_state(
                        request.session_id,
                    )
                    if state is not None:
                        ss = dict(getattr(state, "sandbox_state", None) or {})
                        ss["saved_runtime_revision"] = getattr(
                            sandbox, "saved_runtime_revision", 0,
                        )
                        ss["runtime_revision"] = getattr(
                            sandbox, "runtime_revision", 0,
                        )
                        state.sandbox_state = ss
                except Exception as exc:
                    logger.warning(
                        "[%s] failed to persist saved_runtime_revision: %s",
                        self.session_id, exc,
                    )


            # Step 10 — save session state after publish when requested.
            # Manifest publication is the capture commit point. If session
            # persistence fails after publish, return the committed manifest
            # with session_reconciliation_pending=True — never raise
            # SnapshotCaptureError (retry with a new key could duplicate).
            session_reconciliation_pending = False
            if persist_session:
                try:
                    await self._save_session_state(request.session_id)
                except SnapshotV2Error as exc:
                    logger.error(
                        "[%s] post-publish session save raised SnapshotV2Error "
                        "for %s: %s — marking session_reconciliation_pending",
                        self.session_id, reloaded.snapshot_id, exc,
                    )
                    session_reconciliation_pending = True
                except Exception as exc:
                    logger.error(
                        "[%s] post-publish session save failed for %s: %s — "
                        "manifest already committed; session_reconciliation_pending",
                        self.session_id, reloaded.snapshot_id, exc,
                    )
                    session_reconciliation_pending = True

            # Step 11 — release safe point (only on full success; skip when
            # restore already owns the cooperative safe point).
            if not skip_safe_point_lifecycle:
                await self._release_safe_point()
                safe_point_released = True

            # Step 12 — return committed manifest + pending flag (no private attr)
            return reloaded, session_reconciliation_pending

        except SnapshotV2Error:
            if object_baseline_captured and not manifest_published:
                self._cleanup_unpublished_capture_objects(
                    existing_object_hashes_before_capture,
                )
            raise
        except Exception as exc:
            if object_baseline_captured and not manifest_published:
                self._cleanup_unpublished_capture_objects(
                    existing_object_hashes_before_capture,
                )
            raise SnapshotCaptureError("CAPTURE_FAILED", str(exc))
        finally:
            # The safe point MUST NOT outlive this call, including when the
            # capture task is cancelled. asyncio.CancelledError derives from
            # BaseException, so the two except clauses above never saw it: a user
            # pressing Stop (lifecycle.py task.cancel()) or a workflow teardown
            # mid-capture left the hold in the process-wide SafePointManager
            # forever, and from then on every capture and restore for that
            # session failed SAFE_POINT_BUSY / blocked in await_restore_safe_point
            # until the server was restarted.
            #
            # shield() keeps the release itself from being cancelled partway; if
            # the surrounding task is cancelled again while we wait, the shielded
            # release still runs to completion and the CancelledError propagates
            # as it should.
            if not skip_safe_point_lifecycle and not safe_point_released:
                await asyncio.shield(self._release_safe_point())

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


