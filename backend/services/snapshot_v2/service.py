"""Snapshot V2 capture service — atomic capture transaction.

Lock order (fixed):
  session_lock → safe-point acknowledgement → await_pending_saves → sandbox._exec_lock

Never acquire session_lock while holding _exec_lock.
"""

from __future__ import annotations

import asyncio
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_snapshot_id() -> str:
    return f"snap-{uuid.uuid4().hex[:16]}"


def _as_name_set(raw: Any) -> set[str]:
    """Normalize list/set/tuple/dict keys of dependency names into a set."""
    if not raw:
        return set()
    if isinstance(raw, dict):
        return {str(n) for n in raw if n}
    if isinstance(raw, (list, set, tuple)):
        return {str(n) for n in raw if n}
    return {str(raw)}


@contextlib.asynccontextmanager
async def _null_async_cm():
    yield


class SnapshotService:
    """Atomic snapshot capture and restore orchestration service."""

    def __init__(
        self,
        session_id: str,
        paths: SnapshotV2Paths,
        *,
        session_quota_bytes: int | None = None,
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
            workflow_state: dict[str, Any] = {}
            if workflow is not None:
                for attr in ("_current_graph_state", "_last_workflow_state"):
                    raw = getattr(workflow, attr, None)
                    if isinstance(raw, dict):
                        workflow_state = dict(raw)
                        break
            live_sp = self.safe_point.get_descriptor(self.session_id) or {}
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
                try:
                    pre_rec = await self.pre_restore_capture.capture_locked(context)
                except RequiredSnapshotBlocked as exc:
                    return _status(
                        "failed",
                        failure={
                            "code": exc.code,
                            "message": str(exc),
                            "details": dict(exc.details or {}),
                        },
                    )
                except Exception as exc:
                    return _status(
                        "failed",
                        failure={
                            "code": "PRE_RESTORE_FAILED",
                            "message": str(exc),
                        },
                    )
                if pre_rec.failure is not None:
                    return _status(
                        "failed",
                        failure={
                            "code": pre_rec.failure.code,
                            "message": pre_rec.failure.message,
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
                    failure={
                        "code": "APPLY_FAILED",
                        "message": str(exc),
                        "recovery_required": True,
                    },
                )

            if _cancelled():
                # Cancel after APPLYING → recovery, not cancelled-clean.
                return _status(
                    "failed",
                    recovery_mode=recovery_mode,
                    failure={
                        "code": "CANCELLED_AFTER_APPLYING",
                        "message": "Cancelled after apply; recovery required",
                        "recovery_required": True,
                    },
                )

            # Validate staged checkpoint when present.
            if staged.staged_checkpoint:
                ckpt_path = Path(states_root) / "checkpoint.sqlite"
                if not ckpt_path.is_file() or ckpt_path.stat().st_size <= 0:
                    return _status(
                        "failed",
                        recovery_mode=recovery_mode,
                        failure={
                            "code": "CHECKPOINT_INVALID",
                            "message": "Applied checkpoint missing or empty",
                            "recovery_required": True,
                        },
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
                        failure={
                            "code": "STATE_LOAD_FAILED",
                            "message": str(exc),
                            "recovery_required": True,
                        },
                    )

            if restored_state is None:
                return _status(
                    "failed",
                    recovery_mode=recovery_mode,
                    failure={
                        "code": "MISSING_STATE",
                        "message": "No analysis state after apply",
                        "recovery_required": True,
                    },
                )

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
                    failure={
                        "code": exc.code,
                        "message": str(exc),
                        "recovery_required": True,
                    },
                )

            # Terminal/idle: quarantine unrelated live kernel before exposing session.
            if staged_ckpt.kind in ("terminal", "idle"):
                try:
                    from backend.utils.runtime_apply import quarantine_live_runtime
                    if request.runtime_choice == "reset" or staged_ckpt.kind == "terminal":
                        quarantine_live_runtime(self.session_id)
                    if workflow is not None and hasattr(workflow, "sandbox") and workflow.sandbox:
                        # Do not trust live kernel as continuation runtime.
                        try:
                            stop = getattr(workflow.sandbox, "stop", None)
                            if stop is not None:
                                maybe = stop()
                                if asyncio.iscoroutine(maybe):
                                    await maybe
                        except Exception:
                            pass
                        workflow.sandbox = None
                except Exception as exc:
                    logger.warning(
                        "[%s] terminal/idle kernel quarantine failed: %s",
                        self.session_id, exc,
                    )

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

    async def _persist_restored_state(self, restored_state: Any) -> None:
        """Write restored analysis state to the live session store."""
        from backend.state import sessions

        sessions[self.session_id] = restored_state
        if self.session_persistence is not None:
            await self.session_persistence(self.session_id, restored_state)
        else:
            from backend.services.session_store import async_save_session
            await async_save_session(self.session_id, restored_state)

    def _block_session_after_partial_apply(self, failure: dict[str, Any]) -> None:
        """Guard the in-process session after a post-persist restore failure.

        Once ``_persist_restored_state`` has run, ``sessions[session_id]`` holds
        the target state while the restore journal is still ``APPLYING`` and the
        transaction failed. Mark the session recovery-blocked so forward
        mutation is refused until startup recovery rolls back (or, for
        ``forward_only``, rolls forward) on the next session load — the memory
        state must not be silently trusted as a successful restore.
        """
        try:
            from backend.state import sessions
            from backend.services.session_store import mark_recovery_blocked

            mark_recovery_blocked(
                self.session_id, sessions.get(self.session_id), failure,
            )
        except Exception:
            logger.exception(
                "[%s] failed to mark recovery-blocked after partial restore apply",
                self.session_id,
            )

    async def _verify_live_restore(
        self,
        *,
        target_manifest: Any,
        restored_state: Any,
        staged_checkpoint: Any,
        workspace_root: Path,
        session_dir: Path,
        states_root: Path | None = None,
        runtime_choice: str | None = None,
    ) -> dict[str, Any]:
        """Verify live restore side effects while the journal is still APPLYING.

        Raises ``SnapshotTransactionError("LIVE_VERIFY_FAILED", ...)`` on any
        mismatch. Returns a verification proof dict for the COMMITTED journal.

        Uses the shared contract in ``live_verify.verify_live_restore_components``:
        production residual is ``{session_id}_session.pkl`` + integrity signature
        (not session_state.pkl / residual.pkl); session.json is presence-checked
        only (post-activate state is continuation-mutated).
        """
        from backend.services.snapshot_v2.errors import SnapshotTransactionError
        from backend.services.snapshot_v2.live_verify import (
            verify_live_restore_components,
        )

        states = Path(states_root) if states_root is not None else session_dir / "states"
        try:
            return verify_live_restore_components(
                session_id=self.session_id,
                target_manifest=target_manifest,
                session_dir=session_dir,
                workspace_root=workspace_root,
                states_root=states,
                runtime_choice=runtime_choice,
                restored_state=restored_state,
                staged_checkpoint=staged_checkpoint,
                journal=None,
            )
        except ValueError as exc:
            raise SnapshotTransactionError(
                "LIVE_VERIFY_FAILED",
                str(exc) or "live side effects not verified",
            ) from exc
        except SnapshotTransactionError:
            raise
        except Exception as exc:
            raise SnapshotTransactionError(
                "LIVE_VERIFY_FAILED",
                str(exc) or "live side effects not verified",
            ) from exc

    async def capture_required_snapshot(
        self,
        state: Any,
        request: SnapshotCaptureRequest,
        sandbox: Any = None,
        workflow_state: dict[str, Any] | None = None,
        *,
        workspace_root: str | None = None,
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
        )
        if record.failure is not None:
            failure_payload = record.failure.model_dump(mode="json")
            failure_payload["allowed_actions"] = list(SNAPSHOT_FAILURE_ACTIONS)
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
            await self._send_state_sync({
                "change_type": "snapshot_failure",
                "version": getattr(state, "state_version", 0),
                "status": getattr(state, "status", None),
                "snapshot_failure": failure_payload,
                "snapshot_transaction": state.snapshot_transaction,
            })
            phase = (
                getattr(request, "phase", None)
                or getattr(request, "reason", None)
            )
            details = dict(record.failure.details or {})
            if phase and not details.get("phase"):
                details["phase"] = phase
            raise RequiredSnapshotBlocked(
                record.failure.code,
                record.failure.message,
                details=details,
                snapshot_id=record.snapshot_id,
                phase=phase,
            )

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
        return record


    def compute_protected_snapshot_ids(self) -> dict[str, set[str]]:
        """Return snapshot IDs protected from deletion, keyed by reason.

        Reasons:
          live_head, continuation_source, restore_target, pre_restore,
          import, export

        Sources: committed session state (lineage head / last restored),
        verified create/restore/delete/import/export journals, and the
        in-process restore job registry.
        """
        from pathlib import Path as _Path

        protected: dict[str, set[str]] = {
            "live_head": set(),
            "continuation_source": set(),
            "restore_target": set(),
            "pre_restore": set(),
            "import": set(),
            "export": set(),
        }

        # Live lineage from in-memory session state
        try:
            from backend.state import sessions
            state = sessions.get(self.session_id)
        except Exception:
            state = None
        if state is not None:
            head = getattr(state, "last_snapshot_id", None)
            if head:
                protected["live_head"].add(str(head))
            from backend.workflow.provenance import restore_provenance_id
            cont = restore_provenance_id(state)
            if cont:
                protected["continuation_source"].add(str(cont))

        # In-process restore job registry (active non-terminal jobs)
        try:
            from backend.state import snapshot_restore_jobs
            terminal = {"committed", "failed", "cancelled", "timed_out"}
            for handle in list(snapshot_restore_jobs.values()):
                record = getattr(handle, "record", None)
                if record is None:
                    continue
                if getattr(record, "session_id", None) != self.session_id:
                    continue
                status = getattr(record, "status", None)
                if status in terminal:
                    continue
                sid = getattr(record, "snapshot_id", None)
                if sid:
                    protected["restore_target"].add(str(sid))
        except Exception:
            pass

        # Journals: restore / import / export / delete (signed envelopes)
        journals_dir = _Path(str(self.paths.journals))
        if journals_dir.exists():
            from backend.services.snapshot_v2.journal_store import JournalStore
            from backend.services.snapshot_v2.errors import SnapshotIntegrityError
            from backend.services.snapshot_v2.recovery import RecoveryFailure

            store = JournalStore(self.paths)
            for entry in journals_dir.iterdir():
                if not entry.is_file() or entry.suffix != ".json":
                    continue
                try:
                    env = store.read(entry.stem)
                except (SnapshotIntegrityError, RecoveryFailure, Exception):
                    # Fall back to raw parse for incomplete fixtures only
                    try:
                        data = json.loads(entry.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    # Minimal legacy fallback for unsigned fixtures in tests
                    kind = data.get("kind")
                    payload = data if kind else {}
                    status = str(data.get("status") or "").upper()
                    if kind == "export_protection":
                        for sid in data.get("protected_snapshot_ids") or []:
                            if sid:
                                protected["export"].add(str(sid))
                    elif kind == "import" and status not in ("COMMITTED", "FAILED"):
                        for sid in data.get("protected_snapshot_ids") or []:
                            if sid:
                                protected["import"].add(str(sid))
                    elif data.get("pre_restore_ref") or data.get("target_snapshot_id"):
                        if data.get("pre_restore_ref"):
                            protected["pre_restore"].add(str(data["pre_restore_ref"]))
                        t = data.get("target_snapshot_id")
                        if t:
                            protected["restore_target"].add(str(t))
                    continue

                if env.session_id and env.session_id != self.session_id:
                    continue
                payload = env.payload or {}
                status = str(env.status or "").upper()
                st_lower = str(env.status or "").lower()

                if env.kind == "export_protection":
                    for sid in payload.get("protected_snapshot_ids") or []:
                        if sid:
                            protected["export"].add(str(sid))
                    continue

                if env.kind == "import":
                    if status in ("COMMITTED", "FAILED"):
                        continue
                    for sid in payload.get("protected_snapshot_ids") or []:
                        if sid:
                            protected["import"].add(str(sid))
                    for sid in payload.get("committed_snapshot_ids") or []:
                        if sid:
                            protected["import"].add(str(sid))
                    continue

                if env.kind == "restore_apply":
                    if st_lower in ("cancelled", "timed_out"):
                        continue
                    if st_lower == "committed" and not payload.get("pre_restore_ref"):
                        continue
                    target = payload.get("target_snapshot_id")
                    if target:
                        protected["restore_target"].add(str(target))
                    pre = payload.get("pre_restore_ref")
                    if pre:
                        protected["pre_restore"].add(str(pre))
                    continue

                if env.kind == "restore_job":
                    if st_lower in ("committed", "failed", "cancelled", "timed_out"):
                        continue
                    sid = payload.get("snapshot_id")
                    if sid:
                        protected["restore_target"].add(str(sid))
                    continue

                if env.kind == "delete":
                    # Active delete does not protect delete_set from this path.
                    continue

        return protected

    def delete_snapshot(self, snapshot_id: str, *, branch: bool = False) -> Any:
        """Delete a snapshot with live-lineage and active-transaction protection.

        Raises ``SnapshotDeleteConflict`` with typed ``details["reason"]`` for
        HTTP 409 consumers.
        """
        from backend.services.snapshot_v2.deletion import DeletionService

        by_reason = self.compute_protected_snapshot_ids()
        for reason, ids in by_reason.items():
            if snapshot_id in ids:
                raise SnapshotDeleteConflict(
                    "PROTECTED",
                    f"Snapshot {snapshot_id} is protected ({reason}) and cannot be deleted",
                    details={
                        "reason": reason,
                        "snapshot_id": snapshot_id,
                        "protection": reason,
                    },
                )

        protected_flat: set[str] = set()
        for ids in by_reason.values():
            protected_flat |= ids

        deletion = DeletionService(
            self.session_id,
            self.paths,
            manifest_store=self.manifest_store,
            object_store=self.object_store,
        )
        if branch:
            all_manifests = self.manifest_store.verify_all_committed()
            children: dict[str, list[str]] = {}
            for m in all_manifests:
                pid = m.parent_id
                if pid:
                    children.setdefault(pid, []).append(m.snapshot_id)
            descendant_ids: list[str] = []
            stack = list(children.get(snapshot_id, []))
            while stack:
                sid = stack.pop()
                descendant_ids.append(sid)
                stack.extend(children.get(sid, []))
            for sid in descendant_ids:
                for reason, ids in by_reason.items():
                    if sid in ids:
                        raise SnapshotDeleteConflict(
                            "PROTECTED",
                            f"Snapshot {sid} is protected ({reason}) and cannot be deleted",
                            details={
                                "reason": reason,
                                "snapshot_id": sid,
                                "protection": reason,
                            },
                        )
            return deletion.delete_branch(
                snapshot_id, descendant_ids, protected_ids=protected_flat,
            )
        return deletion.delete_leaf(snapshot_id, protected_ids=protected_flat)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List committed snapshots with typed integrity diagnostics.

        Never silently drops a corrupt committed manifest: each entry is either
        status ``ok`` (with full manifest) or ``corrupt`` (with error code).
        """
        entries = self.manifest_store.list_committed_with_diagnostics()
        out: list[dict[str, Any]] = []
        for e in entries:
            if e.status == "ok" and e.manifest is not None:
                flat = e.manifest.model_dump(mode="json")
                flat["status"] = "ok"
                flat["integrity_status"] = "ok"
                out.append(flat)
            else:
                out.append({
                    "snapshot_id": e.snapshot_id,
                    "status": "corrupt",
                    "integrity_status": "corrupt",
                    "error_code": e.error_code,
                    "error_message": e.error_message,
                    "manifest": None,
                })
        return out

    def preview_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Load a single verified V2 snapshot manifest for detail views."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        try:
            manifest = self.manifest_store.load(snapshot_id)
        except SnapshotIntegrityError:
            raise
        return {"manifest": manifest.model_dump(mode="json")}

    def store_status(self) -> Any:
        """Compute store-level usage statistics from verified manifests."""
        from backend.services.snapshot_v2.management import compute_store_status

        return compute_store_status(self.paths, self.manifest_store)

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
    ) -> Path:
        """Export verified V2 snapshots under the session lock."""
        from backend.services.snapshot_v2.exporter import SnapshotExporter

        lock = get_session_lock(self.session_id)
        async with lock:
            exporter = SnapshotExporter(self.session_id, self.paths)
            return await asyncio.to_thread(
                exporter.export,
                self.session_id,
                include_runtime=include_runtime,
                snapshot_ids=snapshot_ids,
            )

    def _guard_active_safe_point(self) -> None:
        """Raise SnapshotTransactionError if a safe point is active.

        Called by mutation methods (import, delete) before acquiring the
        session lock to prevent concurrent snapshot/restore operations
        from corrupting the session state.
        """
        from backend.services.snapshot_v2.errors import SnapshotTransactionError

        if self.safe_point.is_paused(self.session_id):
            raise SnapshotTransactionError(
                "SAFE_POINT_ACTIVE",
                f"A snapshot/restore operation is in progress for "
                f"session {self.session_id}; rejecting mutation",
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

        lock = get_session_lock(self.session_id)
        async with lock:
            importer = SnapshotImporter(self.session_id, self.paths)
            result = await asyncio.to_thread(
                importer.import_bundle,
                str(archive_path),
                trust_runtime=trust_runtime,
            )
            if getattr(result, "success", False):
                await self._reconcile_analysis_state_snapshots()
            return result

    async def _reconcile_analysis_state_snapshots(self) -> None:
        """Rebuild AnalysisState.snapshots from verified manifests and persist."""
        from backend.services.session_store import async_save_session, update_session_cache
        from backend.state import sessions

        state = sessions.get(self.session_id)
        if state is None:
            return
        try:
            state.snapshots = self.rebuild_snapshots_from_manifests()
        except Exception as exc:
            logger.warning(
                "[%s] rebuild snapshots during reconcile failed: %s",
                self.session_id, exc,
            )
            return
        try:
            await update_session_cache(self.session_id, state)
            await async_save_session(self.session_id, state)
        except Exception as exc:
            logger.warning(
                "[%s] persist after snapshot reconcile failed: %s",
                self.session_id, exc,
            )

    def _find_committed_by_idempotency(
        self, request: SnapshotCaptureRequest,
    ) -> SnapshotManifestV2 | None:
        key = getattr(request, "idempotency_key", None)
        if not key:
            return None
        phase = request.phase or request.reason or "manual"
        for manifest in self.manifest_store.verify_all_committed():
            if (
                manifest.session_id == request.session_id
                and (manifest.phase or "") == phase
                and getattr(manifest, "idempotency_key", None) == key
            ):
                return manifest
        return None

    async def _capture_locked(
        self,
        request: SnapshotCaptureRequest,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
        transaction: str | None = None,
        workflow_state: dict[str, Any] | None = None,
        snapshot_id_override: str | None = None,
        analysis_state: Any = None,
        skip_safe_point_lifecycle: bool = False,
        persist_session: bool = True,
    ) -> SnapshotRecordV2:
        """Capture transaction (caller already holds the session lock).

        ``physical_added_bytes`` is computed from ``ObjectPutResult.created``
        instead of a full-store pre-scan. Each ``put_bytes``/``put_file``
        call returns a boolean ``created`` flag; references whose put
        returned ``created=True`` are counted as physical additions.
        """
        snapshot_id = snapshot_id_override or _new_snapshot_id()
        tx = transaction or snapshot_id
        try:
            manifest, pending = await self._capture_impl(
                request, snapshot_id, tx,
                sandbox=sandbox,
                workspace_root=workspace_root,
                workflow_state=workflow_state,
                analysis_state=analysis_state,
                skip_safe_point_lifecycle=skip_safe_point_lifecycle,
                persist_session=persist_session,
            )
            return SnapshotRecordV2(
                snapshot_id=manifest.snapshot_id,
                manifest=manifest,
                failure=None,
                committed=True,
                session_reconciliation_pending=pending,
            )
        except SnapshotV2Error as exc:
            logger.warning(
                "[snapshot %s] capture failed: %s — %s",
                snapshot_id, exc.code, exc,
            )
            return self._to_failure_record(snapshot_id, request.session_id, exc)
        except Exception as exc:
            logger.exception(
                "[snapshot %s] unexpected capture error: %s", snapshot_id, exc,
            )
            return self._to_failure_record(
                snapshot_id, request.session_id,
                SnapshotCaptureError("UNEXPECTED", str(exc)),
            )

    async def _capture_and_apply(
        self,
        state: Any,
        request: SnapshotCaptureRequest,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
    ) -> SnapshotRecordV2:
        """Capture and apply result to the AnalysisState.

        Unlike ``capture_required_snapshot``, this does not raise
        RequiredSnapshotBlocked (used by unit tests that inspect failure
        fields in-place). Production producers must use
        ``capture_required_snapshot``.
        """
        from backend.services.snapshot_v2.errors import SNAPSHOT_FAILURE_ACTIONS

        if workspace_root is None:
            try:
                from backend.config._config import get_session_workspace
                workspace_root = str(get_session_workspace(request.session_id))
            except Exception:
                workspace_root = None

        record = await self.capture(
            request,
            sandbox=sandbox,
            workspace_root=workspace_root,
            analysis_state=state,
        )
        if record.failure is not None:
            failure_payload = record.failure.model_dump(mode="json")
            failure_payload["allowed_actions"] = list(SNAPSHOT_FAILURE_ACTIONS)
            state.snapshot_failure = failure_payload
            state.snapshot_transaction = {
                "snapshot_id": record.snapshot_id,
                "status": "failed",
                "failed_at": _now_iso(),
            }
            state.wait_input_reason = "snapshot_failure"
            await self._send_state_sync({
                "change_type": "snapshot_failure",
                "version": getattr(state, "state_version", 0),
                "status": getattr(state, "status", None),
                "snapshot_failure": failure_payload,
                "snapshot_transaction": state.snapshot_transaction,
            })
        else:
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
        return record

    async def _capture_locked_for_restore(
        self,
        request: SnapshotCaptureRequest,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
        workflow_state: dict[str, Any] | None = None,
        analysis_state: Any = None,
        snapshot_id_override: str | None = None,
    ) -> SnapshotRecordV2:
        """Capture for restore (caller already holds session lock + safe point).

        Always skips cooperative safe-point create/release so the restore
        transaction's already-held safe point is retained for the full apply.
        Does not persist session state — restore owns final session persistence
        after apply (avoids re-entrant save while the session lock is held).
        """
        return await self._capture_locked(
            request,
            sandbox=sandbox,
            workspace_root=workspace_root,
            workflow_state=workflow_state,
            analysis_state=analysis_state,
            snapshot_id_override=snapshot_id_override,
            skip_safe_point_lifecycle=True,
            persist_session=False,
        )

    def _clear_failure(self, state: Any) -> None:
        """Clear the failure gate state for a retry."""
        state.snapshot_failure = None
        state.snapshot_transaction = None
        if state.wait_input_reason == "snapshot_failure":
            state.wait_input_reason = None

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
        # Step 1 — validate state
        self._validate_state(request)

        # Step 2 — lock is already held by capture() / _capture_locked()

        # Step 3 — create safe point (stores descriptor for kernel_present etc.)
        # Restore path already holds the cooperative safe point; do not
        # create/release it or pre_restore capture would unpause the workflow.
        if not skip_safe_point_lifecycle:
            await self._create_safe_point(workflow_state or {})

        try:
            # Consistency barrier: await pending saves before freezing execution.
            if sandbox is not None and hasattr(sandbox, "await_pending_saves"):
                await sandbox.await_pending_saves()

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
                # physical_added_bytes computed from ObjectPutResult.created
                # instead of full-store pre-scan.

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

            def _add_ref_with_result(put_result) -> ObjectRef:
                """Add an ObjectPutResult to the manifest's objects dict.

                Returns the ObjectRef. The ``created`` flag controls
                whether the size counts toward physical_added_bytes.
                """
                nonlocal physical_added
                ref = put_result.ref
                _record_transaction_evidence(ref)
                if ref.sha256 not in all_objects:
                    all_objects[ref.sha256] = ref
                    if put_result.created:
                        physical_added += ref.size
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
                if ref.sha256 not in all_objects:
                    all_objects[ref.sha256] = ref
                    # Workspace entries were created in this transaction
                    # if they weren't already in the store; use register_existing_ref
                    # fast-path: if the object already existed, don't count it.
                    if not self.object_store.register_existing_ref(ref):
                        physical_added += ref.size
            for ref in runtime.adata_refs.values():
                if ref.sha256 not in all_objects:
                    all_objects[ref.sha256] = ref
                    if not self.object_store.register_existing_ref(ref):
                        physical_added += ref.size
            if runtime.residual_ref is not None:
                rref = runtime.residual_ref
                if rref.sha256 not in all_objects:
                    all_objects[rref.sha256] = rref
                    if not self.object_store.register_existing_ref(rref):
                        physical_added += rref.size
            if runtime_snapshot is not None:
                for ref in (runtime_snapshot.captured_adata_refs or {}).values():
                    if ref.sha256 not in all_objects:
                        all_objects[ref.sha256] = ref
                        if not self.object_store.register_existing_ref(ref):
                            physical_added += ref.size
                if runtime_snapshot.captured_residual_ref is not None:
                    rref = runtime_snapshot.captured_residual_ref
                    if rref.sha256 not in all_objects:
                        all_objects[rref.sha256] = rref
                        if not self.object_store.register_existing_ref(rref):
                            physical_added += rref.size

            # Continuation payload/workflow-state refs must be declared in objects
            # so restore stage can verify them as first-class content.
            for attr in ("workflow_state_ref", "payload_ref"):
                cref = getattr(continuation, attr, None)
                if cref is not None:
                    if cref.sha256 not in all_objects:
                        all_objects[cref.sha256] = cref
                        if not self.object_store.register_existing_ref(cref):
                            physical_added += cref.size

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
            await self._check_quota(manifest)

            # Step 9 — publish manifest
            staging_path = self.manifest_store.write_staging(manifest, prefix=f"{tx}_")
            published = self.manifest_store.publish_staging(staging_path)

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

            # Step 12 — return committed manifest + pending flag (no private attr)
            return reloaded, session_reconciliation_pending

        except SnapshotV2Error:
            if not skip_safe_point_lifecycle:
                await self._release_safe_point()
            raise
        except Exception as exc:
            if not skip_safe_point_lifecycle:
                await self._release_safe_point()
            raise SnapshotCaptureError("CAPTURE_FAILED", str(exc))

    # ── Content capture helpers ──────────────────────────────────────

    async def _capture_analysis_state(
        self,
        request: SnapshotCaptureRequest,
        analysis_state: Any = None,
    ) -> ObjectPutResult:
        """Canonicalize + externalize AnalysisState into the object store.

        Returns ObjectPutResult with ``created`` flag for physical_added accounting.
        """
        if request.analysis_state_ref is not None:
            self.object_store.verify(request.analysis_state_ref)
            return ObjectPutResult(
                ref=request.analysis_state_ref,
                created=False,
            )

        state = analysis_state
        if state is None:
            state = await self._get_session_state(request.session_id)
        if state is None:
            raise SnapshotCaptureError(
                "MISSING_ANALYSIS_STATE",
                "No AnalysisState available to capture",
            )

        # model_dump excluding derived listing fields that are rebuilt
        state_dict = state.model_dump(mode="json")
        state_dict.pop("snapshots", None)
        state_dict.pop("realtime_messages", None)
        for seg in state_dict.get("detailed_report_segments", []) or []:
            if isinstance(seg, dict):
                seg.pop("content_html", None)

        # Externalize large blobs into object store via inline base64 reduction
        # by writing canonical JSON bytes directly (blob refs stay as tokens).
        raw = json.dumps(
            state_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return self.object_store.put_bytes(raw, "application/json")

    async def _capture_notebook(
        self,
        request: SnapshotCaptureRequest,
    ) -> ObjectPutResult:
        if request.notebook_ref is not None:
            self.object_store.verify(request.notebook_ref)
            return ObjectPutResult(
                ref=request.notebook_ref,
                created=False,
            )

        states_dir = get_session_states(request.session_id)
        nb_path = Path(states_dir) / f"{request.session_id}.ipynb"
        if not nb_path.exists():
            # Empty notebook is valid when no kernel work has been done.
            empty_nb = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [],
            }
            raw = json.dumps(
                empty_nb, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            return self.object_store.put_bytes(raw, "application/x-ipynb+json")
        return self.object_store.put_file(nb_path, "application/x-ipynb+json")

    async def _capture_checkpoint(
        self,
        request: SnapshotCaptureRequest,
    ) -> ObjectPutResult | None:
        if request.checkpoint_ref is not None:
            self.object_store.verify(request.checkpoint_ref)
            return ObjectPutResult(
                ref=request.checkpoint_ref,
                created=False,
            )
        # Optional: LangGraph checkpoint sqlite if present
        states_dir = get_session_states(request.session_id)
        ckpt = Path(states_dir) / "checkpoints.sqlite"
        if not ckpt.exists() or not ckpt.is_file():
            return None
        return self.object_store.put_file(ckpt, "application/x-sqlite3")

    # ── Step helpers ─────────────────────────────────────────────────

    def _validate_state(self, request: SnapshotCaptureRequest) -> None:
        if not request.session_id:
            raise SnapshotCaptureError(
                "MISSING_SESSION_ID", "session_id is required",
            )
        if request.session_id != self.session_id:
            raise SnapshotCaptureError(
                "SESSION_ID_MISMATCH",
                f"Request session_id '{request.session_id}' "
                f"does not match service session_id '{self.session_id}'",
            )

    async def _create_safe_point(self, workflow_state: dict[str, Any] | None = None) -> None:
        # Capture lifecycle may store graph-position metadata, but must NOT
        # authorize empty-complete from untrusted capture args.
        # Token is process-local for this service instance's lifecycle hold.
        owner_token = f"capture:{self.session_id}:{id(self)}"
        self._safe_point_owner_token = owner_token
        await self.safe_point.acquire(
            self.session_id,
            workflow_state,
            eligibility_trusted=False,
            owner_token=owner_token,
        )

    async def _release_safe_point(self) -> None:
        owner_token = getattr(self, "_safe_point_owner_token", None) or (
            f"capture:{self.session_id}:{id(self)}"
        )
        await self.safe_point.release(self.session_id, owner_token)
        self._safe_point_owner_token = None

    def _has_active_workflow_ownership(self) -> bool:
        """True when this session owns a live workflow in the process map."""
        try:
            from backend.state import workflows
            return self.session_id in workflows and workflows.get(self.session_id) is not None
        except Exception:
            return False

    def _resolve_empty_runtime_eligibility(
        self,
        phase: str | None,
        sandbox: Any,
        safe_point_descriptor: dict[str, Any] | None,
    ) -> bool:
        """Whether empty-complete runtime may be committed without a sandbox.

        Eligibility requires:
        - no live sandbox object
        - an eligibility-trusted descriptor (ack / explicit producer acquire)
        - ``kernel_present is False`` on that descriptor
        - no active workflow ownership (except inactive post_report/manual paths
          that already proved no kernel)
        - terminal or explicitly inactive semantics for dynamic phases

        Raw ``workflow_state`` passed into capture is never sufficient alone.
        ``pre_restore`` with an active workflow never accepts empty-complete,
        even if a contradictory no-kernel descriptor is present. Absence of a
        trusted descriptor is never proof of inactivity.
        """
        if sandbox is not None:
            return False

        phase_s = str(phase or "manual")
        desc = dict(safe_point_descriptor or {}) if safe_point_descriptor else {}

        # No trusted descriptor → unproven; cannot empty-complete.
        if not desc:
            return False

        if "kernel_present" not in desc or desc.get("kernel_present") is not False:
            return False

        active_workflow = self._has_active_workflow_ownership()

        # Active pre_restore always requires the live sandbox.
        if phase_s == "pre_restore" and active_workflow:
            return False

        inactive = desc.get("inactive") is True
        terminal = (
            desc.get("status") == "completed"
            or desc.get("continuation_kind") == "terminal"
            or desc.get("kind") == "terminal"
        )

        if active_workflow:
            # Active pre_restore / post_chat always require the live sandbox.
            if phase_s in ("post_chat", "pre_restore"):
                return False
            if phase_s == "post_report":
                return True
            return inactive or terminal

        # No active workflow ownership.
        # post_report and idle pre_restore may empty-complete with proven
        # no-kernel descriptor. post_chat/manual still need inactive/terminal.
        if phase_s in ("post_report", "pre_restore"):
            return True
        if phase_s in ("post_chat", "manual"):
            return inactive or terminal

        # Unknown / other phases: only with explicit inactive or terminal.
        return inactive or terminal

    async def _capture_workspace(
        self, workspace_root: str,
    ) -> tuple[list[WorkspaceEntry], list[str]]:
        warnings: list[str] = []
        entries, ws_warnings = self.workspace_capture.capture_scope(
            self.object_store, workspace_root,
        )
        warnings.extend(ws_warnings)
        return entries, warnings

    def _resolve_kernel_present(
        self,
        sandbox: Any,
        workflow_state: dict[str, Any] | None,
    ) -> bool:
        sources: list[Any] = []
        desc = self.safe_point.get_descriptor(self.session_id)
        if desc is not None:
            sources.append(desc)
        if workflow_state is not None:
            sources.append(workflow_state)

        for source in sources:
            if not isinstance(source, dict):
                continue
            if "kernel_present" in source:
                return bool(source["kernel_present"])
            nested = source.get("safe_point_descriptor")
            if isinstance(nested, dict) and "kernel_present" in nested:
                return bool(nested["kernel_present"])

        return getattr(sandbox, "km", None) is not None

    def _collect_checkpoint_dependencies(
        self,
        workflow_state: dict[str, Any] | None,
        sandbox_state: dict[str, Any],
        analysis_state: Any = None,
    ) -> set[str]:
        deps: set[str] = set()
        for source in (workflow_state or {}, sandbox_state):
            if isinstance(source, dict):
                deps |= _as_name_set(source.get("checkpoint_dependencies"))
        if analysis_state is not None:
            ss = getattr(analysis_state, "sandbox_state", None) or {}
            if isinstance(ss, dict):
                deps |= _as_name_set(ss.get("checkpoint_dependencies"))
        return deps

    def _rehydrate_tracker(
        self,
        sandbox: Any,
        sandbox_state: dict[str, Any],
    ) -> RuntimeDependencyTracker | None:
        tracker = getattr(sandbox, "runtime_dependency_tracker", None)
        if tracker is not None:
            return tracker
        data = sandbox_state.get("runtime_dependency_tracker_v2")
        if not data:
            return None
        tracker = RuntimeDependencyTracker.from_state_dict(data)
        try:
            sandbox.runtime_dependency_tracker = tracker
        except Exception:
            pass
        return tracker

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

    async def _compute_root_hash(self, manifest: SnapshotManifestV2) -> None:
        raw = canonical_json(manifest)
        manifest.root_hash = hashlib.sha256(raw).hexdigest()

    async def _check_quota(self, manifest: SnapshotManifestV2) -> None:
        # Session quota counts physical additions for this capture.
        added = manifest.physical_added_bytes
        if added > self.session_quota_bytes:
            raise SnapshotStorageError(
                "snapshot_quota_exceeded",
                f"Snapshot physical addition {added} exceeds quota "
                f"{self.session_quota_bytes}",
            )
        # Also enforce cumulative store size soft bound via logical objects.
        # The v2 root dir is not created during a normal capture (objects/
        # staging/committed are siblings), so probe the nearest existing
        # ancestor — free space is identical across the same filesystem.
        probe = self.paths.root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
            free_after = usage.free - added
            if free_after < self.free_space_reserve_bytes:
                raise SnapshotStorageError(
                    "snapshot_quota_exceeded",
                    f"Insufficient disk space: {usage.free} free, "
                    f"{added} needed, {self.free_space_reserve_bytes} reserve",
                )
        except OSError as exc:
            raise SnapshotStorageError(
                "DISK_USAGE_CHECK_FAILED",
                f"Failed to check disk usage: {exc}",
            )

    async def _save_session_state(self, session_id: str) -> None:
        from backend.services.session_store import async_save_session
        state = await self._get_session_state(session_id)
        if state is not None:
            await async_save_session(session_id, state)

    async def _get_session_state(self, session_id: str) -> Any:
        from backend.state import sessions
        return sessions.get(session_id)

    def _mark_runtime_committed(
        self, sandbox: Any, runtime_revision: int | None = None,
    ) -> None:
        self.runtime_capture.mark_runtime_committed(
            sandbox, runtime_revision=runtime_revision,
        )


    def rebuild_snapshots_from_manifests(self) -> list:
        """Rebuild AnalysisState.snapshots listing from verified committed manifests."""
        from backend.models.schemas import Snapshot
        from datetime import datetime as _dt

        rebuilt = []
        for manifest in self.manifest_store.verify_all_committed():
            try:
                ts = _dt.fromisoformat(
                    manifest.created_at.replace("Z", "+00:00")
                )
            except Exception:
                ts = _dt.now()
            cont_kind = None
            cont = manifest.continuation
            if cont is not None:
                cont_kind = getattr(cont, "kind", None)
            rebuilt.append(
                Snapshot(
                    id=manifest.snapshot_id,
                    name=manifest.name or manifest.snapshot_id,
                    description=manifest.description or "",
                    phase=manifest.phase,
                    round_number=manifest.round_number,
                    timestamp=ts,
                    parent_id=manifest.parent_id,
                    branch_label=manifest.branch_label,
                    branch_id=manifest.branch_id,
                    size_bytes=manifest.logical_size_bytes,
                    format_version=2,
                    runtime_status=manifest.runtime.status if manifest.runtime else None,
                    logical_size_bytes=manifest.logical_size_bytes,
                    physical_added_bytes=manifest.physical_added_bytes,
                    source_restore_transaction_id=manifest.source_restore_transaction_id,
                    continuation_kind=cont_kind,
                    runtime_state_included=bool(
                        manifest.runtime
                        and (
                            manifest.runtime.adata_refs
                            or manifest.runtime.residual_ref is not None
                        )
                    ),
                )
            )
        # Sort by created_at ascending for stable lineage listing
        rebuilt.sort(key=lambda s: s.timestamp)
        return rebuilt

    # ── Failure helpers ──────────────────────────────────────────────

    def _to_failure_record(
        self,
        snapshot_id: str,
        session_id: str,
        error: SnapshotV2Error,
    ) -> SnapshotRecordV2:
        stub_ref = ObjectRef(
            sha256="0" * 64, size=0, media_type="application/octet-stream",
        )
        stub_manifest = SnapshotManifestV2(
            session_id=session_id,
            snapshot_id=snapshot_id,
            name=snapshot_id,
            description="capture-failure-stub",
            phase="manual",
            created_at=_now_iso(),
            analysis_state_ref=stub_ref,
            notebook_ref=stub_ref,
            runtime=empty_complete_runtime(),
            continuation=IdleCapsule(graph_version="2.0"),
            logical_size_bytes=0,
            physical_added_bytes=0,
        )
        return SnapshotRecordV2(
            snapshot_id=snapshot_id,
            manifest=stub_manifest,
            failure=SnapshotFailureRecord(
                snapshot_id=snapshot_id,
                session_id=session_id,
                failed_at=_now_iso(),
                code=error.code,
                message=str(error),
                details=error.details,
                allowed_actions=["retry", "manage", "terminate"],
            ),
            committed=False,
            session_reconciliation_pending=False,
        )

    async def _send_state_sync(self, payload: dict[str, Any]) -> None:
        """Emit a state_sync message for the snapshot failure gate."""
        from backend.state import websockets
        ws = websockets.get(self.session_id)
        if ws is not None:
            try:
                await ws.send_json({
                    "type": "state_sync",
                    "change_type": payload.get("change_type", "snapshot_failure"),
                    **payload,
                })
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to send snapshot failure state sync: %s",
                    self.session_id, exc,
                )
