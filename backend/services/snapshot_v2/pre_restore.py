"""Pre-restore safety capture and audit bundle for Snapshot V2.

Provides ``PreRestoreCapture`` which captures a pre_restore-phase snapshot
derived from the live safe-point descriptor (via ``freeze_dynamic`` /
``ContinuationRegistry.freeze``) that freezes the current session state
before a restore transaction, and ``AuditBundle`` which captures an audit
trail for a failed or discarded restore attempt.

Fail-closed: pre_restore uses the lock-held capture API
(``_capture_locked_for_restore``) and aborts restore on
``RequiredSnapshotBlocked``. Does not hardcode IdleCapsule unconditionally.

Revision-qualified runtime reuse
-------------------------------
Reuse the previous ``RuntimeGenerationManifest`` only when ALL of:

  1. A previous snapshot exists with a non-empty complete ``runtime``
  2. ``runtime_revision == saved_runtime_revision`` (no kernel drift)
  3. The previous manifest is V2 signed (HMAC + root hash verified)
  4. All runtime objects are present and content-addressable
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.services.snapshot_v2.continuation import ContinuationRegistry
from backend.services.snapshot_v2.errors import (
    RequiredSnapshotBlocked,
    SnapshotCaptureError,
    SnapshotIntegrityError,
)
from backend.services.snapshot_v2.models import (
    AuditBundleRef,
    IdleCapsule,
    SnapshotCaptureRequest,
    SnapshotFailureRecord,
    SnapshotManifestV2,
    SnapshotRecordV2,
    SnapshotRestoreRequest,
)
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
from backend.services.snapshot_v2.service import SnapshotService

logger = logging.getLogger(__name__)


@dataclass
class RestoreCaptureContext:
    """Carries everything needed to freeze the current session state before a restore transaction.

    The caller must already hold the per-session lock from ``state.py``.
    """

    session_id: str
    paths: SnapshotV2Paths
    sandbox: Any
    workspace_root: str | None = None
    workflow_state: dict = field(default_factory=dict)
    analysis_state: Any = None
    prev_manifest: SnapshotManifestV2 | None = None
    live_safe_point: dict | None = None
    restore_transaction_id: str | None = None
    restore_request: SnapshotRestoreRequest | None = None


class PreRestoreCapture:
    """Captures a pre_restore-phase snapshot with optional runtime reuse."""

    def __init__(
        self,
        service: SnapshotService,
        object_store: ObjectStore,
        manifest_store: ManifestStore,
    ) -> None:
        self._service = service
        self._object_store = object_store
        self._manifest_store = manifest_store

    def _build_capsule(self, context: RestoreCaptureContext):
        """Derive continuation capsule from live safe point (not unconditional idle)."""
        live_sp = context.live_safe_point
        if live_sp is None:
            live_sp = get_snapshot_safe_point_manager().get_descriptor(
                context.session_id
            ) or {}
        workflow_state = dict(context.workflow_state or {})
        workflow_state.update(live_sp)
        registry = ContinuationRegistry(self._object_store)
        return registry.freeze(
            "pre_restore",
            context.analysis_state,
            workflow_state,
            live_safe_point=live_sp,
        )

    async def _capture_for_restore(
        self,
        req: SnapshotCaptureRequest,
        context: RestoreCaptureContext,
        *,
        sandbox: Any = None,
        snapshot_id_override: str | None = None,
    ) -> SnapshotRecordV2:
        """Lock-held capture only — never reacquires the session lock."""
        capture = getattr(self._service, "_capture_locked_for_restore", None)
        if capture is not None:
            return await capture(
                req,
                sandbox=sandbox,
                workspace_root=context.workspace_root,
                workflow_state=context.workflow_state or {},
                analysis_state=context.analysis_state,
                snapshot_id_override=snapshot_id_override,
            )
        # Fallback: lock-held capture without session persistence (restore owns save).
        return await self._service._capture_locked(
            req,
            sandbox=sandbox,
            workspace_root=context.workspace_root,
            workflow_state=context.workflow_state or {},
            analysis_state=context.analysis_state,
            snapshot_id_override=snapshot_id_override,
            skip_safe_point_lifecycle=True,
            persist_session=False,
        )

    async def capture_locked(self, context: RestoreCaptureContext) -> SnapshotRecordV2:
        """Capture a pre_restore snapshot (caller holds session lock).

        Uses ``freeze`` / ``freeze_dynamic`` for the capsule. On failure of a
        required capture, applies snapshot_failure and raises
        ``RequiredSnapshotBlocked`` so restore aborts.
        """
        capsule = self._build_capsule(context)
        req = SnapshotCaptureRequest(
            session_id=context.session_id,
            reason="pre_restore",
            phase="pre_restore",
            # Declare the round of the state this capture actually freezes.
            # live_verify compares manifest.round_number against the restored
            # session.json current_round and gates on ``is not None``, so the
            # model default (0) is asserted as a real expectation, never
            # skipped. Left defaulted, every pre_restore snapshot taken past
            # round 0 was unrestorable: apply() overwrites the live session
            # first, the round check then fails, and the transaction is refused
            # as LIVE_VERIFY_FAILED with the session recovery-blocked -- and
            # this is the one snapshot a user reaches for to undo a restore.
            round_number=int(
                getattr(context.analysis_state, "current_round", 0) or 0
            ),
            capsule=capsule,
        )

        sandbox = context.sandbox
        if self._can_reuse_runtime(context):
            prev = context.prev_manifest
            req.runtime = prev.runtime
            req.analysis_state_ref = prev.analysis_state_ref
            req.notebook_ref = prev.notebook_ref
            req.checkpoint_ref = prev.checkpoint_ref
            # The reused analysis_state_ref is prev serialized state, so the
            # round must follow that state and not the live one: the revision
            # check only proves the kernel is unchanged, and a session can
            # advance a round without ever dirtying it.
            req.round_number = prev.round_number
            sandbox = None  # skip kernel introspection
            logger.info(
                "[%s] Reusing runtime for pre_restore capture",
                context.session_id,
            )

        # Prefer apply-and-raise path when analysis_state is available so
        # restore aborts on RequiredSnapshotBlocked.
        if context.analysis_state is not None:
            record = await self._capture_for_restore(req, context, sandbox=sandbox)
            if record.failure is not None:
                from backend.services.snapshot_v2.failure_gate import (
                    enter_snapshot_failure_gate,
                )

                state = context.analysis_state
                failure_payload = record.failure.model_dump(mode="json")
                enter_snapshot_failure_gate(
                    state,
                    failure_payload,
                    retry_context={
                        "producer": "pre_restore",
                        "resume_policy": "restore_transaction",
                        "snapshot_request": req.model_dump(mode="json"),
                        "previous_status": (
                            getattr(state, "status", None)
                            and getattr(state.status, "value", str(state.status))
                            or "running"
                        ),
                        "restore_transaction_id": context.restore_transaction_id,
                        "restore_request": (
                            context.restore_request.model_dump(mode="json")
                            if context.restore_request is not None
                            else None
                        ),
                    },
                )
                state.snapshot_transaction = {
                    "snapshot_id": record.snapshot_id,
                    "status": "failed",
                }
                raise RequiredSnapshotBlocked(
                    record.failure.code,
                    record.failure.message,
                    details=dict(record.failure.details or {}),
                    snapshot_id=record.snapshot_id,
                )
            return record

        return await self._capture_for_restore(req, context, sandbox=sandbox)

    def _can_reuse_runtime(self, context: RestoreCaptureContext) -> bool:
        """Return ``True`` when the previous runtime generation can be reused."""
        prev = context.prev_manifest
        if prev is None:
            return False
        if prev.runtime is None or prev.runtime.status != "complete":
            return False
        has_runtime_objects = (
            bool(prev.runtime.adata_refs)
            or bool(prev.runtime.captured_residual_names)
            or prev.runtime.residual_ref is not None
        )
        if not has_runtime_objects:
            return False

        # 1. Revision match
        ss = self._get_sandbox_state(context)
        rt_rev = ss.get("runtime_revision", 0)
        sv_rev = ss.get("saved_runtime_revision", 0)
        if rt_rev != sv_rev:
            return False

        # 2. V2 signature check (load raises if HMAC or root hash fails)
        try:
            self._manifest_store.load(prev.snapshot_id)
        except SnapshotIntegrityError:
            return False

        # 3. Verify all runtime objects exist and are content-addressable
        rt = prev.runtime
        for ref in (rt.adata_refs or {}).values():
            try:
                self._object_store.verify(ref)
            except SnapshotIntegrityError:
                return False
        if rt.residual_ref is not None:
            try:
                self._object_store.verify(rt.residual_ref)
            except SnapshotIntegrityError:
                return False

        return True

    def _get_sandbox_state(self, context: RestoreCaptureContext) -> dict:
        state = context.analysis_state
        if state is None:
            return {}
        if hasattr(state, "sandbox_state"):
            ss = state.sandbox_state
            return ss if isinstance(ss, dict) else {}
        if isinstance(state, dict):
            ss = state.get("sandbox_state", {})
            return ss if isinstance(ss, dict) else {}
        return {}


class AuditBundle:
    """Captures an audit trail for a failed or discarded restore attempt.

    The audit bundle captures state, workspace, and notebook content plus
    structured failure information.  It is signed and retained with the
    transaction but is never listed as a restorable snapshot (its manifest
    snapshot_id carries an ``audit-`` prefix, and ``ManifestStore.list_committed``
    filters those entries).
    """

    AUDIT_PREFIX = "audit-"

    def __init__(self, service: SnapshotService) -> None:
        self._service = service

    async def capture_locked(
        self,
        context: RestoreCaptureContext,
        failures: list[dict],
    ) -> AuditBundleRef:
        """Capture an audit bundle.

        Caller must already hold the per-session lock.  Returns an
        ``AuditBundleRef`` that references the committed manifest and
        the structured failure information.

        Raises ``SnapshotCaptureError`` if the underlying capture fails.
        """
        audit_id = f"{self.AUDIT_PREFIX}{uuid.uuid4().hex[:16]}"

        # Audit bundles are archival only — idle is appropriate.
        capsule = IdleCapsule(graph_version="2.0")
        req = SnapshotCaptureRequest(
            session_id=context.session_id,
            reason="audit_bundle",
            capsule=capsule,
        )

        capture = getattr(self._service, "_capture_locked_for_restore", None)
        if capture is not None:
            record = await capture(
                req,
                sandbox=context.sandbox,
                workspace_root=context.workspace_root,
                workflow_state=context.workflow_state or {},
                analysis_state=context.analysis_state,
                snapshot_id_override=audit_id,
            )
        else:
            record = await self._service._capture_locked(
                req,
                sandbox=context.sandbox,
                workspace_root=context.workspace_root,
                workflow_state=context.workflow_state or {},
                analysis_state=context.analysis_state,
                snapshot_id_override=audit_id,
                skip_safe_point_lifecycle=True,
                persist_session=False,
            )

        if record.failure is not None:
            raise SnapshotCaptureError(
                "AUDIT_CAPTURE_FAILED",
                f"Audit bundle capture failed: {record.failure.message}",
                details={"capture_failure": record.failure.model_dump(mode="json")},
            )

        failure_record = SnapshotFailureRecord(
            snapshot_id=audit_id,
            session_id=context.session_id,
            failed_at=record.manifest.created_at,
            code="RESTORE_FAILURE",
            message="Restore operation failed or was discarded",
            details={"failures": failures},
        )

        return AuditBundleRef(
            audit_id=audit_id,
            snapshot_id=record.manifest.snapshot_id,
            manifest=record.manifest,
            failure=failure_record,
        )
