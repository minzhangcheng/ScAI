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
    empty_complete_runtime,
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
            # Reuse is scoped to exactly what the revision gate proves: the
            # kernel. It says nothing about the analysis state, notebook,
            # checkpoint or round, and a session advances a round without ever
            # dirtying the kernel (the plan-review gate of round N+1, whose
            # lineage head is still round N's capture). Aliasing those four
            # fields to the lineage head made the pre_restore snapshot
            # re-declare the head instead of freezing the live session: a
            # rollback silently discarded every change since that head, and
            # once the rounds diverged the manifest contradicted the capsule
            # this method froze from the live state — apply() installs the
            # head's session.json, activate() writes the capsule's live round
            # back over it, live_verify reports "current_round mismatch" and the
            # transaction is refused as LIVE_VERIFY_FAILED with the session
            # recovery-blocked. This is the one snapshot a user reaches for to
            # undo a restore, so it must serialize the state it claims to
            # freeze. Re-serializing is close to free: the object store is
            # content-addressed, so an unchanged object dedups to the same hash
            # and adds no physical bytes.
            req.runtime = prev.runtime
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

    async def _capture(
        self,
        req: SnapshotCaptureRequest,
        context: RestoreCaptureContext,
        *,
        sandbox: Any,
        snapshot_id_override: str,
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

        record = await self._capture(
            req, context, sandbox=context.sandbox, snapshot_id_override=audit_id,
        )

        runtime_failure: SnapshotFailureRecord | None = None
        if record.failure is not None:
            # This bundle is the exit from a failed pre_restore, and the
            # dominant reason a pre_restore fails is that the live kernel cannot
            # be serialized. Handing the same sandbox to the same capture entry
            # point re-runs the identical introspection against the identical
            # kernel and fails identically, so discard_current — the one way
            # forward PRE_RESTORE_UNAVAILABLE steers the client into — returned
            # AUDIT_FAILED, a code the panel offers no action for. Retry and
            # discard both looped back into the same failure and no API path
            # could complete the restore.
            #
            # The kernel is captured opportunistically, never as a precondition:
            # discard_current is by definition a request to throw it away
            # (runtime_choice="keep" is rejected alongside it), and the bundle's
            # contract is the state, workspace and notebook plus structured
            # failure information. So keep the live attempt — while it succeeds
            # it is the only record of the discarded kernel — and fall back to a
            # runtime-free archival capture when it does not. sandbox=None alone
            # would not do it: with no runtime on the request the capture falls
            # to the empty-runtime eligibility gate, which refuses
            # reason="audit_bundle", trading AUDIT_CAPTURE_FAILED for
            # LIVE_SANDBOX_REQUIRED. Failures that do not originate in the
            # kernel (quota, workspace, analysis state) recur on the retry and
            # still raise — the bundle is never published on a guess.
            runtime_failure = record.failure
            req.runtime = empty_complete_runtime()
            record = await self._capture(
                req, context, sandbox=None, snapshot_id_override=audit_id,
            )

        if record.failure is not None:
            raise SnapshotCaptureError(
                "AUDIT_CAPTURE_FAILED",
                f"Audit bundle capture failed: {record.failure.message}",
                details={"capture_failure": record.failure.model_dump(mode="json")},
            )

        if runtime_failure is not None:
            # A bundle that is the only record of the discarded state must say
            # what it does not contain, and why.
            failures = [
                *failures,
                {
                    "code": "AUDIT_RUNTIME_NOT_CAPTURED",
                    "message": (
                        "Runtime was not captured into the audit bundle: "
                        f"{runtime_failure.message}"
                    ),
                    "cause": runtime_failure.code,
                },
            ]

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
