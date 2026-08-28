"""Snapshot restore transaction flow owned outside the service class."""
from __future__ import annotations

from dataclasses import dataclass

from backend.config._config import (
    get_session_dir,
    get_session_states,
    get_session_workspace,
)
from backend.services.snapshot_v2.errors import (
    RequiredSnapshotBlocked,
    SnapshotCaptureError,
    SnapshotTransactionError,
)
from backend.services.session_lifecycle import assert_session_generation_active
from backend.state import report_workflows, sessions, workflows

from .shared import (  # noqa: F401
    asyncio,
    logger,
    Path,
    Any,
    ObjectRef,
    SnapshotRestoreRequest,
    SnapshotTransactionRecord,
    checkpoint_sqlite_path,
    get_session_lock,
)


@dataclass
class _RestoreFlowContext:
    service: Any
    request: SnapshotRestoreRequest
    transaction_id: str
    expected_lifecycle_generation: int | None
    status: Any
    cancelled: Any
    job_transaction_id: str | None
    session_dir: Any
    workspace_root: Any
    states_root: Any
    workflow_state: Any
    analysis_state: Any
    workflow: Any
    workflow_owners: tuple[Any, ...]
    capsule_kind: str


@dataclass
class _AppliedRestore:
    recovery_mode: str
    staged: Any
    applied_result: Any

async def restore_snapshot(
    service,
    request: SnapshotRestoreRequest,
    transaction_id: str,
    cancel_event: asyncio.Event | None = None,
    *,
    job_transaction_id: str | None = None,
    status_callback: Any = None,
    expected_lifecycle_generation: int | None = None,
    bindings=None,
) -> SnapshotTransactionRecord:
    """Sole orchestration method for restore jobs.

    Sequence under the session lock after safe-point acknowledgement
    (caller must already have obtained the safe point):

      preflight → pre_restore|forward-only audit → stage → cancel check
      → apply → validate checkpoint → activate dispatcher → persist
      → commit journal → emit committed → cleanup

    ``committed`` is only returned after live side effects are verified.
    """
    from backend.services.snapshot_v2.preflight import RestorePreflight
    from backend.services.session_lifecycle import (
        SessionGenerationError,
        SessionGoneError,
        assert_session_generation_active,
    )
    def _status(status: str, **kw: Any) -> SnapshotTransactionRecord:
        rec = SnapshotTransactionRecord(
            transaction_id=transaction_id,
            session_id=service.session_id,
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

    if expected_lifecycle_generation is not None:
        try:
            assert_session_generation_active(
                service.session_id,
                expected_generation=expected_lifecycle_generation,
            )
        except (SessionGoneError, SessionGenerationError) as exc:
            return _status(
                "failed",
                failure={
                    "code": "SESSION_GENERATION_STALE",
                    "message": str(exc),
                },
            )

    # Authoritative preflight (recheck before mutation).
    preflight = RestorePreflight(service.paths).run(request.snapshot_id, request)
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

    lock = get_session_lock(service.session_id)
    async with lock:
        analysis_state = sessions.get(service.session_id)
        analysis_workflow = workflows.get(service.session_id)
        report_workflow = report_workflows.get(service.session_id)
        workflow = analysis_workflow or report_workflow
        workflow_owners = tuple(
            owner
            for owner in (analysis_workflow, report_workflow)
            if owner is not None
        )
        if (
            len(workflow_owners) == 2
            and workflow_owners[0] is workflow_owners[1]
        ):
            workflow_owners = workflow_owners[:1]
        workflow_state = bindings._workflow_state_for_snapshot(workflow)
        live_sp = service.safe_point.get_descriptor(service.session_id) or {}
        if not bindings._has_trusted_workflow_snapshot_context(
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
            workspace_root = str(get_session_workspace(service.session_id))
        except Exception:
            workspace_root = None
        try:
            session_dir = str(get_session_dir(service.session_id))
        except Exception:
            session_dir = str(service.paths.root.parent)
        try:
            states_root = str(get_session_states(service.session_id))
        except Exception:
            states_root = str(Path(session_dir) / "states")

        context = _RestoreFlowContext(
            service=service,
            request=request,
            transaction_id=transaction_id,
            expected_lifecycle_generation=expected_lifecycle_generation,
            status=_status,
            cancelled=_cancelled,
            job_transaction_id=job_transaction_id,
            session_dir=session_dir,
            workspace_root=workspace_root,
            states_root=states_root,
            workflow_state=workflow_state,
            analysis_state=analysis_state,
            workflow=workflow,
            workflow_owners=workflow_owners,
            capsule_kind=capsule_kind,
        )
        return await _restore_locked(context)

async def _restore_locked(
    context: _RestoreFlowContext,
) -> SnapshotTransactionRecord:
    from backend.services.snapshot_v2.pre_restore import RestoreCaptureContext

    service = context.service
    request = context.request
    transaction_id = context.transaction_id
    expected_lifecycle_generation = context.expected_lifecycle_generation
    _status = context.status
    _cancelled = context.cancelled
    job_transaction_id = context.job_transaction_id
    session_dir = context.session_dir
    workspace_root = context.workspace_root
    states_root = context.states_root
    workflow_state = context.workflow_state
    analysis_state = context.analysis_state
    workflow = context.workflow
    workflow_owners = context.workflow_owners
    capsule_kind = context.capsule_kind

    if _cancelled():
        return _status("cancelled")

    sandbox = getattr(workflow, "sandbox", None) if workflow else None
    live_sp = service.safe_point.get_descriptor(service.session_id) or {}

    capture_context = RestoreCaptureContext(
        session_id=service.session_id,
        paths=service.paths,
        sandbox=sandbox,
        workspace_root=workspace_root,
        workflow_state=workflow_state,
        analysis_state=analysis_state,
        prev_manifest=service._resolve_previous_manifest(analysis_state),
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
            audit_ref = await service.audit_bundle.capture_locked(
                capture_context,
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
            service.manifest_store.load(audit_ref.snapshot_id)
            if audit_ref.manifest.analysis_state_ref is not None:
                service.object_store.verify(audit_ref.manifest.analysis_state_ref)
            if audit_ref.manifest.notebook_ref is not None:
                service.object_store.verify(audit_ref.manifest.notebook_ref)
            for we in audit_ref.manifest.workspace or []:
                service.object_store.verify(
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
            pre_rec = await service.pre_restore_capture.capture_locked(
                capture_context,
            )
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
        staged = await service.restore_coordinator.stage(
            request.snapshot_id,
            service.session_id,
            service.paths,
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
            await service.restore_coordinator.cleanup(transaction_id, service.paths)
        except Exception:
            pass
        return _status("cancelled", recovery_mode=recovery_mode)

    _status("applying", recovery_mode=recovery_mode)


    return await _apply_restore(context, recovery_mode, staged)


async def _apply_restore(
    context: _RestoreFlowContext,
    recovery_mode: str,
    staged: Any,
) -> SnapshotTransactionRecord:
    service = context.service
    request = context.request
    transaction_id = context.transaction_id
    expected_lifecycle_generation = context.expected_lifecycle_generation
    _status = context.status
    _cancelled = context.cancelled
    session_dir = context.session_dir
    workspace_root = context.workspace_root
    states_root = context.states_root
    workflow_state = context.workflow_state
    analysis_state = context.analysis_state
    workflow = context.workflow
    workflow_owners = context.workflow_owners
    capsule_kind = context.capsule_kind

    # A terminal/idle capsule resumes nothing, so the kernel that
    # predates the restore must not survive as this session's runtime.
    # Tear it down HERE, before apply.
    #
    # stop(save=False) is mandatory here. The pre-restore capture is the
    # rollback authority and this workflow has already been superseded;
    # asking stop() to save would either publish stale bytes immediately
    # before apply or preserve a failed-save kernel that restore is
    # explicitly replacing.
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
    try:
        for workflow_owner in workflow_owners:
            await service._retire_superseded_workflow(workflow_owner)
    except SnapshotTransactionError as exc:
        # No live bytes have been replaced yet.  A checkpoint owner
        # that could not be closed must remain registered so normal
        # workflow teardown or a retry can still reach it; applying in
        # this state would let its stale WAL shadow the restored DB.
        try:
            await service.restore_coordinator.cleanup(
                transaction_id, service.paths,
            )
        except Exception:
            logger.exception(
                "[%s] Failed to clean staged restore after checkpoint "
                "retirement refusal",
                service.session_id,
            )
        return _status(
            "failed",
            recovery_mode=recovery_mode,
            failure={"code": exc.code, "message": str(exc)},
        )

    if capsule_kind in ("terminal", "idle"):
        if workflow is not None and getattr(workflow, "sandbox", None):
            stopped = False
            try:
                stop = getattr(workflow.sandbox, "stop", None)
                if stop is not None:
                    maybe = stop(save=False)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                    stopped = True
            except Exception as exc:
                logger.warning(
                    "[%s] terminal/idle kernel teardown failed: %s",
                    service.session_id, exc,
                )
            if stopped:
                workflow.sandbox = None

    try:
        from backend.services.session_lifecycle import lifecycle_file_lock

        # Snapshot apply mutates the same live runtime/notebook/state
        # surfaces as regular persistence. The coordinator contains no
        # suspension points, so hold the cross-process lifecycle lock
        # for its complete journaled mutation and revalidate before it
        # publishes any bytes.
        with lifecycle_file_lock(service.session_id):
            assert_session_generation_active(
                service.session_id,
                expected_generation=expected_lifecycle_generation,
            )
            applied_result = await service.restore_coordinator.apply(
                staged,
                service.paths,
                session_dir=session_dir,
                workspace_root=(
                    workspace_root
                    or str(Path(session_dir) / "workspace")
                ),
                states_root=states_root,
                runtime_choice=request.runtime_choice,
            )
            assert_session_generation_active(
                service.session_id,
                expected_generation=expected_lifecycle_generation,
            )
        applied = True
    except Exception as exc:
        # After APPLYING, cancellation/failure → recovery path (not pretend clean).
        return _status(
            "failed",
            recovery_mode=recovery_mode,
            failure=service._partial_apply_failure(
                "APPLY_FAILED", str(exc),
            ),
        )

    if _cancelled():
        # Cancel after APPLYING → recovery, not cancelled-clean.
        return _status(
            "failed",
            recovery_mode=recovery_mode,
            failure=service._partial_apply_failure(
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
                failure=service._partial_apply_failure(
                    "CHECKPOINT_INVALID",
                    "Applied checkpoint missing or empty",
                ),
            )


    applied = _AppliedRestore(
        recovery_mode=recovery_mode,
        staged=staged,
        applied_result=applied_result,
    )
    return await _activate_verify_commit(context, applied)


async def _activate_verify_commit(
    context: _RestoreFlowContext,
    applied: _AppliedRestore,
) -> SnapshotTransactionRecord:
    service = context.service
    request = context.request
    transaction_id = context.transaction_id
    expected_lifecycle_generation = context.expected_lifecycle_generation
    _status = context.status
    _cancelled = context.cancelled
    session_dir = context.session_dir
    workspace_root = context.workspace_root
    states_root = context.states_root
    workflow_state = context.workflow_state
    analysis_state = context.analysis_state
    workflow = context.workflow
    workflow_owners = context.workflow_owners
    capsule_kind = context.capsule_kind

    recovery_mode = applied.recovery_mode
    staged = applied.staged
    applied_result = applied.applied_result
    # Load applied analysis state into memory + activate dispatcher.
    target_manifest = service.manifest_store.load(request.snapshot_id)
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
                failure=service._partial_apply_failure(
                    "STATE_LOAD_FAILED", str(exc),
                ),
            )

    if restored_state is None:
        return _status(
            "failed",
            recovery_mode=recovery_mode,
            failure=service._partial_apply_failure(
                "MISSING_STATE", "No analysis state after apply",
            ),
        )

    if expected_lifecycle_generation is not None:
        restored_state.lifecycle_generation = (
            expected_lifecycle_generation
        )

    from backend.utils.secret_redaction import preserve_runtime_secrets
    preserve_runtime_secrets(analysis_state, restored_state)

    # Exact continuation only — no fresh-run fallback.
    try:
        staged_ckpt = service.continuation_dispatcher.stage_checkpoint(
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
            failure=service._partial_apply_failure(exc.code, str(exc)),
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
    try:
        service.continuation_dispatcher.activate(
            restored_state,
            staged_ckpt,
            workspace_path=Path(workspace_root or session_dir),
        )
    except Exception as exc:
        return _status(
            "failed",
            recovery_mode=recovery_mode,
            failure=service._partial_apply_failure(
                "ACTIVATION_FAILED", str(exc),
            ),
        )

    if getattr(applied_result, "runtime_restore_status", None):
        restored_state.runtime_restore_status = applied_result.runtime_restore_status
    if getattr(applied_result, "runtime_warning", None):
        restored_state.runtime_restore_warning = applied_result.runtime_warning

    # Persist state and branch. Journal remains APPLYING until live verify.
    try:
        assert_session_generation_active(
            service.session_id,
            expected_generation=expected_lifecycle_generation,
        )
        await service._persist_restored_state(restored_state)
    except Exception as exc:
        failure = {
            "code": "PERSIST_FAILED",
            "message": str(exc),
            "recovery_required": True,
        }
        service._block_session_after_partial_apply(failure)
        return _status(
            "failed", recovery_mode=recovery_mode, failure=failure,
        )

    # Journal is still APPLYING here. Any failure remains recoverable.
    try:
        verification_proof = await service._verify_live_restore(
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
        service._block_session_after_partial_apply(failure)
        return _status(
            "failed", recovery_mode=recovery_mode, failure=failure,
        )

    # Only after live verification: HMAC-signed COMMITTED with proof.
    try:
        from backend.services.session_lifecycle import lifecycle_file_lock

        with lifecycle_file_lock(service.session_id):
            assert_session_generation_active(
                service.session_id,
                expected_generation=expected_lifecycle_generation,
            )
            await service.restore_coordinator.commit(
                transaction_id,
                service.paths,
                verification_proof=verification_proof,
            )
            assert_session_generation_active(
                service.session_id,
                expected_generation=expected_lifecycle_generation,
            )
    except Exception as exc:
        failure = {
            "code": "COMMIT_FAILED",
            "message": str(exc),
            "recovery_required": True,
        }
        service._block_session_after_partial_apply(failure)
        return _status(
            "failed", recovery_mode=recovery_mode, failure=failure,
        )

    # Cleanup staging (journal may remain for COMMITTED recovery cleanup).
    try:
        await service.restore_coordinator.cleanup(transaction_id, service.paths)
    except Exception as exc:
        logger.warning(
            "[%s] restore cleanup failed for %s: %s",
            service.session_id, transaction_id, exc,
        )

    return _status("committed", recovery_mode=recovery_mode)
