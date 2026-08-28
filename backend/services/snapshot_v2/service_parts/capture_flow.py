"""Snapshot capture transaction flow owned outside the service class."""
from __future__ import annotations

from dataclasses import dataclass, field

from .shared import (  # noqa: F401
    asyncio,
    Path,
    Any,
    FORMAT_VERSION,
    ObjectPutResult,
    ObjectRef,
    RuntimeGenerationManifest,
    RuntimeManifest,
    SnapshotCaptureError,
    SnapshotStorageError,
    SnapshotManifestV2,
    SnapshotV2Error,
    WorkspaceEntry,
    empty_complete_runtime,
    runtime_manifest_from_generation,
    validate_system_commit,
    _now_iso,
    _null_async_cm,
    logger,
)


@dataclass
class _CaptureFlowState:
    existing_object_hashes_before_capture: set[str] = field(default_factory=set)
    object_baseline_captured: bool = False
    manifest_published: bool = False
    safe_point_released: bool = False


@dataclass
class _FrozenCaptureSources:
    analysis_state_result: Any
    notebook_result: Any
    checkpoint_result: Any
    runtime: RuntimeManifest
    runtime_snapshot: RuntimeGenerationManifest | None
    workspace_entries: list[WorkspaceEntry]
    frozen_runtime_revision: int | None
    current_physical: int
    allow_over_quota_baseline: bool

async def capture_impl(
    service,
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
    bindings=None,
) -> tuple[SnapshotManifestV2, bool]:
    flow = _CaptureFlowState()
    service._validate_state(request)

    if not skip_safe_point_lifecycle:
        await service._create_safe_point(
            workflow_state or {},
            analysis_state=analysis_state,
        )

    try:
        frozen = await _freeze_capture_sources(
            service,
            request,
            flow,
            sandbox=sandbox,
            workspace_root=workspace_root,
            workflow_state=workflow_state,
            analysis_state=analysis_state,
        )
        manifest = _build_capture_manifest(
            service,
            request,
            snapshot_id,
            flow,
            frozen,
        )
        return await _publish_capture_manifest(
            service,
            request,
            tx,
            manifest,
            flow,
            frozen,
            sandbox=sandbox,
            analysis_state=analysis_state,
            skip_safe_point_lifecycle=skip_safe_point_lifecycle,
            persist_session=persist_session,
        )
    except SnapshotV2Error:
        if flow.object_baseline_captured and not flow.manifest_published:
            service._cleanup_unpublished_capture_objects(
                flow.existing_object_hashes_before_capture,
            )
        raise
    except Exception as exc:
        if flow.object_baseline_captured and not flow.manifest_published:
            service._cleanup_unpublished_capture_objects(
                flow.existing_object_hashes_before_capture,
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
        if not skip_safe_point_lifecycle and not flow.safe_point_released:
            await asyncio.shield(service._release_safe_point())

async def _freeze_capture_sources(
    service,
    request,
    flow: _CaptureFlowState,
    *,
    sandbox,
    workspace_root,
    workflow_state,
    analysis_state,
) -> _FrozenCaptureSources:
    # Consistency barrier: await pending saves before freezing execution.
    if sandbox is not None and hasattr(sandbox, "await_pending_saves"):
        await sandbox.await_pending_saves()

    # Snapshot object hashes present before any put in this capture.
    # Used for quota accounting of refs that only expose ObjectRef
    # (workspace/runtime/continuation) — never use register_existing_ref
    # return value as a proxy for "already existed before capture".
    objects_dir = Path(str(service.paths.objects))
    if objects_dir.is_dir():
        existing_object_hashes_before_capture = {
            path.name for path in objects_dir.iterdir() if path.is_file()
        }
    else:
        existing_object_hashes_before_capture = set()
    flow.existing_object_hashes_before_capture = existing_object_hashes_before_capture
    flow.object_baseline_captured = True

    estimated_addition = await service._estimate_capture_addition(
        request, workspace_root, analysis_state, sandbox=sandbox,
    )
    current_physical = service._physical_object_store_bytes()
    phase = str(getattr(request.phase, "value", request.phase))
    allow_over_quota_baseline = phase == "pre_restore"
    if service.session_quota_bytes > 0:
        quota_ceiling = (
            current_physical + service.pre_restore_quota_grace_bytes
            if (
                allow_over_quota_baseline
                and current_physical > service.session_quota_bytes
            )
            else service.session_quota_bytes
        )
        if current_physical + estimated_addition > quota_ceiling:
            raise SnapshotStorageError(
                "snapshot_quota_exceeded",
                "Conservative pre-write estimate would use "
                f"{current_physical + estimated_addition} bytes, "
                f"exceeding capture ceiling {quota_ceiling} bytes "
                f"(session quota {service.session_quota_bytes})",
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
        analysis_state_result = await service._capture_analysis_state(
            request, analysis_state,
        )
        notebook_result = await service._capture_notebook(request)
        checkpoint_result = await service._capture_checkpoint(request)

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
            runtime = await service._capture_runtime(
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
            descriptor = service.safe_point.get_eligibility_descriptor(
                service.session_id,
            )
            if not service._resolve_empty_runtime_eligibility(
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
            workspace_entries, _ = await service._capture_workspace(
                str(workspace_root),
            )

    # Mutable sources frozen — exec lock released.


    return _FrozenCaptureSources(
        analysis_state_result=analysis_state_result,
        notebook_result=notebook_result,
        checkpoint_result=checkpoint_result,
        runtime=runtime,
        runtime_snapshot=runtime_snapshot,
        workspace_entries=workspace_entries,
        frozen_runtime_revision=frozen_runtime_revision,
        current_physical=current_physical,
        allow_over_quota_baseline=allow_over_quota_baseline,
    )


def _build_capture_manifest(
    service,
    request,
    snapshot_id: str,
    flow: _CaptureFlowState,
    frozen: _FrozenCaptureSources,
) -> SnapshotManifestV2:
    analysis_state_result = frozen.analysis_state_result
    notebook_result = frozen.notebook_result
    checkpoint_result = frozen.checkpoint_result
    runtime = frozen.runtime
    runtime_snapshot = frozen.runtime_snapshot
    workspace_entries = frozen.workspace_entries
    existing_object_hashes_before_capture = (
        flow.existing_object_hashes_before_capture
    )
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
            service.object_store._get_evidence(ref.sha256) is None
            and not service.object_store.register_existing_ref(ref)
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
            if service.object_store._get_evidence(ref.sha256) is not None:
                _record_transaction_evidence(ref)
            elif service.object_store.register_existing_ref(ref):
                _record_transaction_evidence(ref)
    for ref in runtime.adata_refs.values():
        if ref.sha256 not in transaction_evidence:
            if service.object_store._get_evidence(ref.sha256) is not None:
                _record_transaction_evidence(ref)
            elif service.object_store.register_existing_ref(ref):
                _record_transaction_evidence(ref)
    if runtime.residual_ref is not None:
        rref = runtime.residual_ref
        if rref.sha256 not in transaction_evidence:
            if service.object_store._get_evidence(rref.sha256) is not None:
                _record_transaction_evidence(rref)
            elif service.object_store.register_existing_ref(rref):
                _record_transaction_evidence(rref)
    for attr in ("workflow_state_ref", "payload_ref"):
        cref = getattr(continuation, attr, None)
        if cref is not None and cref.sha256 not in transaction_evidence:
            if service.object_store._get_evidence(cref.sha256) is not None:
                _record_transaction_evidence(cref)
            elif service.object_store.register_existing_ref(cref):
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
        service.object_store.verify(eref)
        _record_transaction_evidence(eref)

    # Every other manifest ref must already have transaction evidence.
    for sha, ref in all_objects.items():
        if sha not in transaction_evidence:
            raise SnapshotCaptureError(
                "MISSING_TRANSACTION_EVIDENCE",
                f"Object {sha} lacks put/verify evidence in this capture",
            )


    return manifest


async def _publish_capture_manifest(
    service,
    request,
    tx: str,
    manifest: SnapshotManifestV2,
    flow: _CaptureFlowState,
    frozen: _FrozenCaptureSources,
    *,
    sandbox,
    analysis_state,
    skip_safe_point_lifecycle: bool,
    persist_session: bool,
) -> tuple[SnapshotManifestV2, bool]:
    current_physical = frozen.current_physical
    allow_over_quota_baseline = frozen.allow_over_quota_baseline
    frozen_runtime_revision = frozen.frozen_runtime_revision
    # Step 7 — compute root hash
    await service._compute_root_hash(manifest)

    # Step 8 — check quota (uses physical_added + free space)
    await service._check_quota(
        manifest,
        quota_baseline_bytes=current_physical,
        allow_over_quota_baseline=allow_over_quota_baseline,
    )

    # Step 9 — publish manifest
    staging_path = service.manifest_store.write_staging(manifest, prefix=f"{tx}_")
    published = service.manifest_store.publish_staging(staging_path)
    flow.manifest_published = True

    # Re-read verification
    reloaded = service.manifest_store.load(published.snapshot_id)
    if reloaded.root_hash != published.root_hash:
        raise SnapshotCaptureError(
            "MANIFEST_REREAD_MISMATCH",
            "Published manifest failed re-read verification",
        )

    # Mark runtime committed after successful publish
    if sandbox is not None:
        service._mark_runtime_committed(
            sandbox, runtime_revision=frozen_runtime_revision,
        )
        # Persist saved_runtime_revision onto analysis state
        try:
            state = analysis_state or await service._get_session_state(
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
                service.session_id, exc,
            )


    # Step 10 — save session state after publish when requested.
    # Manifest publication is the capture commit point. If session
    # persistence fails after publish, return the committed manifest
    # with session_reconciliation_pending=True — never raise
    # SnapshotCaptureError (retry with a new key could duplicate).
    session_reconciliation_pending = False
    if persist_session:
        try:
            await service._save_session_state(request.session_id)
        except SnapshotV2Error as exc:
            logger.error(
                "[%s] post-publish session save raised SnapshotV2Error "
                "for %s: %s — marking session_reconciliation_pending",
                service.session_id, reloaded.snapshot_id, exc,
            )
            session_reconciliation_pending = True
        except Exception as exc:
            logger.error(
                "[%s] post-publish session save failed for %s: %s — "
                "manifest already committed; session_reconciliation_pending",
                service.session_id, reloaded.snapshot_id, exc,
            )
            session_reconciliation_pending = True

    # Step 11 — release safe point (only on full success; skip when
    # restore already owns the cooperative safe point).
    if not skip_safe_point_lifecycle:
        await service._release_safe_point()
        flow.safe_point_released = True

    # Step 12 — return committed manifest + pending flag (no private attr)
    return reloaded, session_reconciliation_pending
