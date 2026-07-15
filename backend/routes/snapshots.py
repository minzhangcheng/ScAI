"""Snapshot routes: list, restore, delete.

Extracted from backend/main.py to decouple snapshot route handlers.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models.schemas import APIResponse
from backend.models.api_requests import SnapshotTransactionResponse
from backend.services.session_store import (
    async_save_session,
    update_session_cache as _update_session_cache,
)
from backend.services.snapshot_v2.models import SnapshotRestoreRequest as SnapshotV2RestoreRequest
from backend.services.snapshot_v2.jobs import RestoreJobManager
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id
from backend.utils.snapshot import SnapshotManager
from backend.utils.runtime_apply import (
    compute_runtime_status, RuntimeChoiceRequired,
)
from backend.config import get_session_states

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["snapshots"])


from backend.services.snapshot_v2.errors import (
    SnapshotCompatibilityError,
    SnapshotDeleteConflict,
    SnapshotIntegrityError,
    SnapshotStorageError,
    SnapshotTransactionError,
    SnapshotV2Error,
    snapshot_error_detail,
)


def _v2_http_exc(exc) -> HTTPException:
    """Build an HTTPException with the canonical status + structured detail."""
    d = snapshot_error_detail(exc)
    return HTTPException(
        status_code=d["status_code"],
        detail={"code": d["code"], "message": d["message"]},
    )


def _require_v1_snapshot(snapshot) -> None:
    """Reject V2 compatibility projections at legacy management routes."""
    from backend.services.snapshot_v2.legacy import normalize_snapshot_format_version

    format_version = normalize_snapshot_format_version(
        getattr(snapshot, "format_version", None)
    )
    if format_version is not None and format_version >= 2:
        raise HTTPException(
            status_code=409,
            detail="Snapshot V2 must be managed through the V2 snapshot endpoints",
        )
    if not str(getattr(snapshot, "state_file", "") or "").strip():
        raise HTTPException(
            status_code=409,
            detail="Snapshot is not a valid V1 file-backed snapshot",
        )


@router.get("/{session_id}/snapshots")
async def list_snapshots(session_id: str, request: Request):
    """List V1 inventory snapshots (metadata only; not restorable via V1 restore)."""
    from backend.services.snapshot_v2.legacy import (
        build_legacy_inventory,
        normalize_snapshot_format_version,
    )
    from backend.constants import VERSION

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    inventory = build_legacy_inventory(state, warning_version=VERSION)
    # Retain full session snapshot list for delete/export; mark restorable=false.
    snapshots = []
    for s in state.snapshots:
        payload = s.model_dump(mode="json")
        fv = normalize_snapshot_format_version(s.format_version)
        if fv is None or fv < 2:
            payload["restorable"] = False
            payload["unsupported"] = True
        else:
            payload["restorable"] = False  # V1 list is non-authoritative; use V2
            payload["unsupported"] = False
        snapshots.append(payload)
    return APIResponse(
        success=True,
        data={
            "snapshots": snapshots,
            "legacy_inventory": inventory.model_dump(mode="json"),
            "restorable": False,
        },
        message="ok",
    )


@router.post("/{session_id}/snapshots/{snapshot_id}/restore")
async def restore_snapshot(session_id: str, snapshot_id: str, request: Request):
    """V1 restore endpoint — permanently cut over to HTTP 410.

    V1 snapshots remain listable/exportable/deletable. Restorable lifecycle
    is exclusively Snapshot V2 (``restore-v2``). This handler never tears down
    workflows or mutates session state.
    """
    from backend.services.snapshot_v2.legacy import v1_restore_gone_detail

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    # snapshot_id retained for route shape / client logging only.
    _ = snapshot_id
    raise HTTPException(status_code=410, detail=v1_restore_gone_detail())


@router.get("/{session_id}/snapshots/{snapshot_id}/rollback-preview")
async def rollback_preview(session_id: str, snapshot_id: str, request: Request):
    """Preview what will be affected by restoring a snapshot (without actually restoring)."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    snapshot = next((s for s in state.snapshots if s.id == snapshot_id), None)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    _require_v1_snapshot(snapshot)

    runtime_status = compute_runtime_status(snapshot, session_id)

    mgr = SnapshotManager(session_id)
    current_manifest = await asyncio.to_thread(mgr.scan_workspace_manifest)
    snapshot_manifest = snapshot.workspace_manifest or {}

    files_to_delete = {}
    for subdir in ("figures", "tables", "reproducibility", "root"):
        current_set = set(current_manifest.get(subdir, []))
        keep_set = set(snapshot_manifest.get(subdir, []))
        to_delete = current_set - keep_set
        files_to_delete[subdir] = sorted(to_delete)

    # Count notebook cells (current vs snapshot)
    current_cells = 0
    nb_path = get_session_states(session_id) / f"{session_id}.ipynb"
    if nb_path.exists():
        try:
            current_cells = len(json.loads(nb_path.read_text()).get("cells", []))
        except Exception:
            pass

    snapshot_cells = 0
    if snapshot.notebook_file:
        nb_backup = mgr.snapshots_dir / snapshot.notebook_file
        if nb_backup.exists():
            try:
                snapshot_cells = len(json.loads(nb_backup.read_text()).get("cells", []))
            except Exception:
                pass

    # Find descendant snapshots (snapshots on branches after this point)
    descendant_ids = SnapshotManager.find_descendant_ids(state.snapshots, snapshot_id)

    descendant_names = [
        s.name for s in state.snapshots if s.id in descendant_ids
    ]

    return APIResponse(
        success=True,
        data={
            "files_to_delete": files_to_delete,
            "files_to_delete_count": sum(len(v) for v in files_to_delete.values()),
            "notebook_cells_removed": max(0, current_cells - snapshot_cells),
            "current_notebook_cells": current_cells,
            "snapshot_notebook_cells": snapshot_cells,
            "descendant_snapshots": descendant_names,
            "runtime_status": runtime_status,
        },
        message="ok"
    )


@router.delete("/{session_id}/snapshots/{snapshot_id}")
async def delete_snapshot(session_id: str, snapshot_id: str, request: Request):
    """Delete a snapshot."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    snapshot = next((s for s in state.snapshots if s.id == snapshot_id), None)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    _require_v1_snapshot(snapshot)

    mgr = SnapshotManager(session_id)
    await mgr.delete_snapshot(snapshot)
    state.snapshots = [s for s in state.snapshots if s.id != snapshot_id]
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    return APIResponse(success=True, message="Snapshot deleted")


@router.delete("/{session_id}/snapshots/{snapshot_id}/branch")
async def delete_branch(session_id: str, snapshot_id: str, request: Request):
    """Delete a snapshot and all its descendants (entire branch)."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    snapshot = next((s for s in state.snapshots if s.id == snapshot_id), None)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    descendant_ids = SnapshotManager.find_descendant_ids(
        state.snapshots, snapshot_id, include_root=True,
    )
    for candidate in state.snapshots:
        if candidate.id in descendant_ids:
            _require_v1_snapshot(candidate)

    mgr = SnapshotManager(session_id)
    deleted_ids = await mgr.delete_branch(snapshot, state.snapshots)
    state.snapshots = [s for s in state.snapshots if s.id not in deleted_ids]
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=f"Deleted {len(deleted_ids)} snapshots in branch",
        data={"deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}
    )


# ── Snapshot V2 Listing & Status ──────────────────────────────────────
# All V2 mutations go through SnapshotService under the session lock.
# Routes only: ownership checks + HTTP error translation.


@router.get("/{session_id}/snapshots-v2")
async def list_snapshots_v2(session_id: str, request: Request):
    """List V2 snapshots with store-level status and integrity diagnostics."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    paths = SnapshotV2Paths.for_session(session_id)
    from backend.services.snapshot_v2.service import SnapshotService

    service = SnapshotService(session_id, paths)
    try:
        snapshots = service.list_snapshots()
        store_status = service.store_status()
    except SnapshotV2Error as exc:
        raise _v2_http_exc(exc) from exc

    return APIResponse(
        success=True,
        data={
            "snapshots": snapshots,
            "store_status": store_status.model_dump(mode="json"),
        },
        message="ok",
    )


@router.get("/{session_id}/snapshots-v2/{snapshot_id}")
async def get_snapshot_v2(session_id: str, snapshot_id: str, request: Request):
    """Get a single V2 snapshot manifest via SnapshotService."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    paths = SnapshotV2Paths.for_session(session_id)
    from backend.services.snapshot_v2.service import SnapshotService

    service = SnapshotService(session_id, paths)
    try:
        data = service.preview_snapshot(snapshot_id)
    except SnapshotV2Error as exc:
        raise _v2_http_exc(exc) from exc

    return APIResponse(
        success=True,
        data=data,
        message="ok",
    )


@router.delete("/{session_id}/snapshots-v2/{snapshot_id}")
async def delete_snapshot_v2(session_id: str, snapshot_id: str, request: Request, branch: bool = False):
    """Delete a V2 snapshot via SnapshotService (leaf or branch).

    Leaf deletion (default): fails with 409 if the snapshot has children
    or is protected.

    Branch deletion (``?branch=true``): removes the snapshot and all its
    descendants (computed inside the service under the session lock).
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    paths = SnapshotV2Paths.for_session(session_id)
    from backend.services.snapshot_v2.service import SnapshotService

    service = SnapshotService(session_id, paths)
    try:
        if branch:
            result = await service.delete_branch(snapshot_id)
        else:
            result = await service.delete_leaf(snapshot_id)
    except SnapshotV2Error as exc:
        raise _v2_http_exc(exc) from exc

    return APIResponse(
        success=True,
        data=result.model_dump(mode="json"),
        message=f"Deleted {len(result.deleted_ids)} snapshot(s)",
    )


# ── Snapshot V2 Restore Jobs ──────────────────────────────────────────


def _transaction_urls(request: Request, session_id: str, transaction_id: str) -> tuple[str, str]:
    """Build absolute status/cancel URLs for a restore transaction."""
    base = str(request.base_url).rstrip("/")
    status_url = f"{base}/api/session/{session_id}/snapshot-transactions/{transaction_id}"
    cancel_url = f"{status_url}/cancel"
    return status_url, cancel_url


@router.post("/{session_id}/snapshots/{snapshot_id}/restore-v2", status_code=202)
async def restore_snapshot_v2(session_id: str, snapshot_id: str, request: Request):
    """Initiate a background V2 restore job.

    Runs a preflight check synchronously, then enqueues a background
    ``RestoreJobManager`` task that progresses through the restore
    lifecycle.  Returns HTTP 202 with the transaction details so the
    client can poll/cancel the job.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    # Parse the V2 restore request body
    try:
        body = await request.json()
        v2_req = SnapshotV2RestoreRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid restore request: {exc}")

    if v2_req.snapshot_id != snapshot_id:
        raise HTTPException(
            status_code=422,
            detail=f"snapshot_id mismatch: path '{snapshot_id}' != body '{v2_req.snapshot_id}'",
        )

    paths = SnapshotV2Paths.for_session(session_id)
    from backend.services.snapshot_v2.preflight import RestorePreflight

    try:
        preflight = RestorePreflight(paths).run(snapshot_id, v2_req)
    except RuntimeChoiceRequired as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Snapshot runtime is '{exc.status}'. Provide runtime_choice='reset' "
                f"or 'keep'."
            ),
        ) from exc
    except SnapshotV2Error as exc:
        raise _v2_http_exc(exc) from exc

    if preflight.status != "ready":
        if preflight.status == "incompatible":
            raise HTTPException(
                status_code=410,
                detail=(
                    f"Legacy snapshot cannot be restored: "
                    f"{preflight.compatibility_errors or preflight.status}"
                ),
            )
        raise HTTPException(
            status_code=422,
            detail=f"Preflight failed: {preflight.compatibility_errors or preflight.status}",
        )

    try:
        mgr = RestoreJobManager(paths=paths, session_id=session_id)
        record = await mgr.enqueue(session_id, preflight, v2_req)
    except SnapshotV2Error as exc:
        raise _v2_http_exc(exc) from exc

    status_url, cancel_url = _transaction_urls(request, session_id, record.transaction_id)
    return SnapshotTransactionResponse(
        transaction_id=record.transaction_id,
        status=record.status,
        status_url=status_url,
        cancel_url=cancel_url,
    )


@router.get("/{session_id}/snapshot-transactions/{transaction_id}")
async def get_restore_transaction_status(
    session_id: str, transaction_id: str, request: Request,
):
    """Return current status of a V2 restore transaction (shared registry + journal)."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    from backend.services.snapshot_v2.jobs import load_transaction

    paths = SnapshotV2Paths.for_session(session_id)
    record = load_transaction(transaction_id, paths=paths)
    if record is None or record.session_id != session_id:
        raise HTTPException(status_code=404, detail="Restore transaction not found")

    return APIResponse(
        success=True,
        data=record.model_dump(mode="json"),
        message="ok",
    )


@router.post("/{session_id}/snapshot-transactions/{transaction_id}/cancel")
async def cancel_restore_v2(session_id: str, transaction_id: str, request: Request):
    """Cancel a running V2 restore job via the shared job registry."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    from backend.services.snapshot_v2.jobs import cancel_restore_job

    try:
        await cancel_restore_job(transaction_id, session_id=session_id)
    except SnapshotTransactionError as exc:
        # Cancel keeps leak-oracle-safe 404 for missing tx; other cancel
        # failures stay 400 (not the generic transaction 503 mapping).
        status = 404 if exc.code == "TRANSACTION_NOT_FOUND" else 400
        raise HTTPException(
            status_code=status,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    return APIResponse(success=True, message="Restore cancelled")
