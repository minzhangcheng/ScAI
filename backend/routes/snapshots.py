"""Snapshot routes: list, restore, delete.

Extracted from backend/main.py to decouple snapshot route handlers.
"""
import asyncio
import json
import logging
from typing import Dict

from fastapi import APIRouter, HTTPException, Request

from backend.models.schemas import (
    APIResponse, TaskStatus, WebSocketMessage, AgentType,
)
from backend.services.session_store import (
    async_save_session, ensure_session_loaded,
    update_session_cache as _update_session_cache,
)
from backend.state import (
    workflows,
    _run_tasks,
    state_lock as _state_lock,
    initializing_sessions as _initializing_sessions,
    websockets,
    get_session_lock,
)
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id
from backend.utils.snapshot import SnapshotManager
from backend.config import get_session_states

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["snapshots"])


@router.get("/{session_id}/snapshots")
async def list_snapshots(session_id: str, request: Request):
    """List all snapshots for a session."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return APIResponse(
        success=True,
        data={"snapshots": [s.model_dump(mode="json") for s in state.snapshots]},
        message="ok"
    )


@router.post("/{session_id}/snapshots/{snapshot_id}/restore")
async def restore_snapshot(session_id: str, snapshot_id: str, request: Request):
    """Restore analysis state from a snapshot.

    WARNING: This stops any running workflow and replaces the session state.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    snapshot = next((s for s in state.snapshots if s.id == snapshot_id), None)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    # Mark session as initializing + save workflow ref in one lock acquisition.
    # Observer WS connections will wait until this flag is cleared.
    async with _state_lock:
        _initializing_sessions.add(session_id)
        workflow = workflows.get(session_id)

    # Stop running workflow if any.
    # Do NOT signal_continue/annotation_confirmed — signaling would let the
    # graph run nodes, each calling state_callback which writes old state.
    if workflow:
        try:
            await workflow.close()
        except Exception:
            logger.exception(f"[{session_id}] Error closing workflow during restore")

    # Cancel running task. state_callback (runs inside run_coro) completes
    # before wait_for returns. CancelledError handler (runs in the task's
    # exception handler) is guarded by the per-session lock below.
    task = _run_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    _run_tasks.pop(session_id, None)

    # Cancel monitor task
    from backend.state import _monitor_tasks
    monitor = _monitor_tasks.pop(session_id, None)
    if monitor and not monitor.done():
        monitor.cancel()

    # Restore state from snapshot
    mgr = SnapshotManager(session_id)
    restored_state = await mgr.restore_snapshot(snapshot)

    await mgr.restore_notebook(snapshot)
    mgr.cleanup_workspace_files(snapshot.workspace_manifest)
    mgr.cleanup_reports()

    restored_state.exploration_report_html_path = None
    restored_state.detailed_report_html_path = None
    restored_state.annotation_report_html_path = None

    await mgr.delete_checkpoint()

    restored_state.snapshots = state.snapshots
    restored_state.last_restored_snapshot_id = snapshot.id

    # Phase-aware status setting
    phase = snapshot.phase
    if phase == "pre_direction":
        restored_state.status = TaskStatus.WAITING_INPUT
    elif phase == "pre_annotation":
        restored_state.status = TaskStatus.WAITING_ANNOTATION
    elif phase == "pre_round_wait":
        restored_state.status = TaskStatus.WAITING_INPUT
    else:
        restored_state.status = TaskStatus.PAUSED

    # Synthesize HITL messages into realtime_messages so frontend can show dialogs
    if phase == "pre_annotation" and restored_state.annotation_state:
        ann = restored_state.annotation_state
        report_html_url = (
            f"/api/session/{session_id}/annotation-report/html"
            if getattr(restored_state, 'annotation_report_html_path', None) else None
        )
        detailed_report_url = (
            f"/api/session/{session_id}/detailed-report/html"
            if getattr(restored_state, 'detailed_report_html_path', None) else None
        )
        restored_state.realtime_messages.append(WebSocketMessage(
            type="annotation_confirmation",
            agent=AgentType.LITERATURE,
            content={
                "annotations": [a.model_dump(mode="json") for a in ann.cluster_annotations],
                "tissue": ann.tissue,
                "organism": ann.organism,
                "tissue_summary": ann.tissue_summary,
                "analysis_summary": ann.analysis_summary,
                "report_html_url": report_html_url,
                "detailed_report_url": detailed_report_url,
            }
        ))
    elif phase == "pre_annotation" and not restored_state.annotation_state:
        logger.warning(
            f"[{session_id}] pre_annotation snapshot lacks annotation_state "
            f"(legacy snapshot). Falling back to PAUSED status."
        )
        restored_state.status = TaskStatus.PAUSED
    elif phase == "pre_direction" and restored_state.exploration_report:
        exp = restored_state.exploration_report
        html_url = (
            f"/api/session/{session_id}/exploration-report/html"
            if getattr(restored_state, 'exploration_report_html_path', None) else None
        )
        detailed_report_url = (
            f"/api/session/{session_id}/detailed-report/html"
            if getattr(restored_state, 'detailed_report_html_path', None) else None
        )
        last_round = restored_state.round_results[-1] if restored_state.round_results else None
        restored_state.realtime_messages.append(WebSocketMessage(
            type="exploration_report",
            agent=AgentType.PI,
            content={
                "report_a": exp.model_dump(mode="json"),
                "round": last_round.round_number if last_round else 1,
                "summary": last_round.pi_summary if last_round else "",
                "html_url": html_url,
                "detailed_report_url": detailed_report_url,
            }
        ))

    elif phase == "pre_round_wait":
        last_round = restored_state.round_results[-1] if restored_state.round_results else None
        if not last_round:
            logger.warning(
                f"[{session_id}] pre_round_wait snapshot has no round_results. "
                f"Falling back to PAUSED status."
            )
            restored_state.status = TaskStatus.PAUSED
        else:
            restored_state.realtime_messages.append(WebSocketMessage(
                type="round_summary",
                agent=AgentType.PI,
                content={
                    "round": last_round.round_number,
                    "summary": last_round.pi_summary or "",
                    "pathology_findings": last_round.pathology_findings or [],
                    "bioinformatics_findings": last_round.bioinformatics_findings or [],
                    "biochemistry_findings": last_round.biochemistry_findings or [],
                    "literature_findings": last_round.literature_findings or [],
                    "findings": [
                        *(last_round.pathology_findings or []),
                        *(last_round.bioinformatics_findings or []),
                        *(last_round.biochemistry_findings or []),
                        *(last_round.literature_findings or []),
                    ],
                    "data_requests": getattr(last_round, 'data_requests', []) or [],
                    "status": "waiting_input"
                }
            ))
            max_rounds = restored_state.max_rounds if restored_state.max_rounds > 0 else 4
            is_last_round = last_round.round_number >= max_rounds
            if is_last_round:
                status_msg = f"Round {last_round.round_number} complete (final round). You can provide additional data to extend analysis, or skip to generate the final report."
            else:
                status_msg = f"Round {last_round.round_number} complete. Waiting for your input."
            restored_state.realtime_messages.append(WebSocketMessage(
                type="status",
                agent=AgentType.PI,
                content=status_msg
            ))

    # Reset annotation confirmed flag so the restored HITL checkpoint works correctly
    if phase == "pre_annotation" and restored_state.annotation_state:
        restored_state.annotation_state.confirmed = False
        restored_state.annotation_state.pending = True

    # ---- Synchronization point ----
    # Pop workflows AND write cache AND write disk inside the per-session lock.
    # This serializes against state_callback and the CancelledError/Exception
    # handlers in websocket_endpoint, which acquire the same lock before writing.
    # Lock ordering: sess_lock → state_lock (never the reverse).
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        async with _state_lock:
            workflows.pop(session_id, None)

        await _update_session_cache(session_id, restored_state)
        await async_save_session(session_id, restored_state, force=True)

    logger.info(
        f"[{session_id}] Snapshot restored: {snapshot.name} (Round {snapshot.round_number}), "
        f"phase={phase}, status={restored_state.status.value}, "
        f"current_round={restored_state.current_round}"
    )

    # Clear the initializing flag FIRST so the reconnecting WS can fall
    # through to the HITL-hold path (websockets.py:215) with the restored
    # state. If we close the WS before clearing the flag, a fast reconnect
    # may arrive while initializing_sessions still holds the session, enter
    # the observer path (is_restoring=True), and simply return without ever
    # reaching HITL-hold.
    async with _state_lock:
        _initializing_sessions.discard(session_id)

    ws = websockets.get(session_id)
    if ws:
        try:
            await ws.close(code=4000, reason="snapshot_restore")
        except Exception:
            logger.debug(f"[{session_id}] WebSocket close during restore", exc_info=True)

    return APIResponse(
        success=True,
        message=f"State restored to snapshot: {snapshot.name} (Round {snapshot.round_number})"
    )


@router.get("/{session_id}/snapshots/{snapshot_id}/rollback-preview")
async def rollback_preview(session_id: str, snapshot_id: str, request: Request):
    """Preview what will be affected by restoring a snapshot (without actually restoring)."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    snapshot = next((s for s in state.snapshots if s.id == snapshot_id), None)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")

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
