"""Lifecycle routes: run, pause, resume, stop, retry-node, failed-node, resume-bioinformatics.

Extracted from backend/main.py to decouple lifecycle route handlers.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.config import get_session_states
from backend.models.api_requests import RetryNodeRequest, StopAnalysisRequest
from backend.models.schemas import (
    AnalysisPhase, APIResponse, TaskStatus,
)
from backend.services.session_store import async_save_session, ensure_session_loaded
from backend.state import (
    _run_tasks,
    state_lock as _state_lock,
    system_config,
    websockets,
    workflows,
)
from backend.services.session_store import update_session_cache as _update_session_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["lifecycle"])


@router.post("/{session_id}/run")
async def run_analysis(session_id: str):
    """Start running analysis (use WebSocket for real-time updates)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis already running")

    state.status = TaskStatus.RUNNING
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message="Analysis started. Connect to WebSocket for updates.",
        data={"websocket_url": f"/ws/{session_id}"}
    )


@router.post("/{session_id}/pause")
async def pause_analysis(session_id: str):
    """Pause running analysis"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status != TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis not running")

    state.status = TaskStatus.PAUSED
    await async_save_session(session_id, state)

    # Grab workflow reference under lock to avoid race with cleanup_stale_sessions
    async with _state_lock:
        workflow = workflows.get(session_id)
    if workflow:
        await workflow.pause(session_id)

    return APIResponse(
        success=True,
        message="Analysis paused"
    )


@router.post("/{session_id}/resume")
async def resume_analysis(session_id: str):
    """Resume paused or completed analysis (for re-analysis)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status == TaskStatus.COMPLETED:
        # Re-analysis: reset for a new round while keeping existing data
        state.status = TaskStatus.RUNNING
        # Increment max_rounds to allow additional analysis
        state.max_rounds = state.current_round + 1
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        return APIResponse(
            success=True,
            message="Re-analysis started",
            data={"needs_reconnect": True}
        )
    elif state.status == TaskStatus.WAITING_INPUT:
        # Guard: if in innovation phase after round 1, ensure direction is set
        if (state.analysis_phase == AnalysisPhase.INNOVATION
            and state.current_round >= 1
            and not state.user_chosen_direction
            and not state.directions_queue
            and not state.pending_human_input
            and not state.user_chosen_directions):
            # No direction chosen AND user never selected directions — default to PI_DECIDE
            # so analysis doesn't run directionless.
            # If user_chosen_directions is non-empty, all directions have been completed;
            # just let _should_continue() stop naturally at max_rounds.
            state.user_chosen_direction = "[PI_DECIDE]"
            state.pending_human_input = "[PI_DECIDE]"
            logger.info(f"[{session_id}] Resume without direction: defaulting to PI_DECIDE")
            # Extend max_rounds since user explicitly requested continuation
            effective_max = state.max_rounds if state.max_rounds > 0 else 2
            if state.current_round >= effective_max:
                state.max_rounds = state.current_round + 1
                logger.info(f"[{session_id}] Extended max_rounds to {state.max_rounds} for PI_DECIDE resume")
        # Continue from round-end wait (user sent input or skipped)
        # Signal the workflow to continue from _wait_for_user_input
        # Grab workflow reference under lock to avoid race with cleanup_stale_sessions
        async with _state_lock:
            workflow = workflows.get(session_id)
        if workflow:
            state.status = TaskStatus.RUNNING
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
            workflow.signal_continue()
            return APIResponse(
                success=True,
                message="Continuing analysis"
            )
        else:
            # No workflow in memory (e.g. after server restart).
            # Keep status as WAITING_INPUT so the WebSocket endpoint's resume
            # logic detects the correct state when the client reconnects.
            await async_save_session(session_id, state)
            return APIResponse(
                success=True,
                message="Continuing analysis",
                data={"needs_reconnect": True}
            )
    elif state.status == TaskStatus.WAITING_ANNOTATION:
        # Continue from annotation confirmation wait
        state.status = TaskStatus.RUNNING
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        async with _state_lock:
            workflow = workflows.get(session_id)
        if workflow:
            workflow.signal_annotation_confirmed()
            return APIResponse(
                success=True,
                message="Annotation confirmed, continuing analysis"
            )
        else:
            return APIResponse(
                success=True,
                message="Annotation confirmed, continuing analysis",
                data={"needs_reconnect": True}
            )
    elif state.status == TaskStatus.PAUSED:
        state.status = TaskStatus.RUNNING
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        return APIResponse(
            success=True,
            message="Analysis resumed",
            data={"needs_reconnect": True}
        )
    elif state.status in (TaskStatus.GENERATING_REPORT, TaskStatus.FAILED):
        # Interrupted during report generation or failed — allow re-run
        state.status = TaskStatus.RUNNING
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        return APIResponse(
            success=True,
            message="Resuming analysis",
            data={"needs_reconnect": True}
        )
    else:
        raise HTTPException(status_code=400, detail="Analysis not paused, waiting, or completed")


@router.post("/{session_id}/stop")
async def stop_analysis(session_id: str, req: StopAnalysisRequest):
    """Stop a running analysis - either immediately or with summary report"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status not in (TaskStatus.RUNNING, TaskStatus.WAITING_INPUT,
                             TaskStatus.WAITING_ANNOTATION, TaskStatus.GENERATING_REPORT):
        if state.status == TaskStatus.PAUSED:
            return APIResponse(success=True, message="Analysis is already paused")
        return APIResponse(success=True, message=f"Analysis is {state.status.value}")

    # Handle GENERATING_REPORT state
    if state.status == TaskStatus.GENERATING_REPORT:
        if req.action == "immediate":
            state.status = TaskStatus.PAUSED
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
            task = _run_tasks.get(session_id)
            if task and not task.done():
                task.cancel()
            return APIResponse(success=True, message="Report generation cancelled")
        else:  # summarize
            return APIResponse(success=True, message="Report generation already in progress")

    if req.action == "immediate":
        state.status = TaskStatus.PAUSED
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        task = _run_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        # Notify via WebSocket
        ws = websockets.get(session_id)
        if ws:
            try:
                await ws.send_json({
                    "type": "status",
                    "agent": None,
                    "content": "Analysis terminated. State saved.",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception:
                logger.warning("Failed to send termination status via WebSocket", exc_info=True)
        return APIResponse(success=True, message="Analysis terminated")

    elif req.action == "summarize":
        if state.status == TaskStatus.WAITING_INPUT:
            # Workflow is blocked at _wait_for_user_input
            # Set max_rounds = current_round so _should_continue routes to "stop" -> report gen
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
            async with _state_lock:
                workflow = workflows.get(session_id)
            if workflow:
                workflow.signal_continue()
            return APIResponse(success=True, message="Generating report...")
        elif state.status == TaskStatus.WAITING_ANNOTATION:
            # Workflow is blocked at _annotation_event.wait() inside callback
            # Unblock it and let the round finish, then route to report gen
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
            async with _state_lock:
                workflow = workflows.get(session_id)
            if workflow:
                workflow.signal_annotation_confirmed()
            return APIResponse(success=True, message="Generating report...")
        else:
            # RUNNING - mid-agent execution: cancel task, start report generation
            state.status = TaskStatus.GENERATING_REPORT
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
            task = _run_tasks.get(session_id)
            if task and not task.done():
                task.cancel()
            return APIResponse(success=True, message="Generating report...")


@router.post("/{session_id}/retry-node")
async def retry_node(session_id: str, request: RetryNodeRequest):
    """Retry or skip a failed node"""
    node_name = request.node_name
    skip = request.skip

    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status != TaskStatus.FAILED:
        raise HTTPException(status_code=400, detail="Session is not in failed state")

    # Note: The actual retry/skip logic is handled by the workflow's with_retry decorator
    # This endpoint just updates the state to allow resumption

    if skip:
        # Store skip_nodes in sandbox_state (used by workflow on resume)
        if state.sandbox_state is None:
            state.sandbox_state = {}
        if "skip_nodes" not in state.sandbox_state:
            state.sandbox_state["skip_nodes"] = []
        if node_name not in state.sandbox_state["skip_nodes"]:
            state.sandbox_state["skip_nodes"].append(node_name)
        message = f"Node {node_name} will be skipped on resume"
    else:
        # Clear failed status for retry
        state.status = TaskStatus.PAUSED  # Allow resumption
        message = f"Session ready to retry from node {node_name}"

    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=message,
        data={"node": node_name, "action": "skip" if skip else "retry"}
    )


@router.get("/{session_id}/failed-node")
async def get_failed_node(session_id: str):
    """Get information about the failed node"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if we have workflow state with error info
    if session_id in workflows:
        workflow = workflows[session_id]
        config = {"configurable": {"thread_id": session_id}}
        try:
            graph_state = await workflow.compiled_graph.aget_state(config)
            if graph_state and graph_state.values:
                return {
                    "failed_node": graph_state.values.get("failed_node"),
                    "node_errors": graph_state.values.get("node_errors", {}),
                    "retry_counts": graph_state.values.get("node_retry_count", {})
                }
        except Exception as e:
            logger.exception("Error getting workflow state")

    # Fallback: check sandbox_state
    if state.sandbox_state:
        return {
            "failed_node": state.sandbox_state.get("failed_node"),
            "node_errors": state.sandbox_state.get("node_errors", {}),
            "retry_counts": state.sandbox_state.get("node_retry_count", {})
        }

    return {"failed_node": None, "node_errors": {}, "retry_counts": {}}


@router.post("/{session_id}/resume-bioinformatics")
async def resume_bioinformatics(session_id: str):
    """Resume bioinformatics analysis from last checkpoint"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check for checkpoint
    checkpoint_path = get_session_states(session_id) / f"{session_id}_bioinfo_checkpoint.json"
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="No bioinformatics checkpoint found")

    # Read checkpoint info
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return APIResponse(
            success=True,
            message=f"Checkpoint found at step {checkpoint.get('step_number', 0)}. Connect to WebSocket to resume.",
            data={
                "has_checkpoint": True,
                "step_number": checkpoint.get("step_number", 0),
                "total_steps": checkpoint.get("total_steps", 80),
                "findings_count": len(checkpoint.get("findings", [])),
                "figures_count": len(checkpoint.get("figures", []))
            }
        )
    except Exception as e:
        logger.exception("Error reading checkpoint")
        raise HTTPException(status_code=500, detail=f"Error reading checkpoint: {str(e)}")
