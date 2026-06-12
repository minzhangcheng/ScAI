"""Lifecycle routes: run, pause, resume, stop, retry-node, failed-node, resume-bioinformatics.

Extracted from backend/main.py to decouple lifecycle route handlers.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from backend.config import get_session_states
from backend.constants import WaitInputReason
from backend.models.api_requests import LlmFailureActionRequest, RetryNodeRequest, StopAnalysisRequest
from backend.models.schemas import (
    APIResponse, TaskStatus,
)
from backend.services.session_store import async_save_session, ensure_session_loaded, pause_and_restart_ws as _pause_and_restart_ws_impl
from backend.utils.auth import require_session_ownership
from backend.state import (
    _run_tasks,
    state_lock as _state_lock,
    websockets,
    workflows,
    get_session_lock,
)
from backend.services.session_store import update_session_cache as _update_session_cache
from backend.routes._helpers import teardown_running_workflow

logger = logging.getLogger(__name__)

# HITL gates that pause the workflow waiting on explicit user input. Kept as a
# single source of truth so adding a new WAITING_* gate only touches one place.
WAITING_HITL_STATUSES = frozenset({
    TaskStatus.WAITING_INPUT,
    TaskStatus.WAITING_ANNOTATION,
    TaskStatus.WAITING_PLAN_REVIEW,
    TaskStatus.WAITING_SPECIES_INPUT,
})


async def _resume_workflow(session_id, state, signal_fn=None):
    """Transition state to RUNNING, persist, and optionally signal the workflow."""
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)
    async with _state_lock:
        workflow = workflows.get(session_id)
    if workflow and signal_fn:
        signal_fn(workflow)
    return workflow


def resolve_hitl_skip_target(state):
    """Return (node_name, generation) for a WAITING_INPUT gate that must be
    released via signal_llm_failure_action — both the LLM-failure gate and the
    disabled-agent gate use that path. Return None for the round-completion
    gate (released via signal_continue).
    """
    reason = getattr(state, "wait_input_reason", None)
    if reason == WaitInputReason.LLM_FAILURE.value:
        info = state.llm_failure_info or {}
        return info.get("node_name", ""), info.get("generation", 0)
    if reason == WaitInputReason.AGENT_DISABLED.value:
        info = state.agent_disabled_info or {}
        return info.get("node_name", ""), info.get("generation", 0)
    if reason in (WaitInputReason.SPECIES_INPUT.value, WaitInputReason.GENE_SPECIES_MISMATCH.value):
        return "species_input", 0  # sentinel: lifecycle will call signal_species_input("")
    # Unknown or None reason — round-completion gate uses signal_continue.
    # Fail-closed: if a new WaitInputReason is added that has its own gate,
    # this must be updated to handle it (otherwise signal_continue would deadlock).
    if reason is not None:
        logger.warning(f"[lifecycle] resolve_hitl_skip_target: unknown wait_input_reason={reason}, "
                       f"falling back to signal_continue which may not unblock this gate")
    return None


async def _pause_and_restart_ws(session_id: str, state):
    await _pause_and_restart_ws_impl(session_id, state, websockets)


router = APIRouter(prefix="/api/session", tags=["lifecycle"])


@router.post("/{session_id}/run")
async def run_analysis(session_id: str, request: Request):
    """Start running analysis (use WebSocket for real-time updates)"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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
async def pause_analysis(session_id: str, request: Request):
    """Pause running analysis"""
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent resume/stop/llm-failure mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _pause_analysis_locked(session_id, user_id)


async def _pause_analysis_locked(session_id: str, user_id: str):
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    # Idempotent: already paused → no-op success (rapid double-click safe).
    if state.status == TaskStatus.PAUSED:
        return APIResponse(success=True, message="Analysis already paused")

    # Accept RUNNING and any WAITING_* state. Pausing while waiting for HITL
    # input still snapshots sandbox and clears the cooperative gate so when
    # the user later resumes from the gate, the agent loop also honors pause.
    _pausable = {TaskStatus.RUNNING} | WAITING_HITL_STATUSES
    if state.status not in _pausable:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause from status '{state.status}'"
        )

    if state.status in WAITING_HITL_STATUSES:
        state.pre_pause_status = state.status
    else:
        state.pre_pause_status = None
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
async def resume_analysis(session_id: str, request: Request):
    """Resume paused or completed analysis (for re-analysis)"""
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent pause/stop/llm-failure mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _resume_analysis_locked(session_id, user_id)


async def _resume_analysis_locked(session_id: str, user_id: str):
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    # Restore original max_rounds if it was saved during a stop operation
    if state.original_max_rounds is not None:
        state.max_rounds = state.original_max_rounds
        state.original_max_rounds = None

    if state.status == TaskStatus.COMPLETED:
        # Re-analysis: tear down the terminated checkpoint so the WS
        # reconnect triggers a fresh run (core.py "no checkpoint" path)
        # instead of resuming a graph already at END (infinite loop).
        await teardown_running_workflow(session_id)
        state.status = TaskStatus.RUNNING
        state.max_rounds = state.current_round + 1
        state.last_restored_snapshot_id = None
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        return APIResponse(
            success=True,
            message="Re-analysis started",
            data={"needs_reconnect": True}
        )
    elif state.status == TaskStatus.WAITING_INPUT:
        # P3-5: directionless-INNOVATION default lives in workflow's
        # _initialize_round (workflow-state invariant, not a transport concern).
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
            await _pause_and_restart_ws(session_id, state)
            return APIResponse(
                success=True,
                message="Continuing analysis",
                data={"needs_reconnect": True},
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
            await _pause_and_restart_ws(session_id, state)
            return APIResponse(
                success=True,
                message="Annotation confirmed, continuing analysis",
                data={"needs_reconnect": True},
            )
    elif state.status == TaskStatus.WAITING_PLAN_REVIEW:
        # Plan review requires an explicit approve/skip/modify action via /plan-review.
        # Generic resume must not silently convert the gate into "skip".
        return APIResponse(
            success=True,
            message="Session is waiting for plan review. Submit /plan-review to approve, skip, or modify the plan.",
            data={"needs_reconnect": False},
        )
    elif state.status == TaskStatus.WAITING_SPECIES_INPUT:
        # Continue from species-input wait by sending an empty signal
        # (workflow treats empty input as "use suggestion / auto-skip").
        # An explicit submission goes through /interaction/species-input instead.
        state.status = TaskStatus.RUNNING
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        async with _state_lock:
            workflow = workflows.get(session_id)
        if workflow:
            workflow.signal_species_input("")
            return APIResponse(
                success=True,
                message="Continuing analysis (species input auto-skipped)"
            )
        else:
            await _pause_and_restart_ws(session_id, state)
            return APIResponse(
                success=True,
                message="Continuing analysis",
                data={"needs_reconnect": True},
            )
    elif state.status == TaskStatus.PAUSED:
        if state.pre_pause_status in WAITING_HITL_STATUSES:
            restored_status = state.pre_pause_status
            state.status = restored_status
            state.pre_pause_status = None
        else:
            state.status = TaskStatus.RUNNING
            state.pre_pause_status = None
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        async with _state_lock:
            workflow = workflows.get(session_id)
        if workflow:
            workflow.signal_resume()
            # P0-5: do NOT auto-signal HITL gates on generic resume. The user
            # asked to unpause; they must still submit an explicit action via
            # /choose-direction, /confirm-annotation, or /plan-review to
            # advance past the restored WAITING_* gate.
            if state.status in WAITING_HITL_STATUSES:
                message = (
                    "Analysis resumed; awaiting explicit user input via the "
                    "appropriate gate endpoint"
                )
            else:
                message = "Analysis resumed"
            return APIResponse(
                success=True,
                message=message,
                data={"needs_reconnect": False, "status": state.status.value},
            )
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
async def stop_analysis(session_id: str, request: Request, req: StopAnalysisRequest):
    """Stop a running analysis - either immediately or with summary report"""
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent pause/resume/llm-failure mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _stop_analysis_locked(session_id, user_id, req)


async def _stop_analysis_locked(session_id: str, user_id: str, req: StopAnalysisRequest):
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if state.status not in ({TaskStatus.RUNNING, TaskStatus.GENERATING_REPORT} | WAITING_HITL_STATUSES):
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
            skip_target = resolve_hitl_skip_target(state)
            if skip_target is not None:
                node_name, generation = skip_target
                if node_name == "species_input":
                    # Species-input gate — released via signal_species_input
                    if state.original_max_rounds is None:
                        state.original_max_rounds = state.max_rounds
                    state.max_rounds = state.current_round
                    state.status = TaskStatus.RUNNING
                    await _resume_workflow(session_id, state,
                        signal_fn=lambda w: w.signal_species_input(""))
                    return APIResponse(success=True, message="Skipping species input, generating report...")
                # LLM-failure OR disabled-agent gate — both wait on
                # _llm_failure_events[node] and are released by
                # signal_llm_failure_action. Using signal_continue here would
                # leave the gate blocked forever (deadlock).
                if state.original_max_rounds is None:
                    state.original_max_rounds = state.max_rounds
                state.max_rounds = state.current_round
                state.status = TaskStatus.RUNNING
                await _resume_workflow(session_id, state,
                    signal_fn=lambda w, nn=node_name, g=generation: w.signal_llm_failure_action(nn, "skip", g))
                return APIResponse(success=True, message="Skipping blocked step, generating report...")
            else:
                # Round completion wait — existing behavior
                if state.original_max_rounds is None:
                    state.original_max_rounds = state.max_rounds
                state.max_rounds = state.current_round
                state.status = TaskStatus.RUNNING
                await _resume_workflow(session_id, state,
                    signal_fn=lambda w: w.signal_continue())
                return APIResponse(success=True, message="Generating report...")
        elif state.status == TaskStatus.WAITING_ANNOTATION:
            # Workflow is blocked at _annotation_event.wait() inside callback
            # Unblock it and let the round finish, then route to report gen
            if state.original_max_rounds is None:
                state.original_max_rounds = state.max_rounds
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _resume_workflow(session_id, state,
                signal_fn=lambda w: w.signal_annotation_confirmed())
            return APIResponse(success=True, message="Generating report...")
        elif state.status == TaskStatus.WAITING_PLAN_REVIEW:
            # Workflow is blocked at _plan_review_event.wait()
            # Unblock it by skipping the review and finishing the round
            if state.original_max_rounds is None:
                state.original_max_rounds = state.max_rounds
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _resume_workflow(session_id, state,
                signal_fn=lambda w: w.signal_plan_review("skip"))
            return APIResponse(success=True, message="Generating report...")
        elif state.status == TaskStatus.WAITING_SPECIES_INPUT:
            # Workflow is blocked at _species_event.wait() inside planning or
            # gene validation callback. Release via empty signal (auto-skip).
            if state.original_max_rounds is None:
                state.original_max_rounds = state.max_rounds
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _resume_workflow(session_id, state,
                signal_fn=lambda w: w.signal_species_input(""))
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


@router.post("/{session_id}/llm-failure-action")
async def llm_failure_action(session_id: str, request: Request, req: LlmFailureActionRequest):
    """Handle user's choice for an LLM timeout/failure.

    Idempotency: the `generation` field must match the workflow's
    current `_llm_failure_generation`. Duplicate or stale requests
    (e.g. double-click) are rejected with 409.
    """
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent pause/resume/stop mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _llm_failure_action_locked(session_id, user_id, req)


async def _llm_failure_action_locked(session_id: str, user_id: str, req: LlmFailureActionRequest):
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if state.status != TaskStatus.WAITING_INPUT:
        raise HTTPException(
            status_code=409,
            detail=f"Session is {state.status.value}, expected waiting_input"
        )
    if state.wait_input_reason not in (
        WaitInputReason.LLM_FAILURE.value,
        WaitInputReason.AGENT_DISABLED.value,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Session wait_input_reason is {state.wait_input_reason}, expected llm_failure or agent_disabled"
        )

    workflow = workflows.get(session_id)

    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    ok = workflow.signal_llm_failure_action(req.node_name, req.action, req.generation)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Stale or duplicate request (generation {req.generation} != current)"
        )

    return APIResponse(
        success=True,
        message=f"LLM failure action '{req.action}' applied to node '{req.node_name}'"
    )


@router.post("/{session_id}/retry-node")
async def retry_node(session_id: str, request: Request, body: RetryNodeRequest):
    """Retry or skip a failed node"""
    node_name = body.node_name
    skip = body.skip

    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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
async def get_failed_node(session_id: str, request: Request):
    """Get information about the failed node"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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
async def resume_bioinformatics(session_id: str, request: Request):
    """Resume bioinformatics analysis from last checkpoint"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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
