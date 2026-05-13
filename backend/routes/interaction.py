"""Interaction routes: human-input, confirm-annotations, supplementary-analysis, choose-direction, request-input.

Extracted from backend/main.py to decouple human-in-the-loop route handlers.
"""
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from backend.models.api_requests import (
    HumanInputRequest, AnnotationConfirmation, SupplementaryAnalysisRequest,
    DirectionChoice, PlanReviewRequest,
)
from backend.models.schemas import APIResponse, TaskStatus
from backend.services.session_store import (
    async_save_session, ensure_session_loaded,
    update_session_cache as _update_session_cache,
    pause_and_restart_ws as _pause_and_restart_ws_impl,
)
from backend.state import (
    state_lock as _state_lock,
    websockets,
    workflows,
)
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["interaction"])

_UNSAFE_PATTERN = re.compile(r'[{}\[\]\\;`$]|__\w+__|import\s|exec\s*\(|eval\s*\(')


async def _pause_and_restart_ws(session_id: str, state):
    await _pause_and_restart_ws_impl(session_id, state, websockets)


@router.post("/{session_id}/human-input")
async def send_human_input(session_id: str, request: Request, body: HumanInputRequest):
    """Send human input/intervention to the analysis"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    state.pending_human_input = body.message
    state.pending_uploaded_files = [f.model_dump() for f in body.uploaded_files]
    state.human_input_requested = False

    if body.clear_queue and state.directions_queue:
        cleared_count = len(state.directions_queue)
        state.directions_queue = []
        state.max_rounds = state.current_round + 1
        logger.info(f"[{session_id}] Direction queue cleared ({cleared_count} directions removed) by user")

    state.human_messages.append({
        "timestamp": datetime.now().isoformat(),
        "message": body.message,
        "target_agent": body.target_agent,
        "uploaded_files": [f.model_dump() for f in body.uploaded_files]
    })

    if session_id in websockets:
        try:
            await websockets[session_id].send_json({
                "type": "human_input_received",
                "content": {
                    "message": body.message,
                    "uploaded_files": [f.model_dump() for f in body.uploaded_files]
                },
                "target_agent": body.target_agent,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            logger.warning("Failed to buffer human input to WebSocket", exc_info=True)

    async with _state_lock:
        workflow = workflows.get(session_id)
    if workflow and state.status == TaskStatus.WAITING_INPUT:
        workflow.signal_continue()
        logger.info(f"[{session_id}] Human input received, signalling workflow to continue")

    return APIResponse(
        success=True,
        message="Human input received",
        data={
            "target_agent": body.target_agent,
            "files_count": len(body.uploaded_files)
        }
    )


@router.get("/{session_id}/human-input")
async def get_human_input_status(session_id: str, request: Request):
    """Check if system is waiting for human input"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    return {
        "waiting_for_input": state.human_input_requested,
        "pending_input": state.pending_human_input,
        "history": state.human_messages[-10:]
    }


@router.post("/{session_id}/confirm-annotations")
async def confirm_annotations(session_id: str, request: Request, body: AnnotationConfirmation):
    """User confirms or modifies cell type annotations during exploration phase"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if state.status != TaskStatus.WAITING_ANNOTATION:
        raise HTTPException(status_code=400, detail="Session is not waiting for annotation confirmation")

    if state.annotation_state:
        for ua in body.annotations:
            cid = ua.get("cluster_id", "")
            for ca in state.annotation_state.cluster_annotations:
                if ca.cluster_id == cid:
                    if ua.get("user_override"):
                        override = ua["user_override"].strip()
                        override = _UNSAFE_PATTERN.sub('', override)
                        ca.user_override = override[:200]
                    ca.user_confirmed = True
                    break
        state.annotation_state.confirmed = True
        state.annotation_state.pending = False

    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    async with _state_lock:
        workflow = workflows.get(session_id)
    if workflow:
        workflow.signal_annotation_confirmed()
    else:
        await _pause_and_restart_ws(session_id, state)

    return APIResponse(
        success=True,
        message="Annotations confirmed"
    )


@router.post("/{session_id}/supplementary-analysis")
async def request_supplementary_analysis(
    session_id: str, request: Request, body: SupplementaryAnalysisRequest
):
    """User requests additional analysis at a HITL point before finalizing their decision."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if body.target == "annotation" and state.status != TaskStatus.WAITING_ANNOTATION:
        raise HTTPException(status_code=400, detail="Session is not at annotation confirmation point")
    if body.target == "direction" and state.status != TaskStatus.WAITING_INPUT:
        raise HTTPException(status_code=400, detail="Session is not at direction selection point")

    async with _state_lock:
        workflow = workflows.get(session_id)
    if not workflow:
        raise HTTPException(status_code=400, detail="No active workflow for this session")

    workflow.signal_supplementary_analysis(body.request_text, body.target)

    return APIResponse(
        success=True,
        message="Supplementary analysis started"
    )


@router.post("/{session_id}/choose-direction")
async def choose_direction(session_id: str, request: Request, body: DirectionChoice):
    """User chooses research direction(s) for innovation phase after exploration report"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if state.status != TaskStatus.WAITING_INPUT:
        raise HTTPException(status_code=400, detail="Session is not waiting for input")

    if body.let_pi_decide:
        state.user_chosen_direction = "[PI_DECIDE]"
        state.pending_human_input = "[PI_DECIDE]"
    elif body.directions and len(body.directions) > 0:
        state.user_chosen_directions = body.directions
        state.directions_queue = [d.copy() for d in body.directions]
        state.max_rounds = len(body.directions) + 1
        state.user_chosen_direction = None
        state.pending_human_input = json.dumps(
            [d.get("direction", "") for d in body.directions], ensure_ascii=False
        )
        logger.info(f"[{session_id}] Multi-direction mode: {len(body.directions)} directions, max_rounds={state.max_rounds}")
    elif body.direction:
        state.user_chosen_directions = [{"direction": body.direction, "notes": ""}]
        state.directions_queue = [{"direction": body.direction, "notes": ""}]
        state.max_rounds = 2
        state.user_chosen_direction = body.direction
        state.pending_human_input = body.direction
    else:
        raise HTTPException(status_code=400, detail="No direction provided: specify direction, directions list, or let_pi_decide")

    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    async with _state_lock:
        workflow = workflows.get(session_id)
    if workflow:
        workflow.signal_continue()
    else:
        await _pause_and_restart_ws(session_id, state)

    direction_count = len(body.directions) if body.directions else (1 if body.direction else 0)
    return APIResponse(
        success=True,
        message=f"Direction selected: {'PI decides' if body.let_pi_decide else f'{direction_count} direction(s)'}"
    )


@router.post("/{session_id}/request-input")
async def request_human_input(session_id: str, request: Request):
    """Request human input (called by workflow when needing guidance)"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    state.human_input_requested = True

    if session_id in websockets:
        try:
            await websockets[session_id].send_json({
                "type": "human_input_requested",
                "content": "System is requesting human guidance",
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            logger.warning("Failed to send human_input_requested via WebSocket", exc_info=True)


@router.post("/{session_id}/plan-review")
async def submit_plan_review(session_id: str, request: Request, body: PlanReviewRequest):
    """Submit user's review decision for an analysis plan."""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    action = body.action
    user_input = _UNSAFE_PATTERN.sub('', body.user_input) if body.user_input else ""

    async with _state_lock:
        workflow = workflows.get(session_id)
    if not workflow:
        if state.status == TaskStatus.WAITING_PLAN_REVIEW:
            if action == "approve":
                state.current_plan.status = "approved"
                state.current_plan.approved_at = datetime.now()
            await async_save_session(session_id, state)
            await _pause_and_restart_ws(session_id, state)
            return APIResponse(success=True, message="Plan review recorded, restarting")
        raise HTTPException(status_code=404, detail="Session not found")

    workflow.signal_plan_review(action, user_input)
    return APIResponse(success=True, message=f"Plan review action '{action}' received")
