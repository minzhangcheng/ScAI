"""Interaction routes: human-input, confirm-annotations, supplementary-analysis, choose-direction, request-input.

Extracted from backend/main.py to decouple human-in-the-loop route handlers.
"""
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from backend.constants import PI_DECIDE_SIGNAL
from backend.models.api_requests import (
    HumanInputRequest, AnnotationConfirmation, SupplementaryAnalysisRequest,
    DirectionChoice, PlanReviewRequest, SpeciesInputRequest,
)
from backend.models.schemas import AnalysisPhase, APIResponse, TaskStatus
from backend.services.session_store import (
    async_save_session, ensure_session_loaded,
    update_session_cache as _update_session_cache,
    pause_and_restart_ws as _pause_and_restart_ws_impl,
)
from backend.state import (
    state_lock as _state_lock,
    websockets,
    workflows,
    get_session_lock,
)
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id
from backend.services.interaction_record_capture import (
    get_or_open_direction_record, map_selections, build_annotation_record,
    populate_direction_record_from_state, resolve_integration_gate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["interaction"])

_UNSAFE_PATTERN = re.compile(r'[{}\[\]\\;`$]|__\w+__|import\s|exec\s*\(|eval\s*\(')


async def _pause_and_restart_ws(session_id: str, state):
    await _pause_and_restart_ws_impl(session_id, state, websockets)


@router.post("/{session_id}/human-input")
async def send_human_input(session_id: str, request: Request, body: HumanInputRequest):
    """Send human input/intervention to the analysis"""
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize read-modify-save with concurrent mutators on this session.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
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
    # P1-9: serialize read-modify-save with concurrent mutators on this session.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
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
            state.annotation_record = build_annotation_record(state, auto_confirmed=False)

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
    # P1-9: serialize with other mutators on this session.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
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
    # P1-9: serialize read-modify-save with concurrent mutators on this session.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _choose_direction_locked(session_id, user_id, body)


async def _choose_direction_locked(session_id: str, user_id: str, body: DirectionChoice):
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if state.status != TaskStatus.WAITING_INPUT:
        raise HTTPException(status_code=400, detail="Session is not waiting for input")
    exploration_gate = (
        state.current_round == 1
        and state.analysis_phase == AnalysisPhase.INNOVATION
        and not state.user_chosen_direction
    )
    integration_gate = (state.analysis_phase == AnalysisPhase.INTEGRATION)
    if not (exploration_gate or integration_gate):
        raise HTTPException(
            status_code=400,
            detail=("This endpoint is only valid at a direction selection gate "
                    "(post-exploration or integration). For mid-analysis input use "
                    "/supplementary-analysis or /human-input."),
        )
    if integration_gate and (body.direction and not body.directions):
        # Single direction string not allowed at integration gate;
        # use directions list, let_pi_decide, or /finalize-from-integration.
        raise HTTPException(
            status_code=400,
            detail="At the integration gate, select explicit new directions (list) or let PI decide.",
        )

    has_direction_choice = bool(body.let_pi_decide or body.direction or body.directions)
    if not has_direction_choice:
        raise HTTPException(
            status_code=400,
            detail="No direction provided: specify direction, directions list, or let_pi_decide",
        )

    if integration_gate:
        rec = resolve_integration_gate(state, event_type="select")
    else:
        rec = get_or_open_direction_record(
            state, round_number=state.current_round, phase="exploration",
        )
        populate_direction_record_from_state(state, rec, "select")
    if body.let_pi_decide:
        rec.let_pi_decide = True
        rec.event_type = "open"
    elif body.directions:
        rec.selections = map_selections(list(body.directions), rec.candidates)
        rec.event_type = "select"
    elif body.direction:
        rec.selections = map_selections(
            [{"direction": body.direction, "notes": ""}], rec.candidates)
        rec.event_type = "select"

    if body.let_pi_decide:
        state.user_chosen_direction = PI_DECIDE_SIGNAL
        state.pending_human_input = PI_DECIDE_SIGNAL
        if integration_gate:
            state.analysis_phase = AnalysisPhase.INNOVATION
    elif body.directions and len(body.directions) > 0:
        if integration_gate:
            state.user_chosen_directions = (state.user_chosen_directions or []) + list(body.directions)
            state.analysis_phase = AnalysisPhase.INNOVATION
            state.max_rounds = state.current_round + len(body.directions)
        else:
            state.user_chosen_directions = list(body.directions)
            state.max_rounds = len(body.directions) + 1
        state.directions_queue = [d.copy() for d in body.directions]
        state.active_direction_batch = [d.copy() for d in body.directions]
        state.pending_integration = False
        state.user_chosen_direction = None
        state.pending_human_input = json.dumps(
            [d.get("direction", "") for d in body.directions], ensure_ascii=False
        )
        logger.info(f"[{session_id}] Multi-direction mode: {len(body.directions)} directions, max_rounds={state.max_rounds}")
    elif body.direction:
        state.user_chosen_directions = [{"direction": body.direction, "notes": ""}]
        state.directions_queue = [{"direction": body.direction, "notes": ""}]
        state.active_direction_batch = [{"direction": body.direction, "notes": ""}]
        state.pending_integration = False
        state.max_rounds = 2
        state.user_chosen_direction = body.direction
        state.pending_human_input = body.direction

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


@router.post("/{session_id}/finalize-from-integration")
async def finalize_from_integration(session_id: str, request: Request):
    """User stops the integration loop and proceeds to the final report."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        if not (state.analysis_phase == AnalysisPhase.INTEGRATION
                and state.status == TaskStatus.WAITING_INPUT):
            raise HTTPException(
                status_code=400,
                detail="finalize-from-integration is only valid at the integration gate.",
            )
        state.final_report_requested = True
        rec = resolve_integration_gate(state, event_type="finalize")
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
        async with _state_lock:
            workflow = workflows.get(session_id)
        if workflow:
            workflow.signal_continue()
        else:
            # No active workflow — set GENERATING_REPORT so the WS reconnect
            # path detects the finalize request via needs_report_only and
            # runs generate_report directly. Do NOT pause-and-restart
            # (that sets status=PAUSED, which prevents final_report_requested
            # from being consumed by the integration-gate wait logic).
            state.status = TaskStatus.GENERATING_REPORT
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
            # Close any existing WS so the frontend reconnects and triggers
            # the report-only path.
            from ..state import websockets as _ws_map
            old_ws = _ws_map.get(session_id)
            if old_ws:
                try:
                    await old_ws.close(4000, "finalize_from_integration")
                except Exception:
                    pass
        return APIResponse(success=True, message="Finalizing: proceeding to the final report")


@router.post("/{session_id}/request-input")
async def request_human_input(session_id: str, request: Request):
    """Request human input (called by workflow when needing guidance)"""
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize the flag flip with other mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
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

    return APIResponse(success=True, message="Human input requested")


@router.post("/{session_id}/plan-review")
async def submit_plan_review(session_id: str, request: Request, body: PlanReviewRequest):
    """Submit user's review decision for an analysis plan."""
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize plan-state mutation with other mutators on this session.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        state = await require_session_ownership(session_id=session_id, user_id=user_id)

        action = body.action
        user_input = _UNSAFE_PATTERN.sub('', body.user_input) if body.user_input else ""

        async with _state_lock:
            workflow = workflows.get(session_id)
        if not workflow:
            if state.status == TaskStatus.WAITING_PLAN_REVIEW:
                if action == "modify":
                    raise HTTPException(
                        status_code=409,
                        detail="Cannot modify plan without an active workflow. Resume the session first.",
                    )
                if action in ("approve", "skip"):
                    if state.current_plan is None:
                        raise HTTPException(
                            status_code=404,
                            detail="No plan available to approve. Resume the session first.",
                        )
                    state.current_plan.status = "approved"
                    state.current_plan.approved_at = datetime.now()
                await async_save_session(session_id, state)
                await _pause_and_restart_ws(session_id, state)
                return APIResponse(success=True, message="Plan review recorded, restarting")
            raise HTTPException(status_code=404, detail="Session not found")

        workflow.signal_plan_review(action, user_input)
    return APIResponse(success=True, message=f"Plan review action '{action}' received")


@router.post("/{session_id}/species-input")
async def species_input(session_id: str, request: Request, body: SpeciesInputRequest):
    """User submits species-related input (inference or gene validation)."""
    user_id = getattr(request.state, "user_id", "")
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        if state.status != TaskStatus.WAITING_SPECIES_INPUT:
            raise HTTPException(status_code=400, detail="Session is not waiting for species input")

        async with _state_lock:
            workflow = workflows.get(session_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="No active workflow")

        workflow.signal_species_input(body.input)
    return APIResponse(success=True, message="Species input received")
