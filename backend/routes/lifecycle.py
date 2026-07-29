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
from backend.constants import WaitInputReason, SEMANTIC_WAIT_GATES
from backend.models.api_requests import LlmFailureActionRequest, RetryNodeRequest, StopAnalysisRequest
from backend.models.schemas import (
    APIResponse, TaskStatus,
)
from backend.services.session_store import (
    async_save_session,
    assert_recovery_clear,
    ensure_session_loaded,
    pause_and_restart_ws as _pause_and_restart_ws_impl,
)
from backend.utils.auth import require_session_ownership
from backend.utils.runtime_apply import is_forward_resume_blocked
from backend.services.snapshot_v2.errors import (
    SNAPSHOT_FAILURE_ACTIONS,
    SnapshotTransactionError,
    is_snapshot_failure_blocked,
)
from backend.services.snapshot_v2.feature_gate import require_snapshot_v2_enabled
from backend.utils.files import validate_session_id
from backend.state import (
    _run_tasks,
    state_lock as _state_lock,
    websockets,
    workflows,
    get_session_lock,
    get_snapshot_action_lock,
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
    if reason == WaitInputReason.MARKER_DB_MISSING.value:
        info = getattr(state, "llm_failure_info", None) or {}
        return info.get("node_name"), info.get("generation")
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
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    from backend.config import is_snapshot_v2_enabled
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    if not is_snapshot_v2_enabled():
        normalize_disabled_snapshot_state(
            state,
            workflow_active=session_id in workflows,
        )

    if state.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis already running")

    try:
        assert_recovery_clear(state, session_id=session_id)
    except SnapshotTransactionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail=(
                "Session is blocked on a required snapshot failure. "
                "Use Retry, Manage, or Terminate — generic start is refused."
            ),
        )

    # Idle-restored session: signal explicit_start so the graph node
    # preserves the restored round and plan rather than auto-incrementing.
    #
    # The same signal is what makes an *imported* session runnable at all. A
    # session that has already run (current_round > 0) is dispatched to
    # workflow.resume() by the WebSocket handler, and resume() refuses to
    # "invent a fresh initialize_round run" when it finds no checkpoint --
    # which an import never carries unless include_runtime was set. Without an
    # explicit start intent it logs, flips the status back to PAUSED and
    # executes nothing, so Start/Re-run looked like a silent no-op and the
    # session had no way forward at all.
    from backend.config import get_session_states
    from backend.utils.runtime_apply import _live_pkl

    # A checkpoint is only resumable together with the kernel it was written
    # against: it names the graph node to continue from, and the pkl holds the
    # values that node expects to find. Testing the file alone let an archive
    # exported with include_runtime but imported without trust=true — the
    # checkpoint travels unconditionally, the pkl does not — look resumable,
    # which withheld the explicit_start intent and sent Re-run into an exact
    # continuation on an empty kernel instead of a fresh round.
    has_resumable_checkpoint = (
        (get_session_states(session_id) / "checkpoints.sqlite").exists()
        and _live_pkl(session_id).exists()
    )
    if getattr(state, "restore_resume_kind", None) is None and (
        getattr(state, "restore_branch_id", None)
        or (not has_resumable_checkpoint and state.current_round > 0)
    ):
        state.restore_start_intent = "explicit_start"

    # A Skip armed at a boundary the graph never reached again stays on the
    # session forever — consume_snapshot_skip only clears it at a *matching*
    # boundary. Starting a run is a fresh intent, so an unconsumed marker from a
    # previous run must not silently suppress this run's first required capture
    # (the wildcard form, armed when the failing phase is unknown, would
    # suppress whichever capture happens to come first).
    if getattr(state, "skip_required_snapshot_phase", None) is not None:
        state.skip_required_snapshot_phase = None

    # Fresh rerun clears any snapshot restore state — new kernel will be built.
    # Remember *why* it is being cleared: this run is the "re-run from your data"
    # the blocked-resume message promises, so the bioinformatics agent has to be
    # told its kernel is empty rather than left to trip over a missing adata.
    if is_forward_resume_blocked(getattr(state, "runtime_restore_status", "complete")):
        state.runtime_recompute_pending = True
    state.runtime_restore_status = "complete"
    state.runtime_restore_warning = None
    state.status = TaskStatus.RUNNING
    # Forced: this write carries the cleared runtime block and the explicit
    # start intent, and RUNNING is not one of the statuses that bypasses the
    # 5s save throttle. Losing it to the throttle would leave a restarted
    # backend still refusing to start the session the user just started.
    await async_save_session(session_id, state, force=True)

    return APIResponse(
        success=True,
        message="Analysis started. Connect to WebSocket for updates.",
        data={"websocket_url": f"/ws/{session_id}"}
    )


@router.post("/{session_id}/pause")
async def pause_analysis(session_id: str, request: Request):
    """Pause running analysis"""
    validate_session_id(session_id)
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
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent pause/stop/llm-failure mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _resume_analysis_locked(session_id, user_id)


async def _resume_analysis_locked(session_id: str, user_id: str):
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    from backend.config import is_snapshot_v2_enabled
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    if not is_snapshot_v2_enabled():
        normalize_disabled_snapshot_state(
            state,
            workflow_active=session_id in workflows,
        )

    try:
        assert_recovery_clear(state, session_id=session_id)
    except SnapshotTransactionError as exc:
        return APIResponse(
            success=False,
            message=str(exc),
            data={"needs_reconnect": False, "recovery_blocked": True},
        )

    # Snapshot-failure gate: generic resume must not clear or advance.
    # Only dedicated producer-aware snapshot-failure actions may clear it.
    if is_snapshot_failure_blocked(state):
        return APIResponse(
            success=False,
            message=(
                "Session is blocked on a required snapshot failure. "
                "Use a snapshot-failure action — generic resume is refused."
            ),
            data={
                "needs_reconnect": False,
                "snapshot_failure": getattr(state, "snapshot_failure", None),
                "wait_input_reason": getattr(state, "wait_input_reason", None),
            },
        )

    # Restore original max_rounds if it was saved during a stop operation
    if state.original_max_rounds is not None:
        state.max_rounds = state.original_max_rounds
        state.original_max_rounds = None

    if is_forward_resume_blocked(state.runtime_restore_status) and state.status != TaskStatus.COMPLETED:
        return APIResponse(
            success=True,
            message=(
                "This session was restored without a consistent kernel runtime. "
                "Forward analysis is blocked. Re-run the analysis from your data to "
                "recompute kernel state, or view the session read-only."
            ),
            data={"needs_reconnect": False, "runtime_unavailable": True},
        )

    if state.status == TaskStatus.COMPLETED:
        # Re-analysis: tear down the terminated checkpoint so the WS
        # reconnect triggers a fresh run (core.py "no checkpoint" path)
        # instead of resuming a graph already at END (infinite loop).
        from backend.workflow.provenance import consume_restore_provenance

        await teardown_running_workflow(session_id)
        state.status = TaskStatus.RUNNING
        state.max_rounds = state.current_round + 1
        # Clear both durable restore fields so the next run cannot mis-tag
        # plan/RoundResult from a prior restore (terminal/idle leave provenance).
        consume_restore_provenance(state)
        # Same contract as POST /run: a re-analysis that is clearing an
        # unavailable runtime has to tell the agent its kernel is empty, or it
        # carries on as though adata were still loaded from the original run.
        if is_forward_resume_blocked(getattr(state, "runtime_restore_status", "complete")):
            state.runtime_recompute_pending = True
        state.runtime_restore_status = "complete"  # re-run recomputes kernel state
        state.runtime_restore_warning = None  # fresh rerun clears any restore warning
        await _update_session_cache(session_id, state)
        # Forced for the same reason as POST /run: the cleared runtime block and
        # the recompute flag must survive a restart, and RUNNING does not bypass
        # the save throttle on its own.
        await async_save_session(session_id, state, force=True)
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
            if (
                state.restore_resume_kind == "semantic_gate"
                and state.restore_resume_gate in SEMANTIC_WAIT_GATES
            ):
                state.status = TaskStatus.RUNNING
                await _update_session_cache(session_id, state)
                await async_save_session(session_id, state)
                return APIResponse(
                    success=True,
                    message="Continuing restored analysis",
                    data={"needs_reconnect": True, "restored_wait_continuation": True},
                )
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
                message="Continuing analysis",
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
    validate_session_id(session_id)
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
        # Snapshot-failure gate: stop-and-summarize must not skip or advance.
        if is_snapshot_failure_blocked(state):
            return APIResponse(
                success=False,
                message=(
                    "Session is blocked on a required snapshot failure. "
                    "Stop-and-summarize is refused until Retry, Manage, or Terminate."
                ),
                data={
                    "snapshot_failure": getattr(state, "snapshot_failure", None),
                    "wait_input_reason": getattr(state, "wait_input_reason", None),
                },
            )
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
    validate_session_id(session_id)
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
        WaitInputReason.MARKER_DB_MISSING.value,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Session wait_input_reason is {state.wait_input_reason}, expected llm_failure, agent_disabled, or marker_db_missing"
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
    validate_session_id(session_id)
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
    validate_session_id(session_id)
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
    validate_session_id(session_id)
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


@router.post(
    "/{session_id}/snapshot-failure/retry",
    dependencies=[Depends(require_snapshot_v2_enabled)],
)
async def snapshot_failure_retry(session_id: str, request: Request):
    """Retry a snapshot failure by re-invoking the exact failed capture producer.

    Validates ``SnapshotRetryContext`` from the failure gate. Missing or
    invalid context returns 409 and leaves the gate intact. Never defaults
    a missing producer to workflow-boundary.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # Serialize the three snapshot-failure actions. Retry is the longest-running
    # mutator in the system - it re-runs the failed capture while the gate stays
    # armed - and all three actions read-modify-persist the *same* cached
    # AnalysisState. Without this, a Terminate issued mid-capture wins its
    # is_snapshot_failure_blocked() check, tears the workflow down and persists
    # STOPPED, and is then silently overwritten when this handler persists its
    # own outcome below.
    async with get_snapshot_action_lock(session_id):
        return await _snapshot_failure_retry_locked(session_id, user_id)


async def _snapshot_failure_retry_locked(session_id: str, user_id: str):
    from pydantic import ValidationError

    from backend.services.snapshot_v2.failure_gate import (
        default_snapshot_retry_dependencies,
        retry_snapshot_producer,
    )
    from backend.services.snapshot_v2.models import SnapshotRetryContext

    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if not is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail="Session is not blocked on a snapshot failure",
        )

    failure = getattr(state, "snapshot_failure", None) or {}
    raw_context = failure.get("retry_context")
    if not isinstance(raw_context, dict) or not raw_context:
        raise HTTPException(
            status_code=409,
            detail=(
                "Snapshot failure retry context is missing; "
                "gate left intact"
            ),
        )
    try:
        context = SnapshotRetryContext.model_validate(raw_context)
    except ValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Snapshot failure retry context is invalid; "
                "gate left intact"
            ),
        ) from exc

    outcome = await retry_snapshot_producer(
        session_id,
        state,
        context,
        default_snapshot_retry_dependencies(),
    )

    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    # A workflow-boundary gate unwound the graph, so nothing is running and the
    # re-drive only starts when a WebSocket reconnect rebuilds the workflow. The
    # client holding the HITL connection never reconnects on its own, so close
    # it with the same 4000 code the confirm/choose endpoints use to trigger the
    # frontend's auto-reconnect. Skipped when a workflow is live: the capture
    # then already ran in-process and that run owns the socket.
    if outcome.success and context.producer == "workflow_boundary":
        async with _state_lock:
            ws = None if session_id in workflows else websockets.get(session_id)
        if ws is not None:
            try:
                await ws.close(4000, "snapshot_retry_resume")
            except Exception:
                logger.debug(
                    "[%s] Failed to close HITL WebSocket after snapshot retry",
                    session_id,
                    exc_info=True,
                )

    data = {
        "status": outcome.status
        if outcome.status is not None
        else getattr(state.status, "value", str(state.status)),
        "wait_input_reason": (
            outcome.wait_input_reason
            if outcome.wait_input_reason is not None
            or outcome.still_blocked
            else getattr(state, "wait_input_reason", None)
        ),
        "snapshot_failure": (
            outcome.snapshot_failure
            if outcome.still_blocked
            else getattr(state, "snapshot_failure", None)
        ),
    }
    if outcome.still_blocked:
        data["still_blocked"] = True
    if outcome.transaction_id is not None:
        data["transaction_id"] = outcome.transaction_id
    if outcome.snapshot_transaction is not None:
        data["snapshot_transaction"] = outcome.snapshot_transaction

    return APIResponse(
        success=outcome.success,
        message=outcome.message,
        data=data,
    )


@router.post(
    "/{session_id}/snapshot-failure/skip",
    dependencies=[Depends(require_snapshot_v2_enabled)],
)
async def snapshot_failure_skip(session_id: str, request: Request):
    """Skip snapshot creation and continue the analysis.

    Clears the failure gate WITHOUT capturing a snapshot and resumes the
    workflow, forfeiting only the restore point at that boundary. Producer-aware:
    workflow_boundary / post_report / chat are skippable; pre_restore is not
    (skipping would proceed with a restore that overwrites live state without a
    safety point) and returns 409.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # Same gate, same cached state object as Retry/Terminate - must not interleave.
    async with get_snapshot_action_lock(session_id):
        return await _snapshot_failure_skip_locked(session_id, user_id)


async def _snapshot_failure_skip_locked(session_id: str, user_id: str):
    from pydantic import ValidationError

    from backend.services.snapshot_v2.failure_gate import skip_snapshot_producer
    from backend.services.snapshot_v2.models import SnapshotRetryContext

    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if not is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail="Session is not blocked on a snapshot failure",
        )

    failure = getattr(state, "snapshot_failure", None) or {}
    raw_context = failure.get("retry_context")
    if not isinstance(raw_context, dict) or not raw_context:
        raise HTTPException(
            status_code=409,
            detail="Snapshot failure retry context is missing; gate left intact",
        )
    try:
        context = SnapshotRetryContext.model_validate(raw_context)
    except ValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail="Snapshot failure retry context is invalid; gate left intact",
        ) from exc

    outcome = skip_snapshot_producer(state, context)
    if not outcome.success:
        # Skip unsupported for this producer (e.g. pre_restore): gate left intact.
        raise HTTPException(status_code=409, detail=outcome.message)

    # Clear gate + arm the one-shot skip marker + set RUNNING, then persist. The
    # frontend reconnects the WebSocket, which resumes the workflow from the graph
    # checkpoint; the armed marker makes the next required capture at that phase a
    # no-op so the analysis proceeds past the boundary that previously failed.
    # (Mirrors the Retry flow, which likewise defers graph re-drive to reconnect.)
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=outcome.message,
        data={
            "status": outcome.status
            if outcome.status is not None
            else getattr(state.status, "value", str(state.status)),
            "wait_input_reason": None,
            "snapshot_failure": None,
        },
    )


@router.post(
    "/{session_id}/snapshot-failure/terminate",
    dependencies=[Depends(require_snapshot_v2_enabled)],
)
async def snapshot_failure_terminate(session_id: str, request: Request):
    """Terminate the session on snapshot failure.

    Delegates to existing workflow teardown — does NOT create a second
    kernel teardown. Clears the failure gate and stops the workflow.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # Teardown pops the workflow, closes the kernel and deletes the graph
    # checkpoint. Running that under an in-flight Retry would yank the sandbox
    # out from under a live capture and then have the STOPPED write overwritten,
    # so wait for any Retry/Skip on this gate to finish first.
    async with get_snapshot_action_lock(session_id):
        return await _snapshot_failure_terminate_locked(session_id, user_id)


async def _snapshot_failure_terminate_locked(session_id: str, user_id: str):
    from backend.services.snapshot_v2.failure_gate import (
        terminate_snapshot_failure,
    )

    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if not is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail="Session is not blocked on a snapshot failure",
        )

    terminate_snapshot_failure(state)

    # After STOPPED + termination_reason: tear down workflow/kernel resources.
    # Do NOT pause — pause would clobber STOPPED → PAUSED on the same state.
    await teardown_running_workflow(session_id)

    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message="Session terminated due to snapshot failure",
        data={
            "status": getattr(state.status, "value", str(state.status)),
        },
    )
