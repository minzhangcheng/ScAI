"""Lifecycle interventions action owners."""
from __future__ import annotations

from backend.routes._helpers import (
    mark_gate_action_received,
    send_gate_action_received,
)

_OWNED_NAMES = frozenset({'_retry_node_locked', '_llm_failure_action_locked', 'retry_node', 'llm_failure_action', 'get_failed_node', 'resume_bioinformatics'})

bindings = None


def bind_facade(facade) -> None:
    global bindings
    bindings = facade
    globals().update(
        {
            name: value
            for name, value in vars(facade).items()
            if not name.startswith("__") and name not in _OWNED_NAMES
        }
    )


async def llm_failure_action(session_id: str, request: Request, req: LlmFailureActionRequest):
    """Handle user's choice for an LLM timeout/failure.

    Idempotency: the `generation` field must match the workflow's
    current `_llm_failure_generation`. Duplicate or stale requests
    (e.g. double-click) are rejected with 409.
    """
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent pause/resume/stop mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _llm_failure_action_locked(session_id, user_id, req)

async def _llm_failure_action_locked(session_id: str, user_id: str, req: LlmFailureActionRequest):
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

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

    workflow = bindings.workflows.get(session_id)

    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    ok = workflow.signal_llm_failure_action(req.node_name, req.action, req.generation)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Stale or duplicate request (generation {req.generation} != current)"
        )

    await mark_gate_action_received(state)
    await send_gate_action_received(
        session_id, state, f"llm_{req.action}",
    )

    return APIResponse(
        success=True,
        message=f"LLM failure action '{req.action}' applied to node '{req.node_name}'"
    )

async def retry_node(session_id: str, request: Request, body: RetryNodeRequest):
    """Retry or skip a failed node"""
    bindings.validate_session_id(session_id)
    async with get_session_lock(session_id):
        return await _retry_node_locked(session_id, request, body)


async def _retry_node_locked(session_id: str, request: Request, body: RetryNodeRequest):
    from uuid import uuid4
    from backend.constants import WORKFLOW_GRAPH_NODES
    from backend.workflow import SingleCellAnalysisWorkflow

    node_name = body.node_name
    skip = body.skip

    user_id = getattr(request.state, "user_id", "")
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    if state.status != TaskStatus.FAILED:
        raise HTTPException(status_code=400, detail="Session is not in failed state")

    if node_name not in WORKFLOW_GRAPH_NODES or node_name == "restore_entry":
        raise HTTPException(status_code=400, detail="Unknown retry target")
    handler = getattr(SingleCellAnalysisWorkflow, f"_{node_name}", None)
    if skip and getattr(handler, "_retry_node_name", None) != node_name:
        raise HTTPException(
            status_code=400,
            detail="This node has no skip action. Retry it or explicitly stop the analysis.",
        )

    # Save an explicit target, not just a status change. The workflow verifies
    # it against its checkpoint before rearming an already-ended graph.
    state.sandbox_state = {
        **(state.sandbox_state or {}),
        "node_retry": {"id": uuid4().hex, "node_name": node_name, "skip": skip},
    }

    if skip:
        message = f"Node {node_name} will be skipped on resume"
    else:
        # Clear failed status for retry
        state.status = TaskStatus.PAUSED  # Allow resumption
        message = f"Session ready to retry from node {node_name}"

    await bindings.async_save_session(session_id, state, force=True)

    return APIResponse(
        success=True,
        message=message,
        data={"node": node_name, "action": "skip" if skip else "retry"}
    )

async def get_failed_node(session_id: str, request: Request):
    """Get information about the failed node"""
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    # Check if we have workflow state with error info
    if session_id in bindings.workflows:
        workflow = bindings.workflows[session_id]
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

async def resume_bioinformatics(session_id: str, request: Request):
    """Resume bioinformatics analysis from last checkpoint"""
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    # Check for any recognized legacy/scoped executor transcript.  Scoped
    # runs deliberately coexist, so this endpoint only reports the newest
    # available position rather than assuming the legacy filename.
    from backend.agents.bioinformatics.executor_checkpoint import (
        latest_executor_checkpoint_path,
    )

    checkpoint_path = latest_executor_checkpoint_path(
        get_session_states(session_id), session_id=session_id,
    )
    if checkpoint_path is None:
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
