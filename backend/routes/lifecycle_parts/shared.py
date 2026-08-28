"""Lifecycle shared action owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'resolve_hitl_skip_target', '_resume_workflow', '_close_hitl_socket_for_redrive', '_pause_and_restart_ws'})

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


async def _resume_workflow(session_id, state, signal_fn=None):
    """Transition state to RUNNING, persist, and optionally signal the workflow."""
    await bindings._update_session_cache(session_id, state)
    await bindings.async_save_session(session_id, state)
    async with _state_lock:
        workflow = bindings.workflows.get(session_id)
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
    await _pause_and_restart_ws_impl(session_id, state)

async def _close_hitl_socket_for_redrive(session_id: str, reason: str) -> None:
    """Close a parked HITL socket so the frontend reconnects and re-drives.

    A workflow-boundary gate unwound the graph, so nothing is running and the
    re-drive only starts when a WebSocket reconnect rebuilds the workflow. The
    client holding the HITL connection never reconnects on its own, so close it
    with the same 4000 code the confirm/choose endpoints use to trigger the
    frontend's auto-reconnect. Skipped when a workflow is live: the capture then
    already ran in-process and that run owns the socket.
    """
    async with _state_lock:
        workflow_live = session_id in bindings.workflows
    if not workflow_live:
        await close_analysis_websocket(session_id, code=4000)
