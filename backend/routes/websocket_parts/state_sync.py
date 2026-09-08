"""WebSocket state sync owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'_in_integration_loop', '_serialize_realtime_messages', '_send_restore_state_sync', '_preserve_realtime_stream', 'should_resume_workflow', '_resend_wait_input_state_sync', '_needs_report_only', '_session_workflow_activity'})

bindings = None


def bind_facade(facade) -> None:
    global bindings
    bindings = facade
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


def _session_workflow_activity(
    session_id: str,
) -> tuple[bool, bool, bool, bool]:
    """Classify graph and report-only owners for create-or-observe routing."""
    from backend.services.workflow_runner import _report_tasks

    report_task = _report_tasks.get(session_id)
    report_task_running = bool(
        report_task is not None and not report_task.done()
    )
    report_workflow_running = session_id in report_workflows
    detailed_report_running = bool(
        (detailed_task := detailed_report_tasks.get(session_id)) is not None
        and not detailed_task.done()
    )
    analysis_task_running = bool(
        (analysis_task := _run_tasks.get(session_id)) is not None
        and not analysis_task.done()
    )
    workflow_running = bool(
        session_id in workflows
        or report_workflow_running
        or analysis_task_running
    )
    workflow_initializing = bool(
        session_id in initializing_sessions
        or report_task_running
        or detailed_report_running
    )
    return (
        workflow_running,
        workflow_initializing,
        report_task_running,
        report_workflow_running,
    )

def _needs_report_only(state, effective_max_rounds: int, in_integration_loop: bool) -> bool:
    """Whether this attach should generate the report instead of running the graph.

    Scoped to a session that completed its analysis and was interrupted during
    report generation — not to a user-initiated run. An armed
    ``restore_start_intent`` is exactly that ask: ``POST /run`` sets it as the
    "re-run from your data" the blocked-resume message promises, and import arms it
    for a checkpoint-less session. Answering it with a report fabricated a final
    report out of the imported rounds, cleared the runtime warning on the way in,
    and left ``runtime_recompute_pending`` unconsumed — only the bioinformatics
    node clears it — so the finished session's one remaining trace of an empty
    kernel was that flag.

    Suppressing report-only is sufficient on its own: ``_should_continue`` tests
    ``current_round >= max_rounds`` only *after* a round body runs, so the
    preserved round re-executes and then routes to the report exactly once.
    Bumping ``max_rounds`` instead would run an extra round, because
    ``preserve_round_on_initialize`` keeps ``current_round`` where it is.

    ``final_report_requested`` keeps precedence: that is the explicit
    "write the report" intent, and it is already exempt from the integration-loop
    guard.
    """
    if getattr(state, "final_report_requested", False):
        return True
    if getattr(state, "restore_start_intent", None) == "explicit_start":
        return False
    return (
        state.status in (TaskStatus.RUNNING, TaskStatus.WAITING_INPUT,
                         TaskStatus.FAILED, TaskStatus.GENERATING_REPORT)
        and state.current_round >= effective_max_rounds
        and len(state.round_results) >= effective_max_rounds
        and not in_integration_loop
    )

def _in_integration_loop(state) -> bool:
    # User explicitly chose to write the final report — not in the loop.
    if getattr(state, "final_report_requested", False):
        return False
    # Cheap boolean checks first; only fall back to the O(N) batch scan last.
    if getattr(state, "pending_integration", False):
        return True
    if state.analysis_phase == AnalysisPhase.INTEGRATION:
        return True
    if getattr(state, "integration_gate_candidates", None):
        return True
    return active_batch_incomplete(state)

def should_resume_workflow(state) -> bool:
    """Whether this connection continues an existing run or starts a fresh one.

    Named rather than inlined because it is the point where an imported session
    is committed to a path: ``resume()`` refuses to invent a fresh
    ``initialize_round`` run and moves only on an explicit restore intent, while
    ``run()`` always starts one. A session that has executed at least one round
    must take the former even after a runtime-less import, and one that never
    got past round 0 must take the latter, or it holds without executing.
    """
    return state.current_round > 0 or state.status in (
        TaskStatus.PAUSED, TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION
    )

async def _send_restore_state_sync(websocket, state, session_id: str) -> None:
    """Emit a state_sync after a restored session connects.

    Communicates the restore kind and branch identity to the frontend so it
    can render the appropriate banner/dialog (graph-node restore, terminal
    restore, idle restore, etc.).
    """
    kind = getattr(state, "restore_resume_kind", None)
    if kind is None:
        return
    branch_id = getattr(state, "restore_branch_id", None) or ""
    payload = {
        "change_type": f"restore_{kind}",
        "version": getattr(state, "state_version", 0),
        "status": state.status.value if hasattr(state.status, "value") else state.status,
        "current_round": getattr(state, "current_round", 0),
        "analysis_phase": getattr(state, "analysis_phase", None),
        "timestamp": datetime.now().isoformat(),
        "restore_resume_kind": kind,
        "restore_resume_node": getattr(state, "restore_resume_node", None),
        "restore_resume_gate": getattr(state, "restore_resume_gate", None),
        "restore_branch_id": branch_id,
        "restore_branch_label": getattr(state, "restore_branch_label", None) or "",
    }
    try:
        message = {
            "type": "state_sync",
            "agent": None,
            "content": payload,
            "timestamp": datetime.now().isoformat(),
        }
        if isinstance(websocket, WebSocketConnection):
            sent = await bindings.send_connection_json(websocket, message)
        else:  # Backward-compatible test/helper caller.
            await websocket.send_json(message)
            sent = True
        if sent:
            logger.info(
                f"[{session_id}] Sent restore state_sync for kind={kind} "
                f"branch_id={branch_id}"
            )
    except Exception:
        logger.warning(f"[{session_id}] Failed to send restore state_sync", exc_info=True)

async def _resend_wait_input_state_sync(websocket, state, session_id: str) -> None:
    """Re-emit state_sync for pending HITL gates after a reconnect.

    The LlmTimeoutDialog and disabled-agent dialog are driven exclusively by
    state_sync messages (useApi.ts). On page reload, the workflow may already
    be parked in _wait_for_llm_intervention with llm_failure_info set, but the
    reconnect path only replays message_history — which does not feed the
    state_sync handler. Without this resend the dialog never reopens.

    """
    reason = getattr(state, "wait_input_reason", None)
    _HITL_RESEND_REASONS = (
        "llm_failure",
        "agent_disabled",
        "marker_db_missing",
        "species_input",
        "gene_species_mismatch",
    )
    if reason not in _HITL_RESEND_REASONS:
        return
    change_type = (
        "llm_failure" if reason == "marker_db_missing" else reason
    )
    payload = {
        "change_type": change_type,
        "version": getattr(state, "state_version", 0),
        "status": state.status.value if hasattr(state.status, "value") else state.status,
        "current_round": getattr(state, "current_round", 0),
        "analysis_phase": getattr(state, "analysis_phase", None),
        "timestamp": datetime.now().isoformat(),
        "unresolved_errors": len(state.get_unresolved_errors()) if hasattr(state, "get_unresolved_errors") else 0,
        "wait_input_reason": reason,
        "llm_failure_info": getattr(state, "llm_failure_info", None),
        "agent_disabled_info": getattr(state, "agent_disabled_info", None),
        "species_input_info": getattr(state, "species_input_info", None),
    }
    try:
        message = {
            "type": "state_sync",
            "agent": None,
            "content": payload,
            "timestamp": datetime.now().isoformat(),
        }
        if isinstance(websocket, WebSocketConnection):
            sent = await bindings.send_connection_json(websocket, message)
        else:  # Backward-compatible test/helper caller.
            await websocket.send_json(message)
            sent = True
        if sent:
            logger.info(f"[{session_id}] Re-sent state_sync for pending {reason or change_type} on reconnect")
    except Exception:
        logger.warning(f"[{session_id}] Failed to resend wait-input state_sync", exc_info=True)

def _preserve_realtime_stream(prev, new) -> None:
    """Carry the fuller realtime-message stream from the outgoing cached state
    object onto the incoming one before it becomes canonical.

    The running workflow replaces sessions[session_id] with fresh AnalysisState
    objects (LangGraph checkpoint round-trips). Only the WebSocket
    message_callback ever appends to realtime_messages, and it appends to the
    *currently cached* object, so the cached (prev) object always holds the
    fuller stream while the incoming graph object carries a stale/short copy.
    Move the longer stream forward so /load and reconnect keep serving the full
    history instead of the handful of messages that predate the first swap.
    """
    if (
        prev is not None
        and prev is not new
        and len(prev.realtime_messages) > len(new.realtime_messages)
    ):
        new.realtime_messages = prev.realtime_messages

def _serialize_realtime_messages(messages):
    """Serialize a list of WebSocketMessage to JSON-safe dicts."""
    return [serialize_realtime_message(msg, strip_binary=True) for msg in messages]
