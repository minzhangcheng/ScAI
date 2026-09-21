"""SingleCellAnalysisWorkflow execution method owners."""
from __future__ import annotations

from contextlib import AsyncExitStack


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


async def _ainvoke_durably(graph, value, config):
    """Do not enter the next graph step before its checkpoint is durable."""
    return await graph.ainvoke(value, config, durability="sync")


async def _latest_failure_state(self, config, fallback):
    """Keep committed results when a later workflow phase fails or stops."""
    from ...state import sessions as live_sessions

    live = live_sessions.get(self.session_id)
    if live is not None and live.status == TaskStatus.STOPPED:
        return live
    try:
        snapshot = await self.compiled_graph.aget_state(config)
        if snapshot and snapshot.values:
            return snapshot.values.get("analysis_state") or fallback
    except Exception:
        logger.warning("[%s] Could not read latest report checkpoint", self.session_id, exc_info=True)
    return fallback


def _restored_terminal_task_status(state: AnalysisState) -> TaskStatus:
    """Return the exact terminal outcome carried through lifecycle RUNNING."""
    payload = dict(getattr(state, "restore_resume_workflow_state", None) or {})
    raw_status = payload.get("terminal_status")
    if raw_status is None:
        current = getattr(state, "status", None)
        raw_status = getattr(current, "value", current)
    return {
        "completed": TaskStatus.COMPLETED,
        "stopped": TaskStatus.STOPPED,
        "failed": TaskStatus.FAILED,
    }.get(str(raw_status or "").lower(), TaskStatus.COMPLETED)


def _has_idle_restore_identity(state) -> bool:
    """Whether an idle/manual restore has a branch to hold.

    ``restore_branch_id`` is the current restore identity and the primary
    signal. ``last_restored_snapshot_id`` is accepted only as a compatibility
    fallback for sessions written before branch identity existed; durable
    origin never determines round arithmetic, plan resets, or cache resets.
    """
    if getattr(state, "restore_branch_id", None):
        return True
    return bool(getattr(state, "last_restored_snapshot_id", None))


def _has_fresh_start_intent(state) -> bool:
    """Whether lifecycle routing explicitly requested a new graph invocation."""
    return getattr(state, "restore_start_intent", None) in {
        "explicit_start",
        "completed_reanalysis",
    }


def _recursion_limit_for(self, analysis_state) -> int:
    """Size one invocation's superstep ceiling from the run's round budget.

    ``max_rounds <= 0`` means "use the configured default", the same resolution
    ``_check_convergence`` applies when it decides whether to stop.
    """
    from ..core import graph_recursion_limit

    rounds = int(getattr(analysis_state, "max_rounds", 0) or 0)
    if rounds <= 0:
        # Resume can reach the graph before __init__ ran (restore dispatch and
        # its tests build the workflow with __new__), so read the config
        # defensively rather than making the ceiling a hard dependency.
        rounds = int(getattr(getattr(self, "config", None), "max_rounds", 0) or 0)
    return graph_recursion_limit(rounds)


async def run(
    self,
    initial_state: AnalysisState,
    thread_id: str
) -> AnalysisState:
    """
    Run the complete workflow

    Args:
        initial_state: Initial analysis state
        thread_id: Thread ID for checkpointing

    Returns:
        Final analysis state
    """
    from backend.services.legacy_state_sanitizer import sanitize_legacy_state

    sanitize_legacy_state(initial_state, workflow_active=True)

    from ...utils.runtime_pairing import refresh_runtime_availability
    refresh_runtime_availability(initial_state)

    # §1d: never run the graph forward on a kernel that is inconsistent with
    # the restored state. Twin of the lifecycle run guard.
    if is_forward_resume_blocked(
        getattr(initial_state, "runtime_restore_status", "complete"),
        coherence=state_runtime_coherence(initial_state),
        recompute_pending=bool(
            getattr(initial_state, "runtime_recompute_pending", False)
        ),
    ):
        logger.warning(
            f"[{self.session_id}] run blocked: runtime_restore_status="
            f"{getattr(initial_state, 'runtime_restore_status', 'complete')!r} "
            f"coherence={state_runtime_coherence(initial_state)!r}; "
            "refusing to start the graph"
        )
        self._set_status(initial_state, TaskStatus.FAILED)
        return initial_state

        return initial_state

    # Guard: terminal restore → no workflow, preserve its exact outcome.
    if (
        initial_state.restore_resume_kind == "terminal"
        and not _has_fresh_start_intent(initial_state)
    ):
        terminal_status = _restored_terminal_task_status(initial_state)
        logger.info(
            f"[{self.session_id}] Terminal restore — preserving "
            f"{terminal_status.value} "
            f"(current_round={initial_state.current_round})"
        )
        self._set_status(initial_state, terminal_status)
        self._consume_restore_resume_metadata(initial_state)
        return initial_state

    # Guard: idle restore (branch_id set but no resume kind) → persist only.
    # The session was restored from a manual snapshot with no continuation
    # target; the user must explicitly start analysis from the UI.
    # Exception: restore_start_intent="explicit_start" means the user clicked
    # Start — let the graph run (restore_entry will preserve round/plan).
    if (
        initial_state.restore_resume_kind is None
        and _has_idle_restore_identity(initial_state)
        and not _has_fresh_start_intent(initial_state)
    ):
        logger.info(
            f"[{self.session_id}] Idle restore — no auto-start "
            f"(branch={initial_state.restore_branch_id})"
        )
        # Ensure status reflects the idle state — PAUSED so the UI
        # offers a "Start" / "Resume" button rather than "Analysis running".
        if initial_state.status == TaskStatus.RUNNING:
            self._set_status(initial_state, TaskStatus.PAUSED)
        self._consume_restore_resume_metadata(initial_state)
        return initial_state

    # Store state for settings provider initialization
    self._state = initial_state

    try:
        await self._send_message("status", "Initializing analysis system...")
        await self.initialize()
        logger.info(f"Workflow initialized for session {self.session_id}")
    except Exception as e:
        logger.exception("Failed to initialize workflow")
        await self._send_message("error", f"Initialization failed: {str(e)}")
        self._set_status(initial_state, TaskStatus.FAILED)
        return initial_state

    # Guard: restored wait-state snapshot → continuation, not fresh run.
    if (
        initial_state.restore_resume_kind == "semantic_gate"
        and initial_state.restore_resume_gate in SEMANTIC_WAIT_GATES
        and initial_state.status == TaskStatus.RUNNING
    ):
        return await self.resume_restored_wait_state(initial_state)

    # Guard: graph-node restore → seed workflow state and start graph.
    # This handles the case where a snapshot was taken at a specific graph
    # node boundary. The initial workflow state is built from the restore
    # context so the correct planning stage and round are carried forward.
    if initial_state.restore_resume_kind == "graph_node":
        logger.info(
            f"[{self.session_id}] Graph-node restore — starting graph "
            f"from node {initial_state.restore_resume_node} "
            f"(round={initial_state.current_round})"
        )

    # Create initial workflow state
    workflow_state = self._make_initial_workflow_state(initial_state)

    # Run workflow
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _recursion_limit_for(self, initial_state),
    }

    # Verify workflow was properly initialized
    if self.compiled_graph is None:
        error_msg = "Workflow not properly initialized - compiled_graph is None"
        logger.error(f"[{self.session_id}] {error_msg}")
        await self._send_message("error", error_msg)
        self._set_status(initial_state, TaskStatus.FAILED)
        return initial_state

    try:
        logger.info(f"Starting workflow execution for {thread_id}")
        final_state = await _ainvoke_durably(
            self.compiled_graph, workflow_state, config,
        )
        logger.info(f"Workflow completed for {thread_id}")
        return final_state["analysis_state"]

    except WorkflowSupersededError as e:
        # Deliberately does NOT touch session state: the restore already
        # persisted the authoritative state, and a terminal status written
        # here would overwrite exactly what the restore just installed.
        logger.info(f"[{self.session_id}] Run superseded by restore: {e}")
        return self._superseded_state(initial_state)

    except WorkflowTerminatedError as e:
        logger.warning(f"[{self.session_id}] Workflow terminated by user: {e}")
        self._set_status(initial_state, TaskStatus.STOPPED)
        initial_state.wait_input_reason = None
        await self._send_state_sync(initial_state, StateSyncChangeType.LLM_RETRY.value)
        return initial_state

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Workflow execution failed: {error_detail}")
        await self._send_message("error", f"Workflow execution failed: {str(e)}")
        self._set_status(initial_state, TaskStatus.FAILED)
        initial_state.add_message(
            AgentType.PI,
            MessageRole.SYSTEM,
            f"Workflow execution failed: {str(e)}"
        )
        return initial_state

async def generate_report(
    self,
    initial_state: AnalysisState,
    thread_id: str
) -> AnalysisState:
    """
    Run only the report generation phase, bypassing analysis rounds.
    Used when a session completed all analysis rounds but was interrupted
    before report generation finished.
    """
    from backend.services.legacy_state_sanitizer import sanitize_legacy_state

    if sanitize_legacy_state(initial_state, workflow_active=True):
        if initial_state.status == TaskStatus.COMPLETED:
            return initial_state

    # Store state for settings provider initialization
    self._state = initial_state

    # Consume any integration-gate finalize request: once we commit to
    # report-only generation the flag must be cleared, otherwise a later WS
    # reconnect would see final_report_requested=True and re-trigger report
    # generation on every reconnect. The live integration-wait path clears
    # this in _wait_for_user_input, but the no-workflow report-only path
    # bypasses that node.
    initial_state.consume_final_report_request()

    try:
        await self._send_message("status", "Resuming report generation (analysis already complete)...")
        await self.initialize()
        logger.info(f"[{self.session_id}] Workflow initialized for report-only generation")
    except Exception as e:
        logger.exception(f"[{self.session_id}] Failed to initialize for report generation")
        await self._send_message("error", f"Initialization failed: {str(e)}")
        self._set_status(initial_state, TaskStatus.FAILED)
        return initial_state

    # Create workflow state with existing analysis data
    config = {"configurable": {"thread_id": thread_id}}
    if self.compiled_graph is None:
        self._build_graph()

    workflow_state = None
    try:
        snapshot = await self.compiled_graph.aget_state(config)
        if snapshot and snapshot.values:
            workflow_state = snapshot.values
            workflow_state["analysis_state"] = initial_state
            workflow_state.pop("report_blocked", None)
            if initial_state.current_round > len(initial_state.round_results):
                logger.info(f"[{self.session_id}] Found incomplete round {initial_state.current_round}, running pi_synthesis before report")
                workflow_state = await self._pi_synthesis(workflow_state)
    except WorkflowSupersededError as e:
        return self._superseded_report_result(initial_state, e)
    except WorkflowTerminatedError:
        self._set_status(initial_state, TaskStatus.STOPPED)
        initial_state.wait_input_reason = None
        await self._send_state_sync(initial_state, StateSyncChangeType.LLM_RETRY.value)
        return initial_state
    except Exception as e:
        logger.warning(f"[{self.session_id}] Could not load state from checkpointer: {e}")

    if workflow_state is None:
        workflow_state = {
            "analysis_state": initial_state,
            "current_round": initial_state.current_round,
            "pi_instructions": {},
            "pathology_result": None,
            "bioinformatics_result": None,
            "biochemistry_result": None,
            "literature_result": None,
            "round_result": None,
            "should_continue": False,
            "error": None,
            "messages": []
        }

    try:
        logger.info(f"[{self.session_id}] Running report generation only (skipping analysis)")
        # Report-only execution uses the same durable graph as ordinary runs.
        # Otherwise failures are recorded only in session.json while the graph
        # still points at an unrelated analysis node (or END).
        entry_state = {
            **workflow_state,
            "report_blocked": False,
            "failed_node": None,
            "_restore_entry_plan": {
                "intent": "graph_node", "target_node": "generate_final_report",
            },
        }
        entry_state.pop("messages", None)  # This channel appends on update.
        self._current_analysis_state = initial_state
        self._current_graph_state = entry_state
        await self.compiled_graph.aupdate_state(config, entry_state, as_node="restore_entry")
        state = await _ainvoke_durably(self.compiled_graph, None, config)
        logger.info(f"[{self.session_id}] Report generation completed")
        return state["analysis_state"]
    except WorkflowSupersededError as e:
        return self._superseded_report_result(initial_state, e)
    except WorkflowTerminatedError:
        effective = await _latest_failure_state(self, config, initial_state)
        self._set_status(effective, TaskStatus.STOPPED)
        effective.wait_input_reason = None
        await self._send_state_sync(effective, StateSyncChangeType.LLM_RETRY.value)
        return effective
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"[{self.session_id}] Report generation failed: {error_detail}")
        await self._send_message("error", f"Report generation failed: {str(e)}")
        effective = await _latest_failure_state(self, config, initial_state)
        self._set_status(effective, TaskStatus.FAILED)
        return effective

def _superseded_state(self, *fallbacks: Optional[AnalysisState]) -> Optional[AnalysisState]:
    """The state a superseded run must return, and the flag it must drop.

    Shared tail of every ``WorkflowSupersededError`` handler in this class.
    The restore has already persisted the authoritative object, so the live
    session wins over whatever this run was holding; the fallbacks are tried
    in order only when it is gone. The session-scoped superseded flag is
    cleared because this run IS the superseded one and it has now unwound —
    forgetting that in one handler leaks the flag across the process.
    """
    from ...state import sessions as live_sessions

    state = live_sessions.get(self.session_id)
    for fallback in fallbacks:
        if state:
            break
        state = fallback
    return state

def _superseded_report_result(
    self, initial_state: AnalysisState, exc: BaseException,
) -> AnalysisState:
    """Unwind a report run that a committed restore superseded.

    `generate_report` reaches acknowledgement sites (``_pi_synthesis``'s
    node-entry hook, the post-report capture's handover), so it can be
    frozen on a restore safe point and woken with its state already
    replaced. Both of its generic handlers used to absorb that: the
    checkpointer-load one mislabelled it as "could not load state from
    checkpointer" and kept generating — producing reports and a post-report
    snapshot from the state the restore had just discarded — and the
    trailing one wrote FAILED, which is exactly what WorkflowSupersededError
    forbids (see helpers.py).

    Deliberately does not touch session state: the restore already persisted
    the authoritative object. Returning that object rather than
    ``initial_state`` keeps a caller that saves the result from writing the
    pre-restore state back. The session-scoped superseded flag is cleared
    here because this run IS the superseded one and it has now unwound; no
    report path did that, so the flag used to outlive the process.
    """
    logger.info(
        f"[{self.session_id}] Report generation superseded by restore: {exc}"
    )
    # Keep a workflow-scoped record before dropping the session-scoped one.
    # The report runner's `finally` calls close(), which reads both to decide
    # `save=not superseded`; a report workflow is never registered in
    # `state.workflows`, so `_retire_superseded_workflow` cannot set the flag
    # on it and close() would otherwise start a kernel purely to re-dump the
    # runtime the restore just installed and verified.
    self._restore_superseded = True
    return self._superseded_state(initial_state)

async def _invoke_restore_entry(
    self, analysis_state: AnalysisState, thread_id: str,
) -> AnalysisState:
    self._set_status(analysis_state, TaskStatus.RUNNING)
    self._current_analysis_state = analysis_state
    workflow_state = self._make_initial_workflow_state(
        analysis_state,
        current_round=analysis_state.current_round,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _recursion_limit_for(self, analysis_state),
    }
    final_state = await _ainvoke_durably(
        self.compiled_graph, workflow_state, config,
    )
    return final_state["analysis_state"]

async def resume(
    self,
    thread_id: str,
    analysis_state: Optional[AnalysisState] = None
) -> AnalysisState:
    """Resume a paused workflow

    Args:
        thread_id: Thread ID for checkpointing
        analysis_state: Optional analysis state to restore custom config from
    """
    logger.info(f"[{self.session_id}] Resuming workflow for thread {thread_id}")

    if analysis_state is not None:
        from backend.services.legacy_state_sanitizer import sanitize_legacy_state

        sanitize_legacy_state(analysis_state, workflow_active=True)

    from ...utils.runtime_pairing import refresh_runtime_availability
    live_continuation = (
        is_live_runtime_continuation_safe(
            self.session_id, analysis_state, self,
        )
        if analysis_state is not None
        else False
    )
    if analysis_state is not None and not live_continuation:
        refresh_runtime_availability(analysis_state)

    # §1d: never run the graph forward on a kernel that is inconsistent with
    # the restored state. This is the workflow-level twin of the lifecycle
    # resume block and covers the WebSocket-reconnect resume path.
    if (
        analysis_state is not None
        and is_forward_resume_blocked(
            getattr(analysis_state, "runtime_restore_status", "complete"),
            coherence=state_runtime_coherence(analysis_state),
            recompute_pending=bool(
                getattr(analysis_state, "runtime_recompute_pending", False)
            ),
        )
        and not live_continuation
    ):
        logger.warning(
            f"[{self.session_id}] resume blocked: runtime_restore_status="
            f"{getattr(analysis_state, 'runtime_restore_status', 'complete')!r} "
            f"coherence={state_runtime_coherence(analysis_state)!r}; "
            "holding without running the graph"
        )
        # Holding must not leave the session RUNNING. It used to: a held
        # WebSocket reconnect persisted RUNNING, POST /run then answered 400
        # "Analysis already running" and /resume acts only on PAUSED, so the
        # session was reachable only through Pause. PAUSED keeps the one action
        # that can actually resolve this -- re-running from the input data --
        # available, and the run path's twin of this guard likewise sets a
        # status rather than returning untouched.
        if analysis_state.status == TaskStatus.RUNNING:
            self._set_status(analysis_state, TaskStatus.PAUSED)
        return analysis_state

    # Terminal restore → no workflow, preserve its exact outcome.
    if (
        analysis_state is not None
        and analysis_state.restore_resume_kind == "terminal"
        and not _has_fresh_start_intent(analysis_state)
    ):
        terminal_status = _restored_terminal_task_status(analysis_state)
        logger.info(
            f"[{self.session_id}] Terminal restore in resume — preserving "
            f"{terminal_status.value} "
            f"(round={analysis_state.current_round})"
        )
        self._set_status(analysis_state, terminal_status)
        self._consume_restore_resume_metadata(analysis_state)
        return analysis_state

    # Re-initialize agents and components if not already initialized
    if self.compiled_graph is None:
        # Restore state for settings provider initialization
        if analysis_state:
            self._state = analysis_state
        logger.info(f"[{self.session_id}] Re-initializing workflow components for resume")
        await self.initialize()

    # A restore Retry may reconstruct the workflow after its new job has
    # already requested a safe point. Acknowledge before touching the
    if analysis_state is not None:
        from ...state import sessions as live_sessions

        refreshed_state = live_sessions.get(self.session_id)
        if refreshed_state is not None:
            analysis_state = refreshed_state
            self._state = refreshed_state

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _recursion_limit_for(self, analysis_state),
    }

    # Get current state
    state = await self.compiled_graph.aget_state(config)

    retry_request = (
        (analysis_state.sandbox_state or {}).get("node_retry")
        if analysis_state is not None else None
    )
    if retry_request:
        from ...constants import WORKFLOW_GRAPH_NODES

        request_id = retry_request.get("id")
        if not request_id:
            raise ValueError("Node retry request has no identity")
        # Older report-only calls bypassed the graph. Their host-written
        # detailed-report failure marker is the exact continuation point,
        # including when that path never created a graph checkpoint.
        report_retry = (
            (analysis_state.sandbox_state or {}).get("failed_node")
            == "generate_detailed_report"
            and retry_request.get("node_name") in (None, "generate_detailed_report")
            and not retry_request.get("skip")
        )
        if report_retry and (not state or not state.values):
            seed = self._make_initial_workflow_state(
                analysis_state, current_round=analysis_state.current_round,
            )
            seed["_restore_entry_plan"] = {"intent": "idle"}
            seed.pop("messages", None)
            await self.compiled_graph.aupdate_state(config, seed, as_node="restore_entry")
            state = await self.compiled_graph.aget_state(config)
        if not state or not state.values:
            raise ValueError("Cannot retry a node without its saved workflow checkpoint")
        retry_state = analysis_state.model_copy(deep=True)
        retry_state.sandbox_state.pop("node_retry", None)
        if state.values.get("node_retry_id") != request_id:
            failed_node = state.values.get("failed_node")
            pending_nodes = tuple(state.next or ())
            target = retry_request.get("node_name") or (
                "generate_detailed_report" if report_retry else failed_node
            )
            if not target and len(pending_nodes) == 1:
                target = pending_nodes[0]
            if (
                target not in WORKFLOW_GRAPH_NODES
                or target == "restore_entry"
                or (target != failed_node and target not in pending_nodes and not report_retry)
            ):
                raise ValueError("Retry target does not match the failed or pending checkpoint node")
            skip_nodes = list(state.values.get("skip_nodes") or [])
            if retry_request.get("skip"):
                handler = getattr(self, f"_{target}", None)
                if getattr(handler, "_retry_node_name", None) != target:
                    raise ValueError("The failed node does not support Skip")
                if target not in skip_nodes:
                    skip_nodes.append(target)
            else:
                skip_nodes = [node for node in skip_nodes if node != target]
            retry_state.sandbox_state.pop("failed_node", None)
            # Skip decisions belong to the current graph round, not session state.
            retry_state.sandbox_state.pop("skip_nodes", None)
            retry_state.status = TaskStatus.RUNNING
            # Rearm the existing checkpoint through the graph's registered
            # routing entry, retaining the round, plan, feedback and results.
            # The token and scheduled successor commit together, so a stale
            # session.json retry intent cannot replay this action after a crash.
            await self.compiled_graph.aupdate_state(config, {
                "analysis_state": retry_state,
                "node_retry_id": request_id,
                "failed_node": None,
                "should_continue": True,
                "error": None,
                "skip_nodes": skip_nodes,
                "_restore_entry_plan": {"intent": "graph_node", "target_node": target},
            }, as_node="restore_entry")
            state = await self.compiled_graph.aget_state(config)
        else:
            # The checkpoint may already have progressed beyond this request.
            # Never replace its results with the stale session retry payload.
            retry_state = state.values["analysis_state"].model_copy(deep=True)
            retry_state.sandbox_state.pop("node_retry", None)
        analysis_state = retry_state

    # Graph already at END (no pending nodes) — nothing to resume.
    # Continuing would inject stale RUNNING status and ainvoke(None)
    # returns it unchanged, causing an infinite resume loop on every
    # WebSocket reconnect (kernel restart → reconnect → resume → RUNNING).
    if state and state.values and not state.next:
        effective = state.values.get("analysis_state") or analysis_state
        terminal_statuses = {
            TaskStatus.COMPLETED,
            TaskStatus.STOPPED,
            TaskStatus.FAILED,
        }
        if effective and effective.status not in terminal_statuses:
            self._set_status(effective, TaskStatus.COMPLETED)
            logger.info(
                f"[{self.session_id}] Resume requested but graph already "
                "terminated; marking complete"
            )
        else:
            logger.info(
                f"[{self.session_id}] Resume requested but graph already "
                f"terminated; preserving status={getattr(effective, 'status', None)}"
            )
        return effective

    if state and state.values:
        # Restore _current_analysis_state for callbacks
        self._current_analysis_state = state.values.get("analysis_state")
        self._current_graph_state = state.values

        # Inject latest analysis_state if provided (so user changes during pause are picked up)
        # An explicit retry already committed its state and successor atomically.
        # Another unqualified update can infer the old terminal node and erase
        # that successor, or be ambiguous on a newly seeded checkpoint.
        if analysis_state and not retry_request:
            await self.compiled_graph.aupdate_state(config, {"analysis_state": analysis_state})
            self._current_analysis_state = analysis_state
            # Keep in-memory graph state in sync with the updated checkpoint
            self._current_graph_state = {**state.values, "analysis_state": analysis_state}

        # The graph checkpoint owns round-scoped Skip decisions. Legacy
        # session-level lists have no round identity and must not overwrite it.

        # Resume from checkpoint
        logger.info(f"[{self.session_id}] Found saved state, resuming execution")
        try:
            final_state = await _ainvoke_durably(
                self.compiled_graph, None, config,
            )
            logger.info(f"[{self.session_id}] Workflow resumed and completed")
            return final_state["analysis_state"]
        except WorkflowSupersededError as e:
            logger.info(f"[{self.session_id}] Resume superseded by restore: {e}")
            return self._superseded_state(
                analysis_state, state.values.get("analysis_state")
            )

        except WorkflowTerminatedError as e:
            logger.warning(f"[{self.session_id}] Workflow terminated by user: {e}")
            effective = await _latest_failure_state(
                self, config, analysis_state or state.values.get("analysis_state"),
            )
            if effective:
                self._set_status(effective, TaskStatus.STOPPED)
                effective.wait_input_reason = None
            return effective or analysis_state or state.values["analysis_state"]

    # No checkpoint found. Exact continuation only — never fall back to a
    # generic fresh run from initialize_round.
    if analysis_state is None:
        logger.error(f"[{self.session_id}] No checkpoint and no analysis_state for {thread_id}")
        raise ValueError(f"No saved state found for thread {thread_id}")

    if (
        analysis_state.restore_resume_kind == "semantic_gate"
        and analysis_state.restore_resume_gate in SEMANTIC_WAIT_GATES
        and analysis_state.status == TaskStatus.RUNNING
    ):
        return await self.resume_restored_wait_state(analysis_state)

    if analysis_state.restore_resume_kind == "graph_node":
        logger.info(
            f"[{self.session_id}] Graph-node restore without checkpoint — "
            f"seeding exact next node {analysis_state.restore_resume_node} "
            f"(round={analysis_state.current_round})"
        )
        self._set_status(analysis_state, TaskStatus.RUNNING)
        self._current_analysis_state = analysis_state
        workflow_state = self._make_initial_workflow_state(
            analysis_state, current_round=analysis_state.current_round
        )
        if analysis_state.restore_resume_node:
            workflow_state["_restore_resume_node"] = analysis_state.restore_resume_node
        try:
            final_state = await _ainvoke_durably(
                self.compiled_graph, workflow_state, config,
            )
            logger.info(f"[{self.session_id}] Exact graph-node continuation completed")
            self._consume_restore_resume_metadata(final_state["analysis_state"])
            return final_state["analysis_state"]
        except WorkflowSupersededError as e:
            logger.info(f"[{self.session_id}] Resume superseded by restore: {e}")
            return self._superseded_state(analysis_state)

        except WorkflowTerminatedError as e:
            logger.warning(f"[{self.session_id}] Workflow terminated by user: {e}")
            effective = await _latest_failure_state(self, config, analysis_state)
            if effective:
                self._set_status(effective, TaskStatus.STOPPED)
                effective.wait_input_reason = None
            return effective

    if analysis_state.status in (TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION):
        logger.info(
            f"[{self.session_id}] Status is {analysis_state.status.value}, "
            f"deferring to HITL-hold path"
        )
        return analysis_state

    if _has_fresh_start_intent(analysis_state):
        return await self._invoke_restore_entry(analysis_state, thread_id)

    # Idle / unknown: do not invent a fresh initialize_round run.
    logger.info(
        f"[{self.session_id}] No checkpoint and no exact continuation target "
        f"(kind={analysis_state.restore_resume_kind}); holding without fresh run "
        f"(current_round={analysis_state.current_round})"
    )
    if analysis_state.status == TaskStatus.RUNNING:
        self._set_status(analysis_state, TaskStatus.PAUSED)
    self._consume_restore_resume_metadata(analysis_state)
    return analysis_state

async def pause(
    self,
    thread_id: str,
    analysis_state: Optional[AnalysisState] = None,
) -> Dict[str, Any]:
    """Pause the workflow and save sandbox state including Python session.

    Clears the cooperative pause gate FIRST so the bioinformatics agent's
    autonomous loop and the @with_retry wrapper block at their next check,
    then waits for any in-flight sandbox.execute to drain via the shared
    _exec_lock (see LocalJupyterSandbox._init_exec_lock) before persisting state.
    """
    logger.info(f"[{self.session_id}] Pausing workflow for thread {thread_id}")
    # Block waiters before doing anything else — once cleared, the next time
    # the agent loop or a node entry checks the event it will await here.
    self._pause_event.clear()
    # Also clear the HITL gate the workflow is actually blocked on, so
    # resume() can re-set the correct one.  Without this, resume() only
    # sets _pause_event but the workflow stays stuck on the HITL event.
    self._continue_event.clear()
    self._annotation_event.clear()
    self._plan_review_event.clear()
    self._species_event.clear()
    self._species_user_response = None

    config = {"configurable": {"thread_id": thread_id}}
    state = await self.compiled_graph.aget_state(config)

    if state and state.values:
        analysis_state = analysis_state or state.values["analysis_state"]
    if analysis_state is None:
        # Internal callers without an AnalysisState still receive the
        # cooperative-gate result. HTTP lifecycle callers always pass the
        # authoritative object and therefore take the durable path below.
        return {
            "success": True,
            "status": TaskStatus.PAUSED.value,
            "still_blocked": False,
        }

    # The tail lock covers workflow-owned review/follow-up work after the
    # autonomous loop; the turn lock covers one autonomous LLM/tool turn.
    # Acquire in that order so a nested figure-followup run can drain while its
    # enclosing tail phase owns the first lock. Retry gates also drain while
    # either phase is active, so neither phase can wait on the pause event while
    # pause() is waiting for its lock.
    tail_lock = getattr(self, "_pause_bioinformatics_tail_lock", None)
    turn_lock = getattr(
        getattr(self, "bioinformatics_agent", None), "_pause_turn_lock", None,
    )
    async with AsyncExitStack() as pause_boundary:
        if tail_lock is not None:
            await pause_boundary.enter_async_context(tail_lock)
        if turn_lock is not None:
            await pause_boundary.enter_async_context(turn_lock)

        # The first graph checkpoint may not exist yet.  The lifecycle state is
        # nevertheless authoritative and must record the pause before success
        # is exposed; otherwise the cleared cooperative gate leaves a durable
        # RUNNING session that can never advance.
        self._set_status(analysis_state, TaskStatus.PAUSED)

        # Save sandbox state including Python session using dill. The Jupyter
        # kernel is single-threaded, so this queues behind any in-flight
        # execute() and captures a clean post-step snapshot.
        if self.sandbox:
            try:
                saved = await self.sandbox.save_session()
                persistence = self.sandbox.sync_runtime_persistence_state(
                    analysis_state
                )
                if not saved or persistence.get("status") == "failed":
                    raise RuntimeError(
                        "runtime persistence did not commit a current generation"
                    )
            except Exception:
                logger.warning(
                    "[%s] Pause could not persist the runtime; keeping the "
                    "previous committed generation",
                    self.session_id,
                    exc_info=True,
                )
                # The pause could not persist the runtime. The previous committed
                # generation stands, coherence reports the state as stale, and the
                # UI carries a persistent non-blocking warning (spec 5.1). Arming a
                # dialog and holding the pause is the gate this replaced.
                try:
                    self.sandbox.sync_runtime_persistence_state(analysis_state)
                except Exception:
                    logger.warning(
                        "[%s] Could not synchronise runtime persistence after a "
                        "failed pause save",
                        self.session_id,
                        exc_info=True,
                    )
            # Independent of the dump (spec 5.1): a failed runtime save must not
            # take the notebook with it.
            try:
                self.sandbox.notebook_state.save()
            except Exception:
                logger.warning(
                    "[%s] Notebook save failed while pausing",
                    self.session_id,
                    exc_info=True,
                )
            logger.info(f"[{self.session_id}] Sandbox pause persistence attempted")

    logger.info(f"[{self.session_id}] Workflow paused successfully")
    return {
        "success": True,
        "status": analysis_state.status.value,
        "still_blocked": False,
    }
