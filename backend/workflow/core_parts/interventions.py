"""SingleCellAnalysisWorkflow interventions method owners."""
from __future__ import annotations

from ...models.schemas import TaskStatus


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


# Bounded auto-retries for a HITL gate nobody answered.  The gate is only
# reached after the LLM client already exhausted its own retries, so this is a
# second-order budget: enough for a transient upstream to clear, small enough
# that an automated run cannot spend hours re-running an expensive node.
UNATTENDED_GATE_MAX_AUTO_RETRIES = 2


async def _unattended_gate_backoff(event, attempt: int) -> None:
    """Space out unattended auto-retries without going deaf.

    Retrying instantly proves nothing about an upstream that was still degraded
    a moment ago, and every attempt re-runs the caller's work.  The gate stays
    answerable throughout: ``signal_llm_failure_action`` accepts an answer in
    this window and reports success to the caller, so wait on the event rather
    than sleeping through an answer the user was told had landed.
    """
    from ..helpers import _retry_backoff_delay

    try:
        await asyncio.wait_for(event.wait(), _retry_backoff_delay(attempt))
    except asyncio.TimeoutError:
        pass


def _is_non_recoverable_llm_failure(error) -> bool:
    """True when the LLM client already ruled this failure out for retry.

    ``retry_with_backoff`` breaks out of its loop without retrying on auth,
    permission and hard-quota errors, keeping the original as ``__cause__``.
    Re-running the node cannot fix a rejected key or an exhausted budget, and
    in an unattended run nobody is present to replace either.
    """
    from ...utils.llm_client import (
        AuthenticationError,
        PermissionDeniedError,
        _is_hard_quota_error,
    )

    for candidate in (error, getattr(error, "__cause__", None)):
        if candidate is None:
            continue
        if isinstance(candidate, (AuthenticationError, PermissionDeniedError)):
            return True
        if _is_hard_quota_error(candidate):
            return True
    return False


async def _consume_unattended_gate_retry(self, node_name: str, event):
    """Spend one unattended auto-retry for ``node_name``, or None when spent."""
    used = self._unattended_gate_retries.get(node_name, 0)
    if used >= UNATTENDED_GATE_MAX_AUTO_RETRIES:
        logger.warning(
            f"[{self.session_id}] {node_name}: unattended gate exhausted its "
            f"{UNATTENDED_GATE_MAX_AUTO_RETRIES} auto-retries"
        )
        return None
    self._unattended_gate_retries[node_name] = used + 1
    logger.warning(
        f"[{self.session_id}] {node_name}: nobody answered the gate; "
        f"auto-retry {used + 1}/{UNATTENDED_GATE_MAX_AUTO_RETRIES}"
    )
    await _unattended_gate_backoff(event, used)
    return LlmFailureAction.RETRY.value


def _next_generation(self, node: str) -> int:
    """Increment and return the generation counter for a given node."""
    self._llm_failure_generations[node] = self._llm_failure_generations.get(node, 0) + 1
    return self._llm_failure_generations[node]

def _generation_for(self, node: str) -> int:
    """Return the current generation for a given node, or 0 if never set."""
    return self._llm_failure_generations.get(node, 0)

async def _drive_hitl_gate(
    self,
    node_name: str,
    analysis_state: AnalysisState,
    *,
    reason: str,
    info_attr: str,
    info_payload: dict,
    enter_change: str,
    exit_change: str,
    clear_cache_on_retry: bool,
    wait_seconds: int | None = None,
    auto_retry: bool = True,
    require_explicit_resolution: bool = False,
) -> LlmFailureAction:
    """Shared HITL intervention scaffold (lock/event/generation/state_sync).

    Callers parameterize only what differs; behavior is otherwise identical
    across LLM-failure, disabled-agent, and marker-db gates.

    Exceptional gates never inherit routine decision countdowns. After at
    most two automatic retries, only an explicit user action can resolve them.
    ``wait_seconds`` and ``require_explicit_resolution`` remain accepted for
    compatibility; neither can enable an automatic skip.
    """
    # The first two failure dialogs auto-retry; the third stays open. This is
    # a gate-local policy, independent of the routine round decision timeout.
    effective_wait = (
        0
        if auto_retry
        and self._unattended_gate_retries.get(node_name, 0) < UNATTENDED_GATE_MAX_AUTO_RETRIES
        else -1
    )
    async with self._llm_failure_lock:
        gen = self._next_generation(node_name)
        if node_name not in self._llm_failure_events:
            self._llm_failure_events[node_name] = asyncio.Event()
        event = self._llm_failure_events[node_name]
        # A cancelled/superseded prior gate may have left an unconsumed
        # action behind.  A new generation starts with no accepted action.
        self._llm_failure_actions.pop(node_name, None)
        event.clear()

        analysis_state.wait_input_reason = reason
        payload = dict(info_payload)
        payload["node_name"] = node_name
        payload["generation"] = gen
        payload["wait_seconds"] = effective_wait
        setattr(analysis_state, info_attr, payload)
        self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
        await self._send_state_sync(analysis_state, enter_change)

    try:
        action = self._llm_failure_actions.pop(node_name, None)
        if action is None and effective_wait == 0:
            action = await _consume_unattended_gate_retry(self, node_name, event)
            late = self._llm_failure_actions.pop(node_name, None)
            if late is not None:
                # An answer landed during the backoff and the caller was told it
                # was accepted, so it outranks the retry chosen on a timeout.
                action = late
        if action is None:
            logger.warning(
                f"[{self.session_id}] {node_name}: unattended gate requires "
                "explicit resolution after automatic retries"
            )
            # Keep this generation, event, and visible failure state alive
            # until a valid Retry/Skip/Terminate action is delivered.
            while action is None:
                if event.is_set():
                    event.clear()
                await wait_with_skip_time(event, -1, self.session_id, node_name)
                action = self._llm_failure_actions.pop(node_name, None)
        if action != LlmFailureAction.RETRY.value or event.is_set():
            # A user action starts a fresh recovery episode, including Retry
            # after changing credentials. Automatic retries keep their budget.
            self._unattended_gate_retries.pop(node_name, None)
        analysis_state.wait_input_reason = None
        setattr(analysis_state, info_attr, None)
        self._set_status(analysis_state, TaskStatus.RUNNING)
        resolved_exit_change = exit_change
        if exit_change == StateSyncChangeType.LLM_RETRY.value:
            if action == LlmFailureAction.TERMINATE.value:
                resolved_exit_change = StateSyncChangeType.LLM_TERMINATE.value
            elif action != LlmFailureAction.RETRY.value:
                resolved_exit_change = StateSyncChangeType.LLM_SKIP.value
        await self._send_state_sync(analysis_state, resolved_exit_change)
    except (asyncio.CancelledError, WorkflowSupersededError):
        # A superseding run unwinds through this gate the same way a
        # cancellation does, so it needs the same cleanup — otherwise the
        # gate's reason/info stay pinned on a state object nobody will resume.
        analysis_state.wait_input_reason = None
        setattr(analysis_state, info_attr, None)
        raise
    finally:
        self._llm_failure_events.pop(node_name, None)

    if action == LlmFailureAction.RETRY.value:
        if clear_cache_on_retry and self._settings_provider is not None:
            self._settings_provider.clear_client_cache()
        return LlmFailureAction.RETRY
    if action == LlmFailureAction.TERMINATE.value:
        return LlmFailureAction.TERMINATE
    return LlmFailureAction.SKIP

async def _wait_for_llm_intervention(
    self,
    error: UpstreamExhaustedError,
    node_name: str,
    analysis_state: AnalysisState,
    *,
    wait_seconds: int | None = None,
    auto_retry: bool = True,
    require_explicit_resolution: bool = False,
) -> LlmFailureAction:
    """Block workflow, show LlmTimeoutDialog, wait for user action.

    Returns:
        LlmFailureAction.RETRY     — user clicked Retry (caller resumes recovery)
        LlmFailureAction.SKIP      — user clicked Skip
        LlmFailureAction.TERMINATE — user clicked Terminate (caller should abort)
    """
    annotation_failure = error.failure_kind == "annotation_tool_failure"
    log_label = "annotation tool failed" if annotation_failure else "LLM upstream exhausted"
    message_label = "annotation tool failed" if annotation_failure else "LLM call failed"
    logger.error(
        f"[{self.session_id}] Node {node_name} {log_label}: "
        f"agent={error.agent_name} model={error.model} server={error.llm_server}"
    )
    await self._send_message(
        "error",
        f"Node {node_name} {message_label}: {str(error)}",
        AgentType.PI,
    )
    return await self._drive_hitl_gate(
        node_name, analysis_state,
        reason=WaitInputReason.LLM_FAILURE.value,
        info_attr="llm_failure_info",
        info_payload={
            "agent": error.agent_name,
            "model": error.model,
            "llm_server": error.llm_server,
            "error": str(error),
            "failure_kind": error.failure_kind,
        },
        enter_change=StateSyncChangeType.LLM_FAILURE.value,
        exit_change=StateSyncChangeType.LLM_RETRY.value,
        clear_cache_on_retry=not annotation_failure,
        wait_seconds=wait_seconds,
        auto_retry=auto_retry and not annotation_failure and not _is_non_recoverable_llm_failure(error),
        require_explicit_resolution=require_explicit_resolution,
    )

async def _pathology_llm_intervention_callback(
    self,
    error: UpstreamExhaustedError,
    image_label: str,
) -> LlmFailureAction:
    """Image-level LLM-failure HITL for the pathology agent.

    Unlike the node-level @with_retry gate (which restarts the whole node),
    this is invoked from inside PathologyAgent per LLM call, so RETRY re-runs
    only the failing call and SKIP drops only the current image. Reuses the
    same LlmTimeoutDialog infrastructure via _wait_for_llm_intervention.
    """
    logger.warning(
        f"[{self.session_id}] Pathology LLM failure on image '{image_label}': "
        f"{error}"
    )
    analysis_state = self._current_analysis_state
    if analysis_state is None:
        # No state to drive the HITL dialog — fall back to skipping the image.
        logger.error(
            f"[{self.session_id}] Pathology intervention requested but "
            f"_current_analysis_state is None; skipping image '{image_label}'"
        )
        return LlmFailureAction.SKIP
    return await self._wait_for_llm_intervention(
        error, "pathology_analysis", analysis_state
    )

async def _wait_for_disabled_agent_intervention(
    self,
    agent_name: str,
    node_name: str,
    analysis_state: AnalysisState,
) -> LlmFailureAction:
    """Block workflow until the user reconfigures, skips, or terminates.

    Mirrors _wait_for_llm_intervention but for the case where a core agent
    is required by `node_name` but its model is empty (disabled). Uses the
    same event/lock/generation infrastructure; the frontend disambiguates
    via state_sync change_type='agent_disabled'.
    """
    logger.warning(
        f"[{self.session_id}] Node {node_name} requires {agent_name} "
        f"agent but its model is empty (disabled)."
    )
    await self._send_message(
        "error",
        f"Node {node_name} requires {agent_name} agent which is disabled.",
        AgentType.PI,
    )
    return await self._drive_hitl_gate(
        node_name, analysis_state,
        reason=WaitInputReason.AGENT_DISABLED.value,
        info_attr="agent_disabled_info",
        info_payload={"agent": agent_name},
        enter_change=StateSyncChangeType.AGENT_DISABLED.value,
        exit_change=StateSyncChangeType.AGENT_RESUMED.value,
        clear_cache_on_retry=True,
        # Re-running the node against the same empty model cannot help.
        auto_retry=False,
    )

async def _wait_for_marker_db_intervention(
    self, node_name: str, analysis_state: AnalysisState,
) -> LlmFailureAction:
    """缺 marker TSV 时复用 LlmTimeoutDialog 通道。retry 不清 LLM cache。

    The per-source reason travels with the prompt. Asking someone to
    choose between retrying and annotating without DB priors, while
    showing them only "download failed", withholds the one fact the
    choice turns on -- a host that dropped a TLS handshake is worth
    retrying, a 404 is not.
    """
    from backend.services import marker_priors

    reasons = marker_priors.last_fetch_errors()
    detail = "; ".join(f"{name} — {why}" for name, why in reasons.items())
    return await self._drive_hitl_gate(
        node_name, analysis_state,
        reason=WaitInputReason.MARKER_DB_MISSING.value,
        info_attr="llm_failure_info",
        info_payload={
            "agent": "bioinformatics", "model": "", "llm_server": "",
            "error": ("Automatic download of the marker database failed. "
                      "Retry the download, skip (literature-only annotation), "
                      "or terminate."
                      + (f"\n\nLast attempt: {detail}" if detail else "")),
            "reason": "marker_db_missing",
        },
        enter_change=StateSyncChangeType.LLM_FAILURE.value,
        exit_change=StateSyncChangeType.LLM_RETRY.value,
        clear_cache_on_retry=False,
    )

async def _marker_db_intervention_callback(self) -> LlmFailureAction:
    """传给 bio agent 的回调：marker DB 不可用时默认自动联网补齐，不打扰用户；
    仅当下载失败才弹 HITL 让用户选择重试 / 跳过 / 终止。

    "不可用" 包含缺文件与陈旧 revision 两种：后者是表移到运行时数据目录后仍
    会遇到的情况 —— 同一数据卷跨版本复用时，旧投影写下的表还留在原处。
    """
    from backend.services import marker_priors

    # 默认行为：自动下载，无需用户确认
    await self._send_message(
        "status", "Marker database unavailable; downloading automatically...",
        AgentType.BIOINFORMATICS)
    ok = await asyncio.to_thread(marker_priors.trigger_fetch)
    if ok:
        await self._send_message(
            "status", "Marker DB downloaded, computing priors", AgentType.BIOINFORMATICS)
        return LlmFailureAction.RETRY

    # 下载失败才弹窗：重试 / 跳过 / 终止
    while True:
        action = await self._wait_for_marker_db_intervention(
            "bioinformatics_analysis", self._current_analysis_state)
        if action == LlmFailureAction.RETRY:
            ok = await asyncio.to_thread(marker_priors.trigger_fetch)
            if ok:
                await self._send_message(
                    "status", "Marker DB fetched, computing priors", AgentType.BIOINFORMATICS)
                return LlmFailureAction.RETRY
            await self._send_message(
                "status", "Marker DB fetch failed again; awaiting your choice",
                AgentType.BIOINFORMATICS)
            continue  # 失败后重新弹窗
        if action == LlmFailureAction.SKIP:
            await self._send_message("status", "Skipped DB priors", AgentType.BIOINFORMATICS)
        return action

def signal_continue(self):
    """Signal the workflow to continue from wait_for_user_input"""
    self._continue_event.set()

def signal_resume(self):
    """Release the cooperative pause gate so blocked nodes/agent loops can proceed."""
    self._pause_event.set()

def get_snapshot_workflow_state(self) -> dict[str, Any]:
    """Return a detached workflow-state view for snapshot capture."""
    return copy.deepcopy(dict(self._current_graph_state or {}))

def signal_annotation_confirmed(self):
    """Signal that user has confirmed cell type annotations"""
    self._annotation_event.set()

def signal_plan_review(self, action: str, user_input: Optional[str] = None) -> bool:
    """Signal that user has reviewed the plan.

    Args:
        action: "approve", "skip", or "modify"
        user_input: User's modification text (only for "modify")
    """
    if self._plan_review_event.is_set() or self._plan_review_action is not None:
        return False
    self._plan_review_action = {"action": action, "user_input": user_input}
    self._plan_review_event.set()
    return True

def signal_species_input(self, user_input: str) -> bool:
    """Signal that user has submitted species-related input."""
    if self._species_event.is_set():
        return False
    self._species_user_response = user_input
    self._species_event.set()
    return True

def signal_llm_failure_action(self, node_name: str, action: str, generation: int) -> bool:
    """Signal the workflow to handle an LLM failure with the given action.

    Called from the lifecycle API endpoint when the user chooses
    'retry' or 'skip' in the LlmTimeoutDialog.

    Args:
        node_name: The workflow node that failed (e.g. 'pathology_analysis')
        action: "retry" or "skip"
        generation: Monotonic generation counter from the frontend.

    Returns:
        True if the signal was delivered, False if the generation
        doesn't match (stale/duplicate request).
    """
    current_gen = self._generation_for(node_name)
    if generation != current_gen or current_gen == 0:
        logger.warning(
            f"[{self.session_id}] Rejected LLM failure action '{action}' "
            f"for node '{node_name}': generation {generation} != {current_gen}"
        )
        return False

    event = self._llm_failure_events.get(node_name)
    if event is None:
        logger.warning(
            f"[{self.session_id}] No LLM failure event for node '{node_name}'"
        )
        return False

    if event.is_set() or node_name in self._llm_failure_actions:
        logger.warning(
            f"[{self.session_id}] Rejected duplicate/conflicting LLM "
            f"failure action '{action}' for node '{node_name}' generation "
            f"{generation}"
        )
        return False

    self._llm_failure_actions[node_name] = action
    event.set()
    return True
