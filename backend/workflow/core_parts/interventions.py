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
UNATTENDED_GATE_MAX_AUTO_RETRIES = 3


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

    ``wait_seconds=None`` inherits the session's ``round_wait_seconds``, the
    same setting every other HITL gate honours.  A caller that supplies its
    own budget owns its unattended path and should pass ``auto_retry=False``.
    """
    effective_wait = (
        analysis_state.round_wait_seconds if wait_seconds is None else wait_seconds
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
        setattr(analysis_state, info_attr, payload)
        self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
        await self._send_state_sync(analysis_state, enter_change)

    try:
        # This is the seventh HITL gate, and like the six that got the hook
        # in 0.36.12 it blocks *inside* a graph node: the run task stays in
        # flight, so `_acknowledge_if_quiescent` declines, and the graph
        # never reaches the node entry that would otherwise acknowledge. A
        # restore requested while the user sits at the LLM-failure /
        # disabled-agent / marker-DB dialog therefore had no acknowledger at
        # all — and this gate has no timeout, so a user who walks away
        # strands it indefinitely.
        #
        # Ordinary agent gates inherit the session's round_wait_seconds, so
        # an automated run (0 = no wait) never parks on a dialog nobody will
        # answer. Report content generation supplies a finite budget of its
        # own so an unattended run can select its deterministic fallback.
        signaled = await wait_with_skip_time(
            event, effective_wait, self.session_id, node_name,
            safe_point_hook=getattr(self, "await_restore_safe_point", None),
        )
        action = self._llm_failure_actions.pop(node_name, None)
        if action is None and not signaled and auto_retry and effective_wait >= 0:
            # Nobody answered and nobody will. Falling straight through to skip
            # abandons the node on the first blip; blocking strands the run.
            action = await _consume_unattended_gate_retry(self, node_name, event)
            late = self._llm_failure_actions.pop(node_name, None)
            if late is not None:
                # An answer landed during the backoff and the caller was told it
                # was accepted, so it outranks the retry chosen on a timeout.
                action = late
        if action is None and require_explicit_resolution:
            logger.warning(
                f"[{self.session_id}] {node_name}: unattended gate requires "
                "explicit resolution after automatic retries"
            )
            # Keep this generation, event, and visible failure state alive
            # until a valid Retry/Skip/Terminate action is delivered.  The
            # safe-point-aware indefinite wait lets restore coordination make
            # progress while this in-flight graph node remains parked.
            while action is None:
                if event.is_set():
                    event.clear()
                await wait_with_skip_time(
                    event, -1, self.session_id, node_name,
                    safe_point_hook=getattr(self, "await_restore_safe_point", None),
                )
                action = self._llm_failure_actions.pop(node_name, None)
        if action != LlmFailureAction.RETRY.value:
            self._unattended_gate_retries.pop(node_name, None)
        analysis_state.wait_input_reason = None
        setattr(analysis_state, info_attr, None)
        self._set_status(analysis_state, TaskStatus.RUNNING)
        await self._send_state_sync(analysis_state, exit_change)
    except (asyncio.CancelledError, WorkflowSupersededError):
        # The hook re-raises WorkflowSupersededError once a restore commits.
        # That unwinds through this gate the same way a cancellation does,
        # so it needs the same cleanup — otherwise the gate's reason/info
        # stay pinned on a state object nobody will resume.
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
        LlmFailureAction.RETRY     — user clicked Retry (caller should clear caches and retry)
        LlmFailureAction.SKIP      — user clicked Skip
        LlmFailureAction.TERMINATE — user clicked Terminate (caller should abort)
    """
    logger.error(
        f"[{self.session_id}] Node {node_name} LLM upstream exhausted: "
        f"agent={error.agent_name} model={error.model} server={error.llm_server}"
    )
    await self._send_message(
        "error",
        f"Node {node_name} LLM call failed: {str(error)}",
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
        },
        enter_change=StateSyncChangeType.LLM_FAILURE.value,
        exit_change=StateSyncChangeType.LLM_RETRY.value,
        clear_cache_on_retry=True,
        wait_seconds=wait_seconds,
        auto_retry=auto_retry and not _is_non_recoverable_llm_failure(error),
        require_explicit_resolution=require_explicit_resolution,
    )

async def _publish_runtime_persistence_transition(
    self,
    analysis_state: AnalysisState,
    *,
    persist_state: bool,
    abort_on_close_discard: bool = False,
) -> bool:
    """Persist a gate transition before best-effort realtime delivery.

    The return value is a durability proof for close-time transitions. A
    final runtime save gate must not release its live kernel merely because
    the in-memory state changed: reconnect reads the persisted state, so a
    failed callback leaves the transition incomplete and must be retried.
    """
    persisted = not persist_state
    if persist_state and self.state_callback:
        try:
            discard_requested = bool(
                getattr(self, "_discard_runtime_on_close", False)
                or getattr(self, "_restore_superseded", False)
            )
            if abort_on_close_discard and discard_requested:
                return False
            callback_task = asyncio.ensure_future(
                self.state_callback(analysis_state)
            )
            if abort_on_close_discard:
                discard_event = getattr(
                    self, "_close_discard_event", None,
                )
                if discard_event is None:
                    discard_event = asyncio.Event()
                    self._close_discard_event = discard_event
                discard_task = asyncio.create_task(discard_event.wait())
                try:
                    done, _pending = await asyncio.wait(
                        {callback_task, discard_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if discard_task in done and not callback_task.done():
                        callback_task.cancel()
                        await asyncio.gather(
                            callback_task, return_exceptions=True,
                        )
                        return False
                    callback_result = await callback_task
                finally:
                    if not discard_task.done():
                        discard_task.cancel()
                    await asyncio.gather(
                        discard_task, return_exceptions=True,
                    )
            else:
                callback_result = await callback_task
            persisted = callback_result is not False
        except Exception:
            # The live kernel and in-memory gate remain authoritative. Keep
            # the intervention reachable even when the session metadata
            # write itself is temporarily unavailable.
            logger.exception(
                "[%s] Failed to persist runtime-persistence gate transition",
                self.session_id,
            )
            persisted = False
    elif persist_state:
        logger.error(
            "[%s] Runtime-persistence transition has no state callback; "
            "durability is unproven",
            self.session_id,
        )
    try:
        await self._send_state_sync(
            analysis_state,
            StateSyncChangeType.RUNTIME_PERSISTENCE_FAILURE.value,
        )
    except Exception:
        # Realtime delivery is observational. A persisted transition stays
        # authoritative and will be replayed when the client reconnects.
        logger.exception(
            "[%s] Failed to deliver runtime-persistence state sync",
            self.session_id,
        )
    return persisted

async def _wait_for_runtime_persistence_intervention(
    self,
    analysis_state: AnalysisState,
    *,
    successful_status: Optional[TaskStatus] = TaskStatus.RUNNING,
    persist_transitions: bool = False,
    rotate_generation: bool = False,
) -> str:
    """Hold the live kernel until persistence recovery is explicit."""
    from ...services.runtime_persistence_gate import (
        clear_runtime_persistence_gate,
        enter_runtime_persistence_gate,
        prepare_runtime_recompute,
        runtime_persistence_gate_identity,
    )

    def _discard_requested() -> bool:
        return bool(
            getattr(self, "_discard_runtime_on_close", False)
            or getattr(self, "_restore_superseded", False)
        )

    async def _publish_until_durable(*, allow_pending_action: bool) -> bool:
        """Retry a close-time state transition until it is durable.

        A live action may arrive over the current connection even when the
        entry write failed. Let that action advance to its terminal state,
        but require the terminal transition itself to persist before the
        helper returns and the kernel can be shut down.
        """
        delay = 0.1
        while True:
            persisted = await self._publish_runtime_persistence_transition(
                analysis_state,
                persist_state=persist_transitions,
                abort_on_close_discard=persist_transitions,
            )
            if persisted:
                return True
            if _discard_requested():
                return False
            reachable, _registered_here = (
                await self._register_close_persistence_gate(analysis_state)
            )
            if not reachable:
                # A restore or replacement won after the prior reachability
                # check. Retrying this old state's callback would either
                # spin forever on its compare-and-write guard or overwrite
                # the replacement through a less defensive callback.
                self._restore_superseded = True
                return False
            if allow_pending_action and bool(
                getattr(self, "_runtime_persistence_action_pending", False)
            ):
                return False
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 1.0)

    # A restore, deletion, or explicit clean restart may supersede this
    # close between the failed save and gate construction. The discard flag
    # is durable for the lifetime of the workflow, unlike the one-shot
    # action/event pair, so consult it before publishing a gate at all.
    if _discard_requested():
        return "discard"

    event = self._runtime_persistence_event
    self._runtime_persistence_gate_active = True
    self._current_analysis_state = analysis_state
    try:
        persistence = self.sandbox.sync_runtime_persistence_state(analysis_state)
        enter_runtime_persistence_gate(
            analysis_state,
            persistence,
            degraded_recovery=bool(persistence.get("degraded_recovery")),
            rotate_generation=rotate_generation,
        )
        (
            self._runtime_persistence_gate_token,
            self._runtime_persistence_gate_generation,
        ) = runtime_persistence_gate_identity(analysis_state)
        # close(save_runtime=False) may have raced with sync/enter above.
        # Do not clear its wake-up and then wait forever on an unreachable
        # gate; clear the newly published durable gate and discard instead.
        if _discard_requested():
            clear_runtime_persistence_gate(analysis_state)
            return "discard"
        self._runtime_persistence_action = None
        self._runtime_persistence_action_pending = False
        event.clear()
        published = await _publish_until_durable(allow_pending_action=True)
        if not published and _discard_requested():
            clear_runtime_persistence_gate(analysis_state)
            return "discard"
        while True:
            if _discard_requested():
                clear_runtime_persistence_gate(analysis_state)
                return "discard"
            await wait_with_skip_time(
                event,
                -1,
                self.session_id,
                "runtime_persistence_failure",
                safe_point_hook=getattr(self, "await_restore_safe_point", None),
            )
            action = self._runtime_persistence_action
            if action == "retry":
                try:
                    saved = await self.sandbox.save_session()
                except Exception:
                    logger.warning(
                        "[%s] Runtime persistence Retry failed",
                        self.session_id,
                        exc_info=True,
                    )
                    saved = False
                persistence = self.sandbox.sync_runtime_persistence_state(
                    analysis_state
                )
                if not saved or persistence.get("status") == "failed":
                    self._runtime_persistence_action = None
                    self._runtime_persistence_action_pending = False
                    event.clear()
                    enter_runtime_persistence_gate(
                        analysis_state,
                        persistence,
                        rotate_generation=True,
                    )
                    (
                        self._runtime_persistence_gate_token,
                        self._runtime_persistence_gate_generation,
                    ) = runtime_persistence_gate_identity(analysis_state)
                    published = await _publish_until_durable(
                        allow_pending_action=True
                    )
                    if not published and _discard_requested():
                        clear_runtime_persistence_gate(analysis_state)
                        return "discard"
                    continue
                clear_runtime_persistence_gate(analysis_state)
                self._runtime_persistence_action = None
                self._runtime_persistence_action_pending = False
                if successful_status is not None:
                    self._set_status(analysis_state, successful_status)
                if not await _publish_until_durable(
                    allow_pending_action=False
                ):
                    return "discard"
                return "retry"

            if action == "manage":
                prepare_runtime_recompute(analysis_state, self.sandbox)
                self._runtime_persistence_action = None
                self._runtime_persistence_action_pending = False
                if not await _publish_until_durable(
                    allow_pending_action=False
                ):
                    return "discard"
                return "recompute"

            if action == "terminate":
                clear_runtime_persistence_gate(analysis_state)
                self._runtime_persistence_action = None
                self._runtime_persistence_action_pending = False
                self._set_status(analysis_state, TaskStatus.STOPPED)
                analysis_state.termination_reason = (
                    "runtime_persistence_failure"
                )
                if not await _publish_until_durable(
                    allow_pending_action=False
                ):
                    return "discard"
                return "terminate"

            if action == "discard":
                # Internal-only: an authoritative restore/delete has made
                # this live generation obsolete. It is never accepted from
                # the public action endpoint.
                clear_runtime_persistence_gate(analysis_state)
                self._runtime_persistence_action = None
                self._runtime_persistence_action_pending = False
                return "discard"

            logger.warning(
                "[%s] Ignoring unknown runtime persistence action %r",
                self.session_id,
                action,
            )
    finally:
        self._runtime_persistence_gate_active = False
        self._runtime_persistence_action = None
        self._runtime_persistence_action_pending = False
        self._runtime_persistence_gate_token = ""
        self._runtime_persistence_gate_generation = 0

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

async def acknowledge_at_boundary(self) -> bool:
    """Acknowledge a pending restore/post-chat safe-point at a boundary.

    Uses the process-wide SafePointManager (never a private instance).
    Transitions a pending request to acquired so the waiting capture or
    restore job can proceed. Stores a live descriptor including
    ``kernel_present`` and graph position so post-chat / pre_restore
    capture can freeze an exact capsule.

    Does not wait for release — callers that must freeze mutations should
    call ``await_restore_safe_point`` afterward.

    Returns ``True`` if a pending request was acknowledged.
    """
    from ...services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

    sp = get_snapshot_safe_point_manager()
    if not sp.has_pending_request(self.session_id):
        return False

    generation = sp.get_pending_generation(self.session_id)
    if generation is None:
        return False

    graph_state = dict(getattr(self, "_current_graph_state", None) or {})
    sb = getattr(self, "sandbox", None)
    if "kernel_present" not in graph_state:
        graph_state["kernel_present"] = (
            sb is not None and getattr(sb, "km", None) is not None
        )
    if "current_round" not in graph_state:
        analysis = getattr(self, "state", None) or getattr(
            self, "_current_analysis_state", None,
        )
        if analysis is not None:
            graph_state["current_round"] = (
                getattr(analysis, "current_round", 0) or 0
            )

    acked = await sp.acknowledge_safe_point_request(
        self.session_id, generation, graph_state, role="workflow",
    )
    if acked:
        logger.info(
            f"[{self.session_id}] Acknowledged safe-point at boundary "
            f"(generation {generation})"
        )
    return acked

async def await_restore_safe_point(self) -> None:
    """Acknowledge pending restore safe-point and freeze until released.

    Called at graph-node boundaries and between sandbox steps. Freezes
    new workflow mutations while a restore job holds the safe point.
    """
    from ...services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

    await self.acknowledge_at_boundary()
    sp = get_snapshot_safe_point_manager()
    if sp.is_paused(self.session_id):
        logger.info(
            f"[{self.session_id}] Waiting for restore safe-point release"
        )
        await sp.wait_for(self.session_id)
        logger.info(f"[{self.session_id}] Restore safe-point released; resuming")

        # If a restore committed while we were frozen, this run is finished.
        # Its graph position predates the restore, and — decisively — its
        # checkpointer still holds an open connection to the checkpoint
        # database file that the restore replaced by rename. On POSIX that fd
        # still points at the old, now-unlinked inode, so resuming here would
        # read the pre-restore position and write it back, silently undoing
        # the restore. Unwind instead and let the next run rebuild the
        # workflow (and its connection) from the restored checkpoint.
        if (
            sp.is_superseded(self.session_id)
            or getattr(self, "_restore_superseded", False)
        ):
            logger.info(
                f"[{self.session_id}] Run superseded by a snapshot restore; "
                f"unwinding so the restored checkpoint takes effect"
            )
            raise WorkflowSupersededError(
                "snapshot restore replaced this run's checkpoint"
            )

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

def signal_runtime_persistence_action(
    self,
    action: str,
    gate_token: str | None = None,
    generation: int | None = None,
):
    """Arbitrate and deliver one action to the exact live gate generation."""
    from ...services.runtime_persistence_gate import (
        RuntimePersistenceActionOutcome,
        RuntimePersistenceActionResult,
        runtime_persistence_gate_identity,
        validate_runtime_persistence_action,
    )

    if action not in {"retry", "manage", "terminate"}:
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.DUPLICATE_OR_CONFLICT,
            action,
            str(getattr(self, "_runtime_persistence_gate_token", "") or ""),
            int(getattr(self, "_runtime_persistence_gate_generation", 0) or 0),
        )
    if not getattr(self, "_runtime_persistence_gate_active", False):
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.INACTIVE,
            action,
            str(getattr(self, "_runtime_persistence_gate_token", "") or ""),
            int(getattr(self, "_runtime_persistence_gate_generation", 0) or 0),
        )
    state = getattr(self, "_current_analysis_state", None)
    if state is None:
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.INACTIVE,
            action,
            str(getattr(self, "_runtime_persistence_gate_token", "") or ""),
            int(getattr(self, "_runtime_persistence_gate_generation", 0) or 0),
        )
    current_token, current_generation = runtime_persistence_gate_identity(state)
    if (
        not gate_token
        or generation is None
        or str(gate_token) != current_token
        or int(generation) != current_generation
    ):
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.STALE,
            action,
            current_token,
            current_generation,
        )
    if getattr(self, "_runtime_persistence_action_pending", False):
        # The live producer already owns an accepted action. Classify the
        # next request before touching the durable gate so duplicate and
        # conflicting clicks are observationally read-only.
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.DUPLICATE_OR_CONFLICT,
            action,
            current_token,
            current_generation,
        )
    result = validate_runtime_persistence_action(
        state,
        action,
        gate_token=gate_token,
        generation=generation,
        active=True,
    )
    if not result.accepted:
        return result
    self._runtime_persistence_action_pending = True
    self._runtime_persistence_action = action
    self._runtime_persistence_event.set()
    return result
