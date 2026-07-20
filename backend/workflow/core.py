"""
LangGraph workflow for multi-agent single-cell analysis
"""
import asyncio
import copy
import logging
import os
import traceback
from collections import deque
from typing import Callable, Dict, Any, List, Optional, Tuple
from datetime import datetime

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import aiosqlite

# Configure logger
logger = logging.getLogger(__name__)

GRAPH_RECURSION_LIMIT = 100


from .direction_coverage import at_integration_gate
from .helpers import wait_with_skip_time, build_integration_gate_payload
from .retry import with_retry, MAX_NODE_RETRIES
from .state import WorkflowState, make_initial_workflow_state
from .routing import RoutingMixin
from .context import ContextMixin
from .messaging import MessagingMixin
from .callbacks import CallbacksMixin
from .annotation import AnnotationMixin
from .planning import PlanningMixin
from .agents import AgentNodesMixin
from .synthesis import SynthesisMixin
from .initialize import InitializeMixin
from .graph import GraphMixin
from .reporting import ReportingMixin
from .integration import IntegrationMixin

from ..services.snapshot_v2.errors import RequiredSnapshotBlocked
from ..services.snapshot_v2.restore_entry import build_restore_entry_plan

from ..models.schemas import (
    AnalysisState, AgentType, TaskStatus, RoundResult, AnalysisPhase,
    WebSocketMessage, AgentMessage, MessageRole,
    DetailedReportSegment,
)
from ..agents.pi_agent import PIAgent
from ..agents.pathology_agent import PathologyAgent
from ..agents.bioinformatics_agent import BioinformaticsAgent
from ..agents.biochemistry_agent import BiochemistryAgent
from ..agents.literature_agent import LiteratureAgent
from ..agents.painter_agent import PainterAgent
from ..agents.supervisor_agent import SupervisorAgent
from ..agents.technique_agent import TechniqueAgent
from ..utils.llm_client import LLMManager, UpstreamExhaustedError
from ..constants import LlmFailureAction, WaitInputReason, StateSyncChangeType, SEMANTIC_WAIT_GATES
from ..utils.settings_provider import AgentSettingsProvider
from ..utils.sandbox import LocalJupyterSandbox
from ..utils.runtime_apply import is_forward_resume_blocked
from ..config import SystemConfig, get_session_workspace, get_session_states


# Defined in helpers.py (dependency-free) and re-exported here for backward
# compatibility — existing code imports it via `from .core import WorkflowTerminatedError`.
from .helpers import WorkflowTerminatedError  # noqa: E402,F401


class SingleCellAnalysisWorkflow(RoutingMixin, ContextMixin, MessagingMixin, CallbacksMixin, AnnotationMixin, PlanningMixin, AgentNodesMixin, SynthesisMixin, InitializeMixin, GraphMixin, ReportingMixin, IntegrationMixin):
    """
    Multi-agent workflow for single-cell analysis using LangGraph
    """

    @staticmethod
    def _set_status(state: AnalysisState, status: TaskStatus):
        state.status = status
        if status in (TaskStatus.WAITING_INPUT, TaskStatus.WAITING_SPECIES_INPUT):
            state.waiting_input_started_at = datetime.now()
        else:
            state.waiting_input_started_at = None
    
    def _consume_restore_routing_metadata(self, analysis_state: AnalysisState) -> None:
        """Clear one-shot restore routing fields only (not provenance).

        Clears: restore_resume_phase/kind/node/gate/workflow_state and
        restore_start_intent. Does not clear last_restored_snapshot_id or
        other provenance fields owned by Task 4.
        """
        analysis_state.restore_resume_phase = None
        analysis_state.restore_resume_kind = None
        analysis_state.restore_resume_node = None
        analysis_state.restore_resume_gate = None
        analysis_state.restore_resume_workflow_state = None
        analysis_state.restore_start_intent = None

    def _consume_restore_resume_metadata(self, analysis_state: AnalysisState) -> None:
        """Clear restore routing metadata only (not provenance).

        Provenance is consumed only after the canonical first restored
        RoundResult is recorded in synthesis/integration. Prefer
        ``_consume_restore_routing_metadata`` for new call sites.
        """
        self._consume_restore_routing_metadata(analysis_state)

    def _restore_entry(self, state: WorkflowState) -> WorkflowState:
        """Graph entry node that determines and applies the restore entry plan.

        Uses ``restore_entry_loader`` (injectable for testing, defaults to
        ``build_restore_entry_plan``) to produce a ``RestoreEntryPlan`` from
        the analysis state's restore metadata, then:
        - Deep-copies AnalysisState and consumes routing metadata only on the copy
        - Seeds exact frozen WorkflowState for graph_node resumes
        - Sets ``_restore_entry_plan`` for the routing method
        - Handles ``preserve_round_on_initialize`` for explicit_start

        The original input AnalysisState is not mutated before node success so
        loader/validation failures leave metadata intact for retry.
        """
        import copy

        analysis_state = state["analysis_state"]
        loader = self.restore_entry_loader or build_restore_entry_plan

        # Pass restore_start_intent to the plan builder (read-only on original).
        start_intent = getattr(analysis_state, "restore_start_intent", None)
        plan = loader(analysis_state, restore_start_intent=start_intent)

        logger.info(
            f"[{self.session_id}] _restore_entry: intent={plan.intent} "
            f"target={plan.target_node} preserve_round={plan.preserve_round_on_initialize}"
        )

        # Deep-copy AnalysisState before clearing routing metadata so the
        # original input object remains intact until LangGraph checkpoints
        # the successful node return.
        returned_analysis = analysis_state.model_copy(deep=True)
        self._consume_restore_routing_metadata(returned_analysis)

        returned: WorkflowState = {**state, "analysis_state": returned_analysis}

        # Graph-node restore: seed exact frozen WorkflowState (not initialize_round).
        if plan.intent == "graph_node" and plan.target_node:
            returned["_restore_resume_node"] = plan.target_node

        for key, value in plan.workflow_state.items():
            if key not in {"analysis_state", "messages"}:
                returned[key] = copy.deepcopy(value)

        # Explicit-start restore: signal initialize to preserve the round.
        if plan.preserve_round_on_initialize:
            returned["_preserve_round_on_initialize"] = True

        # Store the plan for the routing method.
        returned["_restore_entry_plan"] = plan.model_dump(mode="json")

        return returned

    async def resume_restored_wait_state(self, analysis_state: AnalysisState) -> AnalysisState:
        logger.info("[%s] Continuing restored wait-state snapshot", self.session_id)
        if self.compiled_graph is None:
            self._state = analysis_state
            await self.initialize()

        self._set_status(analysis_state, TaskStatus.RUNNING)
        workflow_state = self._make_initial_workflow_state(
            analysis_state,
            current_round=analysis_state.current_round,
        )

        # Mirror the post-wait max-round extension from _wait_for_user_input().
        has_real_text = bool(
            analysis_state.pending_human_input
            and analysis_state.pending_human_input.strip() not in ("", "[CONTINUE]", "[PI_DECIDE]")
        )
        has_uploaded_files = bool(analysis_state.pending_uploaded_files)
        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        if (has_real_text or has_uploaded_files) and analysis_state.current_round >= max_rounds:
            analysis_state.max_rounds = analysis_state.current_round + 1

        self._consume_restore_resume_metadata(analysis_state)

        if at_integration_gate(analysis_state) and not analysis_state.final_report_requested:
            await self._send_message(
                "integration_gate",
                build_integration_gate_payload(analysis_state, self.session_id),
                AgentType.PI,
            )

        # Integration-gate waiting state: skip the _should_continue /
        # _check_convergence stop guards — they would route to
        # generate_final_report on max_rounds, bypassing the
        # wait_for_user_input checkpoint that lets the user choose new
        # directions. The graph checkpoint resumes at wait_for_user_input.
        is_integration_gate = at_integration_gate(analysis_state)

        if not is_integration_gate:
            if self._should_continue(workflow_state) == "stop":
                final_state = await self._generate_final_report(workflow_state)
                return final_state["analysis_state"]

            workflow_state = await self._check_convergence(workflow_state)
            if self._check_convergence_condition(workflow_state) == "stop":
                final_state = await self._generate_final_report(workflow_state)
                return final_state["analysis_state"]

        config = {"configurable": {"thread_id": self.session_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}
        final_state = await self.compiled_graph.ainvoke(workflow_state, config)
        return final_state["analysis_state"]

    def __init__(
        self,
        config: SystemConfig,
        session_id: str,
        message_callback: Optional[Callable] = None,
        state_callback: Optional[Callable] = None  # Callback to save state after each round
    ):
        # Fail closed: unresolved recovery blocks workflow construction.
        from ..services.session_store import assert_recovery_clear
        assert_recovery_clear(session_id=session_id)

        self.config = config
        self.session_id = session_id
        self.message_callback = message_callback
        self.state_callback = state_callback
        
        # Initialize components
        self.llm_manager: Optional[LLMManager] = None
        self.sandbox: Optional[LocalJupyterSandbox] = None
        
        # Agents
        self.pi_agent: Optional[PIAgent] = None
        self.pathology_agent: Optional[PathologyAgent] = None
        self.bioinformatics_agent: Optional[BioinformaticsAgent] = None
        self.biochemistry_agent: Optional[BiochemistryAgent] = None
        self.literature_agent: Optional[LiteratureAgent] = None
        self.painter_agent: Optional[PainterAgent] = None
        self.supervisor_agent: Optional[SupervisorAgent] = None
        
        # Workflow graph
        self.graph = None
        self.compiled_graph = None
        self.checkpointer = None
        self._db_conn = None

        # Current analysis state for callbacks (updated during workflow execution)
        self._current_analysis_state: Optional[AnalysisState] = None
        self._current_graph_state: Optional[WorkflowState] = None

        # Cooperative pause gate. SET = running (await passes immediately);
        # CLEAR = pause requested (waiters block until resume).
        # Checked by @with_retry at node entry and by the bioinformatics agent's
        # autonomous tool-call loop between steps. Cleared in pause(), set in
        # signal_resume(). Defaults to running so first run() is not blocked.
        self._pause_event = asyncio.Event()
        self._pause_event.set()

        # Event for blocking between rounds (wait for user input or continue signal)
        self._continue_event = asyncio.Event()

        # Event for blocking during annotation confirmation (exploration phase)
        self._annotation_event = asyncio.Event()

        # Event for blocking during plan review
        self._plan_review_event = asyncio.Event()
        self._plan_review_action = None  # "approve" | "skip" | dict with user modifications

        # Event for blocking during species input (inference or gene validation)
        self._species_event = asyncio.Event()
        self._species_user_response: Optional[str] = None

        # LLM failure intervention — serialized via Lock
        self._llm_failure_lock = asyncio.Lock()
        self._llm_failure_events: Dict[str, asyncio.Event] = {}
        self._llm_failure_actions: Dict[str, Optional[str]] = {}
        self._llm_failure_generations: Dict[str, int] = {}  # monotonic counter per node for idempotency

        # Supplementary analysis requests queue (set by API, consumed by HITL loop).
        # Each item is (request_text, target) where target is "annotation" | "direction" | "round".
        # Use a deque so consecutive rapid requests are preserved in order rather than overwritten.
        self._supplementary_queue: deque[Tuple[str, str]] = deque()

        # Settings provider — set in initialize(), provides hot-reloadable settings
        self._settings_provider: Optional[AgentSettingsProvider] = None

        # Technique agent (optional, created in initialize())
        self.technique_agent: Optional[TechniqueAgent] = None

        # Restore entry loader — injectable callable for testing; defaults
        # to build_restore_entry_plan.  Called by _restore_entry() to
        # determine the graph entry plan.
        self.restore_entry_loader: Optional[Callable] = None

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
    ) -> LlmFailureAction:
        """Shared HITL intervention scaffold (lock/event/generation/state_sync).

        Callers parameterize only what differs; behavior is otherwise identical
        across LLM-failure, disabled-agent, and marker-db gates.
        """
        async with self._llm_failure_lock:
            gen = self._next_generation(node_name)
            if node_name not in self._llm_failure_events:
                self._llm_failure_events[node_name] = asyncio.Event()
            event = self._llm_failure_events[node_name]
            event.clear()

            analysis_state.wait_input_reason = reason
            payload = dict(info_payload)
            payload["node_name"] = node_name
            payload["generation"] = gen
            setattr(analysis_state, info_attr, payload)
            self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
            await self._send_state_sync(analysis_state, enter_change)

        try:
            await event.wait()
            action = self._llm_failure_actions.pop(node_name, None)
            analysis_state.wait_input_reason = None
            setattr(analysis_state, info_attr, None)
            self._set_status(analysis_state, TaskStatus.RUNNING)
            await self._send_state_sync(analysis_state, exit_change)
        except asyncio.CancelledError:
            analysis_state.wait_input_reason = None
            setattr(analysis_state, info_attr, None)
            raise
        finally:
            self._llm_failure_events.pop(node_name, None)

        if action == LlmFailureAction.RETRY.value:
            if clear_cache_on_retry and self._settings_provider is not None:
                self._settings_provider._client_cache.clear()
            return LlmFailureAction.RETRY
        if action == LlmFailureAction.TERMINATE.value:
            return LlmFailureAction.TERMINATE
        return LlmFailureAction.SKIP

    async def _wait_for_llm_intervention(
        self,
        error: UpstreamExhaustedError,
        node_name: str,
        analysis_state: AnalysisState,
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
        )

    async def _wait_for_marker_db_intervention(
        self, node_name: str, analysis_state: AnalysisState,
    ) -> LlmFailureAction:
        """缺 marker TSV 时复用 LlmTimeoutDialog 通道。retry 不清 LLM cache。"""
        return await self._drive_hitl_gate(
            node_name, analysis_state,
            reason=WaitInputReason.MARKER_DB_MISSING.value,
            info_attr="llm_failure_info",
            info_payload={
                "agent": "bioinformatics", "model": "", "llm_server": "",
                "error": ("Automatic download of the marker database failed. "
                          "Retry the download, skip (literature-only annotation), or terminate."),
                "reason": "marker_db_missing",
            },
            enter_change=StateSyncChangeType.LLM_FAILURE.value,
            exit_change=StateSyncChangeType.LLM_RETRY.value,
            clear_cache_on_retry=False,
        )

    async def _marker_db_intervention_callback(self) -> LlmFailureAction:
        """传给 bio agent 的回调：缺 marker DB 时默认自动联网补齐，不打扰用户；
        仅当下载失败才弹 HITL 让用户选择重试 / 跳过 / 终止。"""
        from backend.services import marker_priors

        # 默认行为：自动下载，无需用户确认
        await self._send_message(
            "status", "Marker database missing; downloading automatically...",
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
        from ..services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

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
        from ..services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

        await self.acknowledge_at_boundary()
        sp = get_snapshot_safe_point_manager()
        if sp.is_paused(self.session_id):
            logger.info(
                f"[{self.session_id}] Waiting for restore safe-point release"
            )
            await sp.wait_for(self.session_id)
            logger.info(f"[{self.session_id}] Restore safe-point released; resuming")

    def signal_annotation_confirmed(self):
        """Signal that user has confirmed cell type annotations"""
        self._annotation_event.set()

    def signal_plan_review(self, action: str, user_input: Optional[str] = None):
        """Signal that user has reviewed the plan.
        
        Args:
            action: "approve", "skip", or "modify"
            user_input: User's modification text (only for "modify")
        """
        self._plan_review_action = {"action": action, "user_input": user_input}
        self._plan_review_event.set()

    def signal_species_input(self, user_input: str):
        """Signal that user has submitted species-related input."""
        self._species_user_response = user_input
        self._species_event.set()

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

        self._llm_failure_actions[node_name] = action
        event.set()
        return True

    def signal_supplementary_analysis(self, request_text: str, target: str):
        """Queue a supplementary analysis request at the current HITL point.

        Multiple consecutive requests are preserved in arrival order (FIFO) and
        processed one-by-one by the consumer loop in annotation/synthesis nodes.
        """
        self._supplementary_queue.append((request_text, target))
        if target == "annotation":
            self._annotation_event.set()
        elif target in ("direction", "round"):
            self._continue_event.set()

    @property
    def _vision_settings(self) -> Dict[str, bool]:
        """Vision settings — read from Provider so changes are always fresh."""
        return {
            t.value: self._settings_provider.get_vision_setting(t.value)
            for t in AgentType
        }

    def _create_optional_agent(self, agent_name: str):
        """Factory — instantiate a new optional agent by name."""
        if agent_name == "pathology":
            agent = PathologyAgent(
                settings_provider=self._settings_provider,
                tile_size=self.config.image_tile_size,
                max_tiles=50,
            )
            agent.set_session(self.session_id)
            agent.llm_intervention_callback = self._pathology_llm_intervention_callback
            return agent
        if agent_name == "biochemistry":
            agent = BiochemistryAgent(settings_provider=self._settings_provider)
            agent.set_session(self.session_id)
            return agent
        if agent_name == "painter":
            agent = PainterAgent(settings_provider=self._settings_provider)
            agent.set_session(self.session_id)
            return agent
        if agent_name == "supervisor":
            agent = SupervisorAgent(
                settings_provider=self._settings_provider,
                pi_callback=self._pi_callback,
                message_callback=self._supervisor_message_callback,
                session_id=self.session_id,
                technique_agent=self.technique_agent,
            )
            agent.set_sandbox(self.sandbox)
            return agent
        if agent_name == "technique":
            agent = TechniqueAgent(
                settings_provider=self._settings_provider,
                message_callback=self._technique_message_callback,
            )
            agent.set_session(self.session_id)
            return agent
        raise ValueError(f"Unknown optional agent: {agent_name}")

    def ensure_agent_ready(self, agent_name: str) -> bool:
        """Materialize or destroy an optional agent based on its current config.

        Returns True if the agent is ready for use, False if it should be
        skipped (model is empty).
        """
        def sync_technique_dependents():
            if agent_name != "technique":
                return
            for dependent_name in ("bioinformatics_agent", "supervisor_agent"):
                dependent = getattr(self, dependent_name, None)
                if dependent is not None and hasattr(dependent, "technique_agent"):
                    dependent.technique_agent = self.technique_agent

        model = self._settings_provider.get_model(agent_name)
        attr = f"{agent_name}_agent"
        agent = getattr(self, attr, None)
        if not model or not model.strip():
            if agent is not None:
                logger.info(f"[{self.session_id}] {agent_name} agent disabled by empty model")
                setattr(self, attr, None)
                sync_technique_dependents()
            return False
        if agent is None:
            logger.info(f"[{self.session_id}] Creating {agent_name} agent after config enable")
            agent = self._create_optional_agent(agent_name)
            setattr(self, attr, agent)
            sync_technique_dependents()
        if hasattr(agent, "refresh_runtime_config_if_needed"):
            agent.refresh_runtime_config_if_needed()
        return True

    async def initialize(self):
        """Initialize all components"""
        logger.info(f"[{self.session_id}] Initializing workflow components")
        # Close leftover provider from previous initialize() call (e.g. resume)
        if hasattr(self, '_settings_provider') and self._settings_provider:
            await self._settings_provider.cleanup()

        # Initialize checkpointer if not exists
        if self._db_conn is None:
            states_dir = get_session_states(self.session_id)
            states_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(states_dir / "checkpoints.sqlite")
            self._db_conn = await aiosqlite.connect(db_path)
            serde = JsonPlusSerializer(allowed_json_modules=[
                ("backend.models.schemas", "AgentType"),
                ("backend.models.schemas", "AnalysisState"),
                ("backend.models.schemas", "AnalysisPhase"),
                ("backend.models.schemas", "RoundResult"),
                ("backend.models.schemas", "TaskStatus"),
                ("backend.models.schemas", "MessageRole"),
            ])
            self.checkpointer = AsyncSqliteSaver(self._db_conn, serde=serde)
            # Make sure tables are created
            await self.checkpointer.setup()

        # Initialize LLM manager with session context
        self.llm_manager = LLMManager(
            api_key=self.config.api.api_key,
            base_url=self.config.api.llm_base_url,
            proxy_url=self.config.api.proxy_url,
            session_id=self.session_id,
            agent_name="workflow",
        )

        # Initialize local Jupyter sandbox with session-specific paths
        self.sandbox = LocalJupyterSandbox(
            session_id=self.session_id,
            state_dir=str(get_session_states(self.session_id)),
            working_dir=str(get_session_workspace(self.session_id)),
            timeout=self.config.sandbox_timeout
        )

        # Create the settings provider — all agents read model/temp/prompt/client from it
        self._settings_provider = AgentSettingsProvider(
            state=self._state,
            config=self.config,
            global_client=self.llm_manager.openrouter,
            session_id=self.session_id,
        )

        # Install retry callbacks before agents are created so eager external
        # clients, such as Literature deep-research clients, inherit them.
        self._install_retry_notifiers()

        # Technique Agent (lightweight doc lookup, used by Supervisor and Bioinformatics)
        self.technique_agent = None
        self.ensure_agent_ready("technique")

        # Initialize required agents (PI and Bioinformatics — always active)
        self.pi_agent = PIAgent(
            settings_provider=self._settings_provider,
        )
        self.pi_agent.set_session(self.session_id)

        self.bioinformatics_agent = BioinformaticsAgent(
            settings_provider=self._settings_provider,
            sandbox=self.sandbox,
            literature_callback=self._literature_callback,
            pi_callback=self._pi_callback,
            annotate_clusters_callback=self._annotate_clusters_callback,
            message_callback=self._agent_message_callback,
            supervisor_callback=self._supervisor_callback,
            technique_agent=self.technique_agent,
            species_validation_callback=self._species_validation_callback,
            marker_db_intervention_callback=self._marker_db_intervention_callback,
        )
        self.bioinformatics_agent.set_session(self.session_id)
        self.bioinformatics_agent._pause_event = self._pause_event

        # Initialize optional agents — disabled if model is empty
        self.pathology_agent = None
        self.ensure_agent_ready("pathology")
        self.biochemistry_agent = None
        self.ensure_agent_ready("biochemistry")

        # Literature agent with mode-specific configuration
        literature_config = self._settings_provider.get_literature_config()
        self.literature_agent = LiteratureAgent(
            settings_provider=self._settings_provider,
            literature_mode=literature_config.get('mode', 'pubmed_llm'),
            deep_research_model=literature_config.get('deep_research_model', 'o4-mini-deep-research'),
            deep_research_llm_base_url=literature_config.get('deep_research_llm_base_url', ''),
            deep_research_api_key=literature_config.get('deep_research_api_key', ''),
            google_api_key=literature_config.get('google_api_key', ''),
            pubmed_email=literature_config.get('pubmed_email', ''),
            max_search_rounds=literature_config.get('max_search_rounds', 20),
            proxy_url=self.config.api.proxy_url,
        )
        self.literature_agent.set_session(self.session_id)

        # Initialize Painter Agent for graphic abstract generation (optional)
        self.painter_agent = None
        self.ensure_agent_ready("painter")

        # Initialize Supervisor Agent for real-time analysis monitoring (optional)
        self.supervisor_agent = None
        self.ensure_agent_ready("supervisor")

        # Log which agents are active
        pi_model = self._settings_provider.get_model("pi")
        bioinformatics_model = self._settings_provider.get_model("bioinformatics")
        literature_model = self._settings_provider.get_model("literature")
        path_model = self._settings_provider.get_model("pathology")
        biochem_model = self._settings_provider.get_model("biochemistry")
        painter_model = self._settings_provider.get_model("painter")
        sup_model = self._settings_provider.get_model("supervisor")
        logger.info(f"[{self.session_id}] Agents initialized: PI={pi_model}, Pathology={path_model or 'DISABLED'}, Bioinformatics={bioinformatics_model}, Biochemistry={biochem_model or 'DISABLED'}, Literature={literature_model}, Painter={painter_model or 'DISABLED'}, Supervisor={sup_model or 'DISABLED'}")

        # Build workflow graph
        self._build_graph()
        logger.info(f"[{self.session_id}] Workflow graph built")
        
    def _install_retry_notifiers(self):
        """Set on_retry callbacks on all agent LLM clients.

        When an LLM request fails and enters retry-with-backoff, the callback
        pushes a status message through the WebSocket so the frontend knows the
        system is still alive (prevents disconnect due to apparent inactivity).
        """
        async def _on_retry(attempt: int, max_retries: int, delay: float, error_type: str):
            try:
                await self._send_message(
                    "status",
                    f"LLM request retry {attempt + 1}/{max_retries + 1} "
                    f"({error_type}, waiting {delay:.0f}s)...",
                    AgentType.BIOINFORMATICS,
                )
            except Exception:
                pass  # Best-effort; never break retry logic

        # Set on the global client
        self.llm_manager.openrouter.on_retry = _on_retry
        # Set on the provider so all future lazy-created clients also get it
        self._settings_provider.set_retry_callback(_on_retry)
        # Also set on any already-cached clients
        for client in self._settings_provider.cached_clients():
            client.on_retry = _on_retry

    async def _build_and_save_detailed_segment(
        self, state: AnalysisState, phase: str, title: str,
        graph_state=None,
        round_result=None,
    ) -> DetailedReportSegment:
        """Build a detailed report segment and save the combined HTML."""
        from ..services.report import (
            generate_detailed_report_segment_html,
            generate_combined_detailed_report_html,
        )

        content_html = generate_detailed_report_segment_html(
            state, phase, title,
            graph_state=dict(graph_state) if graph_state else None,
            round_result=round_result,
        )
        segment = DetailedReportSegment(
            phase=phase,
            title=title,
            content_html=content_html,
            round_number=state.current_round,
        )
        state.detailed_report_segments.append(segment)

        try:
            generate_combined_detailed_report_html(state)
        except Exception as e:
            logger.warning(f"[{self.session_id}] Combined detailed report generation failed: {e}")

        return segment

    @staticmethod
    def _make_initial_workflow_state(analysis_state, current_round=0):
        return make_initial_workflow_state(analysis_state, current_round)

    async def _create_auto_snapshot(self, state: AnalysisState, phase: str):
        """Create a required workflow snapshot via SnapshotService.

        Fail-closed: on capture failure raises RequiredSnapshotBlocked after
        persisting the snapshot_failure gate (retry/skip/manage/terminate).

        One-shot skip: when the user chose "Skip snapshot & continue" on a prior
        failure at this boundary, the marker is consumed here and the capture is a
        no-op so the graph proceeds past the boundary that previously failed.
        """
        from ..config import is_snapshot_v2_enabled

        if not is_snapshot_v2_enabled():
            return None

        from ..services.snapshot_v2.failure_gate import consume_snapshot_skip

        if consume_snapshot_skip(state, phase):
            logger.warning(
                f"[{self.session_id}] Skipping required snapshot at boundary "
                f"'{phase}' (user chose Skip snapshot & continue)"
            )
            try:
                await self._send_message(
                    "status",
                    f"Skipped snapshot at '{phase}'; continuing analysis "
                    "(this checkpoint will not be restorable).",
                )
            except Exception:
                pass
            return None

        from ..config import get_session_workspace
        from ..services.snapshot_v2.service import SnapshotService
        from ..services.snapshot_v2.continuation import ContinuationRegistry
        from ..services.snapshot_v2.models import SnapshotCaptureRequest
        from ..services.snapshot_v2.lineage import capture_lineage_from_state
        from ..services.snapshot_v2.paths import SnapshotV2Paths
        from ..services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

        # Prefer a long-lived service on the workflow when tests inject one.
        service = getattr(self, "snapshot_service", None)
        if service is None:
            paths = SnapshotV2Paths.for_session(self.session_id)
            service = SnapshotService(self.session_id, paths)
            self.snapshot_service = service

        workflow_state = dict(self._current_graph_state or {})
        # Enrich with kernel presence for runtime capture decisions.
        if "kernel_present" not in workflow_state:
            sb = getattr(self, "sandbox", None)
            workflow_state["kernel_present"] = (
                sb is not None and getattr(sb, "km", None) is not None
            )

        live_sp = get_snapshot_safe_point_manager().get_descriptor(self.session_id)
        capsule = ContinuationRegistry(service.object_store).freeze(
            phase,
            state,
            workflow_state,
            live_safe_point=live_sp,
        )
        request = SnapshotCaptureRequest(
            session_id=self.session_id,
            reason=phase,
            phase=phase,
            capsule=capsule,
            round_number=getattr(state, "current_round", 0) or 0,
            **capture_lineage_from_state(state),
        )
        workspace_root = str(get_session_workspace(self.session_id))
        # capture_required_snapshot applies success/failure and raises on failure
        return await service.capture_required_snapshot(
            state,
            request,
            sandbox=getattr(self, "sandbox", None),
            workflow_state=workflow_state,
            workspace_root=workspace_root,
        )

    def _merge_supplementary_findings(self, agent_key: str, new_findings: List[str]):
        """Merge supplementary findings into the current graph state for an agent."""
        if not self._current_graph_state or not new_findings:
            return
        existing = self._current_graph_state.get(agent_key, {})
        existing_findings = existing.get("findings", []) if isinstance(existing, dict) else []
        existing["findings"] = existing_findings + [f"[Supplementary] {f}" for f in new_findings]
        self._current_graph_state[agent_key] = existing

    async def _run_supplementary_analysis(
        self, state: AnalysisState, request_text: str, context: str
    ):
        """Execute user-requested supplementary analysis at a HITL point."""
        await self._send_message(
            "status",
            f"PI is planning supplementary analysis: {request_text[:200]}...",
            AgentType.PI,
        )

        # PI agent plans supplementary tasks
        supp_plan = await self.pi_agent.plan_supplementary_analysis(
            state, request_text, context
        ) or {}

        # Execute bioinformatics tasks if planned
        bio_tasks = supp_plan.get("bioinformatics_tasks", [])
        if bio_tasks and self.bioinformatics_agent:
            await self._send_message("status", "Running supplementary bioinformatics analysis...", AgentType.BIOINFORMATICS)
            try:
                combined_instruction = "Supplementary analysis tasks:\n" + "\n".join(f"- {t}" for t in bio_tasks)
                supp_result = await self.bioinformatics_agent.run_autonomous_analysis(
                    state, combined_instruction, max_steps=30
                )
                if supp_result:
                    self._merge_supplementary_findings(
                        "bioinformatics_result", supp_result.get("findings", [])
                    )
            except Exception as e:
                logger.error(f"[{self.session_id}] Supplementary bioinformatics failed: {e}")
                await self._send_message("error", f"Supplementary bioinformatics analysis failed: {e}", AgentType.BIOINFORMATICS)

        # Execute literature tasks if planned
        lit_tasks = supp_plan.get("literature_topics", [])
        if lit_tasks and self.literature_agent:
            await self._send_message("status", "Running supplementary literature research...", AgentType.LITERATURE)
            try:
                for topic in lit_tasks[:3]:
                    lit_result = await self.literature_agent.process(
                        state, topic, topic=topic, use_deep_research=False
                    )
                    if lit_result:
                        report_text = lit_result.get("report", "") or lit_result.get("synthesis", "")
                        if report_text:
                            self._merge_supplementary_findings(
                                "literature_result", [report_text[:500]]
                            )
            except Exception as e:
                logger.error(f"[{self.session_id}] Supplementary literature failed: {e}")
                await self._send_message("error", f"Supplementary literature research failed: {e}", AgentType.LITERATURE)

        # Generate new detailed report segment
        seg_index = sum(1 for s in state.detailed_report_segments if "supplementary" in s.phase)
        await self._build_and_save_detailed_segment(
            state,
            f"supplementary_{context}_{seg_index}",
            f"Supplementary Analysis: {request_text[:60]}",
            graph_state=self._current_graph_state,
        )

        await self._send_message(
            "status",
            "Supplementary analysis complete. Updating reports...",
            AgentType.PI,
        )

        # Save state
        if self.state_callback:
            try:
                await self.state_callback(state)
            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to save state after supplementary analysis")

    # --- Round context helpers ---



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
        from ..config import is_snapshot_v2_enabled
        from ..services.snapshot_v2.feature_gate import (
            normalize_disabled_snapshot_state,
        )

        if not is_snapshot_v2_enabled():
            normalize_disabled_snapshot_state(
                initial_state,
                workflow_active=True,
            )

        # §1d: never run the graph forward on a kernel that is inconsistent with
        # the restored state. Twin of the lifecycle run guard.
        if is_forward_resume_blocked(
            getattr(initial_state, "runtime_restore_status", "complete")
        ):
            logger.warning(
                f"[{self.session_id}] run blocked: "
                "runtime_restore_status=unavailable; refusing to start the graph"
            )
            self._set_status(initial_state, TaskStatus.FAILED)
            return initial_state

        # Entry guard: already blocked on required snapshot failure — hold,
        # do not start the graph or clear the gate.
        from ..services.snapshot_v2.errors import is_snapshot_failure_blocked
        if is_snapshot_failure_blocked(initial_state):
            logger.warning(
                f"[{self.session_id}] run blocked: snapshot_failure already present; "
                "holding WAITING_INPUT until Retry/Manage/Terminate"
            )
            self._set_status(initial_state, TaskStatus.WAITING_INPUT)
            initial_state.wait_input_reason = "snapshot_failure"
            try:
                await self._send_state_sync(initial_state, "snapshot_failure")
            except Exception:
                pass
            return initial_state

        # Guard: terminal restore → no workflow, mark complete immediately.
        if initial_state.restore_resume_kind == "terminal":
            logger.info(
                f"[{self.session_id}] Terminal restore — marking complete "
                f"(current_round={initial_state.current_round})"
            )
            self._set_status(initial_state, TaskStatus.COMPLETED)
            self._consume_restore_resume_metadata(initial_state)
            return initial_state

        # Guard: idle restore (branch_id set but no resume kind) → persist only.
        # The session was restored from a manual snapshot with no continuation
        # target; the user must explicitly start analysis from the UI.
        # Exception: restore_start_intent="explicit_start" means the user clicked
        # Start — let the graph run (restore_entry will preserve round/plan).
        if (
            initial_state.restore_resume_kind is None
            and (
                initial_state.restore_branch_id
                or initial_state.last_restored_snapshot_id
            )
            and getattr(initial_state, "restore_start_intent", None) != "explicit_start"
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
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}

        # Verify workflow was properly initialized
        if self.compiled_graph is None:
            error_msg = "Workflow not properly initialized - compiled_graph is None"
            logger.error(f"[{self.session_id}] {error_msg}")
            await self._send_message("error", error_msg)
            self._set_status(initial_state, TaskStatus.FAILED)
            return initial_state

        try:
            logger.info(f"Starting workflow execution for {thread_id}")
            final_state = await self.compiled_graph.ainvoke(
                workflow_state,
                config
            )
            logger.info(f"Workflow completed for {thread_id}")
            return final_state["analysis_state"]

        except WorkflowTerminatedError as e:
            logger.warning(f"[{self.session_id}] Workflow terminated by user: {e}")
            self._set_status(initial_state, TaskStatus.STOPPED)
            initial_state.wait_input_reason = None
            await self._send_state_sync(initial_state, StateSyncChangeType.LLM_RETRY.value)
            return initial_state

        except RequiredSnapshotBlocked as e:
            logger.warning(
                f"[{self.session_id}] Required snapshot blocked workflow: {e}"
            )
            from ..services.snapshot_v2.failure_gate import (
                ensure_workflow_boundary_retry_context,
            )

            ensure_workflow_boundary_retry_context(
                initial_state,
                e,
                previous_status="running",
            )
            try:
                await self._send_state_sync(initial_state, "snapshot_failure")
            except Exception:
                pass
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
        from ..config import is_snapshot_v2_enabled
        from ..services.snapshot_v2.feature_gate import (
            normalize_disabled_snapshot_state,
        )

        if not is_snapshot_v2_enabled() and normalize_disabled_snapshot_state(
            initial_state,
            workflow_active=True,
        ):
            if initial_state.status == TaskStatus.COMPLETED:
                return initial_state

        # Entry guard: snapshot_failure or pending post-report snapshot means
        # reports already ran; hold WAITING_INPUT for capture-only retry.
        # WS reconnect must not re-run report generators.
        from ..services.snapshot_v2.errors import is_snapshot_failure_blocked
        pending = getattr(initial_state, "pending_required_snapshot", None)
        if is_snapshot_failure_blocked(initial_state) or pending is not None:
            logger.warning(
                f"[{self.session_id}] generate_report blocked: "
                f"snapshot_failure={is_snapshot_failure_blocked(initial_state)} "
                f"pending_required_snapshot={pending is not None}; "
                "holding WAITING_INPUT without re-running report generators"
            )
            from ..services.snapshot_v2.failure_gate import (
                ensure_post_report_retry_context,
                enter_snapshot_failure_gate,
            )

            if pending is not None and not initial_state.snapshot_failure:
                enter_snapshot_failure_gate(
                    initial_state,
                    {
                        "code": "pending_required_snapshot",
                        "message": (
                            "Post-report snapshot is pending; use Retry to "
                            "re-enter capture only (reports will not re-run), "
                            "or Skip to complete without this snapshot."
                        ),
                        "allowed_actions": ["retry", "skip", "manage", "terminate"],
                    },
                    retry_context={
                        "producer": "post_report",
                        "resume_policy": "complete",
                        "snapshot_request": dict(pending),
                        "previous_status": "generating_report",
                        "report_generation_id": pending.get("report_generation_id"),
                    },
                )
            else:
                # Always merge typed post_report context (service bare gate
                # may have snapshot_failure without retry_context).
                ensure_post_report_retry_context(
                    initial_state,
                    previous_status="generating_report",
                    snapshot_request=(
                        dict(pending) if isinstance(pending, dict) else {"phase": "post_report"}
                    ),
                )
            try:
                await self._send_state_sync(initial_state, "snapshot_failure")
            except Exception:
                pass
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
            state = await self._generate_final_report(workflow_state)
            state = await self._generate_graphic_abstract(state)
            state = await self._generate_detailed_report(state)
            logger.info(f"[{self.session_id}] Report generation completed")
            return state["analysis_state"]
        except RequiredSnapshotBlocked as e:
            logger.warning(
                f"[{self.session_id}] Required snapshot blocked during report generation: {e}"
            )
            from ..services.snapshot_v2.failure_gate import (
                ensure_post_report_retry_context,
            )

            # Always merge typed post_report context (bare service gate may
            # already set snapshot_failure without retry_context).
            ensure_post_report_retry_context(
                initial_state,
                e,
                previous_status="generating_report",
            )
            try:
                await self._send_state_sync(initial_state, "snapshot_failure")
            except Exception:
                pass
            return initial_state
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"[{self.session_id}] Report generation failed: {error_detail}")
            await self._send_message("error", f"Report generation failed: {str(e)}")
            self._set_status(initial_state, TaskStatus.FAILED)
            return initial_state

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
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        final_state = await self.compiled_graph.ainvoke(workflow_state, config)
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
            from ..config import is_snapshot_v2_enabled
            from ..services.snapshot_v2.feature_gate import (
                normalize_disabled_snapshot_state,
            )

            if not is_snapshot_v2_enabled():
                normalize_disabled_snapshot_state(
                    analysis_state,
                    workflow_active=True,
                )

        # §1d: never run the graph forward on a kernel that is inconsistent with
        # the restored state. This is the workflow-level twin of the lifecycle
        # resume block and covers the WebSocket-reconnect resume path.
        if analysis_state is not None and is_forward_resume_blocked(
            getattr(analysis_state, "runtime_restore_status", "complete")
        ):
            logger.warning(
                f"[{self.session_id}] resume blocked: runtime_restore_status=unavailable; "
                "holding without running the graph"
            )
            return analysis_state

        # Entry guard: already blocked on required snapshot failure.
        from ..services.snapshot_v2.errors import is_snapshot_failure_blocked
        if analysis_state is not None and is_snapshot_failure_blocked(analysis_state):
            logger.warning(
                f"[{self.session_id}] resume blocked: snapshot_failure already present; "
                "holding without running the graph"
            )
            self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
            analysis_state.wait_input_reason = "snapshot_failure"
            return analysis_state

        # Terminal restore → no workflow, mark complete immediately.
        if analysis_state is not None and analysis_state.restore_resume_kind == "terminal":
            logger.info(
                f"[{self.session_id}] Terminal restore in resume — marking complete "
                f"(round={analysis_state.current_round})"
            )
            self._set_status(analysis_state, TaskStatus.COMPLETED)
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
        # checkpointer, then remain frozen until the restore owner releases it.
        await self.await_restore_safe_point()
        if analysis_state is not None:
            from ..state import sessions as live_sessions

            refreshed_state = live_sessions.get(self.session_id)
            if refreshed_state is not None:
                analysis_state = refreshed_state
                self._state = refreshed_state

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}

        # Get current state
        state = await self.compiled_graph.aget_state(config)

        # Graph already at END (no pending nodes) — nothing to resume.
        # Continuing would inject stale RUNNING status and ainvoke(None)
        # returns it unchanged, causing an infinite resume loop on every
        # WebSocket reconnect (kernel restart → reconnect → resume → RUNNING).
        if state and state.values and not state.next:
            effective = state.values.get("analysis_state") or analysis_state
            if effective and effective.status not in (TaskStatus.COMPLETED, TaskStatus.STOPPED):
                self._set_status(effective, TaskStatus.COMPLETED)
            logger.info(f"[{self.session_id}] Resume requested but graph already terminated; marking complete")
            return effective

        if state and state.values:
            # Restore _current_analysis_state for callbacks
            self._current_analysis_state = state.values.get("analysis_state")
            self._current_graph_state = state.values

            # Inject latest analysis_state if provided (so user changes during pause are picked up)
            if analysis_state:
                await self.compiled_graph.aupdate_state(config, {"analysis_state": analysis_state})
                self._current_analysis_state = analysis_state
                # Keep in-memory graph state in sync with the updated checkpoint
                self._current_graph_state = {**state.values, "analysis_state": analysis_state}

            # Sync skip_nodes from sandbox_state to graph state (retry-node API stores
            # skip decisions in analysis_state.sandbox_state, but with_retry checks
            # the top-level WorkflowState)
            skip_nodes = []
            effective_analysis = analysis_state or state.values.get("analysis_state")
            if effective_analysis and effective_analysis.sandbox_state:
                skip_nodes = effective_analysis.sandbox_state.get("skip_nodes", [])
            if skip_nodes:
                await self.compiled_graph.aupdate_state(config, {"skip_nodes": skip_nodes})
                logger.info(f"[{self.session_id}] Injected skip_nodes into graph state: {skip_nodes}")

            # Resume from checkpoint
            logger.info(f"[{self.session_id}] Found saved state, resuming execution")
            try:
                final_state = await self.compiled_graph.ainvoke(
                    None,
                    config
                )
                logger.info(f"[{self.session_id}] Workflow resumed and completed")
                return final_state["analysis_state"]
            except WorkflowTerminatedError as e:
                logger.warning(f"[{self.session_id}] Workflow terminated by user: {e}")
                effective = analysis_state or state.values.get("analysis_state")
                if effective:
                    self._set_status(effective, TaskStatus.STOPPED)
                    effective.wait_input_reason = None
                return effective or analysis_state or state.values["analysis_state"]
            except RequiredSnapshotBlocked as e:
                logger.warning(
                    f"[{self.session_id}] Required snapshot blocked on resume: {e}"
                )
                from ..services.snapshot_v2.failure_gate import (
                    ensure_workflow_boundary_retry_context,
                )

                effective = analysis_state or state.values.get("analysis_state")
                if effective:
                    ensure_workflow_boundary_retry_context(
                        effective,
                        e,
                        previous_status="running",
                    )
                return effective or analysis_state

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
                final_state = await self.compiled_graph.ainvoke(workflow_state, config)
                logger.info(f"[{self.session_id}] Exact graph-node continuation completed")
                self._consume_restore_resume_metadata(final_state["analysis_state"])
                return final_state["analysis_state"]
            except WorkflowTerminatedError as e:
                logger.warning(f"[{self.session_id}] Workflow terminated by user: {e}")
                if analysis_state:
                    self._set_status(analysis_state, TaskStatus.STOPPED)
                    analysis_state.wait_input_reason = None
                return analysis_state
            except RequiredSnapshotBlocked as e:
                logger.warning(
                    f"[{self.session_id}] Required snapshot blocked after restore: {e}"
                )
                from ..services.snapshot_v2.failure_gate import (
                    ensure_workflow_boundary_retry_context,
                )

                if analysis_state:
                    ensure_workflow_boundary_retry_context(
                        analysis_state,
                        e,
                        previous_status="running",
                    )
                return analysis_state

        if analysis_state.status in (TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION):
            logger.info(
                f"[{self.session_id}] Status is {analysis_state.status.value}, "
                f"deferring to HITL-hold path"
            )
            return analysis_state

        if getattr(analysis_state, "restore_start_intent", None) == "explicit_start":
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

    async def pause(self, thread_id: str):
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
            analysis_state = state.values["analysis_state"]
            self._set_status(analysis_state, TaskStatus.PAUSED)

            # Save sandbox state including Python session using dill.
            # The Jupyter kernel is single-threaded, so this queues behind any
            # in-flight execute() and captures a clean post-step snapshot.
            if self.sandbox:
                await self.sandbox.save_session()
                self.sandbox.notebook_state.save()
                logger.info(f"[{self.session_id}] Sandbox state saved")

            # State will be automatically checkpointed
            logger.info(f"[{self.session_id}] Workflow paused successfully")
    
    async def close(self):
        """Clean up resources and save state"""
        logger.info(f"[{self.session_id}] Closing workflow and cleaning up resources")
        if self.sandbox:
            try:
                # This will save the notebook state before stopping
                await self.sandbox.stop()
                logger.info(f"[{self.session_id}] Sandbox stopped")
            except Exception as e:
                logger.exception(f"[{self.session_id}] Error stopping sandbox")
                
        if hasattr(self, '_db_conn') and self._db_conn:
            try:
                await self._db_conn.close()
            except Exception as e:
                logger.exception(f"[{self.session_id}] Error closing sqlite connection")
            finally:
                self._db_conn = None
                
        if self.llm_manager:
            try:
                await self.llm_manager.close()
                logger.info(f"[{self.session_id}] LLM manager closed")
            except Exception as e:
                logger.exception(f"[{self.session_id}] Error closing LLM manager")

        # Close literature-owned runtime clients before provider cleanup
        if hasattr(self, 'literature_agent') and self.literature_agent:
            close_runtime_clients = getattr(self.literature_agent, "close_runtime_clients", None)
            if close_runtime_clients:
                try:
                    await close_runtime_clients()
                except Exception:
                    logger.exception(f"[{self.session_id}] Error closing literature runtime clients")

        # Close per-agent LLM clients managed by the settings provider
        if hasattr(self, '_settings_provider') and self._settings_provider:
            await self._settings_provider.cleanup()
    
    def get_notebook_path(self) -> Optional[str]:
        """Get path to the saved Jupyter notebook"""
        if self.sandbox:
            return self.sandbox.get_notebook_path()
        return None
