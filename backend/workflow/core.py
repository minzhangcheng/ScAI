"""
LangGraph workflow for multi-agent single-cell analysis
"""
import asyncio
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

from ..models.schemas import (
    AnalysisState, AgentType, TaskStatus, RoundResult,
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
from ..constants import LlmFailureAction, WaitInputReason, StateSyncChangeType
from ..utils.settings_provider import AgentSettingsProvider
from ..utils.sandbox import LocalJupyterSandbox
from ..config import SystemConfig, get_session_workspace, get_session_states


class WorkflowTerminatedError(RuntimeError):
    """Raised when the user chooses Terminate in a disabled-agent dialog."""
    pass


class SingleCellAnalysisWorkflow(RoutingMixin, ContextMixin, MessagingMixin, CallbacksMixin, AnnotationMixin, PlanningMixin, AgentNodesMixin, SynthesisMixin, InitializeMixin, GraphMixin, ReportingMixin):
    """
    Multi-agent workflow for single-cell analysis using LangGraph
    """

    @staticmethod
    def _set_status(state: AnalysisState, status: TaskStatus):
        state.status = status
        if status == TaskStatus.WAITING_INPUT:
            state.waiting_input_started_at = datetime.now()
        else:
            state.waiting_input_started_at = None
    
    def __init__(
        self,
        config: SystemConfig,
        session_id: str,
        message_callback: Optional[Callable] = None,
        state_callback: Optional[Callable] = None  # Callback to save state after each round
    ):
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

        # LLM failure intervention — serialized via Lock
        self._llm_failure_lock = asyncio.Lock()
        self._llm_failure_events: Dict[str, asyncio.Event] = {}
        self._llm_failure_actions: Dict[str, Optional[str]] = {}
        self._llm_failure_generation: int = 0  # monotonic counter for idempotency

        # Supplementary analysis requests queue (set by API, consumed by HITL loop).
        # Each item is (request_text, target) where target is "annotation" | "direction" | "round".
        # Use a deque so consecutive rapid requests are preserved in order rather than overwritten.
        self._supplementary_queue: deque[Tuple[str, str]] = deque()

        # Settings provider — set in initialize(), provides hot-reloadable settings
        self._settings_provider: Optional[AgentSettingsProvider] = None

    async def _wait_for_llm_intervention(
        self,
        error: UpstreamExhaustedError,
        node_name: str,
        analysis_state: AnalysisState,
    ) -> LlmFailureAction:
        """Block workflow, show LlmTimeoutDialog, wait for user action.

        Returns:
            LlmFailureAction.RETRY — user clicked Retry (caller should clear caches and retry)
            LlmFailureAction.SKIP  — user clicked Skip or Terminate
        """
        async with self._llm_failure_lock:
            logger.error(
                f"[{self.session_id}] Node {node_name} LLM upstream exhausted: "
                f"agent={error.agent_name} model={error.model} server={error.llm_server}"
            )
            await self._send_message(
                "error",
                f"Node {node_name} LLM call failed: {str(error)}",
                AgentType.PI,
            )

            self._llm_failure_generation += 1
            gen = self._llm_failure_generation

            if node_name not in self._llm_failure_events:
                self._llm_failure_events[node_name] = asyncio.Event()
            event = self._llm_failure_events[node_name]
            event.clear()

            analysis_state.wait_input_reason = WaitInputReason.LLM_FAILURE.value
            analysis_state.llm_failure_info = {
                "agent": error.agent_name,
                "model": error.model,
                "llm_server": error.llm_server,
                "error": str(error),
                "node_name": node_name,
                "generation": gen,
            }
            self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
            await self._send_state_sync(analysis_state, StateSyncChangeType.LLM_FAILURE.value)

            logger.info(
                f"[{self.session_id}] Waiting for user intervention "
                f"on LLM failure (node={node_name}, generation={gen})"
            )

        # CRITICAL: Release lock before awaiting Event so API can acquire it
        try:
            await event.wait()

            action = self._llm_failure_actions.pop(node_name, None)
            analysis_state.wait_input_reason = None
            analysis_state.llm_failure_info = None
            self._set_status(analysis_state, TaskStatus.RUNNING)
            await self._send_state_sync(analysis_state, StateSyncChangeType.LLM_RETRY.value)
        except asyncio.CancelledError:
            analysis_state.wait_input_reason = None
            analysis_state.llm_failure_info = None
            raise
        finally:
            self._llm_failure_events.pop(node_name, None)

        if action == LlmFailureAction.RETRY.value:
            if self._settings_provider is not None:
                self._settings_provider._client_cache.clear()
            return LlmFailureAction.RETRY
        return LlmFailureAction.SKIP

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
        async with self._llm_failure_lock:
            logger.warning(
                f"[{self.session_id}] Node {node_name} requires {agent_name} "
                f"agent but its model is empty (disabled)."
            )
            await self._send_message(
                "error",
                f"Node {node_name} requires {agent_name} agent which is disabled.",
                AgentType.PI,
            )
            self._llm_failure_generation += 1
            gen = self._llm_failure_generation
            if node_name not in self._llm_failure_events:
                self._llm_failure_events[node_name] = asyncio.Event()
            event = self._llm_failure_events[node_name]
            event.clear()

            analysis_state.wait_input_reason = WaitInputReason.AGENT_DISABLED.value
            analysis_state.agent_disabled_info = {
                "agent": agent_name,
                "node_name": node_name,
                "generation": gen,
            }
            self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
            await self._send_state_sync(analysis_state, StateSyncChangeType.AGENT_DISABLED.value)

            logger.info(
                f"[{self.session_id}] Waiting for user to resolve disabled "
                f"agent (agent={agent_name}, node={node_name}, generation={gen})"
            )

        try:
            await event.wait()
            action = self._llm_failure_actions.pop(node_name, None)
            analysis_state.wait_input_reason = None
            analysis_state.agent_disabled_info = None
            self._set_status(analysis_state, TaskStatus.RUNNING)
            await self._send_state_sync(analysis_state, StateSyncChangeType.AGENT_RESUMED.value)
        except asyncio.CancelledError:
            analysis_state.wait_input_reason = None
            analysis_state.agent_disabled_info = None
            raise
        finally:
            self._llm_failure_events.pop(node_name, None)

        if action == LlmFailureAction.RETRY.value:
            if self._settings_provider is not None:
                self._settings_provider._client_cache.clear()
            return LlmFailureAction.RETRY
        if action == LlmFailureAction.TERMINATE.value:
            return LlmFailureAction.TERMINATE
        return LlmFailureAction.SKIP

    def signal_continue(self):
        """Signal the workflow to continue from wait_for_user_input"""
        self._continue_event.set()

    def signal_resume(self):
        """Release the cooperative pause gate so blocked nodes/agent loops can proceed."""
        self._pause_event.set()

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
        current_gen = self._llm_failure_generation
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

        # Technique Agent (lightweight doc lookup, used by Supervisor and Bioinformatics)
        technique_model = self._settings_provider.get_model("technique")
        if technique_model and technique_model.strip():
            self.technique_agent = TechniqueAgent(
                settings_provider=self._settings_provider,
                message_callback=self._technique_message_callback,
            )
            self.technique_agent.set_session(self.session_id)
        else:
            self.technique_agent = None

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
        )
        self.bioinformatics_agent.set_session(self.session_id)
        self.bioinformatics_agent._pause_event = self._pause_event

        # Initialize optional agents — disabled if model is empty
        pathology_model = self._settings_provider.get_model("pathology")
        if pathology_model and pathology_model.strip():
            self.pathology_agent = PathologyAgent(
                settings_provider=self._settings_provider,
                tile_size=self.config.image_tile_size,
                max_tiles=50,
            )
            self.pathology_agent.set_session(self.session_id)
        else:
            self.pathology_agent = None
            logger.info(f"[{self.session_id}] Pathology agent disabled (empty model)")

        biochemistry_model = self._settings_provider.get_model("biochemistry")
        if biochemistry_model and biochemistry_model.strip():
            self.biochemistry_agent = BiochemistryAgent(
                settings_provider=self._settings_provider,
            )
            self.biochemistry_agent.set_session(self.session_id)
        else:
            self.biochemistry_agent = None
            logger.info(f"[{self.session_id}] Biochemistry agent disabled (empty model)")

        # Literature agent with mode-specific configuration
        literature_config = self._settings_provider.get_literature_config()
        self.literature_agent = LiteratureAgent(
            settings_provider=self._settings_provider,
            literature_mode=literature_config.get('mode', 'deep_research'),
            deep_research_model=literature_config.get('deep_research_model', ''),
            deep_research_llm_base_url=literature_config.get('deep_research_llm_base_url', ''),
            deep_research_api_key=literature_config.get('deep_research_api_key', ''),
            google_api_key=literature_config.get('google_api_key', ''),
            pubmed_email=literature_config.get('pubmed_email', ''),
            max_search_rounds=literature_config.get('max_search_rounds', 5),
            proxy_url=self.config.api.proxy_url,
        )
        self.literature_agent.set_session(self.session_id)

        # Register deep research LLM client for cleanup (if literature agent created one)
        if hasattr(self.literature_agent, '_deep_research_llm_client') and self.literature_agent._deep_research_llm_client:
            self._settings_provider.register_client(
                ('deep_research', 'deep_research'),
                self.literature_agent._deep_research_llm_client,
            )

        # Initialize Painter Agent for graphic abstract generation (optional)
        painter_model = self._settings_provider.get_model("painter")
        if painter_model and painter_model.strip():
            self.painter_agent = PainterAgent(
                settings_provider=self._settings_provider,
            )
            self.painter_agent.set_session(self.session_id)
        else:
            self.painter_agent = None
            logger.info(f"[{self.session_id}] Painter agent disabled (empty model)")

        # Initialize Supervisor Agent for real-time analysis monitoring (optional)
        supervisor_model = self._settings_provider.get_model("supervisor")
        if supervisor_model and supervisor_model.strip():
            self.supervisor_agent = SupervisorAgent(
                settings_provider=self._settings_provider,
                pi_callback=self._pi_callback,
                message_callback=self._supervisor_message_callback,
                session_id=self.session_id,
                technique_agent=self.technique_agent,
            )
            self.supervisor_agent.set_sandbox(self.sandbox)
        else:
            self.supervisor_agent = None
            logger.info(f"[{self.session_id}] Supervisor agent disabled (empty model)")

        # Log which agents are active
        pi_model = self._settings_provider.get_model("pi")
        bioinformatics_model = self._settings_provider.get_model("bioinformatics")
        literature_model = self._settings_provider.get_model("literature")
        logger.info(f"[{self.session_id}] Agents initialized: PI={pi_model}, Pathology={pathology_model or 'DISABLED'}, Bioinformatics={bioinformatics_model}, Biochemistry={biochemistry_model or 'DISABLED'}, Literature={literature_model}, Painter={painter_model or 'DISABLED'}, Supervisor={supervisor_model or 'DISABLED'}")

        # Wire on_retry callback into every agent's LLM client so that
        # WebSocket observers receive progress updates during retry waits
        # instead of seeing dead silence.
        self._install_retry_notifiers()

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
        """Create an automatic snapshot at a key analysis point."""
        from ..utils.snapshot import SnapshotManager
        try:
            if self.sandbox:
                if hasattr(self.sandbox, 'notebook_state') and self.sandbox.notebook_state:
                    await asyncio.to_thread(self.sandbox.notebook_state.save)
                await self.sandbox.save_session()

            async def notify(sid, msg_type, content):
                await self._send_message(msg_type, content, AgentType.PI)

            await SnapshotManager.create_and_register_snapshot(
                self.session_id, state, phase, notify_fn=notify,
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] Auto-snapshot failed: {e}")

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
        # Store state for settings provider initialization
        self._state = initial_state

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
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"[{self.session_id}] Report generation failed: {error_detail}")
            await self._send_message("error", f"Report generation failed: {str(e)}")
            self._set_status(initial_state, TaskStatus.FAILED)
            return initial_state

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

        # Re-initialize agents and components if not already initialized
        if self.compiled_graph is None:
            # Restore state for settings provider initialization
            if analysis_state:
                self._state = analysis_state
            logger.info(f"[{self.session_id}] Re-initializing workflow components for resume")
            await self.initialize()

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

        # No checkpoint found — start a fresh run from initialize_round.
        # This is the expected path after snapshot restore (checkpoint was deleted).
        if analysis_state is None:
            logger.error(f"[{self.session_id}] No checkpoint and no analysis_state for {thread_id}")
            raise ValueError(f"No saved state found for thread {thread_id}")

        if analysis_state.status in (TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION):
            logger.info(
                f"[{self.session_id}] Status is {analysis_state.status.value}, "
                f"deferring to HITL-hold path"
            )
            return analysis_state

        logger.info(
            f"[{self.session_id}] No checkpoint found, starting fresh run "
            f"(current_round={analysis_state.current_round})"
        )
        await self._send_message(
            "status",
            f"Resuming from snapshot (Round {analysis_state.current_round})..."
        )

        self._set_status(analysis_state, TaskStatus.RUNNING)
        self._current_analysis_state = analysis_state

        workflow_state = self._make_initial_workflow_state(
            analysis_state, current_round=analysis_state.current_round
        )
        try:
            final_state = await self.compiled_graph.ainvoke(workflow_state, config)
            logger.info(f"[{self.session_id}] Fresh run after restore completed")
            return final_state["analysis_state"]
        except WorkflowTerminatedError as e:
            logger.warning(f"[{self.session_id}] Workflow terminated by user: {e}")
            if analysis_state:
                self._set_status(analysis_state, TaskStatus.STOPPED)
                analysis_state.wait_input_reason = None
            return analysis_state

    async def pause(self, thread_id: str):
        """Pause the workflow and save sandbox state including Python session.

        Clears the cooperative pause gate FIRST so the bioinformatics agent's
        autonomous loop and the @with_retry wrapper block at their next check,
        then waits for any in-flight sandbox.execute to drain (kernel
        serializes execute and save_session naturally) before persisting state.
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

        # Close per-agent LLM clients managed by the settings provider
        if hasattr(self, '_settings_provider') and self._settings_provider:
            await self._settings_provider.cleanup()
    
    def get_notebook_path(self) -> Optional[str]:
        """Get path to the saved Jupyter notebook"""
        if self.sandbox:
            return self.sandbox.get_notebook_path()
        return None
