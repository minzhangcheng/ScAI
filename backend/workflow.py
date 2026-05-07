"""
LangGraph workflow for multi-agent single-cell analysis
"""
import asyncio
import logging
import traceback
import uuid
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import json
import operator
import random

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import aiosqlite

# Configure logger
logger = logging.getLogger(__name__)

# Maximum retry attempts for workflow nodes
MAX_NODE_RETRIES = 3
GRAPH_RECURSION_LIMIT = 100

from .models.schemas import (
    AnalysisState, AnalysisStep, AnalysisPhase, AgentType, TaskStatus, RoundResult,
    WebSocketMessage, AgentMessage, MessageRole, BiochemistryIndicator,
    PublicationFigure, FigureReviewStatus,
    ClusterAnnotation, AnnotationConfirmationState, DetailedReportSegment,
    GraphicAbstract,
)
from .agents.pi_agent import PIAgent
from .agents.pathology_agent import PathologyAgent
from .agents.bioinformatics_agent import BioinformaticsAgent
from .agents.biochemistry_agent import BiochemistryAgent
from .agents.literature_agent import LiteratureAgent
from .agents.painter_agent import PainterAgent
from .agents.supervisor_agent import SupervisorAgent
from .agents.technique_agent import TechniqueAgent
from .agents.base import AgentConfig
from .utils.llm_client import LLMManager, OpenAICompatibleClient, UpstreamExhaustedError
from .constants import LlmFailureAction, WaitInputReason, StateSyncChangeType, NODE_TO_RESULT_KEY
from .utils.settings_provider import AgentSettingsProvider
from .utils.sandbox import LocalJupyterSandbox
from .config import SystemConfig, REPORTS_DIR, WORKSPACE_DIR, get_session_reports, get_session_workspace, get_session_states
from .services.report import DetailedReportGenerator, compress_image_to_base64


def _normalize_tasks(tasks: list) -> List[str]:
    """Convert task items to strings — LLMs sometimes return dicts instead of plain strings."""
    result = []
    for t in tasks:
        if isinstance(t, str):
            result.append(t)
        elif isinstance(t, dict):
            parts = []
            for key in ("task", "description", "topic", "name"):
                if key in t:
                    parts.append(str(t[key]))
                    break
            if not parts:
                parts.append(str(t))
            if "rationale" in t:
                parts.append(f"(Rationale: {t['rationale']})")
            if "recommended_methods" in t and t["recommended_methods"]:
                methods = t["recommended_methods"]
                if isinstance(methods, list):
                    parts.append(f"[Methods: {'; '.join(str(m) for m in methods)}]")
                else:
                    parts.append(f"[Methods: {methods}]")
            if "target_readouts" in t and t["target_readouts"]:
                readouts = t["target_readouts"]
                if isinstance(readouts, list):
                    parts.append(f"[Readouts: {'; '.join(str(r) for r in readouts)}]")
                else:
                    parts.append(f"[Readouts: {readouts}]")
            result.append(" ".join(parts))
        else:
            result.append(str(t))
    return result



def with_retry(node_name: str, max_retries: int = MAX_NODE_RETRIES):
    """Decorator to add retry logic to workflow nodes.

    When UpstreamExhaustedError exhausts all retries (LLM timeout/5xx),
    blocks the workflow and waits for user intervention.

    Uses a while loop (not for) so that on 'retry' the attempt counter
    truly resets to 0.  Python's for-loop rebinds the variable each
    iteration, making in-loop assignment ineffective.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, state: 'WorkflowState') -> 'WorkflowState':
            # Check skip_nodes
            if node_name in state.get("skip_nodes", []):
                logger.info(f"[{self.session_id}] Skipping node {node_name} (not activated by PI)")
                if hasattr(self, '_send_message'):
                    await self._send_message(
                        "status",
                        f"{node_name} skipped this round (not needed)",
                        None
                    )
                result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
                return {**state, "failed_node": None, result_key: {"findings": [], "status": "skipped"}}

            last_error = None
            attempt = 0  # manual counter — while loop, not for

            while attempt <= max_retries:
                try:
                    if attempt > 0:
                        await self._send_message(
                            "status",
                            f"Retrying {node_name} (attempt {attempt + 1}/{max_retries + 1})...",
                            AgentType.PI
                        )
                        delay = min(2 ** attempt + random.uniform(0, 1), 30)
                        await asyncio.sleep(delay)

                    result = await func(self, state)
                    return result

                except UpstreamExhaustedError as e:
                    async with self._llm_failure_lock:
                        last_error = e
                        logger.error(
                            f"[{self.session_id}] Node {node_name} LLM upstream exhausted "
                            f"(node attempt {attempt + 1}/{max_retries + 1}): "
                            f"agent={e.agent_name} model={e.model} server={e.llm_server}"
                        )
                        await self._send_message(
                            "error",
                            f"Node {node_name} LLM call failed: {str(e)}",
                            AgentType.PI
                        )

                        analysis_state = state["analysis_state"]

                        self._llm_failure_generation += 1
                        gen = self._llm_failure_generation

                        if node_name not in self._llm_failure_events:
                            self._llm_failure_events[node_name] = asyncio.Event()
                        event = self._llm_failure_events[node_name]
                        event.clear()

                        analysis_state.wait_input_reason = WaitInputReason.LLM_FAILURE.value
                        analysis_state.llm_failure_info = {
                            "agent": e.agent_name,
                            "model": e.model,
                            "llm_server": e.llm_server,
                            "error": str(e),
                            "node_name": node_name,
                            "generation": gen,
                        }
                        self._set_status(analysis_state, TaskStatus.WAITING_INPUT)

                        await self._send_state_sync(analysis_state, StateSyncChangeType.LLM_FAILURE.value)

                        logger.info(
                            f"[{self.session_id}] Waiting for user intervention "
                            f"on LLM failure (node={node_name}, generation={gen})"
                        )

                    # CRITICAL: Release the Lock before awaiting the Event
                    # so the API endpoint can acquire it to set the Event
                    try:
                        await event.wait()

                        action = self._llm_failure_actions.pop(node_name, None)
                        analysis_state.wait_input_reason = None
                        analysis_state.llm_failure_info = None
                        self._set_status(analysis_state, TaskStatus.RUNNING)
                        await self._send_state_sync(analysis_state, StateSyncChangeType.LLM_RETRY.value)
                    except asyncio.CancelledError:
                        # Only clear LLM fields — do NOT touch status,
                        # which is set to PAUSED by stop_analysis.
                        analysis_state.wait_input_reason = None
                        analysis_state.llm_failure_info = None
                        raise
                    finally:
                        self._llm_failure_events.pop(node_name, None)

                    if action == LlmFailureAction.RETRY.value:
                        # User may have changed model/server config — invalidate cache
                        if self._settings_provider is not None:
                            self._settings_provider._client_cache.clear()
                            logger.info(f"[{self.session_id}] Cleared agent client cache for retry of node {node_name}")

                        attempt = 0
                        continue  # re-enter while loop

                    elif action == LlmFailureAction.SKIP.value:
                        result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
                        return {
                            **state,
                            "failed_node": node_name,
                            result_key: {
                                "error": str(last_error),
                                "findings": [],
                                "status": "failed"
                            }
                        }

                    else:
                        # Unknown or None action — defensive: treat as skip
                        logger.warning(
                            f"[{self.session_id}] Unknown LLM failure action "
                            f"'{action}' for node '{node_name}', defaulting to skip"
                        )
                        result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
                        return {
                            **state,
                            "failed_node": node_name,
                            result_key: {
                                "error": str(last_error),
                                "findings": [],
                                "status": "failed"
                            }
                        }

                except Exception as e:
                    last_error = e
                    error_detail = traceback.format_exc()
                    logger.error(
                        f"[{self.session_id}] Node {node_name} failed "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {error_detail}"
                    )
                    await self._send_message(
                        "error",
                        f"Node {node_name} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}",
                        AgentType.PI
                    )
                    attempt += 1

            # All retries exhausted for non-LLM error
            logger.error(
                f"[{self.session_id}] Node {node_name} failed after {max_retries + 1} attempts"
            )
            result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
            return {
                **state,
                "failed_node": node_name,
                result_key: {"error": str(last_error), "findings": [], "status": "failed"}
            }
        return wrapper
    return decorator


class WorkflowState(TypedDict, total=False):
    """State for the LangGraph workflow"""
    analysis_state: AnalysisState
    current_round: int
    pi_instructions: Dict[str, Any]
    pathology_result: Optional[Dict[str, Any]]
    bioinformatics_result: Optional[Dict[str, Any]]
    biochemistry_result: Optional[Dict[str, Any]]
    literature_result: Optional[Dict[str, Any]]
    round_result: Optional[RoundResult]
    should_continue: bool
    error: Optional[str]
    messages: Annotated[List[WebSocketMessage], operator.add]
    # Human intervention fields
    human_guidance: Optional[str]
    human_stop_requested: bool
    human_guidance_processed: bool
    # User input processing fields
    user_action: Optional[str]
    new_data_files: List[Dict[str, Any]]
    adjusted_instructions: Optional[Dict[str, Any]]
    data_requests: Optional[List[Dict[str, Any]]]
    # Retry mechanism fields
    failed_node: Optional[str]  # Name of failed node (for retry)
    node_retry_count: Dict[str, int]  # Retry count per node
    node_errors: Dict[str, str]  # Error messages per node
    skip_nodes: List[str]  # Nodes to skip on resume


class SingleCellAnalysisWorkflow:
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

        # Event for blocking between rounds (wait for user input or continue signal)
        self._continue_event = asyncio.Event()

        # Event for blocking during annotation confirmation (exploration phase)
        self._annotation_event = asyncio.Event()

        # LLM failure intervention — serialized via Lock
        self._llm_failure_lock = asyncio.Lock()
        self._llm_failure_events: Dict[str, asyncio.Event] = {}
        self._llm_failure_actions: Dict[str, Optional[str]] = {}
        self._llm_failure_generation: int = 0  # monotonic counter for idempotency

        # Supplementary analysis request (set by API, consumed by HITL loop)
        self._supplementary_request: Optional[str] = None
        self._supplementary_target: Optional[str] = None  # "annotation" or "direction"

        # Settings provider — set in initialize(), provides hot-reloadable settings
        self._settings_provider: Optional[AgentSettingsProvider] = None

    def signal_continue(self):
        """Signal the workflow to continue from wait_for_user_input"""
        self._continue_event.set()

    def signal_annotation_confirmed(self):
        """Signal that user has confirmed cell type annotations"""
        self._annotation_event.set()

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
        """Signal that user wants supplementary analysis at current HITL point."""
        self._supplementary_request = request_text
        self._supplementary_target = target
        if target == "annotation":
            self._annotation_event.set()
        elif target == "direction":
            self._continue_event.set()

    @property
    def _vision_settings(self) -> Dict[str, bool]:
        """Vision settings — read from Provider so changes are always fresh."""
        return {
            t.value: self._settings_provider.get_vision_setting(t.value)
            for t in AgentType
        }

    @staticmethod
    def _build_fallback_annotations(
        cluster_markers: Dict[str, Any], error: Exception
    ) -> Dict[str, Any]:
        """Build fallback annotations when annotation service fails."""
        fallback_annotations = {}
        for cid in cluster_markers:
            fallback_annotations[str(cid)] = {
                "cell_type": "Unknown",
                "confidence": "low",
                "key_markers": (
                    list(cluster_markers[cid])[:10]
                    if isinstance(cluster_markers[cid], list)
                    else []
                ),
                "evidence": "",
                "low_confidence_reason": f"Annotation service failed: {str(error)}",
            }
        return {"annotations": fallback_annotations, "tissue_summary": ""}

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

    def _build_graph(self):
        """Build the LangGraph workflow with sequential execution for analysis nodes"""

        # Create graph
        self.graph = StateGraph(WorkflowState)

        # Add nodes
        self.graph.add_node("initialize_round", self._initialize_round)
        self.graph.add_node("pi_planning", self._pi_planning)
        self.graph.add_node("pathology_analysis", self._pathology_analysis)
        self.graph.add_node("bioinformatics_analysis", self._bioinformatics_analysis)
        self.graph.add_node("biochemistry_analysis", self._biochemistry_analysis)
        self.graph.add_node("literature_research", self._literature_research)
        self.graph.add_node("pi_synthesis", self._pi_synthesis)
        self.graph.add_node("wait_for_user_input", self._wait_for_user_input)
        self.graph.add_node("check_convergence", self._check_convergence)
        self.graph.add_node("generate_final_report", self._generate_final_report)
        self.graph.add_node("generate_graphic_abstract", self._generate_graphic_abstract)
        self.graph.add_node("generate_detailed_report", self._generate_detailed_report)

        # Add edges - sequential execution (with early-stop check after init)
        self.graph.add_conditional_edges(
            "initialize_round",
            self._should_continue_after_init,
            {
                "continue": "pi_planning",
                "stop": "generate_final_report"
            }
        )

        # Sequential analysis: biochemistry -> pathology -> bioinformatics -> literature
        self.graph.add_edge("pi_planning", "biochemistry_analysis")
        self.graph.add_edge("biochemistry_analysis", "pathology_analysis")
        self.graph.add_edge("pathology_analysis", "bioinformatics_analysis")
        self.graph.add_edge("bioinformatics_analysis", "literature_research")
        self.graph.add_edge("literature_research", "pi_synthesis")

        # After synthesis, wait for user input (blocks until frontend signals continue)
        self.graph.add_edge("pi_synthesis", "wait_for_user_input")

        # After user input wait, check if we should continue
        self.graph.add_conditional_edges(
            "wait_for_user_input",
            self._should_continue,
            {
                "continue": "check_convergence",
                "stop": "generate_final_report"
            }
        )

        self.graph.add_conditional_edges(
            "check_convergence",
            self._check_convergence_condition,
            {
                "continue": "initialize_round",
                "stop": "generate_final_report"
            }
        )

        # Final report flow: summary -> graphic abstract -> detailed report -> END
        self.graph.add_edge("generate_final_report", "generate_graphic_abstract")
        self.graph.add_edge("generate_graphic_abstract", "generate_detailed_report")
        self.graph.add_edge("generate_detailed_report", END)
        
        # Set entry point
        self.graph.set_entry_point("initialize_round")
        
        # Compile
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
    
    async def _send_message(self, msg_type: str, content: Any, agent: Optional[AgentType] = None):
        """Send message through callback"""
        if self.message_callback:
            msg = WebSocketMessage(
                type=msg_type,
                agent=agent,
                content=content
            )
            try:
                await self.message_callback(msg)
            except Exception as e:
                logger.exception("[Workflow] Failed to send message")

    async def _send_state_sync(self, analysis_state: AnalysisState, change_type: str):
        """Send unified state synchronization message after state changes"""
        # Increment version for synchronization
        await analysis_state.increment_version()

        # Save state via callback first
        if self.state_callback:
            try:
                await self.state_callback(analysis_state)
            except Exception as e:
                logger.exception(f"[{self.session_id}] State callback failed")

        # Then send sync message to frontend
        if self.message_callback:
            sync_msg = WebSocketMessage(
                type="state_sync",
                agent=None,
                content={
                    "change_type": change_type,  # "round_complete", "status_change", "data_added", "error_added", "llm_failure", "llm_retry"
                    "version": analysis_state.state_version,
                    "status": analysis_state.status.value,
                    "current_round": analysis_state.current_round,
                    "analysis_phase": analysis_state.analysis_phase,
                    "timestamp": analysis_state.last_sync_timestamp.isoformat(),
                    "unresolved_errors": len(analysis_state.get_unresolved_errors()),
                    # LLM failure intervention fields (single source of truth)
                    "wait_input_reason": analysis_state.wait_input_reason,
                    "llm_failure_info": analysis_state.llm_failure_info,
                }
            )
            try:
                await self.message_callback(sync_msg)
            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to send state sync")

    async def _literature_callback(self, topic: str) -> str:
        """Callback for bioinformatics agent to request literature support"""
        # Use current analysis state if available, otherwise create minimal state
        state = self._current_analysis_state if self._current_analysis_state else AnalysisState()
        result = await self.literature_agent.process(
            state,
            topic,
            use_deep_research=True
        )
        return result.get("report", "")
    
    async def _pi_callback(self, question: str, context: str) -> str:
        """Callback for bioinformatics agent to consult PI for biological decisions"""
        await self._send_message("status", "Bioinformatics consulting PI...", AgentType.BIOINFORMATICS)
        await self._send_message("pi_question", f"Question: {question}\nContext: {context}", AgentType.BIOINFORMATICS)
        
        # Use PI agent to answer the biological question
        prompt = f"""The bioinformatics analyst is asking for your guidance on a biological decision:

QUESTION: {question}

CONTEXT FROM ANALYSIS:
{context}

Please provide clear guidance on:
1. Your recommended decision/direction
2. The reasoning behind your recommendation
3. Any specific parameters, genes, or pathways to focus on
4. Potential pitfalls to avoid

Be specific and actionable in your response.

IMPORTANT: Respond in English only. Do not use any other language."""

        try:
            response = await self.pi_agent.chat_without_history(prompt, temperature=self._settings_provider.get_temperature('pi'))
            await self._send_message("pi_response", response, AgentType.PI)
            return response
        except Exception as e:
            error_msg = f"PI consultation failed: {str(e)}"
            await self._send_message("error", error_msg, AgentType.PI)
            return "Unable to get PI guidance. Please proceed with standard bioinformatics practices."

    def _build_annotation_context_report(self, state: AnalysisState, graph_state: Optional[WorkflowState]) -> str:
        """Build a comprehensive markdown summary of current analysis context for cell annotation."""
        parts = ["### Analysis Context Report\n"]
        
        # 1. Experimental Design
        parts.append("#### Experimental Design")
        parts.append(state.build_experimental_design_summary())
        parts.append("")

        # 2. Biochemistry & Pathology findings (from graph state if available)
        if graph_state:
            biochem = graph_state.get("biochemistry_result")
            if biochem and biochem.get("findings"):
                parts.append("#### Biochemistry Findings")
                for f in biochem["findings"]:
                    parts.append(f"- {f}")
                parts.append("")

            patho = graph_state.get("pathology_result")
            if patho and patho.get("findings"):
                parts.append("#### Pathology Findings")
                for f in patho["findings"]:
                    parts.append(f"- {f}")
                parts.append("")
        
        # 3. Previous Rounds' Key Findings
        if state.round_results and len(state.round_results) > 0:
            parts.append("#### Previous Analysis Highlights")
            last_round = state.round_results[-1]
            if last_round.pi_summary:
                parts.append(last_round.pi_summary)
            parts.append("")

        # 4. Literature Searches (current round)
        lit_steps = [step for step in state.analysis_log if step.step_type == "search_literature"]
        if lit_steps:
            parts.append("#### Recent Literature Research")
            for step in lit_steps[-3:]:  # Last 3 searches
                topic = step.literature_topic or "Unknown topic"
                result = step.literature_result or "No findings recorded"
                parts.append(f"**Topic:** {topic}")
                parts.append(f"**Findings:** {result[:300]}...\n")

        return "\n".join(parts)

    async def _build_and_save_detailed_segment(
        self, state: AnalysisState, phase: str, title: str,
        graph_state=None,
        round_result=None,
    ) -> DetailedReportSegment:
        """Build a detailed report segment and save the combined HTML."""
        from .services.report import (
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
    def _make_initial_workflow_state(analysis_state: AnalysisState, current_round: int = 0) -> "WorkflowState":
        """Build a clean WorkflowState for a fresh graph invocation."""
        return {
            "analysis_state": analysis_state,
            "current_round": current_round,
            "pi_instructions": {},
            "pathology_result": None,
            "bioinformatics_result": None,
            "biochemistry_result": None,
            "literature_result": None,
            "round_result": None,
            "should_continue": True,
            "error": None,
            "messages": [],
        }

    async def _create_auto_snapshot(self, state: AnalysisState, phase: str):
        """Create an automatic snapshot at a key analysis point."""
        from .utils.snapshot import SnapshotManager
        try:
            if self.sandbox:
                if hasattr(self.sandbox, 'notebook_state') and self.sandbox.notebook_state:
                    self.sandbox.notebook_state.save()
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
        )

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

    @staticmethod
    def _build_cluster_annotations(annotations: Dict) -> List[ClusterAnnotation]:
        """Build ClusterAnnotation list from annotation dict."""
        return [
            ClusterAnnotation(
                cluster_id=cid,
                cell_type=info.get("cell_type", "Unknown"),
                subtypes=info.get("subtypes", []),
                key_markers=info.get("key_markers", []),
                confidence=info.get("confidence", "low"),
                evidence=info.get("evidence", ""),
                alternative_cell_types=info.get("alternative_cell_types", []),
                low_confidence_reason=info.get("low_confidence_reason", ""),
            )
            for cid, info in annotations.items()
            if isinstance(info, dict)
        ]

    async def _annotate_clusters_callback(
        self,
        cluster_markers: Dict,
        tissue: str,
        organism: str,
        analysis_log: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Callback for bioinformatics agent to annotate all clusters via literature.

        When annotations have not yet been confirmed by the user, this callback additionally:
        1. Has PI Agent review annotations and assign confidence scores
        2. Blocks execution until the user confirms annotations via frontend
        """
        n_clusters = len(cluster_markers)
        if n_clusters == 0:
            logger.warning(f"[{self.session_id}] annotate_clusters called with empty cluster_markers")
            return {
                "status": "error",
                "annotations": {},
                "error": "cluster_markers is empty. Run rank_genes_groups and extract markers first."
            }

        state = self._current_analysis_state if self._current_analysis_state else AnalysisState()

        # Early exit: if annotations were already confirmed, return cached results
        # without re-running the expensive deep research + LLM annotation pipeline
        if state.annotation_state and state.annotation_state.confirmed:
            logger.info(f"[{self.session_id}] Annotation already confirmed, returning cached annotations")
            cached_annotations = {}
            for ca in state.annotation_state.cluster_annotations:
                cached_annotations[ca.cluster_id] = {
                    "cell_type": ca.user_override or ca.cell_type,
                    "subtypes": ca.subtypes,
                    "key_markers": ca.key_markers,
                    "confidence": ca.confidence,
                    "evidence": ca.evidence,
                    "alternative_cell_types": ca.alternative_cell_types,
                    "low_confidence_reason": ca.low_confidence_reason,
                }
            return {
                "annotations": cached_annotations,
                "tissue_summary": state.annotation_state.tissue_summary or "",
                "already_confirmed": True,
            }

        # --- PI-guided annotation: PI decides search strategy, Literature executes ---
        # PI guidance has two independent aspects:
        # - Search guidance (query generation + evaluation): only for pubmed_llm mode
        # - Matching guidance (cluster→cell_type): applies to ALL modes
        lit_mode = getattr(self.literature_agent, 'literature_mode', 'pubmed_llm')
        pi_search_guided = False
        pi_query = None
        pi_evaluate = None

        # Step 0a: PI generates search query (only for pubmed_llm mode)
        if lit_mode == "pubmed_llm":
            try:
                pi_query = await self.pi_agent.generate_annotation_search_query(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    state=state,
                )
                logger.info(f"[{self.session_id}] PI generated annotation query: {pi_query[:100]}")

                # Build evaluation closure that captures cluster context for PI
                async def pi_evaluate(articles, search_history):
                    return await self.pi_agent.evaluate_annotation_search_results(
                        cluster_markers=cluster_markers,
                        organism=organism,
                        tissue=tissue,
                        articles=articles,
                        search_history=search_history,
                    )

                pi_search_guided = True
                await self._send_message(
                    "status",
                    f"PI-guided literature search for {n_clusters} clusters in {organism} {tissue}...",
                    AgentType.PI
                )
            except Exception as e:
                logger.warning(f"[{self.session_id}] PI query generation failed, falling back to Literature Agent: {e}")
        else:
            logger.info(
                f"[{self.session_id}] Literature mode is '{lit_mode}', "
                f"PI will guide matching only (search handled by {lit_mode})"
            )

        if not pi_search_guided:
            await self._send_message(
                "status",
                f"Literature-based cluster annotation: {n_clusters} clusters in {organism} {tissue}...",
                AgentType.LITERATURE
            )

        # Step 1: Literature Agent executes search
        # - PI-guided search + skip matching (pubmed_llm with successful PI query)
        # - Self-guided search + skip matching (non-pubmed modes, PI does matching only)
        # - Full self-guided (fallback when PI query generation failed)
        try:
            if pi_search_guided:
                result = await self.literature_agent.annotate_clusters(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    state=state,
                    pi_initial_query=pi_query,
                    pi_evaluate_fn=pi_evaluate,
                    skip_matching=True,  # PI will handle matching in Step 2
                )
            elif lit_mode == "pubmed_llm":
                # PI query generation failed — Literature Agent does everything
                result = await self.literature_agent.annotate_clusters(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    state=state
                )
            else:
                # Non-pubmed mode (deep_research/gemini): search self-guided, PI matches
                result = await self.literature_agent.annotate_clusters(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    state=state,
                    skip_matching=True,  # PI will handle matching in Step 2
                )
        except Exception as e:
            error_msg = f"Cluster annotation failed: {str(e)}"
            logger.error(f"[{self.session_id}] {error_msg}")
            await self._send_message("error", error_msg, AgentType.LITERATURE)
            # Build fallback annotations so the confirmation flow still triggers
            result = self._build_fallback_annotations(cluster_markers, e)

        # Step 2: PI Agent matches clusters to cell types
        # Triggered when skip_matching was used (either PI-guided search or non-pubmed modes)
        if result.get("status") == "literature_only":
            literature_report = result.get("literature_report", "")
            await self._send_message(
                "status",
                f"PI is matching {n_clusters} clusters to cell types using literature...",
                AgentType.PI
            )
            try:
                matching_result = await self.pi_agent.annotate_clusters_from_literature(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    literature_report=literature_report,
                )
                result["status"] = "completed"
                result["annotations"] = matching_result.get("annotations", {})
                result["tissue_summary"] = matching_result.get("tissue_summary", "")
            except Exception as e:
                logger.warning(
                    f"[{self.session_id}] PI matching failed, falling back to Literature Agent: {e}"
                )
                # Fallback: re-run full annotation with Literature Agent
                try:
                    result = await self.literature_agent.annotate_clusters(
                        cluster_markers=cluster_markers,
                        organism=organism,
                        tissue=tissue,
                        state=state,
                    )
                except Exception as e2:
                    logger.error(f"[{self.session_id}] Literature fallback also failed: {e2}")
                    result = self._build_fallback_annotations(cluster_markers, e2)

        annotations = result.get("annotations", {})
        n_annotated = sum(
            1 for v in annotations.values()
            if isinstance(v, dict) and v.get("cell_type", "Unknown") != "Unknown"
        )
        await self._send_message(
            "status",
            f"Cluster annotation complete: {n_annotated}/{n_clusters} clusters identified",
            AgentType.LITERATURE
        )

        # --- PI review + user confirmation (if not already confirmed) ---
        if not (state.annotation_state and state.annotation_state.confirmed):
            # Supplement key_markers with top markers from rank_genes_groups
            # so PI and user can judge cell types with at least 10 genes
            for cid, info in annotations.items():
                if isinstance(info, dict) and cid in cluster_markers:
                    existing = info.get("key_markers", [])
                    if len(existing) < 10:
                        seen = set(existing)
                        supplemented = list(existing)
                        for gene in cluster_markers[cid]:
                            if gene not in seen:
                                supplemented.append(gene)
                                seen.add(gene)
                            if len(supplemented) >= 10:
                                break
                        info["key_markers"] = supplemented

            # Step 1: PI Agent reviews annotations and assigns confidence
            await self._send_message(
                "status",
                "PI is reviewing cluster annotations and assigning confidence scores...",
                AgentType.PI
            )
            try:
                pi_review = await self.pi_agent.review_cluster_annotations(
                    state, annotations, tissue, organism
                )
                for cid, review in pi_review.items():
                    if cid in annotations and isinstance(review, dict):
                        annotations[cid].update(review)
                result["annotations"] = annotations
            except Exception as e:
                logger.warning(f"[{self.session_id}] PI annotation review failed: {e}")
                # Continue with original annotations if PI review fails

            # Step 2: Build AnnotationConfirmationState
            cluster_anns = self._build_cluster_annotations(annotations)

            # Guard: if annotations were already confirmed (e.g. backend restart),
            # apply overrides directly without re-blocking for user confirmation
            if state.annotation_state and state.annotation_state.confirmed:
                logger.info(f"[{self.session_id}] Annotation already confirmed, applying overrides directly")
                for ca in state.annotation_state.cluster_annotations:
                    if ca.user_override and ca.cluster_id in annotations:
                        annotations[ca.cluster_id]["cell_type"] = ca.user_override
                        logger.info(
                            f"[{self.session_id}] User override (restored): cluster {ca.cluster_id} "
                            f"→ {ca.user_override}"
                        )
                result["annotations"] = annotations
                await self._send_message(
                    "status",
                    "Annotation confirmed (restored), continuing analysis...",
                    AgentType.LITERATURE
                )
                return result

            # Temporarily populate state.analysis_log with the in-progress log
            # from the bioinformatics agent (it hasn't returned yet, so the log
            # hasn't been synced to state via _after_bioinformatics)
            original_log = state.analysis_log
            if analysis_log:
                state.analysis_log = analysis_log

            try:
                # Generate the context summary report
                analysis_summary = self._build_annotation_context_report(state, self._current_graph_state)

                # Generate downloadable HTML annotation context report
                from .services.report import generate_annotation_context_report_html
                try:
                    html_path = generate_annotation_context_report_html(state, self._current_graph_state, cluster_anns)
                    state.annotation_report_html_path = html_path
                    report_html_url = f"/api/session/{self.session_id}/annotation-report/html"
                except Exception as e:
                    logger.warning(f"[{self.session_id}] Annotation context report generation failed: {e}")
                    report_html_url = None

                # Generate detailed report segment for annotation checkpoint
                # (must happen before restoring original_log so the segment sees the in-progress log)
                await self._build_and_save_detailed_segment(
                    state, "pre_annotation",
                    "Analysis Summary Before Cell Annotation",
                    graph_state=self._current_graph_state,
                )
                detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if state.detailed_report_html_path else None
            finally:
                # Restore original log — proper sync happens in _after_bioinformatics
                state.analysis_log = original_log

            ann_state = AnnotationConfirmationState(
                cluster_annotations=cluster_anns,
                tissue=tissue,
                organism=organism,
                tissue_summary=result.get("tissue_summary", ""),
                analysis_summary=analysis_summary,
                pending=True,
            )
            state.annotation_state = ann_state

            # Auto-snapshot before annotation HITL (after annotation_state is set,
            # so rollback restores the dialog data needed by the frontend)
            await self._create_auto_snapshot(state, "pre_annotation")

            # Step 3: Annotation confirmation loop (supports supplementary analysis)
            original_status = state.status
            while True:
                self._annotation_event.clear()

                await self._send_message("annotation_confirmation", {
                    "annotations": [a.model_dump() for a in cluster_anns],
                    "tissue": tissue,
                    "organism": organism,
                    "tissue_summary": result.get("tissue_summary", ""),
                    "analysis_summary": analysis_summary,
                    "report_html_url": report_html_url,
                    "detailed_report_url": detailed_report_url,
                }, AgentType.LITERATURE)

                self._set_status(state, TaskStatus.WAITING_ANNOTATION)
                await self._send_state_sync(state, "annotation_confirmation")

                logger.info(f"[{self.session_id}] Waiting for user annotation confirmation...")
                await self._annotation_event.wait()

                # Check: supplementary analysis requested?
                if self._supplementary_request and self._supplementary_target == "annotation":
                    req = self._supplementary_request
                    self._supplementary_request = None
                    self._supplementary_target = None
                    state.status = TaskStatus.RUNNING
                    await self._send_state_sync(state, "supplementary_running")

                    await self._run_supplementary_analysis(state, req, "annotation")

                    # Regenerate context report with new data
                    analysis_summary = self._build_annotation_context_report(state, self._current_graph_state)

                    # Re-run PI review of annotations with supplementary context
                    try:
                        pi_review = await self.pi_agent.review_cluster_annotations(
                            state, annotations, tissue, organism
                        )
                        for cid, review in pi_review.items():
                            if cid in annotations and isinstance(review, dict):
                                annotations[cid].update(review)
                        result["annotations"] = annotations
                        cluster_anns = self._build_cluster_annotations(annotations)
                    except Exception as e:
                        logger.warning(f"[{self.session_id}] PI re-review after supplementary failed: {e}")

                    # Regenerate HTML report
                    try:
                        html_path = generate_annotation_context_report_html(state, self._current_graph_state, cluster_anns)
                        state.annotation_report_html_path = html_path
                        report_html_url = f"/api/session/{self.session_id}/annotation-report/html"
                    except Exception as e:
                        logger.warning(f"[{self.session_id}] Report re-generation failed: {e}")

                    detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if state.detailed_report_html_path else None

                    ann_state = AnnotationConfirmationState(
                        cluster_annotations=cluster_anns,
                        tissue=tissue,
                        organism=organism,
                        tissue_summary=result.get("tissue_summary", ""),
                        analysis_summary=analysis_summary,
                        pending=True,
                    )
                    state.annotation_state = ann_state
                    continue  # Loop back to re-show dialog
                else:
                    # Normal confirmation
                    logger.info(f"[{self.session_id}] Annotation confirmation received")
                    if state.status == TaskStatus.PAUSED:
                        logger.info(f"[{self.session_id}] User paused during annotation wait, returning early")
                        return result
                    state.status = original_status
                    await self._send_state_sync(state, "annotation_resumed")
                    break

            # Step 4: Apply user overrides
            confirmed = state.annotation_state
            if confirmed and confirmed.confirmed:
                for ca in confirmed.cluster_annotations:
                    if ca.user_override and ca.cluster_id in annotations:
                        annotations[ca.cluster_id]["cell_type"] = ca.user_override
                        logger.info(
                            f"[{self.session_id}] User override: cluster {ca.cluster_id} "
                            f"→ {ca.user_override}"
                        )
                result["annotations"] = annotations

            await self._send_message(
                "status",
                "Annotation confirmed, continuing analysis...",
                AgentType.LITERATURE
            )

        return result

    async def _agent_message_callback(self, msg_type: str, content: str):
        """Callback for agent messages"""
        await self._send_message(msg_type, content, AgentType.BIOINFORMATICS)

    async def _supervisor_message_callback(self, msg_type: str, content: str):
        """Callback for supervisor agent messages"""
        await self._send_message(msg_type, content, AgentType.SUPERVISOR)

    async def _technique_message_callback(self, msg_type: str, content: str):
        """Callback for technique agent messages"""
        await self._send_message(msg_type, content, AgentType.TECHNIQUE)

    async def _supervisor_callback(self, step: AnalysisStep, figures: Optional[List[str]] = None) -> Optional[str]:
        """Bridge bioinformatics agent's supervisor_callback to SupervisorAgent."""
        if not self.supervisor_agent:
            return None
        try:
            return await self.supervisor_agent.evaluate_step(step, figures or [])
        except Exception as e:
            logger.warning(f"[{self.session_id}] Supervisor callback failed: {e}")
            return None

    async def _figure_review_callback(self, figure: PublicationFigure) -> Dict[str, Any]:
        """
        Callback for bioinformatics agent to get PI review of generated figures.
        Routes to different reviewers based on vision capability settings.

        Args:
            figure: PublicationFigure to review

        Returns:
            Dict with review results (approved, feedback, suggestions)
        """
        pi_vision = self._vision_settings.get('pi', True)
        bio_vision = self._vision_settings.get('bioinformatics', True)
        sup_vision = (
            self._vision_settings.get('supervisor', True)
            if self.supervisor_agent else False
        )

        # 4.1: All models in the chain lack vision → skip image review
        if not pi_vision and not bio_vision and not sup_vision:
            logger.info(f"[{self.session_id}] All agents lack vision, auto-approving figure: {figure.name}")
            review = {
                "approved": True,
                "quality_score": 7,
                "feedback": "Auto-approved (no vision-capable model in pipeline)",
                "suggestions": [],
                "must_fix": [],
            }
            await self._send_message(
                "figure_approved",
                {
                    "figure_id": figure.id,
                    "figure_name": figure.name,
                    "approved": True,
                    "quality_score": 7,
                    "feedback": review["feedback"],
                    "suggestions": [],
                    "must_fix": []
                },
                AgentType.PI
            )
            return review

        # 4.3.2: PI has no vision
        if not pi_vision:
            if bio_vision:
                # Bioinformatics agent self-reviews its own figure
                logger.info(f"[{self.session_id}] PI lacks vision, bio agent self-reviewing figure: {figure.name}")
                review = await self._bio_self_figure_review(figure)
            elif sup_vision and self.supervisor_agent:
                # 4.3.3: Both PI and bio lack vision → supervisor guides review
                logger.info(f"[{self.session_id}] PI and bio lack vision, supervisor reviewing figure: {figure.name}")
                review = await self._supervisor_figure_review(figure)
            else:
                # No one can see the figure — auto-approve
                review = {
                    "approved": True,
                    "quality_score": 7,
                    "feedback": "Auto-approved (no vision-capable reviewer available)",
                    "suggestions": [],
                    "must_fix": [],
                }

            await self._send_message(
                "figure_approved" if review.get("approved") else "figure_revision",
                {
                    "figure_id": figure.id,
                    "figure_name": figure.name,
                    "approved": review.get("approved", False),
                    "quality_score": review.get("quality_score", 5),
                    "feedback": review.get("feedback", ""),
                    "suggestions": review.get("suggestions", []),
                    "must_fix": review.get("must_fix", [])
                },
                AgentType.BIOINFORMATICS if bio_vision else AgentType.SUPERVISOR
            )
            return review

        # Default: PI reviews with vision (existing flow)
        logger.info(f"[{self.session_id}] PI reviewing figure: {figure.name}")

        await self._send_message(
            "figure_review",
            {
                "figure_id": figure.id,
                "figure_name": figure.name,
                "figure_type": figure.figure_type,
                "description": figure.description,
                "revision_count": figure.revision_count,
                "status": "reviewing"
            },
            AgentType.PI
        )

        state = self._current_analysis_state if self._current_analysis_state else AnalysisState()

        review = await self.pi_agent.review_figure(
            state,
            figure,
            figure.png_base64
        )

        await self._send_message(
            "figure_approved" if review.get("approved") else "figure_revision",
            {
                "figure_id": figure.id,
                "figure_name": figure.name,
                "approved": review.get("approved", False),
                "quality_score": review.get("quality_score", 5),
                "feedback": review.get("feedback", ""),
                "suggestions": review.get("suggestions", []),
                "must_fix": review.get("must_fix", [])
            },
            AgentType.PI
        )

        return review

    async def _vision_figure_review(
        self,
        agent,
        figure: PublicationFigure,
        prompt: str,
        image_b64: str,
        temperature: float = 0.3,
        label: str = "agent",
        **vision_kwargs,
    ) -> Dict[str, Any]:
        """Shared helper: call an agent's vision API to review a figure and parse the result."""
        try:
            agent._set_llm_context()
            response = await agent.llm_client.chat_with_vision(
                model=agent.model,
                text_prompt=prompt,
                images=[image_b64],
                temperature=temperature,
                **vision_kwargs,
            )
            response_text = response.content if response and response.content else ""
        except Exception as e:
            logger.warning(f"[{self.session_id}] {label} figure review vision failed: {e}")
            return {
                "approved": True,
                "quality_score": 7,
                "feedback": f"{label} vision review failed, auto-approved",
                "suggestions": [],
                "must_fix": [],
            }

        if hasattr(agent, '_safe_parse_json'):
            data = agent._safe_parse_json(response_text, default={})
        elif hasattr(agent, '_parse_supervisor_response'):
            data = agent._parse_supervisor_response(response_text) if response_text else {}
        else:
            data = {}
        if data and ("approved" in data or "quality_score" in data):
            score = int(data.get("quality_score", 7))
            return {
                "approved": data.get("approved", score >= 7),
                "quality_score": score,
                "feedback": data.get("feedback", ""),
                "suggestions": data.get("suggestions", []),
                "must_fix": data.get("must_fix", []),
                "requires_reanalysis": data.get("requires_reanalysis", False),
            }
        return {
            "approved": True,
            "quality_score": 7,
            "feedback": "Could not parse review response, auto-approved",
            "suggestions": [],
            "must_fix": [],
        }

    async def _bio_self_figure_review(self, figure: PublicationFigure) -> Dict[str, Any]:
        """Bioinformatics agent self-reviews its own figure using vision."""
        prompt = f"""## Figure Self-Review for Publication Quality

You generated this figure and now need to review it for publication quality.

**Figure Information:**
- Name: {figure.name}
- Type: {figure.figure_type}
- Description: {figure.description}
- Revision count: {figure.revision_count}

**Review Criteria:**
1. Scientific Accuracy: Does the figure accurately represent the data?
2. Clarity: Are labels clear and readable?
3. Aesthetics: Is the color scheme appropriate and layout clean?
4. Completeness: Are all necessary elements present (legend, axis labels, titles)?
5. Publication Quality: Would this be acceptable in a high-impact journal?

**Respond in JSON format:**
```json
{{
    "approved": true/false,
    "quality_score": 1-10,
    "feedback": "Detailed assessment",
    "suggestions": ["suggestion 1"],
    "must_fix": ["critical issue"],
    "requires_reanalysis": false
}}
```
Be strict but fair. Approve if quality_score >= 7."""

        return await self._vision_figure_review(
            self.bioinformatics_agent, figure, prompt, figure.png_base64,
            temperature=0.3, label="Bio self-review",
        )

    async def _supervisor_figure_review(self, figure: PublicationFigure) -> Dict[str, Any]:
        """Supervisor agent reviews figure using vision.

        When the supervisor is the *only* reviewer (PI and bio both lack vision),
        it uses a relaxed approval threshold (score >= 5) instead of the normal 7,
        since a small model shouldn't block the pipeline.  Suggestions are still
        forwarded as non-blocking feedback.

        A per-figure rejection counter (_sup_rejections) tracks consecutive
        rejections.  After 2 consecutive rejections of the same figure, the
        review auto-sets ``requires_reanalysis=True`` so the bioinformatics
        agent breaks out of the revision loop instead of retrying endlessly.
        """
        compressed = self.supervisor_agent._compress_base64_figure(figure.png_base64)

        # Determine if supervisor is the sole reviewer (PI and bio both lack vision)
        pi_vision = self._vision_settings.get('pi', True)
        bio_vision = self._vision_settings.get('bioinformatics', True)
        supervisor_is_sole_reviewer = not pi_vision and not bio_vision

        prompt = f"""## Figure Quality Review

Review this figure for publication quality.
- Name: {figure.name}
- Type: {figure.figure_type}
- Description: {figure.description}

Check for: blank/empty plots, missing labels, overlapping text, poor color choices,
missing legends, unreadable fonts. Score 1-10.

Respond in JSON:
```json
{{
    "approved": true/false,
    "quality_score": 1-10,
    "feedback": "assessment",
    "suggestions": ["suggestion"],
    "must_fix": ["critical issue"]
}}
```
Approve if quality_score >= 7."""

        review = await self._vision_figure_review(
            self.supervisor_agent, figure, prompt, compressed,
            temperature=0.2, label="Supervisor", max_tokens=1024,
        )

        # --- Supervisor-as-sole-reviewer degradation logic ---
        if supervisor_is_sole_reviewer and not review.get("approved", False):
            score = review.get("quality_score", 0)

            # Relaxed threshold: score >= 5 is acceptable when supervisor is the
            # only reviewer, since the small model's judgment is less reliable.
            if score >= 5:
                logger.info(
                    f"[{self.session_id}] Supervisor-only mode: accepting figure "
                    f"{figure.name} at score={score} (relaxed threshold)"
                )
                review["approved"] = True
                review["feedback"] = (
                    f"[Accepted with relaxed threshold (supervisor-only mode, score={score}/10)] "
                    + review.get("feedback", "")
                )
            else:
                # Track consecutive rejections per figure
                if not hasattr(self, "_sup_rejections"):
                    self._sup_rejections: Dict[str, int] = {}
                self._sup_rejections[figure.name] = self._sup_rejections.get(figure.name, 0) + 1
                reject_count = self._sup_rejections[figure.name]

                if reject_count >= 2:
                    logger.warning(
                        f"[{self.session_id}] Supervisor rejected figure {figure.name} "
                        f"{reject_count} times consecutively, escalating to requires_reanalysis"
                    )
                    review["requires_reanalysis"] = True
                    review["feedback"] = (
                        f"[Escalated after {reject_count} consecutive rejections] "
                        + review.get("feedback", "")
                    )

        # Reset rejection counter on approval
        if review.get("approved", False) and hasattr(self, "_sup_rejections"):
            self._sup_rejections.pop(figure.name, None)

        return review

    # --- Round context helpers ---

    def _build_previous_findings_context(
        self,
        analysis_state: AnalysisState,
        agent_type: str,
    ) -> str:
        """Build a text block of an agent's previous round findings.

        Args:
            analysis_state: Current analysis state with round_results
            agent_type: One of "pathology", "bioinformatics", "biochemistry", "literature"

        Returns:
            Formatted string, or "" if no prior findings exist.
        """
        if not analysis_state.round_results:
            return ""

        findings_key = f"{agent_type}_findings"
        parts = []
        for prev_round in analysis_state.round_results:
            findings = getattr(prev_round, findings_key, [])
            if findings:
                parts.append(f"Round {prev_round.round_number}: " + "; ".join(
                    str(f) if not isinstance(f, str) else f for f in findings[:5]
                ))

        if not parts:
            return ""

        return (
            "\n\n## Your Previous Analysis Findings\n"
            "Below are your findings from prior rounds. Build upon these — do NOT repeat the same analyses.\n"
            + "\n".join(parts)
        )

    def _collect_current_round_results(self, state: WorkflowState) -> Dict[str, List[str]]:
        """Collect findings from agents already completed in the current round."""
        collected = {}
        for result_key, findings_key in [
            ("biochemistry_result", "biochemistry_findings"),
            ("pathology_result", "pathology_findings"),
            ("bioinformatics_result", "bioinformatics_findings"),
        ]:
            result = state.get(result_key)
            if (result and result.get("findings")
                    and result.get("status") not in ("skipped", "no_data", "waiting_data")):
                collected[findings_key] = result["findings"]
        return collected

    def _build_code_history_summary(
        self,
        analysis_state: AnalysisState,
        max_entries: int = 20,
    ) -> str:
        """Build a truncated summary of previous code executions for bioinformatics agent.

        Args:
            analysis_state: Current analysis state with code_history
            max_entries: Max number of code cells to include

        Returns:
            Formatted string, or "" if no code history exists.
        """
        if not analysis_state.code_history:
            return ""

        recent = analysis_state.code_history[-max_entries:]
        parts = [
            "\n\n## Previous Code Execution History",
            "Below is a summary of previously executed code. Reference these results — do NOT re-run completed steps.\n",
        ]
        for i, rec in enumerate(recent):
            code_lines = rec.code.strip().split("\n")
            preview = "\n".join(code_lines[:3])
            if len(code_lines) > 3:
                preview += f"\n# ... ({len(code_lines)} lines total)"

            output = (rec.output or "")[:200]
            if len(rec.output or "") > 200:
                output += "..."

            entry = f"**Cell {i + 1}**:\n```python\n{preview}\n```"
            if output:
                entry += f"\nOutput: {output}"
            if rec.figures:
                entry += f"\n[Generated {len(rec.figures)} figure(s)]"
            parts.append(entry)

        return "\n".join(parts)

    # Node functions
    async def _initialize_round(self, state: WorkflowState) -> WorkflowState:
        """Initialize a new analysis round, processing any pending user input"""
        analysis_state = state["analysis_state"]
        # Update current analysis state for callbacks (e.g., literature_callback)
        self._current_analysis_state = analysis_state
        self._current_graph_state = state
        logger.info(f"[{self.session_id}] Initializing round {analysis_state.current_round + 1}")
        
        # Process pending human input (if any)
        human_guidance = None
        user_action = None
        new_data_files = []
        adjusted_instructions = None
        
        if analysis_state.pending_human_input or analysis_state.pending_uploaded_files:
            human_message = analysis_state.pending_human_input or ""
            uploaded_files = analysis_state.pending_uploaded_files or []
            
            await self._send_message(
                "human_input_received",
                f"Processing user input: {human_message[:100]}{'...' if len(human_message) > 100 else ''}" + 
                (f" ({len(uploaded_files)} files)" if uploaded_files else ""),
                AgentType.PI
            )
            
            # Let PI Agent process the user input
            pi_response = await self.pi_agent.process_user_input(
                analysis_state,
                human_message,
                uploaded_files
            )
            
            user_action = pi_response.get("action", "continue")
            human_guidance = human_message
            
            await self._send_message(
                "response",
                pi_response.get("response", "Acknowledged"),
                AgentType.PI
            )
            
            # Check if user requested to stop
            if user_action == "stop":
                await self._send_message(
                    "status",
                    "User requested to conclude analysis. Generating final report...",
                    AgentType.PI
                )
                # Clear pending input
                analysis_state.pending_human_input = None
                analysis_state.pending_uploaded_files = []
                
                return {
                    **state,
                    "analysis_state": analysis_state,
                    "should_continue": False,
                    "human_stop_requested": True
                }
            
            # Process file assignments from PI
            file_assignments = pi_response.get("file_assignments", {})

            # Track new data files for prioritized analysis
            if file_assignments.get("pathology"):
                for file_info in file_assignments["pathology"]:
                    if file_info:
                        new_data_files.append({
                            "section": "pathology",
                            "filename": file_info.get("filename"),
                            "data_type": file_info.get("image_type", "user_upload"),
                            "reason": file_info.get("reason", "")
                        })

            # Extract and add biochemistry values from user input
            biochemistry_values = pi_response.get("biochemistry_values", {})
            if biochemistry_values and isinstance(biochemistry_values, dict):
                for indicator, data in biochemistry_values.items():
                    if isinstance(data, dict) and "value" in data:
                        try:
                            value = float(data["value"])
                            result = BiochemistryIndicator(
                                indicator=indicator,
                                value=value,
                                unit=data.get("unit", ""),
                                interpretation=data.get("interpretation")
                            )
                            analysis_state.biochemistry_indicators.append(result)
                            logger.info(f"[{self.session_id}] Added biochemistry value: {indicator}={value} {data.get('unit', '')}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"[{self.session_id}] Failed to parse biochemistry value for {indicator}: {e}")

            # Track biochemistry data in new_data_files for agent activation logic
            if biochemistry_values:
                new_data_files.append({
                    "section": "biochemistry",
                    "filename": None,
                    "data_type": "indicator_values",
                    "reason": "User-provided biochemistry indicators"
                })

            # Track biochemistry files from file_assignments (reuse from line 1793)
            if file_assignments.get("biochemistry"):
                for file_info in file_assignments["biochemistry"]:
                    if file_info:
                        new_data_files.append({
                            "section": "biochemistry",
                            "filename": file_info.get("filename"),
                            "data_type": file_info.get("image_type", "user_upload"),
                            "reason": file_info.get("reason", "")
                        })

            # Get adjusted instructions
            adjusted_instructions = pi_response.get("adjusted_instructions")
            
            # Clear pending input
            analysis_state.pending_human_input = None
            analysis_state.pending_uploaded_files = []
        
        # Increment round counter (skip if resuming from snapshot restore,
        # since current_round is already correct from the snapshot)
        if not analysis_state.last_restored_snapshot_id:
            analysis_state.current_round += 1
        self._set_status(analysis_state, TaskStatus.RUNNING)

        # Clear residual should_continue from previous convergence check
        # to prevent stale False value from terminating an extended round
        state["should_continue"] = True

        await self._send_message(
            "status",
            f"Starting round {analysis_state.current_round} analysis"
        )

        # Send state sync for round initialization
        await self._send_state_sync(analysis_state, "round_start")

        return {
            **state,
            "analysis_state": analysis_state,
            "current_round": analysis_state.current_round,
            "human_guidance": human_guidance,
            "user_action": user_action,
            "new_data_files": new_data_files,
            "adjusted_instructions": adjusted_instructions,
            "pathology_result": None,
            "bioinformatics_result": None,
            "biochemistry_result": None,
            "literature_result": None,
            "round_result": None,
            # Clear retry-related state for new round
            "node_retry_count": {},
            "node_errors": {},
            "failed_node": None,
            # Reset skip_nodes so stale values from previous round do not carry over
            "skip_nodes": [],
            # Reset continuation flags for new round
            "should_continue": True,
            "human_stop_requested": False,
            "human_guidance_processed": False
        }

    @with_retry("pi_planning")
    async def _pi_planning(self, state: WorkflowState) -> WorkflowState:
        """PI agent plans the round, incorporating human guidance and adjusted instructions"""
        analysis_state = state["analysis_state"]
        human_guidance = state.get("human_guidance")
        adjusted_instructions = state.get("adjusted_instructions")

        # Reset PI message history at start of each round's planning phase.
        # Dynamic planning calls (plan_agent_tasks) will accumulate context
        # within this round via self.chat(), building a coherent chain of thought.
        self.pi_agent.reset_history()

        logger.info(f"[{self.session_id}] PI planning for round {analysis_state.current_round}")
        if human_guidance:
            logger.info(f"[{self.session_id}] Human guidance received: {human_guidance[:100]}...")

        await self._send_message("status", "PI is creating analysis plan...", AgentType.PI)
        
        # If we already have adjusted instructions from user input processing, use them
        # BUT: in innovation phase (round 2+), don't short-circuit direction selection —
        # treat adjusted_instructions as supplementary guidance alongside the direction
        if adjusted_instructions and analysis_state.current_round <= 1:
            await self._send_message(
                "message",
                f"Using adjusted instructions from user input:\n{json.dumps(adjusted_instructions, ensure_ascii=False, indent=2)}",
                AgentType.PI
            )
            return {
                **state,
                "pi_instructions": adjusted_instructions,
                "skip_nodes": [],
                "human_guidance_processed": True
            }
        
        # Normal planning without pre-processed instructions
        skip_nodes = []  # Empty for round 1; populated for round 2+
        if analysis_state.current_round == 1:
            # Round 1: Exploration phase — unbiased comprehensive exploration
            instructions = await self.pi_agent.initial_exploration_instruction(
                analysis_state,
                analysis_state.initial_instruction
            )
        else:
            # Round 2+: Innovation phase — hypothesis-driven targeted analysis
            if analysis_state.analysis_phase != "innovation":
                analysis_state.analysis_phase = AnalysisPhase.INNOVATION

            # Check directions_queue first (multi-direction mode), then single direction
            user_direction = analysis_state.user_chosen_direction
            if analysis_state.directions_queue:
                # Multi-direction mode: peek first, pop only on success
                current_dir = analysis_state.directions_queue[0]
                dir_text = current_dir.get("direction", "")
                dir_notes = current_dir.get("notes", "")
                logger.info(f"[{self.session_id}] Innovation phase: direction from queue → {dir_text[:100]} (remaining: {len(analysis_state.directions_queue) - 1})")
                # Set current direction immediately so retry has correct context
                analysis_state.current_round_direction = dir_text
                try:
                    instructions = await self.pi_agent.innovation_instruction(
                        analysis_state, dir_text, notes=dir_notes
                    )
                except Exception as e:
                    # Direction stays in queue (never popped) — no re-insert needed
                    analysis_state.current_round_direction = None
                    logger.exception(f"[{self.session_id}] innovation_instruction failed, direction stays in queue")
                    raise
                # Pop only after successful instruction generation
                analysis_state.directions_queue.pop(0)
                # Persist state after instructions are successfully generated
                if self.state_callback:
                    try:
                        await self.state_callback(analysis_state)
                    except Exception as e:
                        logger.exception(f"[{self.session_id}] Failed to persist state after direction pop")
                # Clear single direction field
                analysis_state.user_chosen_direction = None
            elif user_direction and user_direction not in ("", "[PI_DECIDE]"):
                logger.info(f"[{self.session_id}] Innovation phase: user-directed → {user_direction[:100]}")
                instructions = await self.pi_agent.innovation_instruction(
                    analysis_state, user_direction
                )
                # Preserve direction for bio agent phase_instruction and synthesis prompt
                analysis_state.current_round_direction = user_direction
                # Consume the direction to avoid reusing in next round
                analysis_state.user_chosen_direction = None
            elif user_direction == "[PI_DECIDE]" or (not user_direction and analysis_state.current_round == 2):
                # PI autonomously selects 2 best directions
                logger.info(f"[{self.session_id}] Innovation phase: PI selecting best directions")
                selected_dirs = await self.pi_agent.select_best_directions(analysis_state)
                # Set up multi-direction queue
                analysis_state.user_chosen_directions = selected_dirs
                # Pop first direction, rest goes to queue
                first_dir = selected_dirs[0] if selected_dirs else {"direction": "Continue analysis", "notes": ""}
                analysis_state.directions_queue = selected_dirs[1:] if len(selected_dirs) > 1 else []
                analysis_state.max_rounds = len(selected_dirs) + 1  # Round 1 + N direction rounds
                dir_text = first_dir.get("direction", "")
                dir_notes = first_dir.get("notes", "")
                instructions = await self.pi_agent.innovation_instruction(
                    analysis_state, dir_text, notes=dir_notes
                )
                analysis_state.current_round_direction = dir_text
                analysis_state.user_chosen_direction = None
            else:
                # Subsequent innovation rounds (3+): use previous round's next_steps
                last_round = analysis_state.get_latest_round()
                instructions = {
                    "pathology_tasks": [],
                    "bioinformatics_tasks": [],
                    "biochemistry_tasks": [],
                    "literature_topics": []
                }

                # Extract tasks for each agent
                prefix_map = {
                    "[Pathology]": "pathology_tasks",
                    "[Bioinformatics]": "bioinformatics_tasks",
                    "[Biochemistry]": "biochemistry_tasks",
                    "[Literature]": "literature_topics",
                }
                for step in (last_round.next_steps if last_round else []):
                    for prefix, key in prefix_map.items():
                        if step.startswith(prefix):
                            instructions[key].append(step.split("] ", 1)[1].strip() if "] " in step else step[len(prefix):].strip())
                            break

                # Add overview for round 3+ so PI dynamic planning has strategic context
                if last_round:
                    summary = last_round.pi_summary or "Continue analysis based on previous round findings."
                    instructions["overview"] = summary[:500]

            last_round = analysis_state.get_latest_round()

            # Fallback: if prefix matching extracted no tasks at all
            all_empty = not any([
                instructions["pathology_tasks"],
                instructions["bioinformatics_tasks"],
                instructions["biochemistry_tasks"],
                instructions["literature_topics"]
            ])
            if all_empty and last_round:
                if last_round.next_steps:
                    # First fallback: assign unprefixed next_steps to bioinformatics
                    instructions["bioinformatics_tasks"] = last_round.next_steps
                    logger.warning(
                        f"[{self.session_id}] No prefixed tasks found in next_steps, "
                        f"falling back to bioinformatics-only with raw next_steps"
                    )
                elif last_round.pi_summary:
                    # Second fallback: use PI summary
                    fallback_context = last_round.pi_summary[:500]
                    fallback = f"Continue analysis based on previous round findings. Context: {fallback_context}"
                    instructions["bioinformatics_tasks"] = [fallback]
                    instructions["literature_topics"] = [f"Literature search related to: {fallback_context[:200]}"]
                    logger.warning(
                        f"[{self.session_id}] No next_steps available, "
                        f"using PI summary as fallback for bioinformatics and literature"
                    )

            # Build skip_nodes from PI's agent activation decision
            agents_to_activate = set(last_round.agents_for_next_round) if last_round else set()
            reanalysis_agents = set(last_round.reanalysis_agents) if last_round else set()
            agents_to_activate.add("bioinformatics")  # Always activate bioinformatics

            # Default disable logic (from round 2 onwards): skip pathology/biochemistry if no new data
            if analysis_state.current_round >= 2:
                new_pathology_files = [f for f in state.get("new_data_files", []) if f.get("section") == "pathology"]
                new_biochemistry_files = [f for f in state.get("new_data_files", []) if f.get("section") == "biochemistry"]

                # Pathology: default disabled unless new data or PI explicitly requests reanalysis
                if not new_pathology_files and "pathology" not in reanalysis_agents:
                    agents_to_activate.discard("pathology")
                    logger.info(f"[{self.session_id}] Default disabling pathology: no new data and no reanalysis request")

                # Biochemistry: default disabled unless new data or PI explicitly requests reanalysis
                if not new_biochemistry_files and "biochemistry" not in reanalysis_agents:
                    agents_to_activate.discard("biochemistry")
                    logger.info(f"[{self.session_id}] Default disabling biochemistry: no new data and no reanalysis request")

            # Reconcile: if prefix parsing found tasks for an agent, ensure it's activated
            task_to_agent = {
                "pathology_tasks": "pathology",
                "bioinformatics_tasks": "bioinformatics",
                "biochemistry_tasks": "biochemistry",
                "literature_topics": "literature",
            }
            for task_key, agent_key in task_to_agent.items():
                if instructions.get(task_key) and agent_key not in agents_to_activate:
                    agents_to_activate.add(agent_key)
                    logger.info(
                        f"[{self.session_id}] Auto-activated {agent_key}: "
                        f"has tasks but was not in agents_for_next_round"
                    )

            skip_nodes = []
            agent_node_map = {
                "pathology": "pathology_analysis",
                "biochemistry": "biochemistry_analysis",
                "literature": "literature_research",
            }
            for agent_key, node_name in agent_node_map.items():
                if agent_key not in agents_to_activate:
                    skip_nodes.append(node_name)

            if skip_nodes:
                activated = [k for k in agent_node_map if k in agents_to_activate]
                logger.info(
                    f"[{self.session_id}] Round {analysis_state.current_round}: "
                    f"PI activated agents: bioinformatics + {activated}, "
                    f"skipping: {skip_nodes}"
                )

        # In innovation phase, merge adjusted_instructions as supplementary human guidance
        # (not replacing direction-based planning, but augmenting it)
        if adjusted_instructions and analysis_state.current_round > 1:
            for key in ("bioinformatics_tasks", "pathology_tasks", "biochemistry_tasks", "literature_topics"):
                extra = adjusted_instructions.get(key, [])
                if extra:
                    existing = instructions.get(key, [])
                    instructions[key] = _normalize_tasks(existing) + [f"[User guidance] {t}" for t in _normalize_tasks(extra)]
            logger.info(f"[{self.session_id}] Merged adjusted_instructions into innovation planning")

        await self._send_message(
            "message",
            json.dumps(instructions, ensure_ascii=False, indent=2),
            AgentType.PI
        )

        return {
            **state,
            "pi_instructions": instructions,
            "skip_nodes": skip_nodes,
            "human_guidance_processed": True,
        }
    
    @with_retry("pathology_analysis")
    async def _pathology_analysis(self, state: WorkflowState) -> WorkflowState:
        """Pathology agent analyzes images"""
        if not self.pathology_agent:
            logger.info(f"[{self.session_id}] Pathology agent disabled, skipping")
            return {**state, "failed_node": None, "pathology_result": {"findings": [], "status": "skipped"}}

        analysis_state = state["analysis_state"]
        instructions = state["pi_instructions"]

        tasks = instructions.get("pathology_tasks", [])
        if not tasks:
            tasks = ["Perform standard pathology analysis"]

        pathology_files = analysis_state.get_files_by_section("pathology")
        logger.info(f"[{self.session_id}] Starting pathology analysis with {len(pathology_files)} images")
        await self._send_message("status", "Pathology analysis starting...", AgentType.PATHOLOGY)
        
        # Check if images are available
        if not pathology_files:
            await self._send_message(
                "result",
                {
                    "status": "no_data",
                    "message": "No pathology images uploaded. Please upload H&E or other stained sections for analysis.",
                    "findings": []
                },
                AgentType.PATHOLOGY
            )
            return {
                **state,
                "analysis_state": analysis_state,
                "pathology_result": {"findings": [], "status": "no_data"}
            }
        
        # Log image count
        await self._send_message(
            "status", 
            f"Analyzing {len(pathology_files)} pathology image(s)...", 
            AgentType.PATHOLOGY
        )
        
        # Dynamic PI planning: refine pathology tasks based on biochemistry results
        overview = instructions.get("overview", "")
        if overview:  # Guard: skip for adjusted_instructions (no overview field)
            completed = self._collect_current_round_results(state)
            if completed:
                await self._send_message("status", "PI is refining pathology tasks...", AgentType.PI)
                tasks = await self.pi_agent.plan_agent_tasks(
                    agent_name="pathology",
                    original_tasks=tasks,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        # Enrich instruction with previous findings for round 2+
        task_instruction = "\n".join(_normalize_tasks(tasks))
        previous_context = self._build_previous_findings_context(analysis_state, "pathology")
        if previous_context:
            task_instruction += previous_context

        # Add biochemistry context for cross-reference
        biochemistry_result = state.get("biochemistry_result")
        if biochemistry_result and biochemistry_result.get("findings"):
            biochem_findings = biochemistry_result["findings"][:5]
            task_instruction += (
                "\n\nCurrent round biochemistry findings:\n"
                + "\n".join(f"- {f}" for f in biochem_findings)
            )

        # Let exceptions propagate to @with_retry decorator for proper retry handling
        result = await self.pathology_agent.process(
            analysis_state,
            task_instruction
        )

        await self._send_message(
            "result",
            result,
            AgentType.PATHOLOGY
        )

        # Add message to state
        analysis_state.add_message(
            AgentType.PATHOLOGY,
            MessageRole.ASSISTANT,
            json.dumps(result, ensure_ascii=False)
        )

        return {
            **state,
            "analysis_state": analysis_state,
            "pathology_result": result
        }
    
    @with_retry("bioinformatics_analysis")
    async def _bioinformatics_analysis(self, state: WorkflowState) -> WorkflowState:
        """
        Bioinformatics analysis with PI-controlled iterative execution

        Flow:
        1. PI provides high-level analysis goals
        2. Bioinformatics agent runs autonomously for technical steps
        3. Agent consults PI only for biological decisions
        4. Agent reports findings when analysis phase is complete
        """
        self._current_graph_state = state
        analysis_state = state["analysis_state"]

        logger.info(f"[{self.session_id}] Starting bioinformatics analysis")
        seq_files = analysis_state.get_files_by_section("sequencing")
        logger.info(f"[{self.session_id}] Sequencing files: {len(seq_files)}")

        await self._send_message("status", "Starting bioinformatics analysis...", AgentType.BIOINFORMATICS)
        
        # Check if data is available
        if not analysis_state.has_sequencing_data():
            await self._send_message(
                "result",
                {"message": "Please upload sequencing data first", "findings": []},
                AgentType.BIOINFORMATICS
            )
            return {
                **state,
                "analysis_state": analysis_state,
                "bioinformatics_result": {"findings": [], "status": "waiting_data"}
            }
        
        # Incorporate human guidance from workflow state (set by _initialize_round)
        human_guidance = ""
        if state.get("human_guidance"):
            human_guidance = f"\n\nHuman researcher guidance: {state['human_guidance']}"
        
        # Step 1: PI provides analysis goals
        await self._send_message("status", "PI is defining analysis goals...", AgentType.PI)
        
        # Get PI instructions from workflow state
        pi_instructions = state.get("pi_instructions", {})
        bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
        merge_strategy = pi_instructions.get("merge_strategy", "")
        cluster_comparison_strategy = pi_instructions.get("cluster_comparison_strategy", "")

        # Dynamic PI planning: refine bioinformatics tasks based on completed results
        overview = pi_instructions.get("overview", "")
        if overview:
            completed = self._collect_current_round_results(state)
            if completed:
                await self._send_message("status", "PI is refining bioinformatics tasks...", AgentType.PI)
                bioinf_tasks = await self.pi_agent.plan_agent_tasks(
                    agent_name="bioinformatics",
                    original_tasks=bioinf_tasks,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        # Build previous findings and code history context for round 2+
        previous_bio_findings = self._build_previous_findings_context(analysis_state, "bioinformatics")
        code_history_context = ""
        if analysis_state.current_round > 1:
            code_history_context = self._build_code_history_summary(analysis_state)

        # Include current round pathology findings if available
        pathology_context = ""
        pathology_result = state.get("pathology_result")
        if pathology_result and pathology_result.get("findings"):
            path_findings = pathology_result["findings"][:5]
            pathology_context = (
                "\n\nCurrent round pathology findings:\n"
                + "\n".join(f"- {f}" for f in path_findings)
            )

        # Include current round biochemistry findings if available
        biochemistry_context = ""
        biochemistry_result = state.get("biochemistry_result")
        if biochemistry_result and biochemistry_result.get("findings"):
            biochem_findings = biochemistry_result["findings"][:5]
            biochemistry_context = (
                "\n\nCurrent round biochemistry findings:\n"
                + "\n".join(f"- {f}" for f in biochem_findings)
            )

        # Build comprehensive instruction for bioinformatics agent
        # Phase-aware instruction for bioinformatics agent
        # Detect data types for dynamic instruction (single pass)
        has_scrna = analysis_state.has_scrnaseq()
        has_bulk = analysis_state.has_bulk_rnaseq()

        phase_instruction = ""
        if analysis_state.analysis_phase == AnalysisPhase.EXPLORATION:
            phase_instruction = """
## EXPLORATION PHASE INSTRUCTIONS
This is Round 1 — comprehensive, unbiased exploration. You MUST:
1. Complete the FULL standard pipeline (QC → normalization → clustering → annotation)
2. After annotation: merge biologically similar clusters (e.g., stress hepatocytes + hepatocytes,
   macrophage subtypes) into unified cell type groups before DEG analysis
3. Perform condition-level DEG (treatment vs control) for ALL merged cell type groups
4. Run GO/KEGG pathway enrichment on ALL DEG results
5. Keep analyses simple and high-throughput (DEG + GO/KEGG)
6. Do NOT skip any cell type — but DO merge related subtypes/states of the same cell type
7. Complete cell communication analysis (liana)
8. For EACH cell type group's DEG results, generate a visualization figure (volcano plot or bar chart)
9. For EACH cell type group's KEGG/GO enrichment, generate a bar plot of top enriched pathways
"""
            if has_bulk:
                phase_instruction += """
10. **MANDATORY — Bulk RNA-seq DEG Analysis** (DO NOT SKIP):
    - Load the bulk RNA-seq count matrix and perform DEG analysis using pydeseq2
      (CRITICAL: If pydeseq2 raises ImportError, fallback to scipy.stats.ttest_ind with statsmodels multipletests for FDR correction)
    - Generate a volcano plot for bulk DEG results
    NOTE: report_findings will be REJECTED if bulk DEG analysis is not performed.

11. **MANDATORY — Bulk/scRNA-seq Cross-Validation** (DO NOT SKIP):
    - Aggregate scRNA-seq data to pseudobulk level OR use condition-level DEGs across all cells
    - Find overlapping DEGs between bulk and scRNA-seq results using set intersection
    - Visualize the overlap with a Venn diagram (from matplotlib_venn import venn2)
    - For overlapping DEGs, visualize their cell-type-specific expression in scRNA-seq (e.g., dotplot or heatmap)
    NOTE: This is a SEPARATE mandatory gate. report_findings will be REJECTED if cross-validation is not performed.

12. **MANDATORY — Cell-Type Deconvolution** (DO NOT SKIP):
    - Use the annotated scRNA-seq data as a reference to perform cell-type deconvolution on the bulk RNA-seq matrix.
    - Prioritize pure Python methods like scipy.optimize.nnls or sklearn.linear_model.LinearRegression(positive=True).
    - If needed and available, use advanced tools like scaden (prioritize scipy.optimize.nnls).
    - Generate a stacked bar plot or box plot showing the inferred cell-type proportions across bulk samples/conditions.
    NOTE: This is a SEPARATE mandatory gate. report_findings will be REJECTED if deconvolution is not performed.
"""
        else:
            direction_ctx = analysis_state.current_round_direction or "PI-directed"
            phase_instruction = f"""
## INNOVATION PHASE INSTRUCTIONS
This is a targeted, hypothesis-driven round.
Research direction: {direction_ctx}

ANALYSIS REQUIREMENTS:
1. Execute ALL tasks assigned by PI below — do not skip any
2. For each task, generate at least one publication-quality figure
3. Use advanced analysis methods as specified (GSEA, gene scoring, trajectory, TF activity, etc.)
4. After completing PI's tasks, if you identify additional relevant analyses, execute them
5. When performing DEG, always include volcano plots and heatmaps of top genes
6. When performing pathway analysis, include bar plots of top enriched terms
7. For any scoring analysis (gene signatures, TF activity), show UMAP feature plots AND violin/box plots by condition
8. Report detailed statistical results (p-values, effect sizes, gene counts) — not just summaries
9. Before computing any score, consult PI about scoring method and gene list source
10. After computing any score, call register_score to document methodology for the report
"""
            if has_bulk:
                phase_instruction += """
11. **MANDATORY — Bulk RNA-seq Direction Validation** (if bulk data exists; DO NOT SKIP):
    Bulk RNA-seq data remains available and loaded from Round 1. When this round
    produces new gene signatures, DEG sets, or pathway conclusions, use bulk data
    as an independent validation layer. Choose validation methods appropriate to
    your specific findings and the bulk sample design. Reuse bulk results from
    Round 1 (see the data context section for the exact variable names) rather than
    re-running the pipeline.
    NOTE: report_findings will be REJECTED if bulk direction validation is not performed.
"""

            phase_instruction += """
You may skip irrelevant cell populations and focus on those relevant to the hypothesis.
Do NOT re-run completed steps from Round 1 (QC, clustering, cell annotation, etc.).
"""
            if has_bulk:
                phase_instruction += """BUT: "completed steps" does NOT include bulk validation — each direction must independently
validate its findings on bulk data, even if Round 1 already ran a bulk DEG pipeline.
"""

        # Build merge strategy context from PI instructions
        merge_context = ""
        if merge_strategy:
            merge_context += f"\n\nPI's cluster merge strategy:\n{merge_strategy}"
        if cluster_comparison_strategy:
            merge_context += f"\n\nPI's cluster comparison strategy:\n{cluster_comparison_strategy}"

        if has_scrna and has_bulk:
            data_desc = "single-cell RNA-seq and bulk RNA-seq"
        elif has_bulk:
            data_desc = "bulk RNA-seq"
        else:
            data_desc = "single-cell RNA-seq"

        if bioinf_tasks:
            goals_text = "\n".join(f"- {task}" for task in _normalize_tasks(bioinf_tasks))
        elif has_bulk and not has_scrna:
            goals_text = "- Perform standard bulk RNA-seq analysis (pydeseq2, pathway enrichment)"
        elif has_bulk and has_scrna:
            goals_text = ("- Perform standard scRNA-seq analysis pipeline\n"
                          "- Perform bulk RNA-seq DEG analysis (pydeseq2) and cross-validate with scRNA-seq pseudobulk DEGs")
        else:
            goals_text = "- Perform standard scRNA-seq analysis pipeline"

        # Ensure bulk direction validation appears in goals even if PI didn't assign a bulk-specific task
        if analysis_state.analysis_phase == AnalysisPhase.INNOVATION and has_bulk and has_scrna:
            goals_lower = goals_text.lower()
            bulk_validation_mentioned = any(
                kw in goals_lower for kw in ["bulk", "ssgsea", "deconvolution", "cross-validat", "pseudobulk"]
            )
            if not bulk_validation_mentioned:
                goals_text += (
                    "\n- Validate direction-level findings on bulk RNA-seq data "
                    "(reuse the bulk variables already loaded in Round 1; see the data context section "
                    "for their exact names)"
                )

        instruction = f"""Analyze {data_desc} data for project: {analysis_state.name}

Research context: {analysis_state.initial_instruction}
{phase_instruction}

PI's analysis goals for this round:
{goals_text}
{merge_context}
{human_guidance}
{pathology_context}
{biochemistry_context}
{previous_bio_findings}
{code_history_context}

This is Round {analysis_state.current_round}. The Jupyter kernel retains all variables from previous rounds.
Focus on PI's specific goals above — do NOT re-run completed steps (QC, clustering, cell annotation, etc.).
Call report_findings ONLY after completing ALL assigned tasks above."""

        await self._send_message("prompt", instruction, AgentType.PI)
        
        # Step 2: Bioinformatics agent runs autonomously with figure review support
        await self._send_message("status", "Bioinformatics agent is running autonomous analysis...", AgentType.BIOINFORMATICS)

        # Reset supervisor state for this round
        if self.supervisor_agent:
            self.supervisor_agent.reset_for_round()

        try:
            result = await self.bioinformatics_agent.run_autonomous_analysis(
                analysis_state,
                instruction,
                max_steps=200,  # Allow sufficient steps for full analysis + figure review loops
                figure_review_callback=self._figure_review_callback,
                supervisor_callback=self._supervisor_callback,
            )
        except UpstreamExhaustedError as e:
            logger.error(f"[{self.session_id}] Bioinformatics analysis halted: upstream exhausted: {e}")
            await self._send_message(
                "session_halted",
                "LLM upstream unavailable. Session checkpointed; you can resume from the dashboard.",
                AgentType.BIOINFORMATICS,
            )
            return

        # Collect supervisor stats
        if self.supervisor_agent:
            supervision_stats = self.supervisor_agent.get_summary_stats()
            await self._send_message("supervisor_summary", json.dumps(supervision_stats), AgentType.SUPERVISOR)
            logger.info(
                f"[{self.session_id}] Supervisor: {supervision_stats['total_steps_monitored']} steps, "
                f"{supervision_stats['steps_with_technical_issues']} issues, "
                f"checkpoints: {supervision_stats['checkpoints_reached']}"
            )

        # Check result status and log accordingly
        result_status = result.get("status", "unknown")
        if result_status == "error":
            logger.error(f"[{self.session_id}] Bioinformatics analysis failed: {result.get('error', 'Unknown error')}")
            await self._send_message("error", f"Bioinformatics analysis encountered an error: {result.get('error', 'Unknown error')}", AgentType.BIOINFORMATICS)
        elif result_status in ("timeout", "max_steps_reached"):
            logger.warning(f"[{self.session_id}] Bioinformatics analysis reached step limit after {result.get('steps_executed', 0)} steps")
            await self._send_message("status", "Bioinformatics analysis reached step limit, proceeding with partial results", AgentType.BIOINFORMATICS)

        # Collect results
        all_findings = result.get("findings", [])
        all_figures = result.get("figures", [])

        # Store code executions (dedup by id AND code content hash to handle
        # both node retries and session restores that assign new UUIDs)
        existing_code_ids = {c.id for c in analysis_state.code_history}
        existing_code_hashes = {hash(c.code) for c in analysis_state.code_history}
        for code_exec in result.get("code_history", []):
            code_hash = hash(code_exec.code)
            if code_exec.id not in existing_code_ids and code_hash not in existing_code_hashes:
                analysis_state.code_history.append(code_exec)
                existing_code_ids.add(code_exec.id)
                existing_code_hashes.add(code_hash)

        # Store analysis log (dedup by step id to handle node retries)
        existing_step_ids = {s.id for s in analysis_state.analysis_log}
        for step in result.get("analysis_log", []):
            if isinstance(step, dict):
                step = AnalysisStep(**step)
            if step.id not in existing_step_ids:
                analysis_state.analysis_log.append(step)
                existing_step_ids.add(step.id)
        # Cap log size to prevent unbounded memory growth
        _MAX_ANALYSIS_LOG = 2000
        if len(analysis_state.analysis_log) > _MAX_ANALYSIS_LOG:
            analysis_state.analysis_log = analysis_state.analysis_log[-_MAX_ANALYSIS_LOG:]

        # Dedup publication_figures by name to handle node retries
        seen_fig_names = {}
        for i, fig in enumerate(analysis_state.publication_figures):
            seen_fig_names[fig.name] = i  # later index wins (keeps latest)
        if len(seen_fig_names) < len(analysis_state.publication_figures):
            analysis_state.publication_figures = [
                analysis_state.publication_figures[i]
                for i in sorted(seen_fig_names.values())
            ]

        # Step 3: PI summarizes results
        await self._send_message("status", "PI is reviewing analysis results...", AgentType.PI)

        summary = await self.pi_agent.summarize_bioinformatics_analysis(
            analysis_state,
            [{"findings": all_findings, "figures": all_figures}]
        )

        final_result = {
            "findings": all_findings,
            "figures": all_figures,
            "summary": summary,
            "steps_executed": result.get("steps_executed", 0),
            "status": result.get("status", "completed")
        }

        logger.info(f"[{self.session_id}] Bioinformatics analysis completed: {len(all_findings)} findings, {len(all_figures)} figures, {result.get('steps_executed', 0)} steps")

        await self._send_message(
            "result",
            {
                "findings": all_findings,
                "summary": summary,
                "steps_executed": result.get("steps_executed", 0)
            },
            AgentType.BIOINFORMATICS
        )

        analysis_state.add_message(
            AgentType.BIOINFORMATICS,
            MessageRole.ASSISTANT,
            json.dumps({"findings": all_findings, "summary": summary}, ensure_ascii=False)
        )

        return {
            **state,
            "analysis_state": analysis_state,
            "bioinformatics_result": final_result
        }
    
    @with_retry("biochemistry_analysis")
    async def _biochemistry_analysis(self, state: WorkflowState) -> WorkflowState:
        """Biochemistry agent analyzes experimental results"""
        if not self.biochemistry_agent:
            logger.info(f"[{self.session_id}] Biochemistry agent disabled, skipping")
            return {**state, "failed_node": None, "biochemistry_result": {"findings": [], "status": "skipped"}}

        analysis_state = state["analysis_state"]
        instructions = state["pi_instructions"]

        tasks = instructions.get("biochemistry_tasks", [])

        bio_files = analysis_state.get_files_by_section("biochemistry")
        logger.info(f"[{self.session_id}] Starting biochemistry analysis with {len(analysis_state.biochemistry_indicators)} indicators and {len(bio_files)} files")
        logger.info(f"[{self.session_id}] Biochemistry tasks: {tasks}")

        await self._send_message("status", "Biochemistry analysis in progress...", AgentType.BIOCHEMISTRY)
        
        # Enrich instruction with previous findings for round 2+
        task_instruction = "\n".join(_normalize_tasks(tasks)) if tasks else "Analyze existing biochemistry indicators"
        previous_context = self._build_previous_findings_context(analysis_state, "biochemistry")
        if previous_context:
            task_instruction += previous_context

        # Biochemistry runs first — no current-round cross-agent results available.
        # The biochemistry agent's _get_other_agent_context() will fall back to
        # previous round results from state.round_results for round 2+.
        current_round_context = {}

        result = await self.biochemistry_agent.process(
            analysis_state,
            task_instruction,
            current_round_context=current_round_context
        )
        
        await self._send_message(
            "result",
            result,
            AgentType.BIOCHEMISTRY
        )

        analysis_state.add_message(
            AgentType.BIOCHEMISTRY,
            MessageRole.ASSISTANT,
            json.dumps(result, ensure_ascii=False)
        )

        return {
            **state,
            "analysis_state": analysis_state,
            "biochemistry_result": result
        }

    @with_retry("literature_research")
    async def _literature_research(self, state: WorkflowState) -> WorkflowState:
        """Literature agent researches relevant topics.

        Deep Research mode: merges all topics into one comprehensive query and
        makes a single DR call. If some topics are insufficiently covered,
        falls back to PubMed+LLM for the gaps.

        PubMed mode: searches each topic individually (up to 3).
        """
        analysis_state = state["analysis_state"]
        instructions = state["pi_instructions"]

        topics = _normalize_tasks(instructions.get("literature_topics", []))
        logger.info(f"[{self.session_id}] Starting literature research with topics: {topics}")

        # Dynamic PI planning: refine literature topics based on all completed results
        overview = instructions.get("overview", "")
        if overview and topics:
            completed = self._collect_current_round_results(state)
            if completed:
                await self._send_message("status", "PI is refining literature topics...", AgentType.PI)
                topics = await self.pi_agent.plan_agent_tasks(
                    agent_name="literature",
                    original_tasks=topics,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        if not topics:
            # Default topics based on other findings
            path_findings = state.get("pathology_result", {}).get("findings", []) if state.get("pathology_result") else []
            bio_findings = state.get("bioinformatics_result", {}).get("findings", []) if state.get("bioinformatics_result") else []
            biochem_findings = state.get("biochemistry_result", {}).get("findings", []) if state.get("biochemistry_result") else []

            all_source_findings = path_findings + bio_findings + biochem_findings
            if all_source_findings:
                # Use truncated findings as search keywords (full findings are too long for search queries)
                short_findings = [
                    (f[:80] if isinstance(f, str) else str(f)[:80])
                    for f in all_source_findings[:3]
                ]
                topics = [f"single cell sequencing {' '.join(short_findings)}"]
            else:
                # No findings from other agents — build topic from research objective and treatment groups
                objective = analysis_state.research_objective or ""
                group_names = [g.name for g in (analysis_state.treatment_groups or []) if g.name]
                if objective:
                    topics = [f"single cell sequencing {objective[:120]}"]
                elif group_names:
                    topics = [f"single cell sequencing {' '.join(group_names[:3])}"]
                else:
                    logger.warning(f"[{self.session_id}] No findings or objective for literature search, skipping")
                    topics = []

        await self._send_message("status", "Literature research in progress...", AgentType.LITERATURE)

        # Collect current round findings from other agents for literature context
        current_round_findings: List[str] = []
        for result_key in ("biochemistry_result", "pathology_result", "bioinformatics_result"):
            result = state.get(result_key)
            if result and result.get("findings"):
                for f in result["findings"][:3]:
                    short = f[:100] if isinstance(f, str) else str(f)[:100]
                    current_round_findings.append(short)

        # Enrich topics with previous findings for round 2+
        previous_lit_context = self._build_previous_findings_context(analysis_state, "literature")

        # Check if Deep Research mode is available (merges topics into one call)
        is_deep_research = self.literature_agent.is_deep_research_available

        limited_topics = topics[:3]

        if is_deep_research and limited_topics:
            # === Deep Research: merge topics, one call ===
            merged_topic = "Please conduct a comprehensive literature review covering the following aspects:\n"
            for i, t in enumerate(limited_topics, 1):
                merged_topic += f"{i}. {t}\n"
            if previous_lit_context:
                merged_topic += f"\n{previous_lit_context}\nFocus on aspects NOT already covered above."

            logger.info(f"[{self.session_id}] Deep Research mode: merging {len(limited_topics)} topics into one query")
            dr_result = await self.literature_agent.process(
                analysis_state, merged_topic, use_deep_research=True,
                current_round_findings=current_round_findings
            )
            results = [dr_result]

            # Evaluate coverage — fallback to PubMed for gaps
            if len(limited_topics) > 1:
                gaps = await self._evaluate_dr_coverage(limited_topics, dr_result)
                if gaps:
                    logger.info(f"[{self.session_id}] Deep Research gaps detected, PubMed fallback for: {gaps}")
                    await self._send_message("status", f"Supplementing {len(gaps)} topic(s) via PubMed...", AgentType.LITERATURE)
                    # Temporarily switch to PubMed mode so _do_research() routes
                    # to _multi_round_pubmed() instead of retrying Deep Research
                    original_mode = self.literature_agent.literature_mode
                    self.literature_agent.literature_mode = "pubmed_llm"
                    try:
                        for gap_topic in gaps:
                            enriched = gap_topic
                            if previous_lit_context:
                                enriched = f"{gap_topic}\n{previous_lit_context}\nFocus on aspects NOT already covered."
                            gap_result = await self.literature_agent.process(
                                analysis_state, enriched, use_deep_research=True,
                                current_round_findings=current_round_findings
                            )
                            results.append(gap_result)
                    finally:
                        self.literature_agent.literature_mode = original_mode
        else:
            # === PubMed mode: per-topic search ===
            results = []
            for topic in limited_topics:
                enriched_topic = topic
                if previous_lit_context:
                    enriched_topic = f"{topic}\n{previous_lit_context}\nFocus on aspects NOT already covered above."
                result = await self.literature_agent.process(
                    analysis_state,
                    enriched_topic,
                    # True routes to _do_research() for full PubMed pipeline,
                    # not the external Deep Research API (guarded by is_deep_research_available).
                    use_deep_research=True,
                    current_round_findings=current_round_findings
                )
                results.append(result)

        # Combine results
        combined_result = {
            "topics": topics,
            "findings": [],
            "reports": []
        }

        for r in results:
            combined_result["findings"].extend(r.get("findings", []))
            if r.get("report"):
                combined_result["reports"].append(r["report"])

        logger.info(f"[{self.session_id}] Literature research completed: {len(combined_result['findings'])} findings from {len(topics)} topics")

        await self._send_message(
            "result",
            combined_result,
            AgentType.LITERATURE
        )

        analysis_state.add_message(
            AgentType.LITERATURE,
            MessageRole.ASSISTANT,
            json.dumps({"findings": combined_result["findings"]}, ensure_ascii=False)
        )

        return {
            **state,
            "analysis_state": analysis_state,
            "literature_result": combined_result
        }

    async def _evaluate_dr_coverage(self, topics: List[str], dr_result: Dict) -> List[str]:
        """Evaluate which topics were NOT sufficiently covered by Deep Research.

        Returns a list of gap topic strings for PubMed+LLM fallback.
        """
        report = dr_result.get("report", "")
        if not report:
            return topics  # No report at all — fall back on everything

        topics_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
        prompt = f"""The following Deep Research report was generated:

{report[:3000]}

The original research topics were:
{topics_list}

Which topics are NOT sufficiently covered in the report? Return a JSON array of topic strings that need additional research. If all topics are well covered, return an empty array [].

Return ONLY the JSON array, nothing else."""

        try:
            response = await self.literature_agent.chat_without_history(prompt, temperature=0.2)
            # Strip markdown code fences if present (e.g. ```json ... ```)
            cleaned = response.strip()
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
                else:
                    cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            gaps = json.loads(cleaned)
            if isinstance(gaps, list):
                return [g for g in gaps if isinstance(g, str)]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{self.session_id}] DR coverage evaluation failed: {e}")
        return []
    
    async def _pi_synthesis(self, state: WorkflowState) -> WorkflowState:
        """PI synthesizes all results and prepares round summary for user"""
        analysis_state = state["analysis_state"]

        logger.info(f"[{self.session_id}] PI synthesis for round {analysis_state.current_round}")

        await self._send_message("status", "PI is synthesizing analysis results...", AgentType.PI)
        
        # Extract findings from each agent, including error/status info for PI awareness
        agent_results = {}
        for agent_name, result_key in [
            ("biochemistry", "biochemistry_result"),
            ("pathology", "pathology_result"),
            ("bioinformatics", "bioinformatics_result"),
            ("literature", "literature_result"),
        ]:
            result = state.get(result_key, {}) or {}
            findings = result.get("findings", [])
            error = result.get("error")
            status = result.get("status")

            # Build result string with error/status context so PI knows what happened
            if error:
                agent_results[agent_name] = json.dumps(
                    {"error": error, "findings": findings}, ensure_ascii=False
                )
                logger.warning(f"[{self.session_id}] {agent_name} had error during this round: {error}")
            elif status in ("skipped", "no_data", "waiting_data"):
                agent_results[agent_name] = json.dumps(
                    {"status": status, "findings": findings}, ensure_ascii=False
                )
            else:
                agent_results[agent_name] = json.dumps(findings, ensure_ascii=False)

        round_result = await self.pi_agent.synthesize_round(
            analysis_state,
            pathology_result=agent_results["pathology"],
            bioinformatics_result=agent_results["bioinformatics"],
            biochemistry_result=agent_results["biochemistry"],
            literature_result=agent_results["literature"],
            research_direction=analysis_state.current_round_direction,
        )

        logger.info(f"[{self.session_id}] Round {round_result.round_number} synthesis complete: new_findings={round_result.new_findings}, {len(round_result.next_steps)} next steps")

        # Tag round with current phase and direction
        round_result.phase = analysis_state.analysis_phase
        round_result.direction = analysis_state.current_round_direction or ""

        # Exploration phase: generate structured Report A and transition to innovation
        if analysis_state.analysis_phase == AnalysisPhase.EXPLORATION:
            await self._send_message(
                "status",
                "Generating exploration report (Report A)...",
                AgentType.PI
            )
            try:
                exploration_report = await self.pi_agent.generate_exploration_report(
                    analysis_state, round_result
                )
                analysis_state.exploration_report = exploration_report
                round_result.exploration_report = exploration_report

                # Generate exploration report HTML
                try:
                    from .services.report import generate_exploration_report_html
                    html_path = generate_exploration_report_html(
                        analysis_state, exploration_report, round_result
                    )
                    analysis_state.exploration_report_html_path = html_path
                    logger.info(f"[{self.session_id}] Exploration report HTML saved to: {html_path}")
                except Exception as html_err:
                    logger.error(f"[{self.session_id}] Exploration report HTML generation failed: {html_err}")

                html_url = f"/api/session/{self.session_id}/exploration-report/html" if analysis_state.exploration_report_html_path else None

                # Generate detailed report segment for exploration completion
                await self._build_and_save_detailed_segment(
                    state=analysis_state,
                    phase="exploration_complete",
                    title="Exploration Phase Complete - Analysis Summary",
                    graph_state=state,
                    round_result=round_result,
                )
                detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if analysis_state.detailed_report_html_path else None

                # Auto-snapshot before direction selection HITL
                await self._create_auto_snapshot(analysis_state, "pre_direction")

                await self._send_message("exploration_report", {
                    "report_a": exploration_report.model_dump(),
                    "round": round_result.round_number,
                    "summary": round_result.pi_summary,
                    "html_url": html_url,
                    "detailed_report_url": detailed_report_url,
                }, AgentType.PI)
            except Exception as e:
                logger.exception(f"[{self.session_id}] Exploration report generation failed")
                await self._send_message(
                    "error",
                    f"Exploration report generation failed: {e}. You can still select a research direction based on the analysis findings above.",
                    AgentType.PI
                )

            # Transition to innovation phase for next round
            analysis_state.analysis_phase = AnalysisPhase.INNOVATION
            logger.info(f"[{self.session_id}] Transitioning to innovation phase")

        analysis_state.round_results.append(round_result)

        # Multi-direction correlation analysis: trigger when queue is empty and multiple directions were selected
        if (
            not analysis_state.directions_queue
            and len(analysis_state.user_chosen_directions) > 1
            and analysis_state.analysis_phase == AnalysisPhase.INNOVATION
            and analysis_state.multi_direction_correlation is None
        ):
            try:
                await self._send_message("status", "Analyzing correlations between research directions...", AgentType.PI)
                # Build directions_with_results by matching direction names (not index)
                directions_with_results = []
                innovation_rounds = [r for r in analysis_state.round_results if r.phase == "innovation"]
                # Build lookup: direction name -> latest round summary
                direction_summaries: Dict[str, str] = {}
                for r in innovation_rounds:
                    if r.direction:
                        direction_summaries[r.direction] = r.pi_summary or ""
                for chosen_dir in analysis_state.user_chosen_directions:
                    dir_name = chosen_dir.get("direction", "")
                    round_summary = direction_summaries.get(dir_name, "")
                    directions_with_results.append({
                        "direction": dir_name,
                        "notes": chosen_dir.get("notes", ""),
                        "round_summary": round_summary
                    })
                correlation = await self.pi_agent.analyze_direction_correlations(
                    analysis_state, directions_with_results
                )
                analysis_state.multi_direction_correlation = correlation
                await self._send_message("direction_correlation", {
                    "correlation": correlation,
                    "directions": [d.get("direction", "") for d in analysis_state.user_chosen_directions]
                }, AgentType.PI)
                logger.info(f"[{self.session_id}] Multi-direction correlation analysis complete")
            except Exception as e:
                logger.exception(f"[{self.session_id}] Direction correlation analysis failed")

        # Save state after each round
        if self.state_callback:
            try:
                await self.state_callback(analysis_state)
            except Exception as e:
                logger.exception("Failed to save state after round")
        
        # Extract data requests from PI
        try:
            data_requests = await self.pi_agent.identify_data_needs(
                analysis_state,
                round_result
            )
        except Exception as e:
            logger.warning(f"Failed to identify data needs: {e}")
            data_requests = []
        
        # Send round summary with data requests
        effective_max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        round_summary = {
            "round": round_result.round_number,
            "max_rounds": effective_max_rounds,
            "summary": round_result.pi_summary,
            "pathology_findings": round_result.pathology_findings,
            "bioinformatics_findings": round_result.bioinformatics_findings,
            "biochemistry_findings": round_result.biochemistry_findings,
            "literature_findings": round_result.literature_findings,
            "findings": [
                *round_result.pathology_findings,
                *round_result.bioinformatics_findings,
                *round_result.biochemistry_findings,
                *round_result.literature_findings,
            ],
            "data_requests": data_requests,
            "status": "waiting_input"
        }
        
        await self._send_message(
            "round_summary",
            round_summary,
            AgentType.PI
        )
        
        # Also send traditional result message for compatibility
        await self._send_message(
            "result",
            {
                "round": round_result.round_number,
                "summary": round_result.pi_summary,
                "next_steps": round_result.next_steps,
                "new_findings": round_result.new_findings
            },
            AgentType.PI
        )
        
        # Auto-snapshot before round wait HITL
        await self._create_auto_snapshot(analysis_state, "pre_round_wait")

        # Clear continue event BEFORE announcing waiting status to prevent race
        # condition where frontend signals between status message and event.wait()
        self._continue_event.clear()

        # Set status to waiting for user input
        self._set_status(analysis_state, TaskStatus.WAITING_INPUT)

        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        is_last_round = analysis_state.current_round >= max_rounds
        if is_last_round:
            await self._send_message(
                "status",
                f"Round {round_result.round_number} complete (final round). You can provide additional data to extend analysis, or skip to generate the final report.",
                AgentType.PI
            )
        else:
            await self._send_message(
                "status",
                f"Round {round_result.round_number} complete. Waiting for your input - you can provide additional data or instructions.",
                AgentType.PI
            )

        # Send state sync for round completion
        await self._send_state_sync(analysis_state, "round_complete")

        return {
            **state,
            "analysis_state": analysis_state,
            "round_result": round_result,
            "data_requests": data_requests
        }

    async def _wait_for_user_input(self, state: WorkflowState) -> WorkflowState:
        """Block between rounds, waiting for user input or continue signal from frontend.

        Behavior depends on analysis_state.round_wait_seconds:
          -1  = manual mode: wait indefinitely for user signal
           0  = no-wait mode: proceed immediately without waiting
          >0  = countdown mode: frontend shows countdown timer, backend uses
                countdown + 60s buffer as safety timeout
        """
        analysis_state = state["analysis_state"]

        # Note: _continue_event is cleared in _pi_synthesis before the waiting
        # status is announced, to prevent a race condition with signal_continue().

        wait_seconds = analysis_state.round_wait_seconds  # -1=manual, 0=no wait, >0=countdown

        # Filter out empty/whitespace-only directions from queue
        if analysis_state.directions_queue:
            analysis_state.directions_queue = [
                d for d in analysis_state.directions_queue if d and d.get("direction", "").strip()
            ]
        # Note: no auto-skip for multi-direction mode — fall through to normal wait logic

        # After exploration phase (Round 1), force manual wait so user must choose direction
        if analysis_state.current_round == 1 and analysis_state.analysis_phase == AnalysisPhase.INNOVATION:
            wait_seconds = -1
            logger.info(f"[{self.session_id}] Exploration phase complete, forcing manual wait for direction selection")

        if wait_seconds == 0:
            # No-wait mode: proceed immediately
            logger.info(f"[{self.session_id}] No-wait mode, proceeding immediately (round {analysis_state.current_round})")
        elif wait_seconds == -1:
            # Manual mode: wait for user signal (with supplementary analysis support)
            if analysis_state.status != TaskStatus.WAITING_INPUT:
                self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                await self._send_state_sync(analysis_state, "waiting_input")

            while True:
                self._continue_event.clear()
                logger.info(f"[{self.session_id}] Manual mode, waiting for user input (round {analysis_state.current_round})")
                await self._continue_event.wait()

                # Check: supplementary analysis requested?
                if self._supplementary_request and self._supplementary_target == "direction":
                    req = self._supplementary_request
                    self._supplementary_request = None
                    self._supplementary_target = None
                    self._set_status(analysis_state, TaskStatus.RUNNING)
                    await self._send_state_sync(analysis_state, "supplementary_running")

                    await self._run_supplementary_analysis(analysis_state, req, "direction")

                    # Regenerate exploration report with supplementary data
                    try:
                        last_round = analysis_state.round_results[-1] if analysis_state.round_results else None
                        if last_round and analysis_state.exploration_report:
                            exploration_report = await self.pi_agent.generate_exploration_report(
                                analysis_state, last_round
                            )
                            analysis_state.exploration_report = exploration_report

                            from .services.report import generate_exploration_report_html
                            html_path = generate_exploration_report_html(
                                analysis_state, exploration_report, last_round
                            )
                            analysis_state.exploration_report_html_path = html_path
                            html_url = f"/api/session/{self.session_id}/exploration-report/html"
                            detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if analysis_state.detailed_report_html_path else None

                            await self._send_message("exploration_report", {
                                "report_a": exploration_report.model_dump(),
                                "round": last_round.round_number,
                                "summary": last_round.pi_summary,
                                "html_url": html_url,
                                "detailed_report_url": detailed_report_url,
                            }, AgentType.PI)
                    except Exception as e:
                        logger.error(f"[{self.session_id}] Exploration report regeneration failed: {e}")
                        await self._send_message("error", f"Report regeneration failed: {e}", AgentType.PI)

                    self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                    await self._send_state_sync(analysis_state, "waiting_input")
                    continue
                elif self._supplementary_request and self._supplementary_target == "round":
                    req = self._supplementary_request
                    self._supplementary_request = None
                    self._supplementary_target = None
                    self._set_status(analysis_state, TaskStatus.RUNNING)
                    await self._send_state_sync(analysis_state, "supplementary_running")

                    await self._run_supplementary_analysis(analysis_state, req, "round")

                    self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                    await self._send_state_sync(analysis_state, "waiting_input")
                    continue
                else:
                    logger.info(f"[{self.session_id}] Continue signal received")
                    break
        else:
            # Countdown mode: backend safety timeout = frontend countdown + 60s buffer
            # Guard: ensure WAITING_INPUT status is synced before blocking
            if analysis_state.status != TaskStatus.WAITING_INPUT:
                self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                await self._send_state_sync(analysis_state, "waiting_input")
            backend_timeout = wait_seconds + 60
            logger.info(f"[{self.session_id}] Waiting for user input (round {analysis_state.current_round}, timeout={backend_timeout}s)")
            try:
                await asyncio.wait_for(self._continue_event.wait(), timeout=backend_timeout)
                logger.info(f"[{self.session_id}] Continue signal received")
            except asyncio.TimeoutError:
                logger.info(f"[{self.session_id}] Wait timeout ({backend_timeout}s), auto-proceeding")
                self._set_status(analysis_state, TaskStatus.RUNNING)
                await self._send_state_sync(analysis_state, "auto_continue")

        # If user provided real input (text or files) on the last round, extend max_rounds to allow one more
        has_real_text = (
            analysis_state.pending_human_input
            and analysis_state.pending_human_input.strip() not in ('', '[CONTINUE]', '[PI_DECIDE]')
        )
        has_uploaded_files = bool(analysis_state.pending_uploaded_files)
        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        if (has_real_text or has_uploaded_files) and analysis_state.current_round >= max_rounds:
            # Don't extend if multi-direction mode has completed all directions
            # AND the user didn't provide real new input/files (only synthetic signals).
            # If the user explicitly typed instructions or uploaded files, honor their intent.
            multi_dir_completed = (
                not analysis_state.directions_queue
                and len(analysis_state.user_chosen_directions) > 1
            )
            if multi_dir_completed and not has_real_text:
                logger.info(f"[{self.session_id}] Multi-direction completed, not extending max_rounds")
            else:
                analysis_state.max_rounds = analysis_state.current_round + 1
                logger.info(f"[{self.session_id}] User input on last round (text={has_real_text}, files={has_uploaded_files}), extending max_rounds to {analysis_state.max_rounds}")

        return {**state, "analysis_state": analysis_state}

    def _should_continue_after_init(self, state: WorkflowState) -> str:
        """Route to report generation if user requested stop during round initialization."""
        if state.get("human_stop_requested", False):
            logger.info(f"[{self.session_id}] Stop requested during init, routing to final report")
            return "stop"
        if state.get("should_continue") is False:
            logger.info(f"[{self.session_id}] should_continue=False after init, routing to final report")
            return "stop"
        return "continue"

    def _should_continue(self, state: WorkflowState) -> str:
        """Determine if analysis should continue"""
        analysis_state = state["analysis_state"]

        # Check if human requested stop (processed by PI Agent)
        if state.get("human_stop_requested", False):
            logger.info(f"[{self.session_id}] Stopping: human stop requested")
            return "stop"
        
        # Check explicit should_continue flag (set by human intervention)
        if state.get("should_continue") is False:
            logger.info(f"[{self.session_id}] Stopping: should_continue flag is False")
            return "stop"

        # Check max rounds (use state.max_rounds if valid, otherwise config default)
        # max_rounds <= 0 means use config default; max_rounds > 0 is user-specified limit
        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        # Ensure max_rounds is always at least 1 to prevent infinite loops
        max_rounds = max(1, max_rounds)
        if analysis_state.current_round >= max_rounds:
            logger.info(f"[{self.session_id}] Stopping: reached max rounds ({max_rounds})")
            return "stop"

        # Check if forced to stop
        if analysis_state.status == TaskStatus.PAUSED:
            logger.info(f"[{self.session_id}] Stopping: analysis is paused")
            return "stop"

        logger.info(f"[{self.session_id}] Continuing analysis (round {analysis_state.current_round}/{max_rounds})")
        return "continue"
    
    async def _check_convergence(self, state: WorkflowState) -> WorkflowState:
        """Check if analysis has converged.

        Note: This node is only reached when _should_continue() returns "continue",
        meaning hard-stop conditions (human stop, max_rounds, paused) have already
        been checked. Here we only evaluate soft convergence (repeated findings).
        """
        analysis_state = state["analysis_state"]

        should_stop = analysis_state.should_stop(self.config.convergence_rounds)

        logger.info(f"[{self.session_id}] Convergence check: round={analysis_state.current_round}, should_stop={should_stop}")

        return {
            **state,
            "should_continue": not should_stop
        }
    
    def _check_convergence_condition(self, state: WorkflowState) -> str:
        """Conditional check for convergence"""
        if "should_continue" not in state:
            logger.warning(f"[{self.session_id}] should_continue key missing from state, defaulting to stop")
        if state.get("should_continue", False):
            return "continue"
        return "stop"
    
    async def _generate_final_report(self, state: WorkflowState) -> WorkflowState:
        """Generate final analysis report including integrated report with figures"""
        analysis_state = state["analysis_state"]

        # Mark status so interrupted report generation can be detected on resume
        self._set_status(analysis_state, TaskStatus.GENERATING_REPORT)
        await self._send_state_sync(analysis_state, "report_generation_start")

        logger.info(f"[{self.session_id}] Generating final report after {analysis_state.current_round} rounds")

        # Step 1: Generate final summary
        await self._send_message("status", "Generating final summary...", AgentType.PI)

        final_summary = await self.pi_agent.generate_final_summary(analysis_state)
        analysis_state.final_summary = final_summary

        await self._send_message(
            "result",
            {"final_summary": final_summary},
            AgentType.PI
        )

        # Step 2: Generate integrated report with figure/table references
        await self._send_message("status", "Generating integrated report with figures...", AgentType.PI)

        try:
            integrated_report = await self.pi_agent.generate_integrated_report(analysis_state)
            analysis_state.integrated_report = integrated_report

            # --- Collect all workspace figures (figures/ subdir + workspace root) ---
            workspace_dir = get_session_workspace(self.session_id)
            pub_figure_names = {fig.name for fig in analysis_state.publication_figures}
            all_workspace_figs: List[Dict[str, Any]] = []
            seen_stems: set = set()

            def _infer_fig_type(stem: str) -> str:
                s = stem.lower()
                if "umap" in s: return "umap"
                if "volcano" in s: return "volcano"
                if "violin" in s: return "violin"
                if "dotplot" in s or "dot_plot" in s: return "dotplot"
                if "heatmap" in s: return "heatmap"
                if "barplot" in s or "bar_plot" in s or "composition" in s or "proportion" in s: return "barplot"
                return "other"

            for scan_dir in [workspace_dir / "figures", workspace_dir]:
                if not scan_dir.exists():
                    continue
                for img in sorted(
                    list(scan_dir.glob("*.png")) + list(scan_dir.glob("*.jpg")),
                    key=lambda p: p.stat().st_mtime,
                ):
                    if img.stem in seen_stems:
                        continue
                    seen_stems.add(img.stem)
                    pdf = img.with_suffix(".pdf")
                    all_workspace_figs.append({
                        "name": img.stem,
                        "type": _infer_fig_type(img.stem),
                        "path": str(img),
                        "pdf_path": str(pdf) if pdf.exists() else None,
                    })

            # Build full figure list for AI selection (workspace + existing pub figures)
            all_figs_for_ai = [{"name": f["name"], "type": f["type"]} for f in all_workspace_figs]
            for fig in analysis_state.publication_figures:
                if fig.name not in seen_stems:
                    all_figs_for_ai.append({"name": fig.name, "type": fig.figure_type})

            # Ask AI to curate the brief report figure set
            selected_names = set(
                await self.pi_agent.select_brief_figures(analysis_state, all_figs_for_ai)
            )
            # Fallback: if AI returned nothing, show all publication figures
            if not selected_names:
                selected_names = pub_figure_names

            # Build figures_data: publication figures first (already have pngBase64),
            # then AI-selected workspace figures not covered by publication figures
            figures_data: List[Dict[str, Any]] = []
            for fig in analysis_state.publication_figures:
                if fig.name in selected_names:
                    figures_data.append({
                        "id": fig.id,
                        "name": fig.name,
                        "description": fig.description,
                        "figure_type": fig.figure_type,
                        "png_base64": fig.png_base64,
                        "pdf_path": fig.pdf_path,
                        "review_status": fig.review_status.value,
                        "revision_count": fig.revision_count
                    })

            for wf in all_workspace_figs:
                if wf["name"] in selected_names and wf["name"] not in pub_figure_names:
                    try:
                        b64 = compress_image_to_base64(wf["path"], max_width=800)
                    except Exception:
                        logger.warning("Failed to compress image", exc_info=True)
                        b64 = None
                    figures_data.append({
                        "id": str(uuid.uuid4()),
                        "name": wf["name"],
                        "description": "",
                        "figure_type": wf["type"],
                        "png_base64": b64,
                        "png_path": wf["path"],
                        "pdf_path": wf["pdf_path"],
                        "review_status": "pending",
                        "revision_count": 0
                    })

            logger.info(
                f"[{self.session_id}] Brief figures: {len(figures_data)} selected "
                f"({len(pub_figure_names)} pub + {len(all_workspace_figs)} workspace scanned)"
            )

            tables_data = [
                {
                    "id": tbl.id,
                    "name": tbl.name,
                    "description": tbl.description,
                    "csv_path": tbl.csv_path,
                    "review_status": tbl.review_status.value
                }
                for tbl in analysis_state.publication_tables
            ]

            await self._send_message(
                "integrated_report",
                {
                    "report": integrated_report,
                    "figures": figures_data,
                    "tables": tables_data,
                    "total_figures": len(figures_data),
                    "approved_figures": sum(1 for f in figures_data if f.get("review_status") == "approved")
                },
                AgentType.PI
            )

            logger.info(f"[{self.session_id}] Integrated report generated with {len(figures_data)} brief figures")

        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to generate integrated report")
            await self._send_message("error", f"Integrated report generation failed: {str(e)}", AgentType.PI)
            # Continue anyway - final summary is still available

        logger.info(f"[{self.session_id}] Final report generated")

        return {
            **state,
            "analysis_state": analysis_state
        }

    async def _generate_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """
        Dispatcher: try image mode first when Painter is available; on failure (or when
        Painter is disabled) fall back to text-mode ASCII diagram drawn by the Bio agent.
        """
        if self.painter_agent:
            try:
                return await self._generate_image_graphic_abstract(state)
            except Exception as e:
                logger.exception(
                    f"[{self.session_id}] Image graphic abstract failed; "
                    f"falling back to text-mode diagram"
                )
                await self._send_message(
                    "status",
                    "Image generation failed, falling back to text-mode diagram...",
                    AgentType.PAINTER,
                )
        else:
            logger.info(
                f"[{self.session_id}] Painter agent disabled; generating text-mode diagram"
            )

        return await self._generate_text_graphic_abstract(state)

    async def _generate_image_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """Generate image graphic abstract with revision loop (max 2 revisions, PI selects best).

        NOTE: must only write `analysis_state.graphic_abstract` on successful completion
        so that exceptions leave the state untouched and the dispatcher can cleanly fall
        back to text mode.
        """
        analysis_state = state["analysis_state"]

        logger.info(f"[{self.session_id}] Generating graphic abstract")
        await self._send_message("status", "Generating graphic abstract...", AgentType.PAINTER)

        try:
            # Step 1: PI generates summary prompt for graphic abstract
            summary_prompt = await self.pi_agent.generate_graphic_abstract_prompt(analysis_state)
            logger.info(f"[{self.session_id}] Generated graphic abstract prompt")

            # Step 2: Painter generates initial graphic abstract
            graphic = await self.painter_agent.generate_graphic_abstract(
                analysis_state,
                summary_prompt
            )

            candidates = [graphic.image_base64]
            revision_history = []

            # Send initial image to frontend
            await self._send_message(
                "graphic_abstract",
                {
                    "image_base64": graphic.image_base64,
                    "prompt": graphic.prompt,
                    "created_at": graphic.created_at.isoformat()
                },
                AgentType.PAINTER
            )
            logger.info(f"[{self.session_id}] Initial graphic abstract generated (candidate 1/3)")

            # Step 3: Revision loop - up to 2 revisions
            for revision_round in range(2):
                # PI reviews the latest image (sees image + all previous feedback)
                await self._send_message(
                    "status",
                    f"PI reviewing graphic abstract (round {revision_round + 1}/2)...",
                    AgentType.PI
                )

                review = await self.pi_agent.review_graphic_abstract(
                    analysis_state,
                    candidates[-1],
                    revision_history
                )

                # Record feedback
                revision_history.append({
                    "round": revision_round + 1,
                    "feedback": review.get("feedback", ""),
                    "suggestions": review.get("suggestions", []),
                    "must_fix": review.get("must_fix", []),
                    "quality_score": review.get("quality_score")
                })

                logger.info(
                    f"[{self.session_id}] PI review round {revision_round + 1}: "
                    f"score={review.get('quality_score', 'N/A')}"
                )

                # Painter revises (sees current image + all feedback)
                try:
                    await self._send_message(
                        "status",
                        f"Revising graphic abstract (revision {revision_round + 1}/2)...",
                        AgentType.PAINTER
                    )

                    new_image = await self.painter_agent.revise_graphic_abstract(
                        analysis_state,
                        candidates[-1],
                        graphic.prompt,
                        review.get("feedback", ""),
                        review.get("suggestions", []),
                        review.get("must_fix", []),
                        revision_history
                    )
                    candidates.append(new_image)

                    # Send revised image to frontend
                    await self._send_message(
                        "graphic_abstract",
                        {
                            "image_base64": new_image,
                            "prompt": graphic.prompt,
                            "created_at": graphic.created_at.isoformat()
                        },
                        AgentType.PAINTER
                    )
                    logger.info(
                        f"[{self.session_id}] Graphic abstract revision {revision_round + 1} "
                        f"successful (candidate {len(candidates)}/3)"
                    )

                except Exception as e:
                    logger.warning(
                        f"[{self.session_id}] Graphic abstract revision "
                        f"{revision_round + 1} failed: {e}"
                    )
                    break  # Stop revisions, PI will pick from available candidates

            # Step 4: PI selects the best from all candidates
            if len(candidates) > 1:
                await self._send_message(
                    "status",
                    f"PI selecting best graphic abstract from {len(candidates)} candidates...",
                    AgentType.PI
                )
                selected_idx = await self.pi_agent.select_best_graphic_abstract(
                    analysis_state,
                    candidates,
                    revision_history
                )
            else:
                selected_idx = 0

            # Update graphic abstract with selected image and metadata
            graphic.image_base64 = candidates[selected_idx]
            graphic.revision_count = len(candidates) - 1
            graphic.revision_history = revision_history
            graphic.all_candidates = candidates
            graphic.selected_index = selected_idx
            analysis_state.graphic_abstract = graphic

            # Send final selected image to frontend
            await self._send_message(
                "graphic_abstract",
                {
                    "image_base64": candidates[selected_idx],
                    "prompt": graphic.prompt,
                    "created_at": graphic.created_at.isoformat()
                },
                AgentType.PAINTER
            )

            logger.info(
                f"[{self.session_id}] Graphic abstract completed: "
                f"selected candidate {selected_idx + 1}/{len(candidates)}, "
                f"{len(candidates) - 1} revisions"
            )

        except Exception:
            logger.exception(f"[{self.session_id}] Failed to generate image graphic abstract")
            # Re-raise so the dispatcher (_generate_graphic_abstract) can fall back to text mode.
            # `analysis_state.graphic_abstract` is only assigned on success, so state is untouched.
            raise

        logger.info(f"[{self.session_id}] Graphic abstract step completed")

        return {
            **state,
            "analysis_state": analysis_state
        }

    # ------------------------------------------------------------------
    # Text-mode graphic abstract (Painter unavailable fallback)
    # ------------------------------------------------------------------

    _DEFAULT_TEXT_DIAGRAM_PROMPT = (
        "Draw an ASCII/Unicode flow diagram summarizing the study's key signaling "
        "pathway and cellular processes. Reason from the final summary provided and "
        "highlight ligand -> receptor -> downstream effector -> transcription factor "
        "-> cellular outcome relationships. Include at least 5 nodes and use "
        "directional arrows. English labels only.\n\n"
        "Study summary:\n{summary}"
    )

    async def _select_diagram_reviewer(
        self,
        state: AnalysisState,
        diagram: str,
        revision_history: List[Dict[str, Any]],
    ):
        """Pick a text-diagram reviewer by exception-based cascade: PI -> Bio -> Supervisor.

        Returns (review_dict, reviewer_name) or (None, None) if all reviewers fail.
        """
        try:
            r = await self.pi_agent.review_text_diagram(state, diagram, revision_history)
            return r, "pi"
        except Exception as e:
            logger.warning(f"[{self.session_id}] PI text diagram review failed: {e}")

        if self.bioinformatics_agent is not None:
            try:
                r = await self.bioinformatics_agent.review_text_diagram(
                    state, diagram, revision_history
                )
                return r, "bioinformatics"
            except Exception as e:
                logger.warning(f"[{self.session_id}] Bio text diagram review failed: {e}")

        if self.supervisor_agent is not None:
            try:
                r = await self.supervisor_agent.review_text_diagram(
                    state, diagram, revision_history
                )
                return r, "supervisor"
            except Exception as e:
                logger.warning(f"[{self.session_id}] Supervisor text diagram review failed: {e}")

        return None, None

    async def _select_best_diagram_across_reviewers(
        self,
        state: AnalysisState,
        candidates: List[str],
        revision_history: List[Dict[str, Any]],
    ) -> int:
        """Pick best text diagram with same PI -> Bio -> Supervisor cascade."""
        try:
            return await self.pi_agent.select_best_text_diagram(
                state, candidates, revision_history
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] PI best-diagram selection failed: {e}")

        # Bio and Supervisor don't have select_best_text_diagram; use PI's as the only
        # LLM-driven selection. On failure, fall back to the most recent candidate.
        return len(candidates) - 1

    async def _generate_text_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """Generate ASCII/Unicode text-mode graphic abstract with tiered review."""
        analysis_state = state["analysis_state"]

        if self.bioinformatics_agent is None:
            logger.error(
                f"[{self.session_id}] Bioinformatics agent missing; cannot generate "
                f"text-mode graphic abstract"
            )
            return state

        await self._send_message(
            "status",
            "Generating text-mode graphic abstract (ASCII diagram)...",
            AgentType.BIOINFORMATICS,
        )

        # Explicit fresh containers — must not inherit any image-mode residue.
        candidates: List[str] = []
        revision_history: List[Dict[str, Any]] = []
        pi_prompt: str = ""

        try:
            # Step 1: PI generates drawing instruction (or hardcoded fallback)
            try:
                pi_prompt = await self.pi_agent.generate_text_diagram_prompt(analysis_state)
            except Exception as e:
                logger.warning(
                    f"[{self.session_id}] PI generate_text_diagram_prompt raised: {e}"
                )
                pi_prompt = ""

            if not pi_prompt:
                summary = (analysis_state.final_summary or "")[:1500]
                pi_prompt = self._DEFAULT_TEXT_DIAGRAM_PROMPT.format(summary=summary)

            # Step 2: Bio draws initial diagram
            initial = await self.bioinformatics_agent.generate_text_diagram(
                analysis_state, pi_prompt
            )
            candidates.append(initial)

            await self._send_message(
                "graphic_abstract",
                {
                    "image_base64": "",
                    "text_diagram": initial,
                    "is_text_mode": True,
                    "prompt": pi_prompt,
                    "created_at": datetime.now().isoformat(),
                },
                AgentType.BIOINFORMATICS,
            )

            # Step 3: revision loop (max 2 rounds)
            for round_idx in range(2):
                review, reviewer_name = await self._select_diagram_reviewer(
                    analysis_state, candidates[-1], revision_history
                )
                if review is None:
                    logger.info(
                        f"[{self.session_id}] No reviewer available for text diagram; "
                        f"ending revision loop at round {round_idx + 1}"
                    )
                    break

                revision_history.append({
                    "round": round_idx + 1,
                    "reviewer": reviewer_name,
                    "feedback": review.get("feedback", ""),
                    "suggestions": review.get("suggestions", []),
                    "must_fix": review.get("must_fix", []),
                    "quality_score": review.get("quality_score"),
                })

                try:
                    await self._send_message(
                        "status",
                        f"Revising text diagram (round {round_idx + 1}/2, reviewer={reviewer_name})...",
                        AgentType.BIOINFORMATICS,
                    )
                    revised = await self.bioinformatics_agent.revise_text_diagram(
                        analysis_state,
                        candidates[-1],
                        pi_prompt,
                        review,
                        revision_history,
                    )
                    candidates.append(revised)

                    await self._send_message(
                        "graphic_abstract",
                        {
                            "image_base64": "",
                            "text_diagram": revised,
                            "is_text_mode": True,
                            "prompt": pi_prompt,
                            "created_at": datetime.now().isoformat(),
                        },
                        AgentType.BIOINFORMATICS,
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.session_id}] Text diagram revision round {round_idx + 1} failed: {e}"
                    )
                    break

            # Step 4: pick best candidate
            if len(candidates) > 1:
                selected_idx = await self._select_best_diagram_across_reviewers(
                    analysis_state, candidates, revision_history
                )
            else:
                selected_idx = 0
            if not (0 <= selected_idx < len(candidates)):
                logger.warning(
                    f"[{self.session_id}] Selected candidate index {selected_idx} is out of bounds, "
                    f"falling back to last candidate."
                )
                selected_idx = len(candidates) - 1

            # Step 5: commit state (LAST, after everything succeeded)
            graphic = GraphicAbstract(
                prompt=pi_prompt,
                image_base64="",
                text_diagram=candidates[selected_idx],
                revision_count=len(candidates) - 1,
                revision_history=revision_history,
                all_candidates=candidates,
                selected_index=selected_idx,
            )
            analysis_state.graphic_abstract = graphic

            if revision_history:
                await self._send_message(
                    "graphic_abstract",
                    {
                        "image_base64": "",
                        "text_diagram": candidates[selected_idx],
                        "is_text_mode": True,
                        "prompt": pi_prompt,
                        "created_at": graphic.created_at.isoformat(),
                    },
                    AgentType.BIOINFORMATICS,
                )

            logger.info(
                f"[{self.session_id}] Text-mode graphic abstract completed: "
                f"selected candidate {selected_idx + 1}/{len(candidates)}, "
                f"{len(candidates) - 1} revisions"
            )

        except Exception as e:
            logger.exception(
                f"[{self.session_id}] Failed to generate text-mode graphic abstract"
            )
            await self._send_message(
                "error",
                f"Text-mode graphic abstract failed: {type(e).__name__}: {e or 'see logs'}. Continuing with detailed report...",
                AgentType.BIOINFORMATICS,
            )
            # Don't fail the workflow - graphic abstract is optional

        return {**state, "analysis_state": analysis_state}

    async def _generate_detailed_report(self, state: WorkflowState) -> WorkflowState:
        """Generate detailed report with figures and tables"""
        analysis_state = state["analysis_state"]

        logger.info(f"[{self.session_id}] Generating detailed report")
        await self._send_message("status", "Generating detailed report with figures and tables...", AgentType.PI)

        try:
            import os
            output_dir = str(get_session_reports(self.session_id))
            os.makedirs(output_dir, exist_ok=True)

            # Progress callback for report generation
            async def progress_callback(data: dict):
                await self._send_message(
                    "progress",
                    {
                        "step": data.get("step", ""),
                        "progress": data.get("progress", 0),
                        "status": data.get("status", "generating")
                    },
                    AgentType.PI
                )

            # Create LLM client (use PI agent's server config for report generation)
            llm_client = self._settings_provider.get_llm_client('pi')

            # Get models from state or use defaults
            pi_model = self._settings_provider.get_model("pi")
            bio_model = self._settings_provider.get_model("bioinformatics")

            workspace_dir = str(get_session_workspace(self.session_id))

            if self.supervisor_agent:
                # Supervisor generates report from its accumulated analysis records.
                # Pass the PI agent's LLM client so the report_model (pi_model) is
                # called on the correct endpoint, not the supervisor's own endpoint.
                report = await self.supervisor_agent.generate_detailed_report(
                    analysis_state=analysis_state,
                    report_model=pi_model,
                    output_dir=output_dir,
                    workspace_dir=str(get_session_workspace(self.session_id)),
                    progress_callback=progress_callback,
                    report_llm_client=self._settings_provider.get_llm_client('pi'),
                )
            else:
                # Fallback: use original DetailedReportGenerator
                supervisor_model = self._settings_provider.get_model("supervisor")
                generator = DetailedReportGenerator(
                    llm_client=llm_client,
                    pi_model=pi_model,
                    bio_model=bio_model,
                    output_dir=output_dir,
                    workspace_dir=workspace_dir,
                    progress_callback=progress_callback,
                    session_id=self.session_id,
                    vision_settings=self._vision_settings,
                    supervisor_model=supervisor_model,
                )
                report = await generator.generate_report(analysis_state)

            await self._send_message(
                "detailed_report",
                {
                    "status": "completed",
                    "report_id": report.get("id", ""),
                    "title": report.get("title", ""),
                    "sections_count": len(report.get("sections", [])),
                    "figures_count": len(report.get("figures", [])),
                    "tables_count": len(report.get("tables", []))
                },
                AgentType.PI
            )

            logger.info(f"[{self.session_id}] Detailed report generated successfully")

        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to generate detailed report")
            import traceback
            logger.error(traceback.format_exc())
            await self._send_message(
                "error",
                f"Detailed report generation failed: {str(e)}. Analysis is still complete.",
                AgentType.PI
            )
            # Don't fail the workflow - detailed report is optional

        # Auto-snapshot after all reports are generated
        await self._create_auto_snapshot(analysis_state, "post_report")

        # Mark analysis as completed
        self._set_status(analysis_state, TaskStatus.COMPLETED)
        await self._send_message("completed", "Analysis completed", AgentType.PI)

        # Send final state sync for completion
        await self._send_state_sync(analysis_state, "analysis_complete")

        logger.info(f"[{self.session_id}] Analysis workflow completed")

        return {
            **state,
            "analysis_state": analysis_state
        }

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

        if state and state.values:
            # Restore _current_analysis_state for callbacks
            self._current_analysis_state = state.values.get("analysis_state")
            self._current_graph_state = state.values

            # Inject latest analysis_state if provided (so user changes during pause are picked up)
            if analysis_state:
                await self.compiled_graph.aupdate_state(config, {"analysis_state": analysis_state})
                self._current_analysis_state = analysis_state

            # Sync skip_nodes from sandbox_state to graph state (retry-node API stores
            # skip decisions in analysis_state.sandbox_state, but with_retry checks
            # the top-level WorkflowState)
            skip_nodes = []
            if analysis_state and analysis_state.sandbox_state:
                skip_nodes = analysis_state.sandbox_state.get("skip_nodes", [])
            if skip_nodes:
                await self.compiled_graph.aupdate_state(config, {"skip_nodes": skip_nodes})
                logger.info(f"[{self.session_id}] Injected skip_nodes into graph state: {skip_nodes}")

            # Resume from checkpoint
            logger.info(f"[{self.session_id}] Found saved state, resuming execution")
            final_state = await self.compiled_graph.ainvoke(
                None,
                config
            )
            logger.info(f"[{self.session_id}] Workflow resumed and completed")
            return final_state["analysis_state"]

        # No checkpoint found — start a fresh run from initialize_round.
        # This is the expected path after snapshot restore (checkpoint was deleted).
        if analysis_state is None:
            logger.error(f"[{self.session_id}] No checkpoint and no analysis_state for {thread_id}")
            raise ValueError(f"No saved state found for thread {thread_id}")

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
        final_state = await self.compiled_graph.ainvoke(workflow_state, config)
        logger.info(f"[{self.session_id}] Fresh run after restore completed")
        return final_state["analysis_state"]

    async def pause(self, thread_id: str):
        """Pause the workflow and save sandbox state including Python session"""
        logger.info(f"[{self.session_id}] Pausing workflow for thread {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        state = await self.compiled_graph.aget_state(config)

        if state and state.values:
            analysis_state = state.values["analysis_state"]
            self._set_status(analysis_state, TaskStatus.PAUSED)
            
            # Save sandbox state including Python session using dill
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
    def get_notebook_path(self) -> Optional[str]:
        """Get path to the saved Jupyter notebook"""
        if self.sandbox:
            return self.sandbox.get_notebook_path()
        return None
