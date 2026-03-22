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

from .models.schemas import (
    AnalysisState, AnalysisStep, AgentType, TaskStatus, RoundResult,
    WebSocketMessage, AgentMessage, MessageRole, BiochemistryIndicator,
    PublicationFigure, FigureReviewStatus,
    ClusterAnnotation, AnnotationConfirmationState,
)
from .agents.pi_agent import PIAgent
from .agents.pathology_agent import PathologyAgent
from .agents.bioinformatics_agent import BioinformaticsAgent
from .agents.biochemistry_agent import BiochemistryAgent
from .agents.literature_agent import LiteratureAgent
from .agents.painter_agent import PainterAgent
from .agents.supervisor_agent import SupervisorAgent
from .agents.base import AgentConfig
from .utils.llm_client import LLMManager, OpenAICompatibleClient
from .utils.sandbox import LocalJupyterSandbox
from .config import SystemConfig, REPORTS_DIR, WORKSPACE_DIR, get_session_reports, get_session_workspace, get_session_states
from .report_generator import DetailedReportGenerator, compress_image_to_base64


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


# Mapping from node names to result keys
NODE_TO_RESULT_KEY = {
    "pathology_analysis": "pathology_result",
    "bioinformatics_analysis": "bioinformatics_result",
    "biochemistry_analysis": "biochemistry_result",
    "literature_research": "literature_result",
}


def with_retry(node_name: str, max_retries: int = MAX_NODE_RETRIES):
    """
    Decorator to add retry logic to workflow nodes.
    Retries the node function internally before propagating failure.

    Args:
        node_name: Name of the node (for logging and tracking)
        max_retries: Maximum retry attempts

    Returns:
        Decorated function with retry capability
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, state: 'WorkflowState') -> 'WorkflowState':
            # Check if node should be skipped (PI decided this agent is not needed this round)
            if node_name in state.get("skip_nodes", []):
                logger.info(f"[{self.session_id}] Skipping node {node_name} (not activated by PI)")
                if hasattr(self, '_send_message'):
                    await self._send_message(
                        "status",
                        f"{node_name} skipped this round (not needed)",
                        None
                    )
                # Clear stale failed_node and set result to skipped (avoid using stale previous round result)
                result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
                return {**state, "failed_node": None, result_key: {"findings": [], "status": "skipped"}}

            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        await self._send_message(
                            "status",
                            f"Retrying {node_name} (attempt {attempt + 1}/{max_retries + 1})...",
                            AgentType.PI
                        )
                        # Exponential backoff with jitter
                        delay = min(2 ** attempt + random.uniform(0, 1), 30)
                        await asyncio.sleep(delay)

                    result = await func(self, state)
                    return result

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

            # All retries exhausted - return error state to allow workflow to continue
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

    def signal_continue(self):
        """Signal the workflow to continue from wait_for_user_input"""
        self._continue_event.set()

    def signal_annotation_confirmed(self):
        """Signal that user has confirmed cell type annotations"""
        self._annotation_event.set()

    async def initialize(self):
        """Initialize all components"""
        logger.info(f"[{self.session_id}] Initializing workflow components")
        
        # Initialize checkpointer if not exists
        if self._db_conn is None:
            states_dir = get_session_states(self.session_id)
            states_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(states_dir / "checkpoints.sqlite")
            self._db_conn = await aiosqlite.connect(db_path)
            serde = JsonPlusSerializer(allowed_msgpack_modules=[
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
            agent_name="workflow"
        )
        
        # Initialize local Jupyter sandbox with session-specific paths
        self.sandbox = LocalJupyterSandbox(
            session_id=self.session_id,
            state_dir=str(get_session_states(self.session_id)),
            working_dir=str(get_session_workspace(self.session_id)),
            timeout=self.config.sandbox_timeout
        )
        
        # Get custom models, prompts, and temperatures from initial state if provided
        custom_models = getattr(self, '_custom_models', {})
        custom_prompts = getattr(self, '_custom_prompts', {})
        custom_temperatures = getattr(self, '_custom_temperatures', {})

        # Merge custom temperatures with defaults from centralized config
        self.temperatures = self.config.temperatures.merge_custom(custom_temperatures)
        
        # Determine models to use (custom takes precedence)
        pi_model = custom_models.get('pi', self.config.models.pi_model)
        pathology_model = custom_models.get('pathology', self.config.models.pathology_model)
        bioinformatics_model = custom_models.get('bioinformatics', self.config.models.bioinformatics_model)
        biochemistry_model = custom_models.get('biochemistry', self.config.models.biochemistry_model)
        literature_model = custom_models.get('literature', self.config.models.literature_model)
        
        # Determine prompts to use (custom takes precedence)
        pi_prompt = custom_prompts.get('pi', self.config.prompts.pi_prompt)
        pathology_prompt = custom_prompts.get('pathology', self.config.prompts.pathology_prompt)
        bioinformatics_prompt = custom_prompts.get('bioinformatics', self.config.prompts.bioinformatics_prompt)
        biochemistry_prompt = custom_prompts.get('biochemistry', self.config.prompts.biochemistry_prompt)
        literature_prompt = custom_prompts.get('literature', self.config.prompts.literature_prompt)
        
        # Initialize agents with resolved models, prompts, and temperatures
        self.pi_agent = PIAgent(
            llm_client=self.llm_manager.openrouter,
            model=pi_model,
            system_prompt=pi_prompt,
            config=AgentConfig(temperature=self.temperatures.get('pi', 0.5))
        )

        self.pathology_agent = PathologyAgent(
            llm_client=self.llm_manager.openrouter,
            model=pathology_model,
            system_prompt=pathology_prompt,
            tile_size=self.config.image_tile_size,
            max_tiles=50,
            config=AgentConfig(temperature=self.temperatures.get('pathology', 0.2))
        )

        self.bioinformatics_agent = BioinformaticsAgent(
            llm_client=self.llm_manager.openrouter,
            model=bioinformatics_model,
            system_prompt=bioinformatics_prompt,
            sandbox=self.sandbox,
            literature_callback=self._literature_callback,
            pi_callback=self._pi_callback,
            annotate_clusters_callback=self._annotate_clusters_callback,
            message_callback=self._agent_message_callback,
            supervisor_callback=self._supervisor_callback,
            config=AgentConfig(temperature=self.temperatures.get('bioinformatics', 0.1))
        )

        self.biochemistry_agent = BiochemistryAgent(
            llm_client=self.llm_manager.openrouter,
            model=biochemistry_model,
            system_prompt=biochemistry_prompt,
            config=AgentConfig(temperature=self.temperatures.get('biochemistry', 0.2))
        )

        # Literature agent with mode-specific configuration
        literature_config = getattr(self, '_literature_config', {})
        self.literature_agent = LiteratureAgent(
            llm_client=self.llm_manager.openrouter,
            model=literature_model,
            system_prompt=literature_prompt,
            config=AgentConfig(temperature=self.temperatures.get('literature', 0.6)),
            literature_mode=literature_config.get('mode', 'deep_research'),
            deep_research_model=literature_config.get('deep_research_model', ''),
            google_api_key=literature_config.get('google_api_key', ''),
            pubmed_email=literature_config.get('pubmed_email', ''),
            max_search_rounds=literature_config.get('max_search_rounds', 5),
            proxy_url=self.config.api.proxy_url,
            fallback_model=pi_model,
        )

        # Initialize Painter Agent for graphic abstract generation (uses OpenRouter like other agents)
        painter_model = custom_models.get('painter', self.config.models.painter_model)
        self.painter_agent = PainterAgent(
            llm_client=self.llm_manager.openrouter,
            model=painter_model,
            config=AgentConfig(temperature=self.temperatures.get('painter', 0.7))
        )
        self.painter_agent.set_session(self.session_id)

        # Initialize Supervisor Agent for real-time analysis monitoring
        supervisor_model = custom_models.get('supervisor', self.config.models.supervisor_model)
        supervisor_prompt = custom_prompts.get('supervisor', getattr(self.config.prompts, 'supervisor_prompt', ''))
        self.supervisor_agent = SupervisorAgent(
            llm_client=self.llm_manager.openrouter,
            model=supervisor_model,
            pi_callback=self._pi_callback,
            message_callback=self._supervisor_message_callback,
            session_id=self.session_id,
            temperature=self.temperatures.get('supervisor', 0.1),
            system_prompt=supervisor_prompt,
        )

        logger.info(f"[{self.session_id}] Agents initialized: PI={pi_model}, Pathology={pathology_model}, Bioinformatics={bioinformatics_model}, Biochemistry={biochemistry_model}, Literature={literature_model}, Painter={painter_model}, Supervisor={supervisor_model}")

        # Build workflow graph
        self._build_graph()
        logger.info(f"[{self.session_id}] Workflow graph built")
        
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
                    "change_type": change_type,  # "round_complete", "status_change", "data_added", "error_added"
                    "version": analysis_state.state_version,
                    "status": analysis_state.status.value,
                    "current_round": analysis_state.current_round,
                    "analysis_phase": analysis_state.analysis_phase,
                    "timestamp": analysis_state.last_sync_timestamp.isoformat(),
                    "unresolved_errors": len(analysis_state.get_unresolved_errors())
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
            response = await self.pi_agent.chat_without_history(prompt, temperature=self.temperatures.get('pi', 0.5))
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

    async def _annotate_clusters_callback(
        self,
        cluster_markers: Dict,
        tissue: str,
        organism: str
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

        await self._send_message(
            "status",
            f"Literature-based cluster annotation: {n_clusters} clusters in {organism} {tissue}...",
            AgentType.LITERATURE
        )

        try:
            result = await self.literature_agent.annotate_clusters(
                cluster_markers=cluster_markers,
                organism=organism,
                tissue=tissue,
                state=state
            )

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
                cluster_anns = []
                for cid, info in annotations.items():
                    if isinstance(info, dict):
                        cluster_anns.append(ClusterAnnotation(
                            cluster_id=cid,
                            cell_type=info.get("cell_type", "Unknown"),
                            subtypes=info.get("subtypes", []),
                            key_markers=info.get("key_markers", []),
                            confidence=info.get("confidence", "low"),
                            evidence=info.get("evidence", ""),
                            alternative_cell_types=info.get("alternative_cell_types", []),
                            low_confidence_reason=info.get("low_confidence_reason", ""),
                        ))

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

                # Generate the context summary report
                analysis_summary = self._build_annotation_context_report(state, self._current_graph_state)

                # Generate downloadable HTML annotation context report
                from .report_generator import generate_annotation_context_report_html
                try:
                    html_path = generate_annotation_context_report_html(state, self._current_graph_state, cluster_anns)
                    state.annotation_report_html_path = html_path
                    report_html_url = f"/api/session/{self.session_id}/annotation-report/html"
                except Exception as e:
                    logger.warning(f"[{self.session_id}] Annotation context report generation failed: {e}")
                    report_html_url = None

                ann_state = AnnotationConfirmationState(
                    cluster_annotations=cluster_anns,
                    tissue=tissue,
                    organism=organism,
                    tissue_summary=result.get("tissue_summary", ""),
                    analysis_summary=analysis_summary,
                    pending=True,
                )
                state.annotation_state = ann_state

                # Step 3: Send annotation confirmation request to frontend and block
                # Clear event BEFORE sending message to prevent race condition:
                # if frontend confirms between send and clear, the set() would be lost.
                self._annotation_event.clear()

                await self._send_message("annotation_confirmation", {
                    "annotations": [a.model_dump() for a in cluster_anns],
                    "tissue": tissue,
                    "organism": organism,
                    "tissue_summary": result.get("tissue_summary", ""),
                    "analysis_summary": analysis_summary,
                    "report_html_url": report_html_url,
                }, AgentType.LITERATURE)

                original_status = state.status
                state.status = TaskStatus.WAITING_ANNOTATION
                await self._send_state_sync(state, "annotation_confirmation")

                logger.info(f"[{self.session_id}] Waiting for user annotation confirmation...")
                await self._annotation_event.wait()
                logger.info(f"[{self.session_id}] Annotation confirmation received")

                # Restore status, but respect pause if user paused during annotation wait
                if state.status == TaskStatus.PAUSED:
                    logger.info(f"[{self.session_id}] User paused during annotation wait, returning early")
                    return result
                state.status = original_status
                await self._send_state_sync(state, "annotation_resumed")

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
        except Exception as e:
            error_msg = f"Cluster annotation failed: {str(e)}"
            logger.error(f"[{self.session_id}] {error_msg}")
            await self._send_message("error", error_msg, AgentType.LITERATURE)
            return {
                "status": "error",
                "annotations": {},
                "error": error_msg
            }

    async def _agent_message_callback(self, msg_type: str, content: str):
        """Callback for agent messages"""
        await self._send_message(msg_type, content, AgentType.BIOINFORMATICS)

    async def _supervisor_message_callback(self, msg_type: str, content: str):
        """Callback for supervisor agent messages"""
        await self._send_message(msg_type, content, AgentType.SUPERVISOR)

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

        Args:
            figure: PublicationFigure to review

        Returns:
            Dict with review results (approved, feedback, suggestions)
        """
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

        # Get current analysis state
        state = self._current_analysis_state if self._current_analysis_state else AnalysisState()

        # Call PI Agent to review the figure
        review = await self.pi_agent.review_figure(
            state,
            figure,
            figure.png_base64
        )

        # Send review result to frontend
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

            # Get adjusted instructions
            adjusted_instructions = pi_response.get("adjusted_instructions")
            
            # Clear pending input
            analysis_state.pending_human_input = None
            analysis_state.pending_uploaded_files = []
        
        # Increment round counter
        analysis_state.current_round += 1
        analysis_state.status = TaskStatus.RUNNING

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
                analysis_state.analysis_phase = "innovation"

            # Check directions_queue first (multi-direction mode), then single direction
            user_direction = analysis_state.user_chosen_direction
            if analysis_state.directions_queue:
                # Multi-direction mode: pop next direction from queue
                current_dir = analysis_state.directions_queue.pop(0)
                dir_text = current_dir.get("direction", "")
                dir_notes = current_dir.get("notes", "")
                logger.info(f"[{self.session_id}] Innovation phase: direction from queue → {dir_text[:100]} (remaining: {len(analysis_state.directions_queue)})")
                # Set current direction immediately so retry has correct context
                analysis_state.current_round_direction = dir_text
                try:
                    instructions = await self.pi_agent.innovation_instruction(
                        analysis_state, dir_text, notes=dir_notes
                    )
                except Exception as e:
                    # Re-insert direction at front of queue so it is not silently lost
                    analysis_state.directions_queue.insert(0, current_dir)
                    # Clear stale direction since this round's instruction failed
                    analysis_state.current_round_direction = None
                    logger.exception(f"[{self.session_id}] innovation_instruction failed, direction re-queued")
                    raise
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
            agents_to_activate.add("bioinformatics")  # Always activate bioinformatics

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
        phase_instruction = ""
        if analysis_state.analysis_phase == "exploration":
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
7. Complete cell communication analysis (CellChat or similar)
8. For EACH cell type group's DEG results, generate a visualization figure (volcano plot or bar chart)
9. For EACH cell type group's KEGG/GO enrichment, generate a bar plot of top enriched pathways
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

You may skip irrelevant cell populations and focus on those relevant to the hypothesis.
Do NOT re-run completed steps from Round 1 (QC, clustering, cell annotation, etc.).
"""

        # Build merge strategy context from PI instructions
        merge_context = ""
        if merge_strategy:
            merge_context += f"\n\nPI's cluster merge strategy:\n{merge_strategy}"
        if cluster_comparison_strategy:
            merge_context += f"\n\nPI's cluster comparison strategy:\n{cluster_comparison_strategy}"

        instruction = f"""Analyze single-cell RNA-seq data for project: {analysis_state.name}

Research context: {analysis_state.initial_instruction}
{phase_instruction}

PI's analysis goals for this round:
{chr(10).join(f"- {task}" for task in _normalize_tasks(bioinf_tasks)) if bioinf_tasks else "- Perform standard scRNA-seq analysis pipeline"}
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

        result = await self.bioinformatics_agent.run_autonomous_analysis(
            analysis_state,
            instruction,
            max_steps=200,  # Allow sufficient steps for full analysis + figure review loops
            figure_review_callback=self._figure_review_callback,
            supervisor_callback=self._supervisor_callback,
        )

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
            response = await self.literature_agent.chat(prompt, temperature=0.2)
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
        if analysis_state.analysis_phase == "exploration":
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
                    from .report_generator import generate_exploration_report_html
                    html_path = generate_exploration_report_html(
                        analysis_state, exploration_report, round_result
                    )
                    analysis_state.exploration_report_html_path = html_path
                    logger.info(f"[{self.session_id}] Exploration report HTML saved to: {html_path}")
                except Exception as html_err:
                    logger.error(f"[{self.session_id}] Exploration report HTML generation failed: {html_err}")

                html_url = f"/api/session/{self.session_id}/exploration-report/html" if analysis_state.exploration_report_html_path else None
                await self._send_message("exploration_report", {
                    "report_a": exploration_report.model_dump(),
                    "round": round_result.round_number,
                    "summary": round_result.pi_summary,
                    "html_url": html_url,
                }, AgentType.PI)
            except Exception as e:
                logger.exception(f"[{self.session_id}] Exploration report generation failed")
                await self._send_message(
                    "error",
                    f"Exploration report generation failed: {e}. You can still select a research direction based on the analysis findings above.",
                    AgentType.PI
                )

            # Transition to innovation phase for next round
            analysis_state.analysis_phase = "innovation"
            logger.info(f"[{self.session_id}] Transitioning to innovation phase")

        analysis_state.round_results.append(round_result)

        # Multi-direction correlation analysis: trigger when queue is empty and multiple directions were selected
        if (
            not analysis_state.directions_queue
            and len(analysis_state.user_chosen_directions) > 1
            and analysis_state.analysis_phase == "innovation"
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
        round_summary = {
            "round": round_result.round_number,
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
        
        # Clear continue event BEFORE announcing waiting status to prevent race
        # condition where frontend signals between status message and event.wait()
        self._continue_event.clear()

        # Set status to waiting for user input
        analysis_state.status = TaskStatus.WAITING_INPUT

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

        # Multi-direction mode: auto-skip wait if more directions remain in queue
        if analysis_state.directions_queue:
            # Filter out empty/whitespace-only directions before deciding to skip
            analysis_state.directions_queue = [
                d for d in analysis_state.directions_queue if d and d.get("direction", "").strip()
            ]
        if analysis_state.directions_queue:
            logger.info(f"[{self.session_id}] Directions queue has {len(analysis_state.directions_queue)} remaining, auto-continuing")
            return {**state, "analysis_state": analysis_state}

        # After exploration phase (Round 1), force manual wait so user must choose direction
        if analysis_state.current_round == 1 and analysis_state.analysis_phase == "innovation":
            wait_seconds = -1
            logger.info(f"[{self.session_id}] Exploration phase complete, forcing manual wait for direction selection")

        if wait_seconds == 0:
            # No-wait mode: proceed immediately
            logger.info(f"[{self.session_id}] No-wait mode, proceeding immediately (round {analysis_state.current_round})")
        elif wait_seconds == -1:
            # Manual mode: wait indefinitely for user signal
            # Guard: ensure WAITING_INPUT status is synced before blocking
            # (prevents deadlock when Resume restores status to RUNNING)
            if analysis_state.status != TaskStatus.WAITING_INPUT:
                analysis_state.status = TaskStatus.WAITING_INPUT
                await self._send_state_sync(analysis_state, "waiting_input")
            logger.info(f"[{self.session_id}] Manual mode, waiting indefinitely for user input (round {analysis_state.current_round})")
            await self._continue_event.wait()
            logger.info(f"[{self.session_id}] Continue signal received")
        else:
            # Countdown mode: backend safety timeout = frontend countdown + 60s buffer
            # Guard: ensure WAITING_INPUT status is synced before blocking
            if analysis_state.status != TaskStatus.WAITING_INPUT:
                analysis_state.status = TaskStatus.WAITING_INPUT
                await self._send_state_sync(analysis_state, "waiting_input")
            backend_timeout = wait_seconds + 60
            logger.info(f"[{self.session_id}] Waiting for user input (round {analysis_state.current_round}, timeout={backend_timeout}s)")
            try:
                await asyncio.wait_for(self._continue_event.wait(), timeout=backend_timeout)
                logger.info(f"[{self.session_id}] Continue signal received")
            except asyncio.TimeoutError:
                logger.info(f"[{self.session_id}] Wait timeout ({backend_timeout}s), auto-proceeding")
                analysis_state.status = TaskStatus.RUNNING
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
        # Default to False (stop) to be conservative - only continue if explicitly set to True
        if state.get("should_continue", False):
            return "continue"
        return "stop"
    
    async def _generate_final_report(self, state: WorkflowState) -> WorkflowState:
        """Generate final analysis report including integrated report with figures"""
        analysis_state = state["analysis_state"]

        # Mark status so interrupted report generation can be detected on resume
        analysis_state.status = TaskStatus.GENERATING_REPORT
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
        """Generate graphic abstract with revision loop (max 2 revisions, PI selects best)"""
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

        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to generate graphic abstract")
            await self._send_message(
                "error",
                f"Graphic abstract generation failed: {str(e)}. Continuing with detailed report...",
                AgentType.PAINTER
            )
            # Don't fail the workflow - graphic abstract is optional

        logger.info(f"[{self.session_id}] Graphic abstract step completed")

        return {
            **state,
            "analysis_state": analysis_state
        }

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

            # Create LLM client
            llm_client = OpenAICompatibleClient(
                api_key=self.config.api.api_key,
                base_url=self.config.api.llm_base_url,
                proxy_url=self.config.api.proxy_url
            )

            # Get models from state or use defaults
            pi_model = analysis_state.custom_models.get("pi", self.config.models.pi_model)
            bio_model = analysis_state.custom_models.get("bioinformatics", self.config.models.bioinformatics_model)

            workspace_dir = str(get_session_workspace(self.session_id))

            if self.supervisor_agent:
                # Supervisor generates report from its accumulated analysis records
                report = await self.supervisor_agent.generate_detailed_report(
                    analysis_state=analysis_state,
                    report_model=pi_model,
                    output_dir=output_dir,
                    workspace_dir=str(get_session_workspace(self.session_id)),
                    progress_callback=progress_callback,
                )
            else:
                # Fallback: use original DetailedReportGenerator
                generator = DetailedReportGenerator(
                    llm_client=llm_client,
                    pi_model=pi_model,
                    bio_model=bio_model,
                    output_dir=output_dir,
                    workspace_dir=workspace_dir,
                    progress_callback=progress_callback,
                    session_id=self.session_id
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

        # Mark analysis as completed
        analysis_state.status = TaskStatus.COMPLETED
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
        # Store custom models, prompts, temperatures, and literature config for initialization
        self._custom_models = initial_state.custom_models
        self._custom_prompts = initial_state.custom_prompts
        self._custom_temperatures = initial_state.custom_temperatures
        self._literature_config = initial_state.literature_config
        
        try:
            await self._send_message("status", "Initializing analysis system...")
            await self.initialize()
            logger.info(f"Workflow initialized for session {self.session_id}")
        except Exception as e:
            logger.exception("Failed to initialize workflow")
            await self._send_message("error", f"Initialization failed: {str(e)}")
            initial_state.status = TaskStatus.FAILED
            return initial_state
        
        # Create initial workflow state
        workflow_state: WorkflowState = {
            "analysis_state": initial_state,
            "current_round": 0,
            "pi_instructions": {},
            "pathology_result": None,
            "bioinformatics_result": None,
            "biochemistry_result": None,
            "literature_result": None,
            "round_result": None,
            "should_continue": True,
            "error": None,
            "messages": []
        }
        
        # Run workflow
        config = {"configurable": {"thread_id": thread_id}}

        # Verify workflow was properly initialized
        if self.compiled_graph is None:
            error_msg = "Workflow not properly initialized - compiled_graph is None"
            logger.error(f"[{self.session_id}] {error_msg}")
            await self._send_message("error", error_msg)
            initial_state.status = TaskStatus.FAILED
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
            initial_state.status = TaskStatus.FAILED
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
        self._custom_models = initial_state.custom_models
        self._custom_prompts = initial_state.custom_prompts
        self._custom_temperatures = initial_state.custom_temperatures
        self._literature_config = initial_state.literature_config

        try:
            await self._send_message("status", "Resuming report generation (analysis already complete)...")
            await self.initialize()
            logger.info(f"[{self.session_id}] Workflow initialized for report-only generation")
        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to initialize for report generation")
            await self._send_message("error", f"Initialization failed: {str(e)}")
            initial_state.status = TaskStatus.FAILED
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
            initial_state.status = TaskStatus.FAILED
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
            # Restore custom models/prompts/temperatures from analysis state
            if analysis_state:
                self._custom_models = analysis_state.custom_models
                self._custom_prompts = analysis_state.custom_prompts
                self._custom_temperatures = analysis_state.custom_temperatures
                self._literature_config = analysis_state.literature_config
            logger.info(f"[{self.session_id}] Re-initializing workflow components for resume")
            await self.initialize()

        config = {"configurable": {"thread_id": thread_id}}

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

        logger.error(f"[{self.session_id}] No saved state found for thread {thread_id}")
        raise ValueError(f"No saved state found for thread {thread_id}")
    
    async def pause(self, thread_id: str):
        """Pause the workflow and save sandbox state including Python session"""
        logger.info(f"[{self.session_id}] Pausing workflow for thread {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        state = await self.compiled_graph.aget_state(config)

        if state and state.values:
            analysis_state = state.values["analysis_state"]
            analysis_state.status = TaskStatus.PAUSED
            
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
    
    def get_notebook_path(self) -> Optional[str]:
        """Get path to the saved Jupyter notebook"""
        if self.sandbox:
            return self.sandbox.get_notebook_path()
        return None
