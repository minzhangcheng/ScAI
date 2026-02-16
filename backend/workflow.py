"""
LangGraph workflow for multi-agent single-cell analysis
"""
import asyncio
import logging
import traceback
from functools import wraps
from typing import Callable, Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import json
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Configure logger
logger = logging.getLogger(__name__)

# Maximum retry attempts for workflow nodes
MAX_NODE_RETRIES = 3

from .models.schemas import (
    AnalysisState, AgentType, TaskStatus, RoundResult,
    WebSocketMessage, AgentMessage, MessageRole, BiochemistryIndicator,
    PublicationFigure, FigureReviewStatus
)
from .agents.pi_agent import PIAgent
from .agents.pathology_agent import PathologyAgent
from .agents.bioinformatics_agent import BioinformaticsAgent
from .agents.biochemistry_agent import BiochemistryAgent
from .agents.literature_agent import LiteratureAgent
from .agents.painter_agent import PainterAgent
from .agents.base import AgentConfig
from .utils.llm_client import LLMManager, OpenAICompatibleClient
from .utils.sandbox import LocalJupyterSandbox
from .config import SystemConfig, REPORTS_DIR, WORKSPACE_DIR
from .report_generator import DetailedReportGenerator


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
                return state

            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        await self._send_message(
                            "status",
                            f"Retrying {node_name} (attempt {attempt + 1}/{max_retries + 1})...",
                            AgentType.PI
                        )
                        # Brief delay before retry
                        await asyncio.sleep(2)

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
                result_key: {"error": str(last_error), "findings": []}
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
        
        # Workflow graph
        self.graph = None
        self.compiled_graph = None
        self.checkpointer = MemorySaver()

        # Current analysis state for callbacks (updated during workflow execution)
        self._current_analysis_state: Optional[AnalysisState] = None

        # Event for blocking between rounds (wait for user input or continue signal)
        self._continue_event = asyncio.Event()
        
    def signal_continue(self):
        """Signal the workflow to continue from wait_for_user_input"""
        self._continue_event.set()

    async def initialize(self):
        """Initialize all components"""
        logger.info(f"[{self.session_id}] Initializing workflow components")
        # Initialize LLM manager with session context
        self.llm_manager = LLMManager(
            api_key=self.config.api.api_key,
            base_url=self.config.api.llm_base_url,
            proxy_url=self.config.api.proxy_url,
            session_id=self.session_id,
            agent_name="workflow"
        )
        
        # Initialize local Jupyter sandbox with session ID for state persistence
        self.sandbox = LocalJupyterSandbox(
            session_id=self.session_id,
            state_dir=self.config.state_dir,
            working_dir=self.config.workspace_dir,
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
            literature_mode=literature_config.get('mode', 'pubmed_llm'),
            google_api_key=literature_config.get('google_api_key', ''),
            pubmed_email=literature_config.get('pubmed_email', ''),
            max_search_rounds=literature_config.get('max_search_rounds', 5),
            proxy_url=self.config.api.proxy_url,
        )

        # Initialize Painter Agent for graphic abstract generation (uses OpenRouter like other agents)
        painter_model = custom_models.get('painter', self.config.models.painter_model)
        self.painter_agent = PainterAgent(
            llm_client=self.llm_manager.openrouter,
            model=painter_model,
            config=AgentConfig(temperature=self.temperatures.get('painter', 0.7))
        )

        logger.info(f"[{self.session_id}] Agents initialized: PI={pi_model}, Pathology={pathology_model}, Bioinformatics={bioinformatics_model}, Biochemistry={biochemistry_model}, Literature={literature_model}, Painter={painter_model}")

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

        # Add edges - sequential execution
        self.graph.add_edge("initialize_round", "pi_planning")

        # Sequential analysis: pathology -> bioinformatics -> biochemistry -> literature
        self.graph.add_edge("pi_planning", "pathology_analysis")
        self.graph.add_edge("pathology_analysis", "bioinformatics_analysis")
        self.graph.add_edge("bioinformatics_analysis", "biochemistry_analysis")
        self.graph.add_edge("biochemistry_analysis", "literature_research")
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
                logger.error(f"[Workflow] Failed to send message: {e}")

    async def _send_state_sync(self, analysis_state: AnalysisState, change_type: str):
        """Send unified state synchronization message after state changes"""
        # Increment version for synchronization
        analysis_state.increment_version()

        # Save state via callback first
        if self.state_callback:
            try:
                await self.state_callback(analysis_state)
            except Exception as e:
                logger.error(f"[{self.session_id}] State callback failed: {e}")

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
                    "timestamp": analysis_state.last_sync_timestamp.isoformat(),
                    "unresolved_errors": len(analysis_state.get_unresolved_errors())
                }
            )
            try:
                await self.message_callback(sync_msg)
            except Exception as e:
                logger.error(f"[{self.session_id}] Failed to send state sync: {e}")

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

Be specific and actionable in your response."""

        try:
            response = await self.pi_agent.chat(prompt, temperature=self.temperatures.get('pi', 0.5))
            await self._send_message("pi_response", response, AgentType.PI)
            return response
        except Exception as e:
            error_msg = f"PI consultation failed: {str(e)}"
            await self._send_message("error", error_msg, AgentType.PI)
            return "Unable to get PI guidance. Please proceed with standard bioinformatics practices."

    async def _annotate_clusters_callback(
        self,
        cluster_markers: Dict,
        tissue: str,
        organism: str
    ) -> Dict[str, Any]:
        """Callback for bioinformatics agent to annotate all clusters via literature."""
        n_clusters = len(cluster_markers)
        await self._send_message(
            "status",
            f"Literature-based cluster annotation: {n_clusters} clusters in {organism} {tissue}...",
            AgentType.LITERATURE
        )

        state = self._current_analysis_state if self._current_analysis_state else AnalysisState()

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
                parts.append(f"Round {prev_round.round_number}: " + "; ".join(findings[:5]))

        if not parts:
            return ""

        return (
            "\n\n## Your Previous Analysis Findings\n"
            "Below are your findings from prior rounds. Build upon these — do NOT repeat the same analyses.\n"
            + "\n".join(parts)
        )

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
            # Reset continuation flags for new round
            "should_continue": True,
            "human_stop_requested": False,
            "human_guidance_processed": False
        }

    async def _pi_planning(self, state: WorkflowState) -> WorkflowState:
        """PI agent plans the round, incorporating human guidance and adjusted instructions"""
        analysis_state = state["analysis_state"]
        human_guidance = state.get("human_guidance")
        adjusted_instructions = state.get("adjusted_instructions")

        logger.info(f"[{self.session_id}] PI planning for round {analysis_state.current_round}")
        if human_guidance:
            logger.info(f"[{self.session_id}] Human guidance received: {human_guidance[:100]}...")

        await self._send_message("status", "PI is creating analysis plan...", AgentType.PI)
        
        # If we already have adjusted instructions from user input processing, use them
        if adjusted_instructions:
            await self._send_message(
                "message",
                f"Using adjusted instructions from user input:\n{json.dumps(adjusted_instructions, ensure_ascii=False, indent=2)}",
                AgentType.PI
            )
            return {
                **state,
                "pi_instructions": adjusted_instructions,
                "human_guidance_processed": True
            }
        
        # Normal planning without pre-processed instructions
        skip_nodes = []  # Empty for round 1; populated for round 2+
        if analysis_state.current_round == 1:
            # First round - generate initial instructions
            instructions = await self.pi_agent.initial_instruction(
                analysis_state,
                analysis_state.initial_instruction
            )
        else:
            # Subsequent rounds - based on previous results
            last_round = analysis_state.get_latest_round()
            instructions = {
                "pathology_tasks": [],
                "bioinformatics_tasks": [],
                "biochemistry_tasks": [],
                "literature_topics": []
            }

            # Extract tasks for each agent
            for step in (last_round.next_steps if last_round else []):
                if step.startswith("[Pathology]"):
                    instructions["pathology_tasks"].append(step[11:].strip())
                elif step.startswith("[Bioinformatics]"):
                    instructions["bioinformatics_tasks"].append(step[16:].strip())
                elif step.startswith("[Biochemistry]"):
                    instructions["biochemistry_tasks"].append(step[14:].strip())
                elif step.startswith("[Literature]"):
                    instructions["literature_topics"].append(step[12:].strip())

            # Build skip_nodes from PI's agent activation decision
            agents_to_activate = set(last_round.agents_for_next_round) if last_round else set()
            agents_to_activate.add("bioinformatics")  # Always activate bioinformatics

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

        await self._send_message(
            "message",
            json.dumps(instructions, ensure_ascii=False, indent=2),
            AgentType.PI
        )

        result = {
            **state,
            "pi_instructions": instructions,
        }
        # Only set skip_nodes for round 2+ (round 1 runs all agents)
        if analysis_state.current_round > 1:
            result["skip_nodes"] = skip_nodes

        return result
    
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
                "pathology_result": {"findings": [], "status": "no_data"}
            }
        
        # Log image count
        await self._send_message(
            "status", 
            f"Analyzing {len(pathology_files)} pathology image(s)...", 
            AgentType.PATHOLOGY
        )
        
        # Enrich instruction with previous findings for round 2+
        task_instruction = "\n".join(tasks)
        previous_context = self._build_previous_findings_context(analysis_state, "pathology")
        if previous_context:
            task_instruction += previous_context

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
                "bioinformatics_result": {"findings": [], "status": "waiting_data"}
            }
        
        # Check for pending human input and incorporate it
        # Note: human input is consumed and cleared in _initialize_round, not here
        human_guidance = ""
        if analysis_state.pending_human_input:
            human_guidance = f"\n\nHuman researcher guidance: {analysis_state.pending_human_input}"
        
        # Step 1: PI provides analysis goals
        await self._send_message("status", "PI is defining analysis goals...", AgentType.PI)
        
        # Get PI instructions from workflow state
        pi_instructions = state.get("pi_instructions", {})
        bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
        
        # Build previous findings and code history context for round 2+
        previous_bio_findings = self._build_previous_findings_context(analysis_state, "bioinformatics")
        code_history_context = ""
        if analysis_state.current_round > 1:
            code_history_context = self._build_code_history_summary(analysis_state)

        # Build comprehensive instruction for bioinformatics agent
        instruction = f"""Analyze single-cell RNA-seq data for project: {analysis_state.name}

Research context: {analysis_state.initial_instruction}

PI's analysis goals for this round:
{chr(10).join(f"- {task}" for task in bioinf_tasks) if bioinf_tasks else "- Perform standard scRNA-seq analysis pipeline"}
{human_guidance}
{previous_bio_findings}
{code_history_context}

This is Round {analysis_state.current_round}. The Jupyter kernel retains all variables from previous rounds.
Focus on PI's specific goals above — do NOT re-run completed steps (QC, clustering, cell annotation, etc.).
Report key findings when you complete each major analysis phase."""

        await self._send_message("prompt", instruction, AgentType.PI)
        
        # Step 2: Bioinformatics agent runs autonomously with figure review support
        await self._send_message("status", "Bioinformatics agent is running autonomous analysis...", AgentType.BIOINFORMATICS)

        result = await self.bioinformatics_agent.run_autonomous_analysis(
            analysis_state,
            instruction,
            max_steps=200,  # Allow sufficient steps for full analysis + figure review loops
            figure_review_callback=self._figure_review_callback
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

        # Store code executions
        for code_exec in result.get("code_history", []):
            if code_exec not in analysis_state.code_history:
                analysis_state.code_history.append(code_exec)

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
        task_instruction = "\n".join(tasks) if tasks else "Analyze existing biochemistry indicators"
        previous_context = self._build_previous_findings_context(analysis_state, "biochemistry")
        if previous_context:
            task_instruction += previous_context

        result = await self.biochemistry_agent.process(
            analysis_state,
            task_instruction
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

        topics = instructions.get("literature_topics", [])
        logger.info(f"[{self.session_id}] Starting literature research with topics: {topics}")
        if not topics:
            # Default topics based on other findings
            path_findings = state.get("pathology_result", {}).get("findings", []) if state.get("pathology_result") else []
            bio_findings = state.get("bioinformatics_result", {}).get("findings", []) if state.get("bioinformatics_result") else []

            if path_findings or bio_findings:
                # Use findings as search topics (context from actual analysis)
                topics = [f"single cell sequencing {' '.join((path_findings + bio_findings)[:3])}"]

        await self._send_message("status", "Literature research in progress...", AgentType.LITERATURE)

        # Enrich topics with previous findings for round 2+
        previous_lit_context = self._build_previous_findings_context(analysis_state, "literature")

        # Check if Deep Research mode is available
        is_deep_research = (
            self.literature_agent.literature_mode == "gemini_deep_research"
            and self.literature_agent._deep_research is not None
        )

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
                analysis_state, merged_topic, use_deep_research=True
            )
            results = [dr_result]

            # Evaluate coverage — fallback to PubMed for gaps
            if len(limited_topics) > 1:
                gaps = await self._evaluate_dr_coverage(limited_topics, dr_result)
                if gaps:
                    logger.info(f"[{self.session_id}] Deep Research gaps detected, PubMed fallback for: {gaps}")
                    await self._send_message("status", f"Supplementing {len(gaps)} topic(s) via PubMed...", AgentType.LITERATURE)
                    for gap_topic in gaps:
                        enriched = gap_topic
                        if previous_lit_context:
                            enriched = f"{gap_topic}\n{previous_lit_context}\nFocus on aspects NOT already covered."
                        gap_result = await self.literature_agent.process(
                            analysis_state, enriched, use_deep_research=False
                        )
                        results.append(gap_result)
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
                    use_deep_research=True
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
            # Strip markdown code fences if present
            cleaned = response.strip().strip('`')
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
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
        
        # Extract findings from each agent
        path_findings = state.get("pathology_result", {}).get("findings", []) if state.get("pathology_result") else []
        bio_findings = state.get("bioinformatics_result", {}).get("findings", []) if state.get("bioinformatics_result") else []
        chem_findings = state.get("biochemistry_result", {}).get("findings", []) if state.get("biochemistry_result") else []
        lit_findings = state.get("literature_result", {}).get("findings", []) if state.get("literature_result") else []
        
        round_result = await self.pi_agent.synthesize_round(
            analysis_state,
            pathology_result=json.dumps(path_findings, ensure_ascii=False),
            bioinformatics_result=json.dumps(bio_findings, ensure_ascii=False),
            biochemistry_result=json.dumps(chem_findings, ensure_ascii=False),
            literature_result=json.dumps(lit_findings, ensure_ascii=False)
        )

        logger.info(f"[{self.session_id}] Round {round_result.round_number} synthesis complete: new_findings={round_result.new_findings}, {len(round_result.next_steps)} next steps")

        analysis_state.round_results.append(round_result)
        
        # Save state after each round
        if self.state_callback:
            try:
                await self.state_callback(analysis_state)
            except Exception as e:
                logger.error(f"Failed to save state after round: {e}")
        
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
                *round_result.biochemistry_findings
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
        
        # Set status to waiting for user input
        analysis_state.status = TaskStatus.WAITING_INPUT

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

        Frontend manages a 2-minute visual countdown timer. When timer expires,
        user clicks skip, or user sends input, frontend calls the resume API
        which signals this event to unblock.
        Backend uses a 10-minute safety timeout to prevent infinite hangs.
        """
        analysis_state = state["analysis_state"]

        # Clear the event for this wait cycle
        self._continue_event.clear()

        logger.info(f"[{self.session_id}] Waiting for user input (round {analysis_state.current_round})")

        # Wait for signal from frontend (via resume API) or safety timeout
        try:
            await asyncio.wait_for(self._continue_event.wait(), timeout=600)
            logger.info(f"[{self.session_id}] Continue signal received")
        except asyncio.TimeoutError:
            logger.info(f"[{self.session_id}] Wait timeout (10min safety), auto-proceeding")

        # If user provided real input (text or files) on the last round, extend max_rounds to allow one more
        has_real_text = (
            analysis_state.pending_human_input
            and analysis_state.pending_human_input.strip() not in ('', '[CONTINUE]')
        )
        has_uploaded_files = bool(analysis_state.pending_uploaded_files)
        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        if (has_real_text or has_uploaded_files) and analysis_state.current_round >= max_rounds:
            analysis_state.max_rounds = analysis_state.current_round + 1
            logger.info(f"[{self.session_id}] User input on last round (text={has_real_text}, files={has_uploaded_files}), extending max_rounds to {analysis_state.max_rounds}")

        return {**state, "analysis_state": analysis_state}

    def _should_continue(self, state: WorkflowState) -> str:
        """Determine if analysis should continue"""
        round_result = state.get("round_result")
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
        """Check if analysis has converged"""
        analysis_state = state["analysis_state"]

        # Only check convergence if should_continue is currently True
        # This preserves max_rounds and human stop decisions from _should_continue
        current_should_continue = state.get("should_continue", True)
        if not current_should_continue:
            logger.info(f"[{self.session_id}] Convergence check: skipped (already stopping)")
            return state

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

            # Send integrated report with figure info to frontend
            figures_data = [
                {
                    "id": fig.id,
                    "name": fig.name,
                    "description": fig.description,
                    "figure_type": fig.figure_type,
                    "png_base64": fig.png_base64,
                    "pdf_path": fig.pdf_path,
                    "review_status": fig.review_status.value,
                    "revision_count": fig.revision_count
                }
                for fig in analysis_state.publication_figures
            ]

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
                    "total_figures": len(analysis_state.publication_figures),
                    "approved_figures": sum(1 for f in analysis_state.publication_figures if f.review_status == FigureReviewStatus.APPROVED)
                },
                AgentType.PI
            )

            logger.info(f"[{self.session_id}] Integrated report generated with {len(analysis_state.publication_figures)} figures")

        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to generate integrated report: {e}")
            await self._send_message("error", f"Integrated report generation failed: {str(e)}", AgentType.PI)
            # Continue anyway - final summary is still available

        logger.info(f"[{self.session_id}] Final report generated")

        return {
            **state,
            "analysis_state": analysis_state
        }

    async def _generate_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """Generate graphic abstract using Painter Agent"""
        analysis_state = state["analysis_state"]

        logger.info(f"[{self.session_id}] Generating graphic abstract")
        await self._send_message("status", "Generating graphic abstract...", AgentType.PAINTER)

        try:
            # Step 1: PI generates summary prompt for graphic abstract
            summary_prompt = await self.pi_agent.generate_graphic_abstract_prompt(analysis_state)
            logger.info(f"[{self.session_id}] Generated graphic abstract prompt")

            # Step 2: Painter Agent generates the graphic abstract
            graphic = await self.painter_agent.generate_graphic_abstract(
                analysis_state,
                summary_prompt
            )

            analysis_state.graphic_abstract = graphic

            await self._send_message(
                "graphic_abstract",
                {
                    "image_base64": graphic.image_base64,
                    "prompt": graphic.prompt,
                    "created_at": graphic.created_at.isoformat()
                },
                AgentType.PAINTER
            )

            logger.info(f"[{self.session_id}] Graphic abstract generated successfully")

        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to generate graphic abstract: {e}")
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
            output_dir = str(REPORTS_DIR / self.session_id)
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
                base_url=self.config.api.llm_base_url
            )

            # Get models from state or use defaults
            pi_model = analysis_state.custom_models.get("pi", self.config.models.pi_model)
            bio_model = analysis_state.custom_models.get("bioinformatics", self.config.models.bioinformatics_model)

            workspace_dir = str(WORKSPACE_DIR / self.session_id)

            generator = DetailedReportGenerator(
                llm_client=llm_client,
                pi_model=pi_model,
                bio_model=bio_model,
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                progress_callback=progress_callback
            )

            # Generate report
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
            logger.error(f"[{self.session_id}] Failed to generate detailed report: {e}")
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
            logger.error(f"Failed to initialize workflow: {e}")
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
            logger.error(f"[{self.session_id}] Failed to initialize for report generation: {e}")
            await self._send_message("error", f"Initialization failed: {str(e)}")
            initial_state.status = TaskStatus.FAILED
            return initial_state

        # Create workflow state with existing analysis data
        workflow_state: WorkflowState = {
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
                logger.error(f"[{self.session_id}] Error stopping sandbox: {e}")
        if self.llm_manager:
            try:
                await self.llm_manager.close()
                logger.info(f"[{self.session_id}] LLM manager closed")
            except Exception as e:
                logger.error(f"[{self.session_id}] Error closing LLM manager: {e}")
    
    def get_notebook_path(self) -> Optional[str]:
        """Get path to the saved Jupyter notebook"""
        if self.sandbox:
            return self.sandbox.get_notebook_path()
        return None
