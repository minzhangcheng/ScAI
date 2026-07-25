# Core-agent guard plus the pathology and biochemistry analysis nodes.
# Moved verbatim out of backend/workflow/agents.py; composed into AgentNodesMixin.
"""Agent node methods: pathology, bioinformatics, biochemistry, literature."""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...models.schemas import AgentType, AnalysisPhase, AnalysisStep, MessageRole, TaskStatus
from ...utils.llm_client import UpstreamExhaustedError
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...agents.bioinformatics_agent import count_saved_figures
from ..helpers import _get_task_text, _has_stage1_keyword, _normalize_tasks, cache_hit, cache_invalidate, cache_store, decide_bioinfo_post_execution, prepend_species, safe_pi_brief, STAGE1_KEYWORDS
from ...utils.blob_store import BLOB_REF_PREFIX
from ..retry import with_retry

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict

from .shared import (  # noqa: F401
    asyncio,
    hashlib,
    json,
    logging,
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    AgentType,
    AnalysisPhase,
    AnalysisStep,
    MessageRole,
    TaskStatus,
    UpstreamExhaustedError,
    MECHANISM_EVIDENCE_POLICY,
    count_saved_figures,
    _get_task_text,
    _has_stage1_keyword,
    _normalize_tasks,
    cache_hit,
    cache_invalidate,
    cache_store,
    decide_bioinfo_post_execution,
    prepend_species,
    safe_pi_brief,
    STAGE1_KEYWORDS,
    BLOB_REF_PREFIX,
    with_retry,
    _reset_pi_instructions_for_stage2,
    _stable_code_hash,
    _STAGE1_INSTRUCTION,
    _MECHANISM_CAPTURE_INSTRUCTION,
    _build_exploration_instruction,
    _build_innovation_instruction,
    _build_stage2_instruction,
    build_phase_instruction,
)


logger = logging.getLogger(__name__)


class ModalityNodesMixin:
    """Core-agent guard plus the pathology and biochemistry analysis nodes."""

    async def _require_core_agent(
        self,
        state: 'WorkflowState',
        agent_name: str,
        node_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Block until the named core agent is configured.

        Returns None to indicate the caller may proceed. If the user chose to
        skip, returns a partial state dict the caller should `return` immediately
        (the corresponding result_key set to {'status': 'skipped'}). If the user
        chose to terminate, raises WorkflowTerminatedError.
        """
        from ...constants import CORE_AGENTS, LlmFailureAction, NODE_TO_RESULT_KEY
        from ..core import WorkflowTerminatedError

        if agent_name not in CORE_AGENTS:
            return None

        provider = getattr(self, "_settings_provider", None)
        if provider is None:
            return None

        while True:
            model = provider.get_model(agent_name)
            if model:
                return None
            action = await self._wait_for_disabled_agent_intervention(
                agent_name, node_name, state["analysis_state"]
            )
            if action == LlmFailureAction.RETRY:
                continue
            if action == LlmFailureAction.TERMINATE:
                raise WorkflowTerminatedError(
                    f"User terminated workflow due to disabled {agent_name}"
                )
            # SKIP
            result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
            skip_nodes = list(state.get("skip_nodes", []))
            if node_name not in skip_nodes:
                skip_nodes.append(node_name)
            return {
                **state,
                "skip_nodes": skip_nodes,
                "failed_node": None,
                result_key: {"findings": [], "status": "skipped"},
            }

    @with_retry("pathology_analysis")
    async def _pathology_analysis(self, state: WorkflowState) -> WorkflowState:
        """Pathology agent analyzes images"""
        analysis_state = state["analysis_state"]
        # Required by _pathology_llm_intervention_callback to drive the
        # image-level LLM-failure HITL dialog.
        self._current_analysis_state = analysis_state

        cached = cache_hit(analysis_state, AgentType.PATHOLOGY)
        if cached is not None:
            logger.info(f"[{self.session_id}] pathology cache hit, skipping re-execution")
            await self._send_message("result", cached, AgentType.PATHOLOGY)
            return {
                **state,
                "analysis_state": analysis_state,
                "pathology_result": cached,
                "failed_node": None,
            }

        if not self.ensure_agent_ready("pathology"):
            logger.info(f"[{self.session_id}] Pathology agent disabled, skipping")
            skipped_result = {"findings": [], "status": "skipped"}
            cache_store(analysis_state, AgentType.PATHOLOGY, skipped_result)
            return {**state, "failed_node": None, "pathology_result": skipped_result}

        instructions = state["pi_instructions"]

        tasks = instructions.get("pathology_tasks", [])
        if not tasks:
            tasks = ["Perform standard pathology analysis"]

        pathology_files = analysis_state.get_files_by_section("pathology")
        logger.info(f"[{self.session_id}] Starting pathology analysis with {len(pathology_files)} images")
        await self._send_message("status", "Pathology analysis starting...", AgentType.PATHOLOGY)
        
        # Check if images are available
        if not pathology_files:
            no_data_result = {"findings": [], "status": "no_data"}
            await self._send_message(
                "result",
                {
                    "status": "no_data",
                    "message": "No pathology images uploaded. Please upload stained tissue sections (e.g., H&E, IHC, or special stains) appropriate to your study.",
                    "findings": []
                },
                AgentType.PATHOLOGY
            )
            cache_store(analysis_state, AgentType.PATHOLOGY, no_data_result)
            return {
                **state,
                "analysis_state": analysis_state,
                "pathology_result": no_data_result
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

        task_instruction = prepend_species(task_instruction, analysis_state.species)

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

        cache_store(analysis_state, AgentType.PATHOLOGY, result)
        return {
            **state,
            "analysis_state": analysis_state,
            "pathology_result": result
        }

    @with_retry("biochemistry_analysis")
    async def _biochemistry_analysis(self, state: WorkflowState) -> WorkflowState:
        """Biochemistry agent analyzes experimental results"""
        analysis_state = state["analysis_state"]

        cached = cache_hit(analysis_state, AgentType.BIOCHEMISTRY)
        if cached is not None:
            logger.info(f"[{self.session_id}] biochemistry cache hit, skipping re-execution")
            await self._send_message("result", cached, AgentType.BIOCHEMISTRY)
            return {
                **state,
                "analysis_state": analysis_state,
                "biochemistry_result": cached,
                "failed_node": None,
            }

        if not self.ensure_agent_ready("biochemistry"):
            logger.info(f"[{self.session_id}] Biochemistry agent disabled, skipping")
            skipped_result = {"findings": [], "status": "skipped"}
            cache_store(analysis_state, AgentType.BIOCHEMISTRY, skipped_result)
            return {**state, "failed_node": None, "biochemistry_result": skipped_result}

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

        task_instruction = prepend_species(task_instruction, analysis_state.species)

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

        cache_store(analysis_state, AgentType.BIOCHEMISTRY, result)
        return {
            **state,
            "analysis_state": analysis_state,
            "biochemistry_result": result
        }
