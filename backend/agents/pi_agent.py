"""
PI Agent - Principal Investigator that coordinates all agents
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
import re
import logging

from .base import BaseAgent
from ..utils.settings_provider import AgentSettingsProvider
from ..utils.direction_evidence import build_fallback_direction_evidence
from ..utils.formatting import as_list, safe_parse_json_array
from ..utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from .prompt_blocks import build_integration_block

# Configure logger
logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisPhase, AnalysisState, MessageRole, RoundResult, ExplorationReport,
    PublicationFigure, FigureReviewStatus, GraphicAbstract,
    DATA_TYPE_SCRNASEQ,
    CrossModalSynthesis, CrossModalMechanismCandidate, CrossModalEvidenceItem,
)
from ..utils.llm_client import OpenAICompatibleClient, UpstreamExhaustedError
from ..constants import (
    CONTINUE_SIGNAL,
    PI_DECIDE_SIGNAL,
    PI_REVIEW_FALLBACK_DECISION,
    NO_OUTPUT_TOKEN_CAP,
    STOP_KEYWORDS,
)

# Re-exported so `backend.agents.pi_agent.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .pi.shared import (  # noqa: F401
    _CONTROL_SIGNAL_RESULTS,
    _BRIEF_PREFERRED_TYPES,
    _BRIEF_QC_MARKERS,
    _DIRECTION_CANDIDATE_LIST_FIELDS,
    _normalize_supporting_data,
    _normalize_direction_candidate,
    _normalize_integration_scored_candidates,
    _bounded_brief_fallback,
    _format_database_priors_block,
    _build_default_bio_tasks,
)
from .pi import (  # noqa: F401
    InstructionsMixin,
    SpeciesMixin,
    AnnotationMixin,
    ResearchPlanningMixin,
    DirectionCandidatesMixin,
    DirectionSelectionMixin,
    SynthesisMixin,
    BioinfoLiaisonMixin,
    FiguresMixin,
    ReportsMixin,
    InteractionMixin,
    ExecutionReviewMixin,
)
class PIAgent(
    InstructionsMixin,
    SpeciesMixin,
    AnnotationMixin,
    ResearchPlanningMixin,
    DirectionCandidatesMixin,
    DirectionSelectionMixin,
    SynthesisMixin,
    BioinfoLiaisonMixin,
    FiguresMixin,
    ReportsMixin,
    InteractionMixin,
    ExecutionReviewMixin,
    BaseAgent,
):
    """
    Principal Investigator Agent

    Responsibilities:
    - Coordinate overall research strategy
    - Analyze and synthesize results from all agents
    - Resolve conflicts between agents
    - Make decisions on next steps
    - Determine when to stop analysis
    """

    _LITERATURE_TOPIC_FORMAT_RULE = (
        "Format rule for `literature_topics`: output a JSON array of PLAIN "
        "natural-language English strings — NOT Python dicts and NOT dict-like "
        "strings. Each topic is one self-contained descriptive sentence, e.g. "
        "\"Association between the observed pathway change and the measured phenotype\". "
        "Do NOT wrap topics as {'Topic': ..., 'Rationale': ...}; "
        "if you want to convey rationale, fold it into the sentence itself."
    )

    def __init__(
        self,
        settings_provider: 'AgentSettingsProvider',
    ):
        super().__init__(
            agent_type=AgentType.PI,
            settings_provider=settings_provider,
        )
        
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process instruction and coordinate analysis"""
        
        self.refresh_runtime_config_if_needed()
        # Get context from state
        context = self._build_context(state)
        
        # Build prompt
        prompt = f"""
{context}

Current instruction: {instruction}

Based on the above information:
1. Analyze results from all aspects
2. Identify key findings and potential issues
3. Arbitrate if there are conflicts
4. Plan next steps for analysis

Reply in JSON format with the following fields:
- summary: Summary of the current round
- findings: List of key findings
- conflicts: Conflicts found and solutions
- next_steps: Tasks for each Agent to execute next
- should_continue: Whether to continue analysis (true/false)
- reason: Reason for continuing/stopping
"""
        
        response = await self.chat(prompt, temperature=0.5)
        
        # Parse response
        result = self._parse_response(response)
        
        return result
    
    












    # Shared "Rules:" footer for PI-authored Deep Research briefs. The
    # citation-format directive is intentionally excluded — the Literature Agent
    # appends its own (parser-coupled) footer.
    _BRIEF_RULES = (
        "Rules:\n"
        "- Write the brief as direct instructions to a deep-research agent (what to search and synthesize).\n"
        "- Do NOT include any citation or reference-format instructions.\n"
        "- Do NOT quote or restate the full experimental background."
    )






























    
    
    




    def _build_context(self, state: AnalysisState) -> str:
        """Build context string from state"""
        context_parts = []
        
        # Add project info
        context_parts.append(f"## Project Information")
        context_parts.append(f"Project Name: {state.name}")
        context_parts.append(f"Research Objective: {state.initial_instruction}")
        
        # Add current round info
        context_parts.append(f"\nCurrent Round: {state.current_round}")
        
        # Add previous round summaries
        if state.round_results:
            context_parts.append("\n## Historical Analysis Results")
            for result in state.round_results[-3:]:  # Last 3 rounds
                context_parts.append(f"\n### Round {result.round_number}\n{result.pi_summary}")
        
        return "\n".join(context_parts)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse PI response to extract structured data"""
        # Use base class safe JSON parser
        result = self._safe_parse_json(response, default={})

        if result and "summary" in result:
            return result

        return {
            "summary": response if response else "",
            "findings": [],
            "conflicts": [],
            "next_steps": [],
            "should_continue": True,
            "reason": ""
        }
    
    def _extract_next_steps(self, data: Dict[str, Any]) -> List[str]:
        """Extract next steps from parsed data"""
        steps = []
        
        if "next_biochemistry" in data:
            for task in data["next_biochemistry"]:
                steps.append(f"[Biochemistry] {task}")

        if "next_pathology" in data:
            for task in data["next_pathology"]:
                steps.append(f"[Pathology] {task}")

        if "next_bioinformatics" in data:
            for task in data["next_bioinformatics"]:
                steps.append(f"[Bioinformatics] {task}")

        if "next_literature" in data:
            for topic in data["next_literature"]:
                steps.append(f"[Literature] {topic}")
        
        return steps
    
    # ============================================
    # Iterative Bioinformatics Analysis Control
    # ============================================
    
    
    
    

    # ============================================
    # Publication Figure Review
    # ============================================



    # ============================================
    # Graphic Abstract Review & Selection
    # ============================================




    # ============================================
    # Text-Mode Graphic Abstract (ASCII diagram)
    # ============================================




    # ============================================
    # Integrated Report Generation
    # ============================================



    # ============================================
    # Graphic Abstract Prompt Generation
    # ============================================

