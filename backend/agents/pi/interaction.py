"""Human guidance and free-form user input handling.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List
import logging
import re


# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState
from ...constants import STOP_KEYWORDS
from ...utils.formatting import as_list, coerce_json_text

from .shared import (  # noqa: F401
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


_ADJUSTED_INSTRUCTION_FIELDS = (
    "pathology_tasks",
    "bioinformatics_tasks",
    "biochemistry_tasks",
    "literature_topics",
)


def _contains_stop_command(value: str) -> bool:
    """Match explicit stop commands with boundaries and clause-local negation."""
    text = str(value or "").casefold()
    for keyword in STOP_KEYWORDS:
        token = str(keyword or "").casefold().strip()
        if not token:
            continue
        is_cjk = any("\u4e00" <= char <= "\u9fff" for char in token)
        matches = (
            re.finditer(re.escape(token), text)
            if is_cjk
            else re.finditer(rf"(?<![\w]){re.escape(token)}(?![\w])", text)
        )
        for match in matches:
            prefix = text[:match.start()]
            clause_prefix = re.split(
                r"[.!?。！？,，;；]|\b(?:but|however)\b|(?:但是|不过|然而|但)",
                prefix,
            )[-1]
            if is_cjk:
                negated = bool(re.search(
                    r"(?:(?:请\s*)?勿|不要|别|无需|暂不|先不|不必|不用|不可|不)",
                    clause_prefix,
                ))
            else:
                negated = bool(re.search(
                    r"(?:\bdo\s+not|\bdon['’]?t|\bdont|\bnever|\bnot|\bno)\b",
                    clause_prefix,
                ))
            if not negated:
                return True
    return False


def _normalize_adjusted_instructions(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = dict(value)
    for field in _ADJUSTED_INSTRUCTION_FIELDS:
        normalized[field] = as_list(normalized.get(field))
    if "overview" in normalized:
        normalized["overview"] = coerce_json_text(normalized.get("overview"))
    return normalized


def _normalize_file_assignments(value: Any) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for key, items in value.items():
        assignments = [
            dict(item) for item in as_list(items) if isinstance(item, dict)
        ]
        if assignments:
            normalized[str(key)] = assignments
    return normalized


class InteractionMixin:
    """Human guidance and free-form user input handling."""

    async def process_human_guidance(
        self,
        state: AnalysisState,
        human_input: str
    ) -> Dict[str, Any]:
        """
        Process human researcher's guidance and adjust analysis direction

        Args:
            state: Current analysis state
            human_input: Human researcher's guidance/instruction
            
        Returns:
            Dict with:
            - response: PI's response to human
            - should_stop: Whether to stop analysis
            - adjusted_instructions: New instructions for agents (if applicable)
        """
        # Build context from current state
        context = self._build_context(state)
        
        prompt = f"""
## Current Analysis Status
Project: {state.name}
Current Round: {state.current_round}
Research Objective: {state.initial_instruction}

{context}

## Human Researcher Guidance
The human researcher has provided the following guidance:
"{human_input}"

As the PI Agent, please:
1. Carefully interpret what the human researcher wants
2. Decide if they want to stop/conclude the analysis, or adjust the direction
3. If adjusting direction, provide new specific instructions for each agent

Common stop/conclude keywords: stop, terminate, end, finish, conclude, summarize

Respond in JSON format:
{{
    "interpretation": "Your interpretation of what the human wants",
    "response": "Your response to the human researcher (be helpful and professional)",
    "should_stop": true/false,
    "reason": "Why you decided to stop or continue",
    "adjusted_instructions": {{
        "pathology_tasks": ["task1", "task2"],
        "bioinformatics_tasks": ["task1", "task2"],
        "biochemistry_tasks": ["task1", "task2"],
        "literature_topics": ["topic1", "topic2"]
    }},
    "focus_areas": ["key area 1", "key area 2"],
    "deprioritize": ["area to skip or reduce focus"]
}}

If should_stop is true, adjusted_instructions can be empty.
If adjusting direction, provide specific and actionable tasks.
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        if data and "interpretation" in data:
            return {
                "interpretation": coerce_json_text(data.get("interpretation")),
                "response": coerce_json_text(data.get("response"))
                or "Acknowledged. Adjusting analysis direction.",
                "should_stop": data.get("should_stop") is True,
                "reason": coerce_json_text(data.get("reason")),
                "adjusted_instructions": _normalize_adjusted_instructions(
                    data.get("adjusted_instructions")
                ),
                "focus_areas": as_list(data.get("focus_areas")),
                "deprioritize": as_list(data.get("deprioritize")),
            }
        else:
            logger.warning(f"[PI] No valid JSON found in response, using fallback parsing. Response: {response[:200] if response else 'None'}...")
        
        # Fallback: check for stop keywords manually (English + Chinese)
        if _contains_stop_command(human_input):
            return {
                "interpretation": "Human requested to stop the analysis",
                "response": "Understood. I will conclude the current analysis and generate a final report.",
                "should_stop": True,
                "reason": "Human explicitly requested to stop/conclude",
                "adjusted_instructions": None
            }
        
        # Default: treat as guidance to continue with adjustments
        return {
            "interpretation": human_input,
            "response": f"Acknowledged. I will incorporate your guidance: {human_input}",
            "should_stop": False,
            "reason": "Continuing with adjusted focus based on human guidance",
            "adjusted_instructions": {
                "pathology_tasks": [f"Focus on: {human_input}"],
                "bioinformatics_tasks": [f"Prioritize analysis related to: {human_input}"],
                "biochemistry_tasks": [f"Review biochemistry aspects of: {human_input}"],
                "literature_topics": [f"Search literature on: {human_input}"]
            }
        }

    async def process_user_input(
        self,
        state: AnalysisState,
        user_message: str,
        uploaded_files: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input after a round completes.
        Classify uploaded files and determine next actions.
        
        Args:
            state: Current analysis state
            user_message: User's text message
            uploaded_files: List of uploaded file info (type, filename, id)
            
        Returns:
            Dict with:
            - action: 'continue', 'stop', 'analyze_new_data'
            - file_assignments: Which files go to which agents
            - instructions: Updated instructions for agents
            - response: Response to user
        """
        uploaded_files = uploaded_files or []
        logger.info(f"[PI] Processing user input: {user_message[:100] if user_message else 'no message'}... ({len(uploaded_files)} files)")

        # Quick path: synthetic control sentinels must deterministically continue
        # and NOT be sent to the stop/continue LLM classifier — that path can
        # non-deterministically return action="stop" and end the analysis after
        # Round 1. [PI_DECIDE] (set by no-wait mode in synthesis.py) means "let
        # the PI autonomously pick research directions"; _pi_planning's
        # select_best_directions() later consumes it via user_chosen_direction.
        if user_message and user_message.strip() in _CONTROL_SIGNAL_RESULTS:
            response, reason = _CONTROL_SIGNAL_RESULTS[user_message.strip()]
            logger.info(f"[PI] Control sentinel {user_message.strip()} detected, proceeding to next round")
            return {
                "action": "continue",
                "response": response,
                "file_assignments": {},
                "biochemistry_values": {},
                "adjusted_instructions": None,
                "focus_areas": [],
                "reason": reason,
            }

        # Build file description
        file_descriptions = []
        for f in uploaded_files:
            file_descriptions.append(f"{f.get('type', 'unknown')}: {f.get('filename', 'unknown')}")
        
        prompt = f"""
## Project Context
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## User Input Processing

The user has provided input after Round {state.current_round} analysis.

User message: "{user_message}"

Uploaded files:
{chr(10).join(file_descriptions) if file_descriptions else "None"}

## Task
1. Determine what the user wants:
   - Continue to next round?
   - Stop and generate final report?
   - Analyze new data first?
   - Adjust analysis direction?

2. If files were uploaded, classify each file:
   - Images of H&E, Sirius Red, Oil Red O, etc. → pathology
   - Images of Western blots, gels → biochemistry
   - Text describing biochemistry results → biochemistry
   - Other data files → determine appropriate agent

3. Extract any specific instructions from user message

Respond in JSON format:
{{
    "action": "continue" | "stop" | "analyze_new_data",
    "response": "Your response to the user",
    "file_assignments": {{
        "pathology": [
            {{"filename": "example_image.png", "image_type": "user-specified stain", "reason": "Assess the feature defined by the study"}}
        ],
        "biochemistry": [
            {{"filename": "xxx.txt", "data_type": "metabolic markers", "reason": "Enzyme activity values"}}
        ]
    }},
    "biochemistry_values": {{
        "Marker1": {{"value": 120, "unit": "U/L"}},
        "Marker2": {{"value": 95, "unit": "U/L"}}
    }},
    "adjusted_instructions": {{
        "pathology_tasks": ["specific task"],
        "bioinformatics_tasks": ["specific task"],
        "biochemistry_tasks": ["specific task"],
        "literature_topics": ["specific topic"]
    }},
    "focus_areas": ["area to emphasize"],
    "reason": "Why this action was chosen"
}}

Note: Extract biochemistry values from user message if they mention specific numbers (e.g., "ALT is 120 U/L").
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        parsed = self._safe_parse_json(response, default={})
        data = dict(parsed) if isinstance(parsed, dict) else {}
        action = (
            coerce_json_text(data.get("action")).casefold()
            if "action" in data
            else ""
        )

        if action in {"continue", "stop", "analyze_new_data"}:
            return {
                "action": action,
                "response": coerce_json_text(data.get("response")) or "Acknowledged",
                "file_assignments": _normalize_file_assignments(
                    data.get("file_assignments")
                ),
                "biochemistry_values": (
                    dict(data["biochemistry_values"])
                    if isinstance(data.get("biochemistry_values"), dict)
                    else {}
                ),
                "adjusted_instructions": _normalize_adjusted_instructions(
                    data.get("adjusted_instructions")
                ),
                "focus_areas": as_list(data.get("focus_areas")),
                "reason": coerce_json_text(data.get("reason")),
            }
        
        logger.warning(
            "[PI] Structured user-input action was missing or unsupported; "
            "falling back to the original user message"
        )

        if _contains_stop_command(user_message):
            fallback_action = "stop"
        elif uploaded_files:
            fallback_action = "analyze_new_data"
        else:
            fallback_action = "continue"
        fallback_assignments = _normalize_file_assignments(
            data.get("file_assignments")
        )
        if not fallback_assignments:
            fallback_assignments = {
                "pathology": [
                    {"filename": f["filename"], "image_type": "user_upload"}
                    for f in uploaded_files
                    if f.get("type") == "image" and f.get("filename")
                ]
            }
            if not fallback_assignments["pathology"]:
                fallback_assignments = {}
        return {
            "action": fallback_action,
            "response": coerce_json_text(data.get("response")) or (
                "Understood. Generating final report."
                if fallback_action == "stop"
                else "Acknowledged. Proceeding with analysis."
            ),
            "file_assignments": fallback_assignments,
            "biochemistry_values": (
                dict(data["biochemistry_values"])
                if isinstance(data.get("biochemistry_values"), dict)
                else {}
            ),
            "adjusted_instructions": _normalize_adjusted_instructions(
                data.get("adjusted_instructions")
            ),
            "focus_areas": as_list(data.get("focus_areas")),
            "reason": coerce_json_text(data.get("reason")) or (
                "User requested to stop"
                if fallback_action == "stop"
                else "Continuing analysis"
            ),
        }
