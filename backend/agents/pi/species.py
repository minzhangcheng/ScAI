"""Species inference from gene symbols and the mismatch decision gate.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any
import logging


# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState
from ...utils.llm_client import UpstreamExhaustedError

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


class SpeciesMixin:
    """Species inference from gene symbols and the mismatch decision gate."""

    async def _infer_species(self, prompt: str, log_context: str) -> Dict[str, Any]:
        """Shared helper for species inference calls.

        Args:
            prompt: The LLM prompt to send
            log_context: Context string for logging (e.g., "project: X" or "user hint: Y")

        Returns:
            Dict with species, confidence, suggestions, reasoning
        """
        logger.info(f"[PI] Inferring species: {log_context}")
        try:
            response = await self.chat(prompt, temperature=0.1)
            result = self._safe_parse_json(response, default={})
            if not result.get("species"):
                return {"species": "", "confidence": "low",
                        "suggestions": result.get("suggestions", []),
                        "reasoning": result.get("reasoning", "Could not determine species from project info")}
            return {
                "species": result.get("species", ""),
                "confidence": result.get("confidence", "low"),
                "suggestions": result.get("suggestions", [result["species"]]),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning(f"[PI] Species inference failed: {e}")
            return {"species": "", "confidence": "low", "suggestions": [],
                    "reasoning": f"Auto-inference failed: {e}"}

    async def infer_species(self, state: AnalysisState) -> Dict[str, Any]:
        """Infer species from project info.

        Returns {"species": str, "confidence": "high"|"low",
                 "suggestions": [...], "reasoning": str}
        """
        prompt = f"""Determine the species/organism for this research project.

Project Name: {state.name}
Research Objective: {state.initial_instruction}
Available Data:
{state.build_experimental_design_summary()}

Based on ALL available information (project name, research description, sample/group names, data file names), determine the species.

Common species in single-cell research:
- Homo sapiens (human) — gene names are ALL UPPERCASE (TP53, BRCA1, CD4)
- Mus musculus (mouse) — gene names Capitalize first letter (Tp53, Brca1, Cd4)
- Rattus norvegicus (rat) — similar to mouse naming

Return JSON:
{{
    "species": "Scientific name (e.g., Homo sapiens, Mus musculus)",
    "confidence": "high" or "low",
    "suggestions": ["Most likely species", "Second option if uncertain"],
    "reasoning": "Brief explanation of how you determined the species"
}}

Set confidence to "high" ONLY if you are very certain from the project information.
If there is any ambiguity, set confidence to "low".
"""
        return await self._infer_species(prompt, f"project: {state.name}")

    async def infer_species_from_hint(self, state: AnalysisState, user_hint: str) -> Dict[str, Any]:
        """Re-infer species incorporating user hint. Same return format as infer_species."""
        prompt = f"""Determine the species/organism for this research project based on the user's input.

Project Name: {state.name}
Research Objective: {state.initial_instruction}
User's input about the species: {user_hint}

Based on the user's input and project information, determine the species.
Use the scientific name (e.g., "Homo sapiens", "Mus musculus", "Rattus norvegicus").

Return JSON:
{{
    "species": "Scientific name",
    "confidence": "high" or "low",
    "suggestions": ["Most likely species"],
    "reasoning": "Brief explanation"
}}
"""
        return await self._infer_species(prompt, f"user hint: {user_hint[:100]}")

    async def decide_species_action(
        self, user_input: str, current_species: str, assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide what action to take based on user's free-text input about gene-species mismatch.

        Returns {"action": "change_species"|"keep"|"custom", "new_species": str}
        """
        prompt = f"""A gene-species mismatch was detected. The current species is "{current_species}".
Assessment: {assessment.get('reasoning', '')}
Likely actual species: {assessment.get('likely_species', '')}

The user responded: "{user_input}"

Classify the user's intent into one of these actions:
1. "change_species" — user wants to change the species. Extract the new species name.
2. "keep" — user wants to keep the current species and proceed.
3. "custom" — user gave a specific instruction that doesn't fit the above.

If the user's input looks like a species name (e.g., "human", "mouse", "Homo sapiens"), 
treat it as "change_species" and convert to scientific name.

Return JSON: {{"action": "change_species"|"keep"|"custom", "new_species": "scientific name if changing"}}
"""
        try:
            response = await self.chat(prompt, temperature=0.1)
            result = self._safe_parse_json(response, default={"action": "keep"})
            return {
                "action": result.get("action", "keep"),
                "new_species": result.get("new_species", ""),
            }
        except UpstreamExhaustedError:
            raise
        except Exception:
            return {"action": "keep", "new_species": ""}
