"""Instruction, evaluation, and summarization of bioinformatics work.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List
import json
import logging


# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState

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


class BioinfoLiaisonMixin:
    """Instruction, evaluation, and summarization of bioinformatics work."""

    async def generate_bioinformatics_instruction(
        self,
        state: AnalysisState,
        current_step: str,
        previous_results: List[Dict[str, Any]],
        step_index: int,
        total_steps: int
    ) -> str:
        """
        Generate specific instruction for bioinformatics agent
        
        Args:
            state: Current analysis state
            current_step: The current step description
            previous_results: Results from previous steps
            step_index: Current step number (0-indexed)
            total_steps: Total number of planned steps
            
        Returns:
            Specific instruction string for the bioinformatics agent
        """
        # Build context from previous results
        prev_context = ""
        if previous_results:
            prev_context = "\n## Previous Analysis Results\n"
            for i, result in enumerate(previous_results[-5:]):  # Last 5 results
                findings = result.get("findings", [])
                desc = result.get("description", "")
                if findings or desc:
                    prev_context += f"\nStep {i+1} ({desc}):\n"
                    for finding in findings:
                        prev_context += f"- {finding}\n"
        
        # Build data path info for first step
        data_paths_info = ""
        if step_index == 0 or "load" in current_step.lower() or "read" in current_step.lower():
            seq_files = [f for f in state.data_files if f.section == "sequencing"]
            data_paths = []
            for f in seq_files:
                group = state.get_group_by_id(f.group_id)
                group_label = group.name if group else "Ungrouped"
                sample_label = f"Sample {f.sample_number}" if f.sample_number else ""
                label = f"{group_label} {sample_label}".strip()
                data_paths.append(f"{label} ({f.data_type}): {f.file_path}")
            if data_paths:
                design = state.build_experimental_design_summary()
                data_paths_info = f"\n## Experimental Design\n{design}\n\n## Data File Paths\n" + "\n".join(data_paths) + "\n"
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Current Progress
Step {step_index + 1}/{total_steps}: {current_step}
{data_paths_info}
{prev_context}

Please generate specific analysis instructions for the Bioinformatics Agent to execute "{current_step}".

Instruction requirements:
1. Be specific and clear about what the Agent should do
2. If loading data, must use the exact file paths provided above
3. Include necessary parameter suggestions (thresholds, method choices, etc.)
4. Specify expected outputs (charts, numbers, etc.)
5. If performing differential analysis, specify comparison groups (treatment vs control)

Return the instruction content directly, no additional formatting needed.
"""
        
        instruction = await self.chat(prompt, temperature=0.3)
        return instruction.strip()

    async def evaluate_bioinformatics_result(
        self,
        state: AnalysisState,
        step_description: str,
        result: Dict[str, Any],
        remaining_steps: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate bioinformatics result and decide next action
        
        Args:
            state: Current analysis state
            step_description: What step was just completed
            result: Result from the bioinformatics agent
            remaining_steps: List of remaining planned steps
            
        Returns:
            Dict with:
            - continue_analysis: bool - whether to continue
            - next_step: str - next step to execute (if continue)
            - modify_plan: bool - whether to modify the remaining steps
            - new_steps: List[str] - new steps to insert (if modifying)
            - findings: str - PI's interpretation of the result
            - issues: List[str] - any issues or concerns
        """
        findings = result.get("findings", [])
        error = result.get("error", "")
        success = result.get("success", False)
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Completed Analysis Step
{step_description}

## Execution Result
Success: {"Yes" if success else "No"}
Error: {error or "None"}
Findings: {json.dumps(findings, ensure_ascii=False) if findings else "No specific findings"}

## Remaining Planned Steps
{json.dumps(remaining_steps, ensure_ascii=False) if remaining_steps else "No more planned steps"}

Please evaluate this step's result and decide the next action.

Return in JSON format:
{{
    "continue_analysis": true/false,
    "findings": "PI's interpretation of this step's results",
    "issues": ["issue 1", "issue 2"],
    "need_retry": true/false,
    "retry_instruction": "If retry needed, provide modified instruction",
    "insert_steps": ["If additional analysis steps needed, list here"],
    "skip_steps": ["If some planned steps are no longer needed, list here"]
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        if data and "continue_analysis" in data:
            return {
                "continue_analysis": data.get("continue_analysis", True),
                "findings": data.get("findings", ""),
                "issues": data.get("issues", []),
                "need_retry": data.get("need_retry", False),
                "retry_instruction": data.get("retry_instruction", ""),
                "insert_steps": data.get("insert_steps", []),
                "skip_steps": data.get("skip_steps", [])
            }

        # Default: continue if successful
        return {
            "continue_analysis": success and len(remaining_steps) > 0,
            "findings": response[:500] if response else "",
            "issues": [] if success else [error],
            "need_retry": not success,
            "retry_instruction": "",
            "insert_steps": [],
            "skip_steps": []
        }

    async def summarize_bioinformatics_analysis(
        self,
        state: AnalysisState,
        all_results: List[Dict[str, Any]]
    ) -> str:
        """
        Summarize the complete bioinformatics analysis

        Args:
            state: Current analysis state
            all_results: All results from the bioinformatics analysis

        Returns:
            Summary string
        """
        # Collect all findings
        all_findings = []
        all_figures = 0

        for result in all_results:
            findings = result.get("findings", [])
            figures = result.get("figures", [])
            desc = result.get("description", "")

            if findings:
                all_findings.append(f"**{desc}**: " + "; ".join(findings))
            all_figures += len(figures)

        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Bioinformatics Analysis Completed
Total {len(all_results)} analysis steps executed
Generated {all_figures} figures

## Findings from Each Step
{chr(10).join(all_findings) if all_findings else "No specific findings recorded"}

Please synthesize the above results and write a concise bioinformatics analysis summary (200-300 words).
Focus on:
1. Cell type composition and changes
2. Major differential expression patterns
3. Key findings related to research objectives
4. Questions requiring further verification or analysis

Respond in English.
"""

        summary = await self.chat(prompt, temperature=0.3)
        return summary.strip()
