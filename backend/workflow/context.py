"""Context builders for planning, findings, and code history."""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models.schemas import AnalysisState

if TYPE_CHECKING:
    from .state import WorkflowState

logger = logging.getLogger(__name__)


class ContextMixin:
    """Mixin providing context-building methods. Composed into SingleCellAnalysisWorkflow."""

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

    def _collect_current_round_results(self, state: "WorkflowState") -> Dict[str, List[str]]:
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

    def _build_planning_context(self, state: "WorkflowState") -> Optional[Dict[str, Any]]:
        """Build combined previous_results for planning_dialogue.

        Merges:
        - planning_context (stage1 bioinfo results, set after stage1 execution)
        - Current-round biochemistry, pathology, and bioinformatics findings (via _collect_current_round_results)
        - Fallback to previous round findings from round_results if wet-lab agents were skipped
        """
        combined = dict(state.get("planning_context") or {})

        # Reuse existing method for current-round wet-lab findings
        current_findings = self._collect_current_round_results(state)
        for key in ("biochemistry_findings", "pathology_findings", "bioinformatics_findings"):
            if key in current_findings:
                combined[key] = current_findings[key]

        # Fallback: if wet-lab results are empty, look up previous round findings
        analysis_state = state["analysis_state"]
        if analysis_state.round_results:
            for prev_round in reversed(analysis_state.round_results):
                if "biochemistry_findings" not in combined and prev_round.biochemistry_findings:
                    combined["biochemistry_findings"] = prev_round.biochemistry_findings
                if "pathology_findings" not in combined and prev_round.pathology_findings:
                    combined["pathology_findings"] = prev_round.pathology_findings
                if "bioinformatics_findings" not in combined and prev_round.bioinformatics_findings:
                    combined["bioinformatics_findings"] = prev_round.bioinformatics_findings
                if "biochemistry_findings" in combined and "pathology_findings" in combined and "bioinformatics_findings" in combined:
                    break

        return combined if combined else None

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
