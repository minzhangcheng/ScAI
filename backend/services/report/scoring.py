"""Scoring mixin for DetailedReportGenerator."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.models.schemas import AnalysisState


class ScoringMixin:
    """Provides scoring/score definition methods for report generation."""

    def _build_score_definitions_context(self, state: "AnalysisState") -> str:
        """Build score definitions context string for prompts."""
        if not state.score_definitions:
            return ""
        score_parts = []
        for sd in state.score_definitions:
            genes_str = ", ".join(sd.gene_list[:20]) if sd.gene_list else "N/A"
            params_str = ", ".join(f"{k}={v}" for k, v in sd.parameters.items()) if sd.parameters else "default"
            score_parts.append(
                f"- **{sd.score_name}**: {sd.method} — {sd.description}\n"
                f"  Parameters: {params_str} | Genes ({len(sd.gene_list)} from {sd.gene_list_source or 'unspecified'}): {genes_str}"
            )
        return "\n## Registered Score Definitions\n" + "\n".join(score_parts)
