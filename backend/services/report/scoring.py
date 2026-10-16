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
            params_str = ", ".join(f"{k}={v}" for k, v in sd.parameters.items()) if sd.parameters else "default"
            kind = getattr(sd, "definition_kind", "unspecified") or "unspecified"
            if kind == "unspecified" and sd.gene_list:
                kind = "gene_panel"
            if kind == "gene_panel":
                genes_str = ", ".join(sd.gene_list[:20]) if sd.gene_list else "not recorded"
                definition = (
                    f"Panel genes ({len(sd.gene_list)} from "
                    f"{sd.gene_list_source or 'unspecified'}): {genes_str}"
                )
            else:
                references = ", ".join(
                    f"{item.role}:{item.table_id}"
                    for item in (getattr(sd, "definition_artifacts", None) or [])
                ) or "not registered"
                definition = (
                    f"Definition: {kind} "
                    f"({getattr(sd, 'definition_status', 'legacy')}); "
                    f"registered tables: {references}"
                )
            score_parts.append(
                f"- **{sd.score_name}**: {sd.method} — {sd.description}\n"
                f"  Parameters: {params_str} | {definition}"
            )
        return "\n## Registered Score Definitions\n" + "\n".join(score_parts)
