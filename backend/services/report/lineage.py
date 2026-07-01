"""Lineage filtering helpers for report generation."""

from backend.models.schemas import AnalysisState
from backend.workflow.direction_coverage import normalize_legacy_round_result


def build_lineage_report_state(state: AnalysisState, target_lineage_id: str) -> AnalysisState:
    """Return a deep-copied AnalysisState with round_results limited to one lineage."""
    target = target_lineage_id or "main"
    report_state = state.model_copy(deep=True)
    normalized = [normalize_legacy_round_result(r) for r in report_state.round_results]
    report_state.round_results = [r for r in normalized if r.lineage_id == target]
    return report_state
