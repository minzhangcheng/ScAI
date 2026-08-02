"""Lineage filtering helpers for report generation.

Legacy analysis logs and plain workspace images predate lineage provenance. Main
reports keep accepting those files for backwards compatibility, but a restored
branch must not guess that an unowned file belongs to it: the same workspace can
also contain outputs produced before or after a different restore. Branch
reports therefore admit images only when a target-lineage ``PublicationFigure``
record vouches for them.
"""

from pathlib import Path
from typing import Any

from backend.models.schemas import AnalysisState
from backend.services.report.artifacts import artifact_stems
from backend.services.report.figure_discovery import discover_figures_in_workspace
from backend.workflow.direction_coverage import normalize_legacy_round_result


REPORT_TARGET_LINEAGE_STATE_KEY = "_report_target_lineage_id"


def clone_publication_figures_for_branch(
    figures,
    *,
    branch_id: str,
    source_lineage_id: str = "main",
):
    """Preserve source records and add branch-local copies of inherited figures."""
    target = branch_id or "main"
    source = source_lineage_id or "main"
    output = list(figures or [])
    existing = {
        ((getattr(figure, "lineage_id", None) or "main"), figure.name)
        for figure in output
    }
    if target == source:
        return output
    for original in list(output):
        if (getattr(original, "lineage_id", None) or "main") != source:
            continue
        key = (target, original.name)
        if key in existing:
            continue
        clone = original.model_copy(deep=True)
        clone.lineage_id = target
        # IDs are API/download identities. Keep inherited records addressable in
        # both lineages without making two records share one route key.
        clone.id = f"{original.id}:{target}"
        output.append(clone)
        existing.add(key)
    return output


def _restrict_unowned_branch_figures(
    report_state: AnalysisState,
    workspace_path: Any,
) -> None:
    """Exclude legacy image files that cannot be assigned to this branch.

    The exclusions live only on the deep-copied report state. They reuse the
    existing stale-stem reconciliation path, so a current target-lineage record
    still wins even when an old flat invalidation set contains the same stem.
    """
    workspace = Path(workspace_path) if workspace_path else None
    owned_stems = artifact_stems(report_state.publication_figures)
    unowned_stems = {
        Path(str(name)).stem
        for step in (report_state.analysis_log or [])
        for name in (step.saved_figure_filenames or [])
        if Path(str(name)).stem not in owned_stems
    }

    # A legacy log entry proves only that a filename was once saved. Retain a
    # name on a non-main branch only when a lineage-filtered artifact record owns
    # that stem; collectors may still recover the record's verified preview.
    for step in report_state.analysis_log or []:
        step.saved_figure_filenames = [
            name
            for name in (step.saved_figure_filenames or [])
            if Path(str(name)).stem in owned_stems
        ]

    if workspace is not None and workspace.exists():
        unowned_stems.update(
            str(candidate.get("name") or "")
            for candidate in discover_figures_in_workspace(workspace)
            if candidate.get("name")
            and str(candidate.get("name")) not in owned_stems
        )
    report_state.invalidated_artifact_stems = sorted(
        set(report_state.invalidated_artifact_stems or []) | unowned_stems
    )


def build_lineage_report_state(
    state: AnalysisState,
    target_lineage_id: str,
    *,
    workspace_path: Any = None,
) -> AnalysisState:
    """Return a report snapshot limited to one analysis lineage."""
    target = target_lineage_id or "main"
    report_state = state.model_copy(deep=True)
    sandbox_state = dict(report_state.sandbox_state or {})
    sandbox_state[REPORT_TARGET_LINEAGE_STATE_KEY] = target
    report_state.sandbox_state = sandbox_state
    normalized = [normalize_legacy_round_result(r) for r in report_state.round_results]
    report_state.round_results = [r for r in normalized if r.lineage_id == target]
    report_state.publication_figures = [
        figure
        for figure in report_state.publication_figures
        if (getattr(figure, "lineage_id", None) or "main") == target
    ]
    if target != "main":
        _restrict_unowned_branch_figures(report_state, workspace_path)
    return report_state
