"""Reporting mixin — final report, graphic abstract, and detailed report generation."""
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict

from ...models.schemas import (
    AnalysisState, AgentType, TaskStatus, GraphicAbstract,
)
from ...utils.llm_client import UpstreamExhaustedError
from ...constants import LlmFailureAction
from ...config import get_session_reports, get_session_workspace
from ...services.report import DetailedReportGenerator, compress_image_to_base64
from ...services.report.figure_discovery import discover_workspace_figures
from ...services.report.artifacts import (
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
    gallery_artifacts,
    has_blocking_layout_defects,
)
from ...services.report.figure_attribution import attribute_figures_to_rounds
from ..direction_coverage import current_lineage_id, direction_coverage_status
from ...services.report.paths import resolve_report_output_dir
from ...services.report.lineage import build_lineage_report_state


def report_coverage_observation(state: AnalysisState) -> str:
    """Return a direction-coverage fact for PI review, or an empty string.

    Coverage metadata never decides whether report generation proceeds or
    whether a report is scientifically acceptable.
    """
    from ..direction_coverage import active_batch_incomplete
    if active_batch_incomplete(state):
        return (
            "The active research direction batch does not have complete or "
            "coverage-eligible evidence for every direction. The PI must decide "
            "how this affects the report and its limitations."
        )
    if len(state.user_chosen_directions) <= 1:
        return ""
    coverage = direction_coverage_status(state, lineage_id=current_lineage_id(state))
    if coverage.complete:
        return ""
    missing = ", ".join(sorted(coverage.missing_keys))
    return (
        "Selected directions are missing completed evidence: "
        f"{missing}. The PI must decide how this affects the report and its limitations."
    )
