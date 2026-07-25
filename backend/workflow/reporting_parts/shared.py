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
from ...services.report.artifacts import artifact_stems, eligible_artifacts
from ..direction_coverage import current_lineage_id, direction_coverage_status
from ...services.report.paths import resolve_report_output_dir
from ...services.report.lineage import build_lineage_report_state


def validate_complete_report_allowed(state: AnalysisState) -> tuple[bool, str]:
    if state.allow_partial_report:
        return True, ""
    from ..direction_coverage import active_batch_incomplete
    if active_batch_incomplete(state):
        return False, (
            "The active research direction batch does not have complete or "
            "coverage-eligible evidence for every direction; only a partial "
            "report can be produced."
        )
    if len(state.user_chosen_directions) <= 1:
        return True, ""
    coverage = direction_coverage_status(state, lineage_id=current_lineage_id(state))
    if coverage.complete:
        return True, ""
    missing = ", ".join(sorted(coverage.missing_keys))
    return False, f"Complete report blocked because selected directions are missing completed evidence: {missing}"
