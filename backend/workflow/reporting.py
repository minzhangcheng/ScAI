"""Reporting mixin — final report, graphic abstract, and detailed report generation."""
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

from ..models.schemas import (
    AnalysisState, AgentType, TaskStatus, GraphicAbstract,
)
from ..utils.llm_client import UpstreamExhaustedError
from ..constants import LlmFailureAction
from ..config import get_session_reports, get_session_workspace
from ..services.report import DetailedReportGenerator, compress_image_to_base64
from ..services.report.figure_discovery import discover_workspace_figures
from ..services.report.artifacts import artifact_stems, eligible_artifacts
from .direction_coverage import current_lineage_id, direction_coverage_status
from ..services.report.paths import resolve_report_output_dir
from ..services.report.lineage import build_lineage_report_state

# Re-exported so `backend.workflow.reporting.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .reporting_parts.shared import (  # noqa: F401
    logging,
    os,
    uuid,
    datetime,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    AnalysisState,
    AgentType,
    TaskStatus,
    GraphicAbstract,
    UpstreamExhaustedError,
    LlmFailureAction,
    get_session_reports,
    get_session_workspace,
    DetailedReportGenerator,
    compress_image_to_base64,
    discover_workspace_figures,
    artifact_stems,
    eligible_artifacts,
    current_lineage_id,
    direction_coverage_status,
    resolve_report_output_dir,
    build_lineage_report_state,
    report_coverage_observation,
)
from .reporting_parts import (  # noqa: F401
    FinalReportMixin,
    GraphicAbstractMixin,
)


logger = logging.getLogger(__name__)


class ReportingMixin(
    FinalReportMixin,
    GraphicAbstractMixin,
):
    """Mixin providing report generation node methods.

    Composed into SingleCellAnalysisWorkflow. Relies on self.session_id,
    self.pi_agent, self.painter_agent, self.bioinformatics_agent,
    self.supervisor_agent, self._settings_provider, self._vision_settings,
    self._send_message, self._send_state_sync, self._set_status,
    self._wait_for_llm_intervention,
    self.sandbox, and self.llm_manager.
    """

    # ------------------------------------------------------------------
    # Text-mode graphic abstract (Painter unavailable fallback)
    # ------------------------------------------------------------------

    _DEFAULT_TEXT_DIAGRAM_PROMPT = (
        "Draw an ASCII/Unicode flow diagram summarizing what this study actually found. "
        "Reason ONLY from the final summary provided: the nodes and the relationships "
        "between them must be ones the summary names. Let the summary decide the shape "
        "- a signalling cascade, a comparison between groups, a compositional shift, a "
        "temporal or spatial progression, or a set of associations are all valid "
        "diagrams. Do NOT impose a ligand/receptor/effector/transcription-factor chain "
        "on a study that did not measure one, and do not invent intermediate nodes to "
        "reach a node count. Use directional arrows only where the summary supports "
        "direction. English labels only.\n\n"
        "Study summary:\n{summary}"
    )
