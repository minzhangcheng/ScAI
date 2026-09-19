"""Cluster annotation helpers and HITL confirmation flow."""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models.schemas import (
    AgentType, AnalysisState, AnnotationConfirmationState,
    ClusterAnnotation, TaskStatus,
)
from ..services.interaction_record_capture import build_annotation_record
from .helpers import safe_pi_brief, wait_with_skip_time
from .annotation_validation import compute_validation_attempt_id, should_run_validation

if TYPE_CHECKING:
    from .state import WorkflowState

import sys
from .annotation_parts import callback as annotation_callback
from .annotation_parts import payloads as annotation_payloads
from .annotation_parts import validation as annotation_validation

logger = logging.getLogger(__name__)


class AnnotationMixin:
    """Mixin providing cluster-annotation methods. Composed into SingleCellAnalysisWorkflow."""

    @staticmethod
    def _build_fallback_annotations(cluster_markers: Dict, error: Exception) -> Dict[str, Any]:
        return annotation_payloads._build_fallback_annotations(cluster_markers, error)

    def _build_annotation_context_report(self, state, graph_state) -> str:
        annotation_payloads.bind_facade(sys.modules[__name__])
        return annotation_payloads._build_annotation_context_report(self, state, graph_state)

    @staticmethod
    def _build_cluster_annotations(annotations: Dict) -> List[ClusterAnnotation]:
        annotation_payloads.bind_facade(sys.modules[__name__])
        return annotation_payloads._build_cluster_annotations(annotations)

    @staticmethod
    def _apply_group_proportions(cluster_annotations, proportion_meta) -> None:
        return annotation_payloads._apply_group_proportions(cluster_annotations, proportion_meta)

    @staticmethod
    def _annotations_from_cluster_annotations(cluster_annotations) -> Dict[str, Any]:
        return annotation_payloads._annotations_from_cluster_annotations(cluster_annotations)

    async def _generate_annotation_brief(self, cluster_markers, organism, tissue, state):
        annotation_callback.bind_facade(sys.modules[__name__])
        return await annotation_callback._generate_annotation_brief(
            self, cluster_markers, organism, tissue, state
        )

    async def _annotate_clusters_callback(
        self, cluster_markers: Dict, tissue: str, organism: str,
        analysis_log: Optional[List] = None, database_priors: Optional[Dict] = None,
        annotation_source: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        annotation_callback.bind_facade(sys.modules[__name__])
        return await annotation_callback.annotate_clusters_callback(
            self, cluster_markers, tissue, organism, analysis_log, database_priors, annotation_source
        )

    def _validation_enabled(self) -> bool:
        annotation_validation.bind_facade(sys.modules[__name__])
        return annotation_validation._validation_enabled(self)

    @staticmethod
    def _empty_validation_meta(status: str, summary: str = "", attempt_id=None):
        return annotation_payloads._empty_validation_meta(status, summary, attempt_id)

    @staticmethod
    def _build_validation_text_summary(val_result):
        return annotation_payloads._build_validation_text_summary(val_result)

    @staticmethod
    def _build_annotation_confirmation_payload(
        state: AnalysisState, cluster_anns: List[ClusterAnnotation],
        tissue: str, organism: str, tissue_summary: str,
        analysis_summary: str, report_html_url: Optional[str],
        detailed_report_url: Optional[str],
    ) -> Dict[str, Any]:
        return annotation_payloads._build_annotation_confirmation_payload(
            state, cluster_anns, tissue, organism, tissue_summary,
            analysis_summary, report_html_url, detailed_report_url,
        )

    def _validation_limits(self):
        annotation_validation.bind_facade(sys.modules[__name__])
        return annotation_validation._validation_limits(self)

    async def _run_annotation_validation_round(
        self, state, annotations, cluster_markers, tissue, organism,
        triage=None, database_priors=None, focused_evidence=None,
        literature_report="", force=False,
    ):
        annotation_validation.bind_facade(sys.modules[__name__])
        return await annotation_validation._run_annotation_validation_round(
            self, state, annotations, cluster_markers, tissue, organism,
            triage=triage, database_priors=database_priors,
            focused_evidence=focused_evidence, literature_report=literature_report,
            force=force,
        )
