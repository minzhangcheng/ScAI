# Round results, publication artifacts, and the API request/response envelopes.
# Moved verbatim out of backend/models/schemas.py.
"""
Data models for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


from .analysis import CrossModalSynthesis, DataDrivenMechanismProgram
from .enums import AgentType, FigureReviewStatus
from .session import tolerant_from_llm_dict


class ExplorationReport(BaseModel):
    """Structured exploration report (Report A) generated after Phase 1"""
    cluster_overview: str = ""
    data_quality_warnings: List[str] = Field(default_factory=list)
    strongest_heterogeneity: List[str] = Field(default_factory=list)
    key_pathway_changes: List[str] = Field(default_factory=list)
    abnormal_markers: List[str] = Field(default_factory=list)
    low_confidence_annotations: List[str] = Field(default_factory=list)
    pi_direction_assessment: str = ""
    research_direction_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    raw_direction_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    direction_literature_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    direction_dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    direction_selection_summary: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_llm_dict(cls, data: Dict[str, Any]) -> "ExplorationReport":
        """Build a report from loosely-typed LLM JSON, dropping any field whose
        value fails type validation so a single malformed field cannot crash the
        whole report.  Delegates to tolerant_from_llm_dict.
        """
        return tolerant_from_llm_dict(cls, data)

class RoundResult(BaseModel):
    """Result from one analysis round"""
    round_number: int
    pathology_findings: List[str] = Field(default_factory=list)
    bioinformatics_findings: List[str] = Field(default_factory=list)
    biochemistry_findings: List[str] = Field(default_factory=list)
    literature_findings: List[str] = Field(default_factory=list)
    pi_summary: str = ""
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    new_findings: bool = True
    # The PI's own verdict on whether the direction is answered. It has always
    # been asked for in the synthesis prompt and was previously dropped on parse,
    # so reports could not distinguish "the science concluded" from "the round
    # budget ran out". Advisory only: convergence remains bookkeeping-driven,
    # because in practice this is False almost every round.
    should_stop: bool = False
    stop_reason: str = ""
    agents_for_next_round: List[str] = Field(
        default_factory=lambda: ["bioinformatics", "pathology", "biochemistry", "literature"]
    )
    reanalysis_agents: List[str] = Field(default_factory=list)  # Agents needing reanalysis of existing data
    phase: str = "exploration"  # "exploration" | "innovation" | "integration"
    direction: str = ""  # Research direction for innovation rounds
    exploration_report: Optional[ExplorationReport] = None  # Only for Phase 1
    cross_modal_synthesis: Optional[CrossModalSynthesis] = None
    data_driven_mechanism_programs: List[DataDrivenMechanismProgram] = Field(default_factory=list)
    integration_report_summary: str = ""  # Excerpt from cross-direction integration report (integration phase only)
    integration_candidates: List[Dict[str, Any]] = Field(default_factory=list)  # Scored candidate directions from integration (integration phase only)
    lineage_id: str = "main"
    branch_id: Optional[str] = None
    branch_label: Optional[str] = None
    source_snapshot_id: Optional[str] = None
    restored_from_snapshot_id: Optional[str] = None
    direction_key: str = ""
    result_identity_key: str = ""
    round_status: Literal["completed", "partial", "failed", "skipped"] = "completed"
    evidence_status: Literal["substantive", "empty", "failed"] = "empty"
    bioinformatics_evidence_verified: Optional[bool] = None
    evidence_execution_ids: List[str] = Field(default_factory=list)
    evidence_notes: List[str] = Field(default_factory=list)
    # Whether this innovation round has enough core-agent evidence to satisfy
    # direction-batch coverage. Optional modality degradation may leave the
    # overall round_status as "partial" while the direction is still covered.
    # None preserves the legacy interpretation for older persisted results.
    direction_coverage_eligible: Optional[bool] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class PublicationFigure(BaseModel):
    """Publication-quality figure with PNG thumbnail and PDF high-res"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    figure_type: str  # umap, heatmap, violin, dotplot, barplot, trajectory, volcano, other
    png_path: str  # Path to PNG thumbnail
    pdf_path: str  # Path to high-res PDF
    png_base64: Optional[str] = None  # Base64 encoded PNG for preview (compressed)
    interpretation: Optional[str] = None  # Scientific interpretation of the figure
    review_status: FigureReviewStatus = FigureReviewStatus.PENDING
    pi_feedback: Optional[str] = None
    revision_history: List[Dict[str, Any]] = Field(default_factory=list)  # Accumulated feedback from all revision rounds
    revision_count: int = 0
    # Renderer-measured layout defects still present in the saved image
    # (see agents/bioinformatics/layout_lint.py). Blocking kinds mean some text
    # is covered by another artist and cannot be read at all.
    layout_violations: List[Dict[str, Any]] = Field(default_factory=list)
    # Reviewer must_fix items that were never addressed before the revision
    # loop gave up. Recorded so the report can say so instead of presenting the
    # figure as reviewed and fine.
    unresolved_must_fix: List[str] = Field(default_factory=list)
    round_number: int = 0
    analysis_generation: int = 0
    expression_provenance: Dict[str, Any] = Field(default_factory=dict)
    is_current: bool = True
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class PublicationTable(BaseModel):
    """Publication-quality table"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    csv_path: str
    columns: List[str] = Field(default_factory=list)
    preview_data: List[Dict[str, Any]] = Field(default_factory=list)  # First few rows for preview
    interpretation: Optional[str] = None  # Scientific interpretation of the table data
    review_status: FigureReviewStatus = FigureReviewStatus.PENDING
    pi_feedback: Optional[str] = None
    round_number: int = 0
    analysis_generation: int = 0
    expression_provenance: Dict[str, Any] = Field(default_factory=dict)
    is_current: bool = True
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class GraphicAbstract(BaseModel):
    """Graphic abstract generated by Painter Agent, or text-mode ASCII diagram drawn by Bio agent when Painter is unavailable"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str  # The prompt used to generate the image / diagram
    image_base64: str  # Base64 encoded image (final selected); empty string in text mode
    text_diagram: Optional[str] = None  # ASCII/Unicode text diagram (only populated in text mode)
    @property
    def is_text_mode(self) -> bool:
        """True when falling back to text-mode diagram (image_base64 == "")"""
        return bool(self.text_diagram and not self.image_base64)
    revision_count: int = 0  # Number of revisions performed
    revision_history: List[Dict[str, Any]] = Field(default_factory=list)  # Reviewer feedback from each round
    all_candidates: List[str] = Field(default_factory=list)  # All candidates: base64 strings in image mode, ASCII strings in text mode
    selected_index: int = 0  # Index of selected best candidate
    created_at: datetime = Field(default_factory=datetime.now)

class APIRequest(BaseModel):
    """Request to start or resume analysis"""
    session_id: Optional[str] = None
    instruction: Optional[str] = None
    action: Literal["start", "resume", "pause", "stop", "update_prompts"] = "start"
    custom_prompts: Optional[Dict[str, str]] = None

class APIResponse(BaseModel):
    """Response from API"""
    success: bool
    message: str
    session_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class UploadRequest(BaseModel):
    """Request to upload data"""
    session_id: str
    section: Literal["pathology", "sequencing", "biochemistry"]
    data_type: str
    group_id: Optional[str] = None
    sample_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebSocketMessage(BaseModel):
    """Message for WebSocket communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal[
        "status", "message", "code", "result", "error", "figure", "progress",
        "step", "prompt", "response", "thinking", "tool_call",
        # New types for autonomous bioinformatics agent
        "literature", "pi_question", "pi_response", "findings",
        # Human intervention
        "human_input_received", "human_input_requested",
        # Round management
        "round_summary", "data_request",
        # Workflow lifecycle
        "completed", "output",
        # Publication figure review
        "figure_review", "figure_approved", "figure_revision",
        # Integrated report and graphic abstract
        "integrated_report", "graphic_abstract", "detailed_report",
        # State synchronization and error tracking
        "state_sync", "parallel_errors",
        # WebSocket reconnection recovery
        "message_history",
        # Cluster annotation
        "annotation_result", "annotation_confirmation",
        # Exploration report
        "exploration_report",
        # Multi-direction correlation analysis
        "direction_correlation",
        # Chat delegation
        "chat_delegation", "chat_result", "chat_error",
        # Supervisor monitoring
        "supervisor_feedback", "supervisor_checkpoint", "supervisor_summary",
        "snapshot_count",
        # Plan-vs-evidence audit and the bounded supplementary pass that follows it
        "post_execution_review",
        # Collaborative planning
        "plan_review", "plan_approved",
        # Exploration failure (likely a data problem) — prompt user to check data
        "exploration_failed",
        # Modality warning (e.g. imaging data uploaded but its agent is disabled)
        "modality_warning",
        # Final report blocked (e.g. insufficient direction coverage) — awaits user
        "report_blocked",
        # Integration (cross-direction) round: gate for direction selection, and
        # terminal completion when the PI selects no further directions / cap hit
        "integration_gate", "integration_complete",
    ]
    agent: Optional[AgentType] = None
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)
