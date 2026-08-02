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
    # Version 1 means the completion/evidence fields below were derived from the
    # retained node results by the current workflow.  ``None`` is intentional for
    # sessions written before that derivation existed: historical model defaults
    # were serialized as if they were measured verdicts, so consumers must not
    # turn those defaults into new blocking failures.
    evidence_schema_version: Optional[int] = None
    round_status: Literal["completed", "partial", "failed", "skipped"] = "completed"
    evidence_status: Literal["substantive", "empty", "failed"] = "empty"
    bioinformatics_evidence_verified: Optional[bool] = None
    evidence_gate_degraded: bool = False
    evidence_execution_ids: List[str] = Field(default_factory=list)
    evidence_notes: List[str] = Field(default_factory=list)
    evidence_gate_findings: List[Dict[str, Any]] = Field(default_factory=list)
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
    figure_type: str  # umap, heatmap, violin, dotplot, barplot, trajectory, volcano, qc, other
    png_path: str  # Path to PNG thumbnail
    pdf_path: str  # Path to high-res PDF
    png_base64: Optional[str] = None  # Base64 encoded PNG for preview (compressed)
    # Digests bind the reviewed record to the bytes that were actually rendered.
    # They prevent a later plain ``savefig`` using the same filename from being
    # mistaken for the reviewed publication artifact.  Optional preserves
    # compatibility with sessions persisted before integrity tracking existed.
    render_sha256: Optional[str] = None
    preview_sha256: Optional[str] = None
    pdf_sha256: Optional[str] = None
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
    # Rendered by the platform rather than by the analysis agent, and therefore
    # never offered to the figure-review loop. Read by ``is_report_eligible``,
    # which must not withhold such a figure for lacking a sign-off that nobody was
    # asked to give. Defaults False, so sessions written before this field
    # deserialize unchanged.
    diagnostic: bool = False
    # Link the render back to the approved plan when the producer knows it.
    # Older sessions and platform diagnostics legitimately have no step id.
    plan_step_id: Optional[int] = None
    # Stable plan-level provenance prevents a render from an earlier plan that
    # reused the same numeric step id from closing a revised visual contract.
    # None preserves deserialization of pre-provenance session records.
    plan_id: Optional[str] = None
    # ``result`` is the default.  Platform baselines use ``baseline`` and QC
    # diagnostics use ``diagnostic`` so completeness checks can distinguish
    # scientific deliverables from supporting visual context.
    purpose: str = "result"
    # Stable semantic identity across separately rendered attempts.  Filenames are
    # presentation details and must never be parsed for ``_v2``/``_final`` style
    # suffixes to decide which records supersede one another.
    logical_figure_id: Optional[str] = None
    render_attempt_id: Optional[str] = None
    # Producer lifecycle is independent of PI review.  A pending/rejected review
    # can still refer to a physically complete render, while a process crash can
    # leave a review-pending record whose staged files no longer exist.  Older
    # persisted records predate this distinction and represent completed
    # producers, hence the backward-compatible ``published`` default.
    producer_status: Literal[
        "pending", "publishing", "published", "failed", "abandoned"
    ] = "published"
    # Attempt lineage is distinct from snapshot branch ownership.  ``revision_of``
    # records what this attempt tried to replace; ``supersedes`` is set only after
    # the new attempt becomes report-eligible, so a failed/pending redraw cannot
    # withdraw the last good archival render.
    revision_of_id: Optional[str] = None
    supersedes_id: Optional[str] = None
    is_published_head: bool = True
    # Snapshot/branch lineage that owns this render.  Report generation filters
    # on this field just as it filters RoundResult; otherwise equal round numbers
    # from two restore branches can leak figures across reports.
    lineage_id: str = "main"
    round_number: int = 0
    analysis_generation: int = 0
    expression_provenance: Dict[str, Any] = Field(default_factory=dict)
    is_current: bool = True
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class LegacyArtifactRecord(BaseModel):
    """Explicit inventory entry for a pre-registry unreviewed image.

    These records preserve old-session report compatibility without pretending a
    plain workspace file passed publication review.  New strict-registry sessions
    never create them from an unregistered current-run file.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    relative_path: str
    render_sha256: str
    pdf_relative_path: Optional[str] = None
    pdf_sha256: Optional[str] = None
    source: str = "legacy_workspace"
    provenance_kind: Literal["legacy_unregistered"] = "legacy_unregistered"
    lineage_id: str = "main"
    discovered_at: datetime = Field(default_factory=datetime.now)

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
