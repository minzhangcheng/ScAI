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
from .round_report import RoundReportChapter


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
    direction_dependency_repairs: List[str] = Field(default_factory=list)
    direction_selection_summary: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_llm_dict(cls, data: Dict[str, Any]) -> "ExplorationReport":
        """Build a report from loosely-typed LLM JSON, dropping any field whose
        value fails type validation so a single malformed field cannot crash the
        whole report.  Delegates to tolerant_from_llm_dict.
        """
        return tolerant_from_llm_dict(cls, data)


class PathologyCoverageRecord(BaseModel):
    """Host-retained tile coverage for one pathology image."""

    image_id: str = ""
    image_type: str = ""
    status: Literal["completed", "partial", "error", "failed", "skipped"] = "error"
    expected_tiles: Optional[int] = Field(default=None, ge=0)
    valid_tiles: Optional[int] = Field(default=None, ge=0)
    coverage_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    invalid_batches: List[Dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class RoundResult(BaseModel):
    """Result from one analysis round"""
    round_number: int
    pathology_findings: List[str] = Field(default_factory=list)
    pathology_coverage: List[PathologyCoverageRecord] = Field(default_factory=list)
    bioinformatics_findings: List[str] = Field(default_factory=list)
    bioinformatics_finding_records: List[Dict[str, Any]] = Field(
        default_factory=list
    )
    biochemistry_findings: List[str] = Field(default_factory=list)
    literature_findings: List[str] = Field(default_factory=list)
    pi_summary: str = ""
    report_chapter: Optional[RoundReportChapter] = None
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
    # Host-stamped source/output lineage for the settled round. A round with
    # multiple sources retains every preparation id; the final plan's source
    # must not overwrite earlier execution provenance.
    adata_source_preparation_ids: List[str] = Field(default_factory=list)
    canonical_adata_output_checkpoint_id: Optional[str] = None
    source_snapshot_id: Optional[str] = None
    restored_from_snapshot_id: Optional[str] = None
    # The execution/finding records in an inherited branch result keep the
    # lineage where they were actually produced.  This identifies that lineage
    # without relabelling the immutable receipts as branch-native work.
    inherited_from_lineage_id: Optional[str] = None
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
    wait_policy: Optional[Literal["manual", "countdown", "no_wait"]] = None
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
    # True means no visual review verdict was obtained. Keep PENDING so older
    # readers do not mistake this for approval; this flag distinguishes the
    # terminal observation from a review that is still in progress.
    review_unavailable: bool = False
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
    # The PI explicitly selected this candidate for report use after ordinary
    # review did not clear the deliverable. Measured defects and must-fix
    # observations remain recorded; this flag only carries that PI disposition.
    pi_degraded_acceptance: bool = False
    pi_degraded_acceptance_reason: Optional[str] = None
    # Rendered by the platform rather than by the analysis agent, and therefore
    # never offered to the figure-review loop. Read by ``is_report_eligible``,
    # which must not withhold such a figure for lacking a sign-off that nobody was
    # asked to give. Defaults False, so sessions written before this field
    # deserialize unchanged.
    diagnostic: bool = False
    submitted_plan_step_id: Optional[int] = None
    plan_requirement_key: Optional[str] = None
    supervisor_binding_decision: Optional[Literal[
        "accept", "rebind", "supplementary", "reject",
    ]] = None
    supervisor_binding_reason: Optional[str] = None
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
    # A restored branch owns this figure record, while its source execution may
    # still belong to the lineage where the frozen render was produced.
    inherited_from_lineage_id: Optional[str] = None
    round_number: int = 0
    analysis_generation: int = 0
    cohort_generation: int = 0
    cohort_digest: Optional[str] = None
    source_execution_id: Optional[str] = None
    # Version 2 binds a new plan-result figure to a validated execution receipt.
    # Optional defaults preserve historical artifacts without inventing lineage.
    provenance_schema_version: Optional[int] = None
    producer_run_kind: Optional[Literal[
        "ordinary", "supplementary", "figure_followup", "platform",
    ]] = None
    producer_run_scope_id: Optional[str] = None
    source_execution_run_kind: Optional[str] = None
    source_execution_validated: Optional[bool] = None
    source_preparation_id: Optional[str] = None
    figure_followup_task_id: Optional[str] = None
    provenance_validation_reason: Optional[str] = None
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
    csv_sha256: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    preview_data: List[Dict[str, Any]] = Field(default_factory=list)  # First few rows for preview
    total_rows: int = 0
    interpretation: Optional[str] = None  # Scientific interpretation of the table data
    review_status: FigureReviewStatus = FigureReviewStatus.PENDING
    pi_feedback: Optional[str] = None
    lineage_id: str = "main"
    plan_id: Optional[str] = None
    plan_step_ids: List[int] = Field(default_factory=list)
    plan_binding_status: Literal[
        "candidate", "confirmed", "supplementary", "unverified",
    ] = "unverified"
    plan_binding_reason: Optional[str] = None
    round_number: int = 0
    analysis_generation: int = 0
    cohort_generation: int = 0
    cohort_digest: Optional[str] = None
    source_execution_id: Optional[str] = None
    producer_status: Literal["published", "failed"] = "published"
    expression_provenance: Dict[str, Any] = Field(default_factory=dict)
    is_current: bool = True
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class FigureRevisionTask(BaseModel):
    """Durable work item for a figure that did not clear PI review."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    logical_figure_id: str
    plan_id: Optional[str] = None
    plan_requirement_key: Optional[str] = None
    current_figure_id: str = ""
    plan_step_id: Optional[int] = None
    lineage_id: Optional[str] = None
    round_number: int = 0
    required_count: int = 1
    found_count: int = 0
    source_execution_id: Optional[str] = None
    status: Literal[
        "pending", "in_progress", "resolved", "blocked", "abandoned",
        "exhausted_excluded",
    ] = "pending"
    reason: Literal[
        "visual_revision", "reanalysis_required", "revision_failed",
        "attempts_exhausted", "cohort_changed", "missing_figure",
    ]
    must_fix: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = Field(default=2, ge=1, le=6)
    pi_disposition: Optional[Literal[
        "targeted_redraw", "accept_best", "omit", "needs_new_analysis",
    ]] = None
    pi_rationale: Optional[str] = None
    pi_report_note: Optional[str] = None
    selected_figure_id: Optional[str] = None
    pi_instruction: Optional[str] = None
    followup_started_at: Optional[datetime] = None
    followup_finished_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class PIDecisionRecord(BaseModel):
    """Durable PI guidance that may refine a plan during execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    context: str = ""
    response: str
    plan_id: Optional[str] = None
    round_number: int = 0
    planning_stage: Optional[str] = None
    source: Literal["consult_pi"] = "consult_pi"
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
    # Optional tri-state availability observations. ``None`` preserves the
    # unknown/not-applicable meaning of older persisted graphic abstracts.
    review_available: Optional[bool] = None
    selection_available: Optional[bool] = None
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
        "state_sync", "parallel_errors", "snapshot_listing_changed",
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
        # Plan-vs-evidence audit and the bounded supplementary pass that follows it
        "post_execution_review",
        # Collaborative planning
        "plan_review", "plan_approved",
        # Exploration failure (likely a data problem) — prompt user to check data
        "exploration_failed",
        # Modality warning (e.g. imaging data uploaded but its agent is disabled)
        "modality_warning",
        # Legacy report-blocked event retained for stored-session compatibility;
        # current reporting records coverage for PI review and does not emit it.
        "report_blocked",
        # Integration (cross-direction) round: gate for direction selection, and
        # terminal completion when the PI selects no further directions / cap hit
        "integration_gate", "integration_complete",
    ]
    agent: Optional[AgentType] = None
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)
