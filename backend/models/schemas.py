"""
Data models for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid
import asyncio
import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_ROUND_WAIT_SECONDS: int = 300

# Sequencing data type constants
DATA_TYPE_SCRNASEQ = "scRNA-seq"
DATA_TYPE_BULK_RNASEQ = "bulk RNA-seq"

# Module-level lock for async-safe state version increments
_version_lock = asyncio.Lock()

_RUNTIME_GATE_TOKEN_RE = re.compile(r"^runtime-(\d+)-[0-9a-f]{32}$")


def _sanitize_snapshot_failure(payload: Dict[str, Any]) -> Any:
    """Sanitize diagnostics without destroying a runtime-gate nonce.

    ``gate_token`` is normally a credential-shaped key, but the runtime
    persistence failure payload uses it as a public, single-generation nonce.
    Keep only the strictly generated nonce already present in the independent
    ``runtime_persistence`` record.  This also repairs states written by
    0.38.10 where the diagnostic copy was persisted as ``<redacted-secret>``.
    """
    from backend.utils.error_parsing import sanitize_error_payload

    failure = sanitize_error_payload(payload.get("snapshot_failure"), max_chars=3000)
    if not isinstance(failure, dict):
        return failure
    if not str(failure.get("code") or "").startswith("runtime_persistence_"):
        return failure

    persistence = payload.get("runtime_persistence")
    if not isinstance(persistence, dict):
        return failure
    token = str(persistence.get("gate_token") or "")
    match = _RUNTIME_GATE_TOKEN_RE.fullmatch(token)
    generation = int(persistence.get("gate_generation") or 0)
    if match is None or int(match.group(1)) != generation:
        return failure

    details = dict(failure.get("details") or {})
    details["gate_token"] = token
    details["gate_generation"] = generation
    retry_context = dict(failure.get("retry_context") or {})
    retry_context["gate_token"] = token
    retry_context["generation"] = generation
    failure = dict(failure)
    failure["details"] = details
    failure["retry_context"] = retry_context
    return failure


# Implementations live in ./schema_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from .schema_parts.enums import (  # noqa: F401
    AgentType,
    MessageRole,
    TaskStatus,
    AnalysisPhase,
    SnapshotPhase,
    AnalysisPhaseType,
    PlanStage,
    PlanStatus,
    FigureReviewStatus,
)
from .schema_parts.analysis import (  # noqa: F401
    ScoringPlan,
    AgentError,
    AgentMessage,
    CodeExecution,
    AnalysisStep,
    ScoreDefinition,
    TreatmentGroup,
    DataFile,
    BiochemistryIndicator,
    LiteratureMechanismSignal,
    DataDrivenMechanismProgram,
    LiteratureReport,
    CrossModalEvidenceItem,
    CrossModalMechanismCandidate,
    CrossModalSynthesis,
)
from .schema_parts.session import (  # noqa: F401
    ClusterAnnotation,
    AnnotationConfirmationState,
    SelectionEntry,
    DirectionSelectionRecord,
    AnnotationClusterEntry,
    AnnotationRecord,
    AgentRoundCache,
    DetailedReportSegment,
    Snapshot,
    FigureRequirement,
    PlanStep,
    AnalysisPlan,
    SpeciesTissueQualification,
    DirectionSearchQuery,
    PubMedArticle,
    PubMedEvidenceItem,
    DirectionLiteratureEvidence,
    ResearchDirectionCandidate,
    DirectionSelectionDecision,
    tolerant_from_llm_dict,
)
from .schema_parts.outputs import (  # noqa: F401
    ExplorationReport,
    PathologyCoverageRecord,
    RoundResult,
    PublicationFigure,
    LegacyArtifactRecord,
    PublicationTable,
    GraphicAbstract,
    APIRequest,
    APIResponse,
    UploadRequest,
    WebSocketMessage,
)










    

























































class AnalysisState(BaseModel):
    """Complete state of an analysis session"""

    @model_validator(mode="before")
    @classmethod
    def _open_persisted_secrets(cls, value):
        """Decrypt credentials emitted by the matching state serializer."""
        if isinstance(value, dict):
            from backend.utils.secret_redaction import open_analysis_state_payload

            payload = open_analysis_state_payload(value)
            if payload.get("snapshot_failure") is not None:
                payload = dict(payload)
                payload["snapshot_failure"] = _sanitize_snapshot_failure(payload)
            return payload
        return value

    @model_serializer(mode="wrap")
    def _seal_persisted_secrets(self, handler):
        """Ensure model_dump/checkpoint/snapshot payloads never contain raw keys."""
        from backend.utils.secret_redaction import seal_analysis_state_payload

        payload = seal_analysis_state_payload(handler(self))
        if payload.get("snapshot_failure") is not None:
            payload = dict(payload)
            payload["snapshot_failure"] = _sanitize_snapshot_failure(payload)
        return payload

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lifecycle_generation: int = 0
    name: str = "Untitled Analysis"
    user_id: str = Field(..., min_length=1)  # Owner user ID (required, non-empty)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # State version control for synchronization
    state_version: int = 0
    last_sync_timestamp: datetime = Field(default_factory=datetime.now)

    # Status
    status: TaskStatus = TaskStatus.PENDING
    pre_pause_status: Optional[TaskStatus] = None
    current_round: int = 0
    max_rounds: int = 3  # User-configurable maximum rounds
    original_max_rounds: Optional[int] = Field(default=None)  # Preserved during stop to avoid clobbering max_rounds
    round_wait_seconds: int = DEFAULT_ROUND_WAIT_SECONDS  # Countdown seconds between rounds (-1=manual, 0=no wait)
    plan_review_wait_seconds: int = -1
    auto_approve_plans: bool = False
    allow_partial_report: bool = False
    partial_report_requested_at: Optional[datetime] = None
    waiting_input_started_at: Optional[datetime] = None

    # Two-phase workflow state
    analysis_phase: AnalysisPhase = AnalysisPhase.EXPLORATION
    annotation_state: Optional[AnnotationConfirmationState] = None
    exploration_report: Optional[ExplorationReport] = None
    user_chosen_direction: Optional[str] = None  # User's chosen direction for innovation phase (backward compat)
    current_round_direction: str = ""  # Active direction for current innovation round (preserved after user_chosen_direction is consumed)

    @field_validator("current_round_direction", mode="before")
    @classmethod
    def _coerce_none_round_direction(cls, v):
        return "" if v is None else v

    # Multi-direction workflow
    user_chosen_directions: List[Dict[str, Any]] = Field(default_factory=list)
    # Each dict: {"direction": str, "notes": str, "is_custom": bool, "candidate_index": int|None}
    directions_queue: List[Dict[str, Any]] = Field(default_factory=list)
    # Remaining directions to process (popped one by one)
    multi_direction_correlation: Optional[str] = None
    integration_cycle: int = 0
    max_integration_cycles: int = 3
    pending_integration: bool = False
    active_direction_batch: List[Dict[str, Any]] = Field(default_factory=list)
    last_integrated_round: int = 0
    integration_gate_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    direction_selection_records: List[DirectionSelectionRecord] = Field(default_factory=list)
    annotation_record: Optional[AnnotationRecord] = None
    annotation_supplementary_requests: List[str] = Field(default_factory=list)
    integration_gate_literature_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    integration_reports: List[str] = Field(default_factory=list)
    final_report_requested: bool = False

    def consume_final_report_request(self) -> bool:
        """Atomically read-and-clear the one-shot final-report request flag."""
        requested = bool(self.final_report_requested)
        self.final_report_requested = False
        return requested

    # Correlation analysis result for multi-direction mode
    exploration_report_html_path: Optional[str] = None
    # Path to exploration report HTML file
    annotation_report_html_path: Optional[str] = None
    integration_report_html_path: Optional[str] = None
    # Path to annotation context report HTML file

    # Incremental detailed report segments (built at each HITL point)
    detailed_report_segments: List[DetailedReportSegment] = Field(default_factory=list)
    detailed_report_html_path: Optional[str] = None

    # Snapshots for debugging rollback
    snapshots: List[Snapshot] = Field(default_factory=list)
    last_restored_snapshot_id: Optional[str] = None  # Tracks which snapshot was last restored (for parent chain)
    # Durable first-round restore provenance; survives routing consume until
    # the canonical RoundResult records it (then consume_restore_provenance).
    restore_provenance_snapshot_id: Optional[str] = None
    last_snapshot_id: Optional[str] = None  # Lineage head: id of the most recently created/restored snapshot, used as the forward parent
    restore_start_intent: Optional[Literal["explicit_start"]] = None
    restore_resume_phase: Optional[str] = None
    restore_resume_kind: Optional[Literal["graph_node", "semantic_gate", "terminal"]] = None
    restore_resume_node: Optional[str] = None
    restore_resume_gate: Optional[str] = None
    # Verified frozen WorkflowState for graph-node restore (internal; excluded from API)
    restore_resume_workflow_state: Optional[Dict[str, Any]] = None
    restore_branch_id: Optional[str] = None
    restore_branch_label: Optional[str] = None

    # Snapshot runtime consistency (Stage 1 foundation safety).
    # "complete": kernel runtime matches restored business state.
    # "unavailable": restored without a consistent runtime; forward resume is blocked.
    # "intentionally_kept": user override kept a mismatched live runtime (risk accepted).
    runtime_restore_status: Literal["complete", "unavailable", "intentionally_kept"] = "complete"
    runtime_restore_warning: Optional[str] = None
    # Set when a run is started specifically to recompute a kernel that is gone
    # (the usual cause is an import without runtime state). Consumed by the
    # bioinformatics data context so the agent reloads its inputs instead of
    # referencing variables from a session it can no longer see.
    runtime_recompute_pending: bool = False

    # Audit reason when a session is terminated (e.g. snapshot_failure).
    termination_reason: Optional[str] = None

    # Snapshot V2: failure and transaction tracking
    snapshot_failure: Optional[Dict[str, Any]] = None
    snapshot_transaction: Optional[Dict[str, Any]] = None

    # One-shot "skip snapshot & continue" marker. Set when the user chooses Skip
    # on a snapshot-failure gate; consumed by the next required workflow-boundary
    # capture for the matching phase ("*" matches any) so the workflow proceeds
    # without capturing at that boundary. See failure_gate.consume_snapshot_skip.
    skip_required_snapshot_phase: Optional[str] = None

    # One-shot "capture anyway" marker. Set when the user chooses Proceed on a
    # memory-preflight gate, accepting the OOM risk; consumed by the next runtime
    # capture, which then skips the preflight. Cleared on consumption so a later
    # capture is judged on its own measurements.
    # See runtime_parts.capture._consume_memory_preflight_override.
    force_snapshot_memory_override: bool = False

    # Recovery failure marker: set when journal recovery fails, blocks mutations
    recovery_blocked: bool = False

    # Post-report snapshot idempotency (Task 6)
    report_generation_id: Optional[str] = None
    pending_required_snapshot: Optional[Dict[str, Any]] = None
    # Durable handoff from a bounded detailed-report LLM intervention. Once
    # armed, a restarted workflow must bypass the failed upstream and finish
    # with the deterministic report-content fallback.
    detailed_report_fallback_pending: bool = False

    # Collaborative analysis planning
    current_plan: Optional[AnalysisPlan] = None
    plan_dialogue_history: List[Dict[str, Any]] = Field(default_factory=list)  # PI↔生信对话记录

    # Initial inputs
    initial_instruction: str = ""

    # Species identification (global, visible to all agents)
    species: str = ""
    species_resolved: bool = False
    species_genes_validated: bool = False
    species_input_info: Optional[Dict[str, Any]] = None

    # Agent messages history
    messages: List[AgentMessage] = Field(default_factory=list)

    # Realtime messages for WebSocket (persisted for session restoration)
    realtime_messages: List["WebSocketMessage"] = Field(default_factory=list)
    
    # Code execution history (for reproducibility)
    code_history: List[CodeExecution] = Field(default_factory=list)

    # Comprehensive analysis log (ALL steps with agent reasoning, chronological)
    analysis_log: List[AnalysisStep] = Field(default_factory=list)

    # Treatment groups and uploaded data (multi-group, multi-sample)
    treatment_groups: List[TreatmentGroup] = Field(default_factory=list)
    data_files: List[DataFile] = Field(default_factory=list)
    biochemistry_indicators: List[BiochemistryIndicator] = Field(default_factory=list)
    
    # Literature reports
    literature_reports: List[LiteratureReport] = Field(default_factory=list)

    # Resolved label of the literature model(s) actually used (DR vs PubMed), for the report
    literature_actual_model: Optional[str] = None
    
    # Round results
    round_results: List[RoundResult] = Field(default_factory=list)
    current_round_cache: Optional[AgentRoundCache] = None

    # Data-driven mechanism programs captured across rounds
    data_driven_mechanism_programs: List[DataDrivenMechanismProgram] = Field(default_factory=list)

    # Final summary
    final_summary: Optional[str] = None

    # Publication-quality outputs
    publication_figures: List[PublicationFigure] = Field(default_factory=list)
    # ``None`` means the session was persisted before artifact registration was
    # enforced.  New session factories explicitly stamp the current version;
    # deserialization must not guess from mtimes or today's application version.
    artifact_registry_version: Optional[int] = None
    legacy_artifacts: List[LegacyArtifactRecord] = Field(default_factory=list)
    publication_tables: List[PublicationTable] = Field(default_factory=list)
    invalidated_artifact_stems: List[str] = Field(default_factory=list)
    graphic_abstract: Optional[GraphicAbstract] = None
    integrated_report: Optional[str] = None  # Integrated report with figure/table references

    # Score definitions (methodology tracking for report transparency)
    score_definitions: List[ScoreDefinition] = Field(default_factory=list)
    # Monotonic session audit: at least one mandatory evidence gate exhausted
    # its retry ceiling and was force-released.
    evidence_gate_degraded: bool = False
    evidence_gate_findings: List[Dict[str, Any]] = Field(default_factory=list)

    # Scoring plans frozen before their results were known. register_score cites
    # one of these; the handler re-hashes the panel to detect post-hoc revision.
    scoring_plans: List[ScoringPlan] = Field(default_factory=list)

    # Post-execution review records, one per audited bioinformatics execution.
    # Carries the plan-versus-evidence audit and, when a supplementary pass ran,
    # its plan and outcome. Keyed by a deterministic attempt id so node retry and
    # snapshot restore reuse a completed review instead of paying for it twice.
    post_execution_reviews: List[Dict[str, Any]] = Field(default_factory=list)

    # Agent prompts (can be customized per session)
    custom_prompts: Dict[str, str] = Field(default_factory=dict)
    
    # Agent models (can be customized per session)
    custom_models: Dict[str, str] = Field(default_factory=dict)
    
    # Agent temperatures (can be customized per session)
    custom_temperatures: Dict[str, float] = Field(default_factory=dict)

    # Per-agent LLM server overrides (agent_name -> {"llm_base_url": ..., "api_key": ...})
    agent_server_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # P2-11: Session-configurable DEG significance thresholds.
    # Different research domains use different cutoffs (e.g. tumor heterogeneity uses |logFC|>0.5).
    deg_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"logfc": 1.0, "padj": 0.05}
    )

    # Agent vision capability settings (agent_name -> bool, True = supports vision)
    agent_vision_settings: Dict[str, bool] = Field(default_factory=dict)

    # Literature agent configuration
    literature_config: Dict[str, Any] = Field(default_factory=lambda: {
        "mode": "pubmed_llm",                 # deep_research | gemini_deep_research | pubmed_llm
        "deep_research_model": "o4-mini-deep-research",
        "deep_research_llm_base_url": "",    # Separate LLM server for deep research (optional)
        "deep_research_api_key": "",         # Separate API key for deep research (optional)
        "google_api_key": "",                # For Gemini Deep Research mode
        "pubmed_email": "",                  # For PubMed mode (NCBI requirement)
        "max_search_rounds": 20,             # Max iterative search rounds
    })
    literature_search_cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    environment_api_notes: Dict[str, str] = Field(default_factory=dict)
    
    # Chat messages (PI Agent free-form chat, separate from workflow)
    chat_messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Sandbox state (for potential restoration)
    sandbox_state: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"runtime_revision": 0, "saved_runtime_revision": 0}
    )
    runtime_persistence: Dict[str, Any] = Field(default_factory=lambda: {
        "status": "stale",
        "current_revision": 0,
        "saved_revision": 0,
        "dirty_since": None,
        "last_saved_at": None,
        "last_restored_at": None,
        "last_failed_at": None,
        "failed_variables": [],
        "error_summary": None,
        "requires_recompute": False,
        "recompute_variables": [],
        "recommended_action": None,
        "commit_source": None,
        "runtime_digest": None,
        "budget_violations": [],
        "top_offenders": [],
    })
    
    # Config snapshot
    config_snapshot: Optional[Dict[str, Any]] = None
    
    # Human intervention support
    pending_human_input: Optional[str] = None  # Human message waiting to be processed
    pending_uploaded_files: List[Dict[str, Any]] = Field(default_factory=list)  # Files uploaded with human input
    human_input_requested: bool = False  # Whether system is waiting for human input
    human_messages: List[Dict[str, Any]] = Field(default_factory=list)  # History of human interventions

    # LLM failure user intervention
    wait_input_reason: Optional[str] = None    # Pending HITL gate identifier.
    llm_failure_info: Optional[Dict[str, Any]] = None
    # llm_failure_info keys when set:
    #   "agent": str       — agent name (e.g. "pathology")
    #   "model": str       — model identifier
    #   "llm_server": str  — base URL
    #   "error": str       — error message
    #   "node_name": str   — workflow node name
    #   "generation": int  — monotonic counter

    agent_disabled_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Info about a disabled core agent the workflow is waiting on. "
                    "Keys: agent (str), node_name (str), generation (int).",
    )

    # Error tracking for parallel analysis
    agent_errors: List[AgentError] = Field(default_factory=list)

    _API_KEY_MAP: Dict[str, str] = {
        "id": "session_id",
        "user_id": "user_id",
        "name": "name",
        "status": "status",
        "current_round": "current_round",
        "max_rounds": "max_rounds",
        "round_wait_seconds": "round_wait_seconds",
        "waiting_input_started_at": "waiting_input_started_at",
        "created_at": "created_at",
        "updated_at": "updated_at",
        "initial_instruction": "initial_instruction",
        "treatment_groups": "treatment_groups",
        "data_files": "data_files",
        "biochemistry_indicators": "biochemistry_indicators",
        "round_results": "round_results",
        "final_summary": "final_summary",
        "analysis_phase": "analysis_phase",
        # The annotation gate is rebuilt from this on reload; the
        # annotation_confirmation message alone is too lossy to rely on
        # (replay window, stream cap, snapshot restore all drop it).
        "annotation_state": "annotation_state",
        "realtime_messages": "messages",
        "custom_prompts": "custom_prompts",
        "custom_models": "custom_models",
        "custom_temperatures": "custom_temperatures",
        "deg_thresholds": "deg_thresholds",
        "state_version": "state_version",
        "last_sync_timestamp": "last_sync_timestamp",
        "current_plan": "current_plan",
        "plan_dialogue_history": "plan_dialogue_history",
        "agent_errors": "agent_errors",
        "runtime_persistence": "runtime_persistence",
        "evidence_gate_degraded": "evidence_gate_degraded",
        "evidence_gate_findings": "evidence_gate_findings",
        "publication_figures": "publication_figures",
        "publication_tables": "publication_tables",
        "graphic_abstract": "graphic_abstract",
        "integrated_report": "integrated_report",
        "species": "species",
        "species_resolved": "species_resolved",
        "species_genes_validated": "species_genes_validated",
        "species_input_info": "species_input_info",
        # Pending HITL gates must be part of the authoritative load response.
        # WebSocket state_sync delivery is best-effort and cannot be the only
        # way a refreshed client learns how to reopen an intervention dialog.
        "wait_input_reason": "wait_input_reason",
        "llm_failure_info": "llm_failure_info",
        "agent_disabled_info": "agent_disabled_info",
        "literature_config": "literature_config",
        "runtime_restore_status": "runtime_restore_status",
        "runtime_restore_warning": "runtime_restore_warning",
        "termination_reason": "termination_reason",
        "snapshot_failure": "snapshot_failure",
        "snapshot_transaction": "snapshot_transaction",
        "pending_required_snapshot": "pending_required_snapshot",
        "detailed_report_fallback_pending": "detailed_report_fallback_pending",
    }

    # --- Data helper methods ---

    def get_files_by_section(self, section: str) -> List[DataFile]:
        """Get all data files for a section (pathology/sequencing/biochemistry)"""
        return [f for f in self.data_files if f.section == section]

    def get_files_by_group(self, group_id: Optional[str], section: Optional[str] = None) -> List[DataFile]:
        """Get data files for a specific group, optionally filtered by section"""
        files = [f for f in self.data_files if f.group_id == group_id]
        if section:
            files = [f for f in files if f.section == section]
        return files

    def has_sequencing_data(self) -> bool:
        """Check if any sequencing data is uploaded"""
        return any(f.section == "sequencing" for f in self.data_files)

    def has_bulk_rnaseq(self) -> bool:
        """Check if bulk RNA-seq data is available"""
        return any(f.data_type == DATA_TYPE_BULK_RNASEQ for f in self.get_files_by_section("sequencing"))

    def has_scrnaseq(self) -> bool:
        """Check if scRNA-seq data is available"""
        return any(f.data_type == DATA_TYPE_SCRNASEQ for f in self.get_files_by_section("sequencing"))

    def get_group_by_id(self, group_id: Optional[str]) -> Optional[TreatmentGroup]:
        """Lookup a treatment group by ID"""
        if group_id is None:
            return None
        for g in self.treatment_groups:
            if g.id == group_id:
                return g
        return None

    def build_experimental_design_summary(self) -> str:
        """Generate a text description of the experimental design for agent prompts."""
        from collections import Counter
        if not self.treatment_groups:
            # Check if there are ungrouped files
            if self.data_files:
                parts = ["Experimental Design: No groups defined (ungrouped data)"]
                for section in ["pathology", "sequencing", "biochemistry"]:
                    section_files = self.get_files_by_section(section)
                    if section_files:
                        types = Counter(f.data_type for f in section_files)
                        parts.append(f"  {section}: {dict(types)}")
                return "\n".join(parts)
            return "No experimental groups or data defined."

        parts = [f"Experimental Design: {len(self.treatment_groups)} group(s)"]
        for g in sorted(self.treatment_groups, key=lambda x: x.order):
            parts.append(f"  - {g.name} ({g.sample_count} sample(s)): {g.description}")
            group_files = self.get_files_by_group(g.id)
            for section in ["pathology", "sequencing", "biochemistry"]:
                section_files = [f for f in group_files if f.section == section]
                if section_files:
                    types = Counter(f.data_type for f in section_files)
                    parts.append(f"    {section}: {dict(types)}")

        # Also show ungrouped data
        ungrouped = self.get_files_by_group(None)
        if ungrouped:
            parts.append("  - Ungrouped data:")
            for section in ["pathology", "sequencing", "biochemistry"]:
                section_files = [f for f in ungrouped if f.section == section]
                if section_files:
                    types = Counter(f.data_type for f in section_files)
                    parts.append(f"    {section}: {dict(types)}")

        # Show biochemistry indicators summary
        if self.biochemistry_indicators:
            parts.append(f"  Biochemistry indicators: {len(self.biochemistry_indicators)} manual entries")

        return "\n".join(parts)

    async def increment_version(self):
        """Atomically increment state version for synchronization

        Uses a module-level async lock to ensure safe increment in concurrent async environments.
        """
        async with _version_lock:
            self.state_version += 1
            self.last_sync_timestamp = datetime.now()
            self.updated_at = datetime.now()

    def add_error(self, agent: AgentType, error_type: str, message: str,
                  details: Optional[str] = None) -> AgentError:
        """Add a structured error record"""
        error = AgentError(
            agent=agent,
            error_type=error_type,
            message=message,
            details=details,
            round_number=self.current_round
        )
        self.agent_errors.append(error)
        self.updated_at = datetime.now()
        return error

    def get_unresolved_errors(self) -> List[AgentError]:
        """Get all unresolved errors"""
        return [e for e in self.agent_errors if not e.resolved]

    def get_errors_by_agent(self, agent: AgentType) -> List[AgentError]:
        """Get errors for a specific agent"""
        return [e for e in self.agent_errors if e.agent == agent]

    def resolve_error(self, error_id: str, resolution: Optional[str] = None):
        """Mark an error as resolved"""
        for error in self.agent_errors:
            if error.id == error_id:
                error.resolved = True
                error.resolution = resolution
                self.updated_at = datetime.now()
                break

    def add_message(self, agent: AgentType, role: MessageRole, content: str, **metadata):
        """Add a message to history"""
        msg = AgentMessage(
            agent=agent,
            role=role,
            content=content,
            metadata=metadata
        )
        self.messages.append(msg)
        self.updated_at = datetime.now()
        return msg
    
    def get_agent_messages(self, agent: AgentType) -> List[AgentMessage]:
        """Get all messages for a specific agent"""
        return [m for m in self.messages if m.agent == agent]
    
    def get_latest_round(self) -> Optional[RoundResult]:
        """Get the latest round result"""
        if self.round_results:
            return self.round_results[-1]
        return None
    
    def should_stop(self, convergence_rounds: int = 2) -> bool:
        """Check if analysis should stop based on convergence"""
        if len(self.round_results) < convergence_rounds:
            return False

        # Check if last N rounds had no new findings
        recent_rounds = self.round_results[-convergence_rounds:]
        return all(not r.new_findings for r in recent_rounds)

    @staticmethod
    def _serialize_dialogue_history(history: list) -> list:
        """Serialize plan dialogue history, converting nested datetime objects."""
        def _convert(value):
            if hasattr(value, 'isoformat'):
                return value.isoformat()
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_convert(v) for v in value]
            return value

        return [_convert(entry) if isinstance(entry, dict) else entry for entry in history]

    def to_api_dict(self, *, strip_message_base64=None, include_messages: bool = True) -> dict:
        """Serialize state for the load_session API response.

        Uses model_dump(mode="json") as baseline, then applies snake_case →
        camelCase key renaming via _API_KEY_MAP and special transformations.
        Fields not in _API_KEY_MAP are excluded from the API response.

        Args:
            strip_message_base64: Optional callable(msg_type, content) -> content
                to strip large base64 data from messages before sending.
        """
        dumped = self.model_dump(
            mode="json",
            exclude=None if include_messages else {"realtime_messages"},
        )

        result = {}
        for field_name, api_key in self._API_KEY_MAP.items():
            if field_name not in dumped:
                continue
            val = dumped[field_name]
            if field_name == "treatment_groups":
                result[api_key] = self._serialize_treatment_groups(val)
            elif field_name == "data_files":
                result[api_key] = self._serialize_data_files(val)
            elif field_name == "biochemistry_indicators":
                result[api_key] = self._serialize_biochemistry_indicators(val)
            elif field_name == "round_results":
                result[api_key] = self._serialize_round_results(val)
            elif field_name == "agent_errors":
                result[api_key] = self._serialize_agent_errors(val)
            elif field_name == "publication_figures":
                result[api_key] = self._serialize_publication_figures(
                    val,
                    lineage_id=(self.restore_branch_id or "main"),
                )
            elif field_name == "publication_tables":
                result[api_key] = self._serialize_publication_tables(val)
            elif field_name == "graphic_abstract":
                result[api_key] = self._serialize_graphic_abstract(val)
            elif field_name == "plan_dialogue_history":
                result[api_key] = self._serialize_dialogue_history(self.plan_dialogue_history)
            elif field_name == "literature_config":
                from backend.utils.settings_provider import AgentSettingsProvider
                result[api_key] = AgentSettingsProvider.mask_literature_config(val)
            elif field_name == "realtime_messages":
                if include_messages:
                    result[api_key] = self._serialize_messages(strip_message_base64)
            else:
                result[api_key] = val

        result["exploration_report_url"] = (
            f"/api/session/{self.id}/exploration-report/html"
            if getattr(self, 'exploration_report_html_path', None) else None
        )
        result["annotation_report_url"] = (
            f"/api/session/{self.id}/annotation-report/html"
            if getattr(self, 'annotation_report_html_path', None) else None
        )
        result["integration_report_url"] = (
            f"/api/session/{self.id}/integration-report/html"
            if getattr(self, 'integration_report_html_path', None) else None
        )
        result["detailed_report_url"] = (
            f"/api/session/{self.id}/detailed-report/html"
            if getattr(self, 'detailed_report_html_path', None) else None
        )
        result["snapshot_count"] = len(self.snapshots)

        return result

    @staticmethod
    def _serialize_treatment_groups(groups):
        return [
            {"id": g["id"], "name": g["name"], "description": g["description"],
             "sampleCount": g["sample_count"], "order": g["order"]}
            for g in groups
        ]

    @staticmethod
    def _serialize_data_files(files):
        return [
            {"id": f["id"], "filename": f["filename"], "filePath": f["file_path"],
             "section": f["section"], "dataType": f["data_type"],
             "groupId": f["group_id"], "sampleNumber": f["sample_number"],
             "description": f["description"], "timestamp": f["timestamp"]}
            for f in files
        ]

    @staticmethod
    def _serialize_biochemistry_indicators(indicators):
        return [
            {"id": r["id"], "indicator": r["indicator"], "value": r["value"], "unit": r["unit"],
             "referenceRange": r["reference_range"], "interpretation": r["interpretation"],
             "groupId": r["group_id"], "sampleNumber": r["sample_number"]}
            for r in indicators
        ]

    @staticmethod
    def _serialize_round_results(results):
        def _simplify_candidates(cms):
            if not cms:
                return []
            # Same 5 fields as CrossModalMechanismCandidate.to_summary_dict().
            # Per-field defaults match the Pydantic model defaults.
            _SUMMARY_DEFAULTS = {
                "theme": "", "mechanism_statement": "",
                "confidence": "low",
                "relation_to_current_direction": "uncovered",
                "report_recommendation": "future_direction",
            }
            return [{k: c.get(k, v) for k, v in _SUMMARY_DEFAULTS.items()}
                    for c in (cms.get("candidates") or [])]

        return [
            {"round": r["round_number"], "pi_summary": r["pi_summary"],
             "phase": r.get("phase", "exploration"),
             "pathology_findings": r["pathology_findings"],
             "pathology_coverage": r.get("pathology_coverage", []),
             "bioinformatics_findings": r["bioinformatics_findings"],
             "biochemistry_findings": r["biochemistry_findings"],
             "literature_findings": r["literature_findings"],
             "next_steps": r["next_steps"], "conflicts": r["conflicts"],
             "cross_modal_candidates": _simplify_candidates(r.get("cross_modal_synthesis")),
             "integration_report_summary": r.get("integration_report_summary", ""),
             "integration_candidates": r.get("integration_candidates", []),
             "round_status": r.get("round_status", "completed"),
             "evidence_status": r.get("evidence_status", "empty"),
             "direction_coverage_eligible": r.get("direction_coverage_eligible"),
             "bioinformatics_evidence_verified": r.get("bioinformatics_evidence_verified"),
             "evidence_gate_degraded": r.get("evidence_gate_degraded", False),
             "evidence_execution_ids": r.get("evidence_execution_ids", []),
             "evidence_notes": r.get("evidence_notes", []),
             "evidence_gate_findings": r.get("evidence_gate_findings", []),
             "lineage_id": r.get("lineage_id", "main"),
             "branch_id": r.get("branch_id"),
             "branch_label": r.get("branch_label"),
             "restored_from_snapshot_id": r.get("restored_from_snapshot_id"),
             "direction": r.get("direction", ""),
             "direction_key": r.get("direction_key", ""),
             "result_identity_key": r.get("result_identity_key", ""),
             "timestamp": r["timestamp"]}
            for r in results
        ]

    @staticmethod
    def _serialize_agent_errors(errors):
        return [
            {"id": e["id"], "agent": e["agent"],
             "error_type": e["error_type"], "message": e["message"],
             "details": e["details"], "round_number": e["round_number"],
             "timestamp": e["timestamp"],
             "resolved": e["resolved"], "resolution": e["resolution"]}
            for e in errors
        ]

    @staticmethod
    def _serialize_publication_figures(figures, *, lineage_id: str = "main"):
        from backend.services.figure_layout import blocking_violations
        from backend.services.report.artifacts import (
            archival_render_available,
            gallery_artifacts,
        )

        # round / layout_blocking / diagnostic are the gallery's badges. The
        # integrated_report WebSocket message carries them for the figures it
        # curated, but a reload rebuilds the gallery from this payload alone —
        # so they have to travel with the record too, or the badges disappear
        # on the first refresh. ``round`` comes from the record's own
        # round_number (source 2 of services/report/figure_attribution.py),
        # which is set for every promoted figure; 0 means unattributed and is
        # reported as such rather than as round zero.
        target = lineage_id or "main"
        gallery_figures = gallery_artifacts([
            figure for figure in figures
            if (figure.get("lineage_id") or "main") == target
        ])
        return [
            {"id": f["id"], "name": f["name"], "description": f["description"],
             "figure_type": f["figure_type"], "png_path": f["png_path"],
             "pdf_path": f["pdf_path"], "png_base64": f["png_base64"],
             "review_status": f["review_status"],
             "pi_feedback": f["pi_feedback"], "revision_count": f["revision_count"],
             "round": f.get("round_number") or None,
             "plan_step_id": f.get("plan_step_id"),
             "plan_id": f.get("plan_id"),
             "purpose": f.get("purpose", "result"),
             "logical_figure_id": f.get("logical_figure_id"),
             "render_attempt_id": f.get("render_attempt_id"),
             "producer_status": f.get("producer_status", "published"),
             "revision_of_id": f.get("revision_of_id"),
             "supersedes_id": f.get("supersedes_id"),
             "is_published_head": bool(f.get("is_published_head", True)),
             "is_publication": True,
             "layout_blocking": bool(blocking_violations(f.get("layout_violations"))),
             "diagnostic": bool(f.get("diagnostic")),
             "created_at": f["created_at"]}
            for f in gallery_figures
            if archival_render_available(f)
        ]

    @staticmethod
    def _serialize_publication_tables(tables):
        return [
            {"id": t["id"], "name": t["name"], "description": t["description"],
             "csv_path": t["csv_path"], "columns": t["columns"],
             "preview_data": t["preview_data"],
             "review_status": t["review_status"],
             "pi_feedback": t["pi_feedback"], "created_at": t["created_at"]}
            for t in tables
        ]

    @staticmethod
    def _serialize_graphic_abstract(ga):
        if not ga:
            return None
        return {
            "id": ga["id"], "prompt": ga["prompt"],
            "image_base64": ga["image_base64"],
            "text_diagram": ga["text_diagram"],
            "is_text_mode": bool(ga.get("text_diagram") and not ga.get("image_base64")),
            "created_at": ga["created_at"],
        }

    def _serialize_messages(self, strip_message_base64):
        from backend.services.report.artifacts import (
            archival_render_available,
            eligible_artifacts,
        )

        current_figures = [
            figure for figure in eligible_artifacts(self.publication_figures)
            if (getattr(figure, "lineage_id", None) or "main")
            == (self.restore_branch_id or "main")
            and archival_render_available(figure)
        ]
        current_ids = {str(figure.id) for figure in current_figures}
        current_names = {str(figure.name) for figure in current_figures}

        def _registered_figure_is_current(raw) -> bool:
            if not isinstance(raw, dict):
                return True
            if raw.get("is_publication") is False:
                return True
            figure_id = str(raw.get("id") or raw.get("figure_id") or "")
            name = str(raw.get("name") or raw.get("figure_name") or "")
            if figure_id:
                return figure_id in current_ids
            return bool(name and name in current_names)

        serialized = []
        for msg in self.realtime_messages:
            content = (
                strip_message_base64(msg.type, msg.content)
                if strip_message_base64 else msg.content
            )
            if msg.type in {"figure_review", "figure_approved", "figure_revision"}:
                nested = content.get("figure") if isinstance(content, dict) else None
                if nested is not None and not _registered_figure_is_current(nested):
                    continue
            elif msg.type == "integrated_report" and isinstance(content, dict):
                content = dict(content)
                content["figures"] = [
                    raw for raw in (content.get("figures") or [])
                    if _registered_figure_is_current(raw)
                ]
            serialized.append({
                "id": msg.id,
                "type": msg.type,
                "agent": msg.agent.value if msg.agent else None,
                "content": content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else "",
            })
        return serialized
