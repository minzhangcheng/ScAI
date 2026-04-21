"""
Data models for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid
import asyncio

# Sequencing data type constants
DATA_TYPE_SCRNASEQ = "scRNA-seq"
DATA_TYPE_BULK_RNASEQ = "bulk RNA-seq"

# Module-level lock for async-safe state version increments
_version_lock = asyncio.Lock()

class AgentType(str, Enum):
    PI = "pi"
    PATHOLOGY = "pathology"
    BIOINFORMATICS = "bioinformatics"
    BIOCHEMISTRY = "biochemistry"
    LITERATURE = "literature"
    PAINTER = "painter"
    SUPERVISOR = "supervisor"
    TECHNIQUE = "technique"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_INPUT = "waiting_input"
    WAITING_ANNOTATION = "waiting_annotation"  # Paused for user annotation confirmation
    PAUSED = "paused"
    GENERATING_REPORT = "generating_report"

class AnalysisPhase(str, Enum):
    """Two-phase workflow: exploration (Round 1) then innovation (Round 2+)"""
    EXPLORATION = "exploration"
    INNOVATION = "innovation"

class FigureReviewStatus(str, Enum):
    """Status of figure review by PI Agent"""
    PENDING = "pending"
    APPROVED = "approved"
    ACCEPTED_WITH_RESERVATIONS = "accepted_with_reservations"
    REVISION_REQUESTED = "revision_requested"


class AgentError(BaseModel):
    """Structured agent error record for tracking analysis failures"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: AgentType
    error_type: str  # "execution_error", "timeout", "data_missing", "llm_error"
    message: str
    details: Optional[str] = None
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None


class AgentMessage(BaseModel):
    """Message from/to an agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: AgentType
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class CodeExecution(BaseModel):
    """Record of code execution"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    code: str
    output: Optional[str] = None
    error: Optional[str] = None
    figures: List[str] = Field(default_factory=list)  # Base64 encoded images
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class AnalysisStep(BaseModel):
    """A single step in the bioinformatics agent analysis log, covering all tool call types"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    step_type: str  # "execute_code" | "consult_pi" | "search_literature" | "annotate_clusters" | "report_findings" | "generate_publication_figure" | "text_response"
    timestamp: datetime = Field(default_factory=datetime.now)

    # Agent reasoning (LLM response.content, truncated to 2000 chars)
    agent_reasoning: Optional[str] = None

    # execute_code fields
    code: Optional[str] = None
    code_output: Optional[str] = None
    code_error: Optional[str] = None
    code_description: Optional[str] = None
    code_step_type: Optional[str] = None  # "technical" | "biological" (from tool arg step_type)
    saved_figure_filenames: List[str] = Field(default_factory=list)  # disk PNG filenames, no base64

    # consult_pi fields
    pi_question: Optional[str] = None
    pi_context: Optional[str] = None
    pi_options: List[str] = Field(default_factory=list)
    pi_response: Optional[str] = None

    # search_literature fields
    literature_topic: Optional[str] = None
    literature_result: Optional[str] = None

    # annotate_clusters fields
    cluster_markers_summary: Optional[str] = None
    annotation_tissue: Optional[str] = None
    annotation_organism: Optional[str] = None
    annotation_result_summary: Optional[str] = None

    # report_findings fields
    findings: List[str] = Field(default_factory=list)
    findings_summary: Optional[str] = None
    findings_suggestions: List[str] = Field(default_factory=list)

    # generate_publication_figure fields
    pub_figure_name: Optional[str] = None
    pub_figure_description: Optional[str] = None
    pub_figure_type: Optional[str] = None
    pub_figure_code: Optional[str] = None
    pub_figure_result: Optional[str] = None

    # text_response fields
    text_content: Optional[str] = None

class ScoreDefinition(BaseModel):
    """Definition of a computed score for methodology transparency in the final report"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score_name: str          # e.g., "Fibrosis_Score", "EMT_signature"
    method: str              # e.g., "sc.tl.score_genes", "gseapy.ssgsea"
    description: str         # What the score measures biologically
    parameters: Dict[str, Any] = Field(default_factory=dict)
    gene_list: List[str] = Field(default_factory=list)
    gene_list_source: str = ""  # e.g., "MSigDB Hallmark EMT"
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

class BioinformaticsCheckpoint(BaseModel):
    """Checkpoint for bioinformatics analysis progress - enables step-level recovery"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    step_number: int
    total_steps: int = 80

    # Conversation state for LLM context restoration
    messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Analysis progress
    findings: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)  # Figure IDs or paths
    code_history: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_log: List[Dict[str, Any]] = Field(default_factory=list)  # Serialized AnalysisStep dicts via model_dump()
    analysis_context: Dict[str, Any] = Field(default_factory=dict)

    # Resumption info
    last_tool_call: Optional[Dict[str, Any]] = None
    completed: bool = False

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class TreatmentGroup(BaseModel):
    """A treatment/experimental group definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str                          # e.g., "Control", "NASH Model", "Treatment A"
    description: str = ""              # Detailed description of what this group represents
    sample_count: int = 1              # Number of samples/replicates in this group
    order: int = 0                     # Display order

class DataFile(BaseModel):
    """Universal file record for all uploaded data (pathology, sequencing, biochemistry)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str                     # Absolute path on disk

    # Section: which data section this belongs to
    section: Literal["pathology", "sequencing", "biochemistry"]

    # Data type within the section (see DATA_TYPE_* constants for sequencing types)
    # pathology: "H&E", "Sirius Red", "Perls Blue", "Oil Red O", "PAS", or custom
    # sequencing: DATA_TYPE_SCRNASEQ, DATA_TYPE_BULK_RNASEQ, "loom"
    # biochemistry: "image", "file"
    data_type: str

    # Group and sample assignment
    group_id: Optional[str] = None     # TreatmentGroup.id, or None for "No group distinction"
    sample_number: Optional[int] = None  # 1-based sample number, or None for "No sample distinction"

    # Section-specific metadata
    tiles: List[str] = Field(default_factory=list)       # For pathology: tile paths
    analysis: Optional[str] = None                        # For pathology: analysis result
    description: Optional[str] = None                     # User-provided description
    extracted_data: Optional[Dict[str, Any]] = None       # For biochemistry: extracted values

    timestamp: datetime = Field(default_factory=datetime.now)

class BiochemistryIndicator(BaseModel):
    """Manual biochemistry entry (indicator name + value + unit) with group/sample association"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    indicator: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None

    # Group and sample
    group_id: Optional[str] = None
    sample_number: Optional[int] = None

    timestamp: datetime = Field(default_factory=datetime.now)

class LiteratureReport(BaseModel):
    """Literature research report"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    report: str
    references: List[Dict[str, str]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

class ClusterAnnotation(BaseModel):
    """Single cluster annotation for user review during exploration phase"""
    cluster_id: str
    cell_type: str
    subtypes: List[str] = Field(default_factory=list)
    key_markers: List[str] = Field(default_factory=list)
    confidence: str = "low"  # "high" / "medium" / "low"
    evidence: str = ""
    n_cells: int = 0  # Number of cells in this cluster
    alternative_cell_types: List[str] = Field(default_factory=list)  # Other possible cell types
    low_confidence_reason: str = ""  # Why confidence is low/medium
    user_confirmed: bool = False
    user_override: Optional[str] = None  # User-provided cell type override

class AnnotationConfirmationState(BaseModel):
    """State for cell annotation confirmation pause during exploration phase"""
    cluster_annotations: List[ClusterAnnotation] = Field(default_factory=list)
    tissue: str = ""
    organism: str = ""
    tissue_summary: str = ""
    analysis_summary: str = ""
    pending: bool = False
    confirmed: bool = False

class DetailedReportSegment(BaseModel):
    """One segment of the incrementally-built detailed report at each HITL checkpoint"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phase: str  # "pre_annotation" | "exploration_complete" | "supplementary_annotation_N" | "supplementary_direction_N"
    title: str  # Human-readable, e.g. "Pre-Annotation Analysis Summary"
    content_html: str = ""  # Pre-rendered HTML
    timestamp: datetime = Field(default_factory=datetime.now)
    round_number: int = 0

class Snapshot(BaseModel):
    """A saved snapshot of analysis state for debugging rollback"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    phase: str  # "pre_annotation" | "pre_direction" (auto-created at key analysis points)
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    state_file: str = ""  # Filename within snapshots/ directory
    size_bytes: int = 0
    parent_id: Optional[str] = None  # Parent snapshot ID for tree view

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
    # Each candidate: {"direction": str, "rationale": str, "expected_innovation": str, "expected_novelty": str, "data_reliability": str}

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
    agents_for_next_round: List[str] = Field(
        default_factory=lambda: ["bioinformatics", "pathology", "biochemistry", "literature"]
    )
    reanalysis_agents: List[str] = Field(default_factory=list)  # Agents needing reanalysis of existing data
    phase: str = "exploration"  # "exploration" or "innovation"
    direction: str = ""  # Research direction for innovation rounds
    exploration_report: Optional[ExplorationReport] = None  # Only for Phase 1
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

class AnalysisState(BaseModel):
    """Complete state of an analysis session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Analysis"
    user_id: str = Field(..., min_length=1)  # Owner user ID (required, non-empty)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # State version control for synchronization
    state_version: int = 0
    last_sync_timestamp: datetime = Field(default_factory=datetime.now)

    # Status
    status: TaskStatus = TaskStatus.PENDING
    current_round: int = 0
    max_rounds: int = 3  # User-configurable maximum rounds
    round_wait_seconds: int = 300  # Countdown seconds between rounds (-1=manual, 0=no wait)

    # Two-phase workflow state
    analysis_phase: AnalysisPhase = AnalysisPhase.EXPLORATION
    annotation_state: Optional[AnnotationConfirmationState] = None
    exploration_report: Optional[ExplorationReport] = None
    user_chosen_direction: Optional[str] = None  # User's chosen direction for innovation phase (backward compat)
    current_round_direction: str = ""  # Active direction for current innovation round (preserved after user_chosen_direction is consumed)

    # Multi-direction workflow
    user_chosen_directions: List[Dict[str, str]] = Field(default_factory=list)
    # Each dict: {"direction": str, "notes": str}
    directions_queue: List[Dict[str, str]] = Field(default_factory=list)
    # Remaining directions to process (popped one by one)
    multi_direction_correlation: Optional[str] = None
    # Correlation analysis result for multi-direction mode
    exploration_report_html_path: Optional[str] = None
    # Path to exploration report HTML file
    annotation_report_html_path: Optional[str] = None
    # Path to annotation context report HTML file

    # Incremental detailed report segments (built at each HITL point)
    detailed_report_segments: List[DetailedReportSegment] = Field(default_factory=list)
    detailed_report_html_path: Optional[str] = None

    # Snapshots for debugging rollback
    snapshots: List[Snapshot] = Field(default_factory=list)
    last_restored_snapshot_id: Optional[str] = None  # Tracks which snapshot was last restored (for parent chain)

    # Initial inputs
    initial_instruction: str = ""
    
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
    
    # Round results
    round_results: List[RoundResult] = Field(default_factory=list)
    
    # Final summary
    final_summary: Optional[str] = None

    # Publication-quality outputs
    publication_figures: List[PublicationFigure] = Field(default_factory=list)
    publication_tables: List[PublicationTable] = Field(default_factory=list)
    graphic_abstract: Optional[GraphicAbstract] = None
    integrated_report: Optional[str] = None  # Integrated report with figure/table references

    # Score definitions (methodology tracking for report transparency)
    score_definitions: List[ScoreDefinition] = Field(default_factory=list)

    # Agent prompts (can be customized per session)
    custom_prompts: Dict[str, str] = Field(default_factory=dict)
    
    # Agent models (can be customized per session)
    custom_models: Dict[str, str] = Field(default_factory=dict)
    
    # Agent temperatures (can be customized per session)
    custom_temperatures: Dict[str, float] = Field(default_factory=dict)

    # Per-agent LLM server overrides (agent_name -> {"llm_base_url": ..., "api_key": ...})
    agent_server_configs: Dict[str, Dict[str, str]] = Field(default_factory=dict)

    # Agent vision capability settings (agent_name -> bool, True = supports vision)
    agent_vision_settings: Dict[str, bool] = Field(default_factory=dict)

    # Literature agent configuration
    literature_config: Dict[str, Any] = Field(default_factory=lambda: {
        "mode": "deep_research",              # deep_research | gemini_deep_research | pubmed_llm
        "deep_research_model": "",           # e.g., "openai/o4-mini-deep-research"
        "deep_research_llm_base_url": "",    # Separate LLM server for deep research (optional)
        "deep_research_api_key": "",         # Separate API key for deep research (optional)
        "google_api_key": "",                # For Gemini Deep Research mode
        "pubmed_email": "",                  # For PubMed mode (NCBI requirement)
        "max_search_rounds": 5,              # Max iterative search rounds
    })
    
    # Chat messages (PI Agent free-form chat, separate from workflow)
    chat_messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Sandbox state (for potential restoration)
    sandbox_state: Optional[Dict[str, Any]] = None
    
    # Config snapshot
    config_snapshot: Optional[Dict[str, Any]] = None
    
    # Human intervention support
    pending_human_input: Optional[str] = None  # Human message waiting to be processed
    pending_uploaded_files: List[Dict[str, Any]] = Field(default_factory=list)  # Files uploaded with human input
    human_input_requested: bool = False  # Whether system is waiting for human input
    human_messages: List[Dict[str, Any]] = Field(default_factory=list)  # History of human interventions

    # Error tracking for parallel analysis
    agent_errors: List[AgentError] = Field(default_factory=list)

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

    def to_api_dict(self, *, strip_message_base64=None) -> dict:
        """Serialize state for the load_session API response.

        Centralizes the snake_case → camelCase mapping and enum/datetime
        conversions that the frontend expects, so the route handler is a
        one-liner and field additions only need updating here.

        Args:
            strip_message_base64: Optional callable(msg_type, content) -> content
                to strip large base64 data from messages before sending.
        """
        def _dt(v) -> str:
            return v.isoformat() if v else ""

        def _enum(v) -> str:
            return v.value if hasattr(v, 'value') else str(v)

        result = {
            "session_id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "status": _enum(self.status),
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "round_wait_seconds": self.round_wait_seconds,
            "created_at": _dt(self.created_at),
            "updated_at": _dt(self.updated_at),
            "initial_instruction": self.initial_instruction,
            "treatment_groups": [
                {"id": g.id, "name": g.name, "description": g.description,
                 "sampleCount": g.sample_count, "order": g.order}
                for g in self.treatment_groups
            ],
            "data_files": [
                {"id": f.id, "filename": f.filename, "filePath": f.file_path,
                 "section": f.section, "dataType": f.data_type,
                 "groupId": f.group_id, "sampleNumber": f.sample_number,
                 "description": f.description, "timestamp": _dt(f.timestamp)}
                for f in self.data_files
            ],
            "biochemistry_indicators": [
                {"id": r.id, "indicator": r.indicator, "value": r.value, "unit": r.unit,
                 "referenceRange": r.reference_range, "interpretation": r.interpretation,
                 "groupId": r.group_id, "sampleNumber": r.sample_number}
                for r in self.biochemistry_indicators
            ],
            "round_results": [
                {"round": r.round_number, "pi_summary": r.pi_summary,
                 "pathology_findings": r.pathology_findings,
                 "bioinformatics_findings": r.bioinformatics_findings,
                 "biochemistry_findings": r.biochemistry_findings,
                 "literature_findings": r.literature_findings,
                 "next_steps": r.next_steps, "conflicts": r.conflicts,
                 "timestamp": _dt(r.timestamp)}
                for r in self.round_results
            ],
            "final_summary": self.final_summary,
            "exploration_report_url": (
                f"/api/session/{self.id}/exploration-report/html"
                if getattr(self, 'exploration_report_html_path', None) else None
            ),
            "annotation_report_url": (
                f"/api/session/{self.id}/annotation-report/html"
                if getattr(self, 'annotation_report_html_path', None) else None
            ),
            "analysis_phase": self.analysis_phase,
            "messages": [
                {"id": msg.id, "type": msg.type,
                 "agent": _enum(msg.agent) if msg.agent else None,
                 "content": (strip_message_base64(msg.type, msg.content)
                             if strip_message_base64 else msg.content),
                 "timestamp": _dt(msg.timestamp)}
                for msg in self.realtime_messages
            ],
            "custom_prompts": self.custom_prompts,
            "custom_models": self.custom_models,
            "custom_temperatures": self.custom_temperatures,
            "state_version": self.state_version,
            "last_sync_timestamp": _dt(self.last_sync_timestamp),
            "agent_errors": [
                {"id": err.id, "agent": _enum(err.agent),
                 "error_type": err.error_type, "message": err.message,
                 "details": err.details, "round_number": err.round_number,
                 "timestamp": _dt(err.timestamp),
                 "resolved": err.resolved, "resolution": err.resolution}
                for err in self.agent_errors
            ],
            "publication_figures": [
                {"id": fig.id, "name": fig.name, "description": fig.description,
                 "figure_type": fig.figure_type, "png_path": fig.png_path,
                 "pdf_path": fig.pdf_path, "png_base64": fig.png_base64,
                 "review_status": _enum(fig.review_status) if fig.review_status else None,
                 "pi_feedback": fig.pi_feedback, "revision_count": fig.revision_count,
                 "created_at": _dt(fig.created_at)}
                for fig in self.publication_figures
            ],
            "publication_tables": [
                {"id": tbl.id, "name": tbl.name, "description": tbl.description,
                 "csv_path": tbl.csv_path, "columns": tbl.columns,
                 "preview_data": tbl.preview_data,
                 "review_status": _enum(tbl.review_status) if tbl.review_status else None,
                 "pi_feedback": tbl.pi_feedback, "created_at": _dt(tbl.created_at)}
                for tbl in self.publication_tables
            ],
            "graphic_abstract": (
                {"id": self.graphic_abstract.id, "prompt": self.graphic_abstract.prompt,
                 "image_base64": self.graphic_abstract.image_base64,
                 "text_diagram": self.graphic_abstract.text_diagram,
                 "is_text_mode": self.graphic_abstract.is_text_mode,
                 "created_at": _dt(self.graphic_abstract.created_at)}
                if self.graphic_abstract else None
            ),
            "integrated_report": self.integrated_report,
        }
        return result

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
        "supervisor_feedback", "supervisor_checkpoint", "supervisor_summary"
    ]
    agent: Optional[AgentType] = None
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)
