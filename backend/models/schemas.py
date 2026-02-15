"""
Data models for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid
import threading

# Module-level lock for thread-safe state version increments
_version_lock = threading.RLock()

class AgentType(str, Enum):
    PI = "pi"
    PATHOLOGY = "pathology"
    BIOINFORMATICS = "bioinformatics"
    BIOCHEMISTRY = "biochemistry"
    LITERATURE = "literature"
    PAINTER = "painter"

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
    PAUSED = "paused"
    GENERATING_REPORT = "generating_report"

class FigureReviewStatus(str, Enum):
    """Status of figure review by PI Agent"""
    PENDING = "pending"
    APPROVED = "approved"
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
    analysis_context: Dict[str, Any] = Field(default_factory=dict)

    # Resumption info
    last_tool_call: Optional[Dict[str, Any]] = None
    completed: bool = False

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class PathologyImage(BaseModel):
    """Pathology image and analysis"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    image_type: str  # H&E, Sirius Red, Perls Blue, etc.
    original_path: str
    tiles: List[str] = Field(default_factory=list)  # Paths to tiles
    analysis: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class BiochemistryResult(BaseModel):
    """Biochemistry experiment result"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    indicator: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class BiochemistryFile(BaseModel):
    """Biochemistry file (image or data file)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str  # 'image' or 'file' (csv, xlsx, txt, etc.)
    file_path: str
    description: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None  # Extracted values from file/image
    timestamp: datetime = Field(default_factory=datetime.now)

class LiteratureReport(BaseModel):
    """Literature research report"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    report: str
    references: List[Dict[str, str]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

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
    review_status: FigureReviewStatus = FigureReviewStatus.PENDING
    pi_feedback: Optional[str] = None
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
    review_status: FigureReviewStatus = FigureReviewStatus.PENDING
    pi_feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class GraphicAbstract(BaseModel):
    """Graphic abstract generated by Painter Agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str  # The prompt used to generate the image
    image_base64: str  # Base64 encoded image
    created_at: datetime = Field(default_factory=datetime.now)

class AnalysisState(BaseModel):
    """Complete state of an analysis session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Analysis"
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
    
    # Initial inputs
    initial_instruction: str = ""
    
    # Agent messages history
    messages: List[AgentMessage] = Field(default_factory=list)

    # Realtime messages for WebSocket (persisted for session restoration)
    realtime_messages: List["WebSocketMessage"] = Field(default_factory=list)
    
    # Code execution history (for reproducibility)
    code_history: List[CodeExecution] = Field(default_factory=list)
    
    # Uploaded data
    pathology_images: List[PathologyImage] = Field(default_factory=list)
    biochemistry_results: List[BiochemistryResult] = Field(default_factory=list)
    biochemistry_files: List[BiochemistryFile] = Field(default_factory=list)  # Uploaded files/images
    control_seq_path: Optional[str] = None     # Path to control group scRNA-seq data (.h5ad)
    treatment_seq_path: Optional[str] = None   # Path to treatment group scRNA-seq data (.h5ad)
    control_loom_path: Optional[str] = None    # Path to control group loom file (for RNA velocity)
    treatment_loom_path: Optional[str] = None  # Path to treatment group loom file (for RNA velocity)
    
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

    # Agent prompts (can be customized per session)
    custom_prompts: Dict[str, str] = Field(default_factory=dict)
    
    # Agent models (can be customized per session)
    custom_models: Dict[str, str] = Field(default_factory=dict)
    
    # Agent temperatures (can be customized per session)
    custom_temperatures: Dict[str, float] = Field(default_factory=dict)

    # Literature agent configuration
    literature_config: Dict[str, Any] = Field(default_factory=lambda: {
        "mode": "pubmed_llm",                # gemini_deep_research | pubmed_llm
        "google_api_key": "",                 # For Gemini Deep Research mode
        "pubmed_email": "",                   # For PubMed mode (NCBI requirement)
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

    def increment_version(self):
        """Atomically increment state version for synchronization

        Uses a module-level lock to ensure thread-safe increment in concurrent environments.
        """
        with _version_lock:
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
    upload_type: Literal["pathology", "biochemistry", "snrna_seq"]
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
        "annotation_result",
        # Chat delegation
        "chat_delegation", "chat_result", "chat_error"
    ]
    agent: Optional[AgentType] = None
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)
