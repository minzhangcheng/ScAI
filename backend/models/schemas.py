"""
Data models for the Multi-Agent Liver Analysis System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid

class AgentType(str, Enum):
    PI = "pi"
    PATHOLOGY = "pathology"
    BIOINFORMATICS = "bioinformatics"
    BIOCHEMISTRY = "biochemistry"
    LITERATURE = "literature"

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
    timestamp: datetime = Field(default_factory=datetime.now)

class AnalysisState(BaseModel):
    """Complete state of an analysis session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Analysis"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    current_round: int = 0
    max_rounds: int = 3  # User-configurable maximum rounds
    
    # Initial inputs
    initial_instruction: str = ""
    
    # Agent messages history
    messages: List[AgentMessage] = Field(default_factory=list)
    
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
    
    # Agent prompts (can be customized per session)
    custom_prompts: Dict[str, str] = Field(default_factory=dict)
    
    # Agent models (can be customized per session)
    custom_models: Dict[str, str] = Field(default_factory=dict)
    
    # Agent temperatures (can be customized per session)
    custom_temperatures: Dict[str, float] = Field(default_factory=dict)
    
    # Sandbox state (for potential restoration)
    sandbox_state: Optional[Dict[str, Any]] = None
    
    # Config snapshot
    config_snapshot: Optional[Dict[str, Any]] = None
    
    # Human intervention support
    pending_human_input: Optional[str] = None  # Human message waiting to be processed
    pending_uploaded_files: List[Dict[str, Any]] = Field(default_factory=list)  # Files uploaded with human input
    human_input_requested: bool = False  # Whether system is waiting for human input
    human_messages: List[Dict[str, Any]] = Field(default_factory=list)  # History of human interventions
    
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
    type: Literal[
        "status", "message", "code", "result", "error", "figure", "progress", 
        "step", "prompt", "response", "thinking", "tool_call",
        # New types for autonomous bioinformatics agent
        "literature", "pi_question", "pi_response", "findings",
        # Human intervention
        "human_input_received", "human_input_requested",
        # Round management
        "round_summary", "data_request"
    ]
    agent: Optional[AgentType] = None
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)
