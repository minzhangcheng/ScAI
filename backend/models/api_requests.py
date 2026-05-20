"""Pydantic request bodies used across API routes.

Extracted verbatim from backend/main.py to decouple route handlers from
the FastAPI app module.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from backend.models.schemas import DEFAULT_ROUND_WAIT_SECONDS


VALID_AGENTS = Literal["pi", "pathology", "bioinformatics", "biochemistry", "literature", "painter", "supervisor", "technique"]


class ConfigRequest(BaseModel):
    api_key: str  # Main LLM API key
    llm_base_url: str = "https://openrouter.ai/api/v1"
    proxy_url: Optional[str] = None
    google_api_key: str = ""  # Deprecated: kept for backward compat, now in literature_config


class ValidateKeyRequest(BaseModel):
    api_key: str
    llm_base_url: str = "https://openrouter.ai/api/v1"
    proxy_url: Optional[str] = None


class StartRequest(BaseModel):
    instruction: str
    session_name: Optional[str] = None
    round_wait_seconds: int = Field(ge=-1, le=3600, default=DEFAULT_ROUND_WAIT_SECONDS)


class PromptUpdateRequest(BaseModel):
    agent: VALID_AGENTS
    prompt: str = Field(max_length=50000)


class ModelUpdateRequest(BaseModel):
    agent: VALID_AGENTS
    model: str


class TemperatureUpdateRequest(BaseModel):
    agent: VALID_AGENTS
    temperature: float = Field(ge=0.0, le=2.0)


class VisionSettingUpdateRequest(BaseModel):
    agent: VALID_AGENTS
    vision_enabled: bool


class LiteratureConfigRequest(BaseModel):
    mode: Literal["deep_research", "gemini_deep_research", "pubmed_llm", "disabled"] = Field(default="deep_research")
    deep_research_model: str = ""
    deep_research_llm_base_url: str = ""
    deep_research_api_key: str = ""
    google_api_key: str = ""
    pubmed_email: str = ""
    max_search_rounds: int = Field(default=5, ge=1, le=20)


class AgentServerConfigRequest(BaseModel):
    agent: Literal["pi", "pathology", "bioinformatics", "biochemistry", "literature", "painter", "supervisor", "technique"]
    llm_base_url: str = ""
    api_key: str = ""


class AgentConfigUpdateRequest(BaseModel):
    """Unified request for updating a single agent config field.

    Only one value field should be provided per request.
    The `field` parameter specifies which field to update.
    """
    agent: VALID_AGENTS
    field: Literal["prompt", "model", "temperature", "vision_enabled"]
    value: Union[str, float, bool]


class BiochemistryInput(BaseModel):
    indicator: str = Field(..., min_length=1, max_length=200, description="Biochemistry indicator name")
    value: float = Field(..., description="Indicator value")
    unit: str = Field(..., max_length=50, description="Measurement unit")
    group_id: Optional[str] = None
    sample_number: Optional[int] = None


class SessionUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    instruction: Optional[str] = None


class GroupCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    sample_count: int = Field(1, ge=1, le=100)


class GroupUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    sample_count: Optional[int] = Field(None, ge=1, le=100)


class StopAnalysisRequest(BaseModel):
    """Request to stop an analysis"""
    action: Literal["immediate", "summarize"]


class RetryNodeRequest(BaseModel):
    """Request to retry or skip a failed node"""
    node_name: str
    skip: bool = False  # If True, skip instead of retry


class UploadedFileInfo(BaseModel):
    """Info about an uploaded file"""
    type: str  # 'image' or 'file'
    filename: str
    id: Optional[str] = None


class HumanInputRequest(BaseModel):
    """Request for human intervention"""
    message: str
    target_agent: str = "pi"  # Which agent to send the message to
    uploaded_files: List[UploadedFileInfo] = Field(default_factory=list)
    clear_queue: bool = False  # Clear remaining direction queue (multi-direction mode)


class AnnotationConfirmation(BaseModel):
    """Request body for confirming cluster annotations"""
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    # Each item: {"cluster_id": str, "user_override": Optional[str], "user_confirmed": bool}


class SupplementaryAnalysisRequest(BaseModel):
    """Request body for supplementary analysis at HITL points"""
    request_text: str = Field(..., min_length=1, max_length=2000)
    target: str = Field(..., pattern=r"^(annotation|direction)$")


class DirectionChoice(BaseModel):
    """Request body for choosing innovation phase direction"""
    direction: str = ""  # backward compat: single direction
    let_pi_decide: bool = False
    # Multi-direction support
    directions: List[Dict[str, str]] = Field(default_factory=list)
    # Each dict: {"direction": str, "notes": str}


class LlmFailureActionRequest(BaseModel):
    """Request to handle an LLM timeout/failure after user intervention."""
    action: Literal["retry", "skip", "terminate"]
    node_name: str
    generation: int  # monotonic counter from frontend for idempotency


class PlanReviewRequest(BaseModel):
    """Request body for plan review actions."""
    action: Literal["approve", "skip", "modify"] = "skip"
    user_input: str = Field(default="", max_length=5000)


