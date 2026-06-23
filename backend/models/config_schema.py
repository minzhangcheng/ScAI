"""Configuration JSON schema v1.0 for ScAI export/import."""
from typing import Literal, Optional, List, Dict, get_args
from pydantic import BaseModel, Field, model_validator


AgentName = Literal[
    "pi", "pathology", "bioinformatics", "biochemistry",
    "literature", "painter", "supervisor", "technique",
]

AGENT_NAMES = list(get_args(AgentName))
AGENT_NAME_SET = set(AGENT_NAMES)


class AnalysisSection(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    initial_instruction: str
    round_wait_seconds: int = Field(ge=-1, le=3600)


class DefaultLlm(BaseModel):
    llm_base_url: str
    api_key: str
    model: str = ""
    proxy_url: Optional[str] = None


class LiteratureConfig(BaseModel):
    mode: Literal["pubmed_llm", "deep_research", "gemini_deep_research", "disabled"]
    deep_research_model: Optional[str] = None
    deep_research_llm_base_url: Optional[str] = None
    deep_research_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    pubmed_email: Optional[str] = None
    max_search_rounds: int = Field(ge=1, le=50, default=10)


class ApiConfigSection(BaseModel):
    default_llm: DefaultLlm
    literature: LiteratureConfig


class GroupEntry(BaseModel):
    key: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    sample_count: int = Field(ge=1, le=100)
    order: int = 0


class FileEntry(BaseModel):
    local_path: str
    section: Literal["pathology", "sequencing", "biochemistry"]
    data_type: str
    group: Optional[str] = None
    sample_number: Optional[int] = Field(default=None, ge=1)
    description: str = ""
    filename: Optional[str] = None  # Set on export, optional on import


class IndicatorEntry(BaseModel):
    indicator: str = Field(min_length=1, max_length=200)
    value: float
    unit: str = Field(min_length=1, max_length=50)
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None
    group: Optional[str] = None
    sample_number: Optional[int] = Field(default=None, ge=1)


class DataSection(BaseModel):
    groups: List[GroupEntry] = []
    files: List[FileEntry] = []
    indicators: List[IndicatorEntry] = []


class AgentOverride(BaseModel):
    model: Optional[str] = None
    llm_base_url: Optional[str] = None
    api_key: Optional[str] = None
    custom_prompt: Optional[str] = None
    temperature: Optional[float] = None
    vision: Optional[bool] = None


class ConfigV1(BaseModel):
    """Top-level configuration schema v1.0."""
    schema_version: Literal["1.0"]
    analysis: AnalysisSection
    api_config: ApiConfigSection
    data: DataSection = Field(default_factory=DataSection)
    agents: Dict[str, AgentOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_agent_names(self) -> "ConfigV1":
        for name in self.agents:
            if name not in AGENT_NAME_SET:
                raise ValueError(f"Unknown agent name: {name!r}")
        return self

    @model_validator(mode="after")
    def validate_file_group_refs(self) -> "ConfigV1":
        group_keys = {g.key for g in self.data.groups}
        for f in self.data.files:
            if f.group and f.group not in group_keys:
                raise ValueError(
                    f"File {f.local_path!r} references unknown group {f.group!r}"
                )
        for ind in self.data.indicators:
            if ind.group and ind.group not in group_keys:
                raise ValueError(
                    f"Indicator {ind.indicator!r} references unknown group {ind.group!r}"
                )
        return self
