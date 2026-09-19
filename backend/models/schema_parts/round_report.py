"""Persisted PI-authored round chapters, independent of report filenames."""
from typing import Literal
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


class RoundReportArtifact(BaseModel):
    artifact_id: str
    kind: Literal["figure", "table"]
    filename: str
    caption: str = ""


class RoundReportBlock(BaseModel):
    text: str
    artifact_ids: list[str] = Field(default_factory=list)


class RoundReportChapter(BaseModel):
    title: str
    blocks: list[RoundReportBlock] = Field(default_factory=list)
    conclusions: str = ""
    artifacts: list[RoundReportArtifact] = Field(default_factory=list)
    assembly_notes: list[str] = Field(default_factory=list)
    source: Literal["pi", "fallback"] = "pi"


class StageReviewReport(BaseModel):
    """A frozen decision-time export, independent of the final round chapter."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=datetime.now)
    session_id: str
    lineage_id: str = "main"
    stage: Literal["stage1", "stage2"]
    revision: int
    status: Literal["ready", "partial", "failed"] = "failed"
    chapter: RoundReportChapter | None = None
    html: str = ""
    error: str = ""
