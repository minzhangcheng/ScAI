"""Persisted PI-authored round chapters, independent of report filenames."""
from typing import Literal

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
