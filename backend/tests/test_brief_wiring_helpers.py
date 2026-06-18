"""Workflow helpers that fetch PI briefs with try/except + empty→None."""
import pytest
from unittest.mock import AsyncMock
from backend.workflow.annotation import AnnotationMixin
from backend.workflow.agents import AgentNodesMixin


def _annotation_obj():
    obj = AnnotationMixin.__new__(AnnotationMixin)
    obj.session_id = "test-session"
    obj.pi_agent = AsyncMock()
    return obj


MARKERS = {"0": ["Cd3d", "Cd3e"]}


@pytest.mark.asyncio
async def test_annotation_brief_helper_returns_brief():
    obj = _annotation_obj()
    obj.pi_agent.generate_annotation_research_brief = AsyncMock(return_value="A BRIEF")
    out = await obj._generate_annotation_brief(MARKERS, "mouse", "liver", state=None)
    assert out == "A BRIEF"


@pytest.mark.asyncio
async def test_annotation_brief_helper_none_on_empty():
    obj = _annotation_obj()
    obj.pi_agent.generate_annotation_research_brief = AsyncMock(return_value="")
    out = await obj._generate_annotation_brief(MARKERS, "mouse", "liver", state=None)
    assert out is None


@pytest.mark.asyncio
async def test_annotation_brief_helper_none_on_exception():
    obj = _annotation_obj()
    obj.pi_agent.generate_annotation_research_brief = AsyncMock(side_effect=RuntimeError("PI down"))
    out = await obj._generate_annotation_brief(MARKERS, "mouse", "liver", state=None)
    assert out is None


def _agents_obj():
    obj = AgentNodesMixin.__new__(AgentNodesMixin)
    obj.session_id = "test-session"
    obj.pi_agent = AsyncMock()
    return obj


@pytest.mark.asyncio
async def test_round_brief_helper_returns_brief():
    obj = _agents_obj()
    obj.pi_agent.generate_round_research_brief = AsyncMock(return_value="ROUND BRIEF")
    out = await obj._generate_round_brief("topic", ["f1"], state=None)
    assert out == "ROUND BRIEF"


@pytest.mark.asyncio
async def test_round_brief_helper_none_on_empty():
    obj = _agents_obj()
    obj.pi_agent.generate_round_research_brief = AsyncMock(return_value="   ")
    out = await obj._generate_round_brief("topic", None, state=None)
    assert out is None


@pytest.mark.asyncio
async def test_round_brief_helper_none_on_exception():
    obj = _agents_obj()
    obj.pi_agent.generate_round_research_brief = AsyncMock(side_effect=RuntimeError("PI down"))
    out = await obj._generate_round_brief("topic", None, state=None)
    assert out is None
