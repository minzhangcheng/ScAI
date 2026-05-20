"""Tests for _pi_planning early-return planning_stage propagation."""
import asyncio
from unittest.mock import MagicMock, AsyncMock
import pytest

from backend.models.schemas import AnalysisState, AnalysisPlan
from backend.workflow.planning import PlanningMixin
from backend.workflow.agents import AgentNodesMixin


class _Harness(AgentNodesMixin, PlanningMixin):
    """Minimal harness exposing the mixin method under test."""
    def __init__(self):
        self.session_id = "t"
        self.pi_agent = MagicMock()
        self.pi_agent.reset_history = MagicMock()
        self._send_message = AsyncMock()

    async def call_pi_planning(self, state):
        fn = getattr(self._pi_planning, "__wrapped__", self._pi_planning)
        return await fn(self, state)


@pytest.mark.asyncio
async def test_pi_planning_early_return_propagates_stage1():
    s = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    s.current_plan = AnalysisPlan(phase="exploration", stage="stage1", status="approved")
    h = _Harness()
    state = {"analysis_state": s}
    out = await h.call_pi_planning(state)
    assert out["plan_approved"] is True
    assert out["planning_stage"] == "stage1"


@pytest.mark.asyncio
async def test_pi_planning_early_return_propagates_stage2():
    s = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    s.current_plan = AnalysisPlan(phase="innovation", stage="stage2", status="approved")
    h = _Harness()
    state = {"analysis_state": s}
    out = await h.call_pi_planning(state)
    assert out["planning_stage"] == "stage2"


@pytest.mark.asyncio
async def test_pi_planning_early_return_propagates_full():
    s = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    s.current_plan = AnalysisPlan(phase="exploration", stage="full", status="approved")
    h = _Harness()
    state = {"analysis_state": s}
    out = await h.call_pi_planning(state)
    assert out["planning_stage"] == "full"


@pytest.mark.asyncio
async def test_round1_adjusted_instructions_preserve_stage1():
    s = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    s.current_round = 1
    h = _Harness()
    state = {
        "analysis_state": s,
        "adjusted_instructions": {
            "bioinformatics_tasks": ["focus on macrophages"],
            "pathology_tasks": [],
            "biochemistry_tasks": [],
            "literature_topics": [],
        },
    }

    out = await h.call_pi_planning(state)

    assert out["pi_instructions"]["bioinformatics_tasks"] == ["focus on macrophages"]
    assert out["human_guidance_processed"] is True
    assert out["planning_stage"] == "stage1"
    assert out["plan_approved"] is False