"""Tests for make_initial_workflow_state planning_stage seeding."""
from backend.models.schemas import AnalysisState, AnalysisPlan
from backend.workflow.state import make_initial_workflow_state


def _make_state_with_plan(stage: str, status: str = "approved") -> AnalysisState:
    s = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    s.current_plan = AnalysisPlan(phase="exploration", stage=stage, status=status)
    return s


def test_planning_stage_seeded_from_approved_stage1_plan():
    s = _make_state_with_plan("stage1", "approved")
    ws = make_initial_workflow_state(s)
    assert ws["planning_stage"] == "stage1"
    assert ws["plan_approved"] is True


def test_planning_stage_seeded_from_approved_stage2_plan():
    s = _make_state_with_plan("stage2", "approved")
    ws = make_initial_workflow_state(s)
    assert ws["planning_stage"] == "stage2"
    assert ws["plan_approved"] is True


def test_planning_stage_defaults_to_full_when_no_plan():
    s = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    ws = make_initial_workflow_state(s)
    assert ws["planning_stage"] == "full"
    assert ws["plan_approved"] is False


def test_planning_stage_seeded_even_when_plan_not_approved():
    s = _make_state_with_plan("stage1", "draft")
    ws = make_initial_workflow_state(s)
    assert ws["planning_stage"] == "stage1"
    assert ws["plan_approved"] is False
