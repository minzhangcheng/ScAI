from backend.models.schemas import AnalysisPhase, AnalysisPlan, AnalysisState
from backend.workflow.planning import approved_plan_matches_current_context, populate_plan_scope


def _state(**kwargs):
    defaults = {
        "id": "s",
        "user_id": "u",
        "analysis_phase": AnalysisPhase.INNOVATION,
        "current_round": 3,
        "current_round_direction": "Direction A",
        "restore_branch_id": None,
        "last_restored_snapshot_id": None,
    }
    defaults.update(kwargs)
    return AnalysisState(**defaults)


def test_same_context_approved_plan_matches():
    state = _state()
    plan = AnalysisPlan(phase="innovation", status="approved")
    populate_plan_scope(plan, state)
    state.current_plan = plan
    assert approved_plan_matches_current_context(state) is True


def test_round_mismatch_invalidates_plan():
    state = _state()
    plan = AnalysisPlan(phase="innovation", status="approved", round_number=2, direction="Direction A", direction_key="direction a")
    state.current_plan = plan
    assert approved_plan_matches_current_context(state) is False


def test_direction_mismatch_invalidates_plan():
    state = _state(current_round_direction="Direction B")
    plan = AnalysisPlan(phase="innovation", status="approved", round_number=3, direction="Direction A", direction_key="direction a")
    state.current_plan = plan
    assert approved_plan_matches_current_context(state) is False


def test_branch_mismatch_invalidates_plan():
    state = _state(restore_branch_id="branch-1")
    plan = AnalysisPlan(phase="innovation", status="approved", round_number=3, direction="Direction A", direction_key="direction a", branch_id=None)
    state.current_plan = plan
    assert approved_plan_matches_current_context(state) is False
