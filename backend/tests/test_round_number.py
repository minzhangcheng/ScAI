import pytest
from backend.models.schemas import AnalysisStep, AnalysisState

def test_analysis_step_has_round_number_default_one():
    step = AnalysisStep(step_number=1, step_type="execute_code")
    assert step.round_number == 1

def test_analysis_step_round_number_can_be_set():
    step = AnalysisStep(step_number=1, step_type="execute_code", round_number=3)
    assert step.round_number == 3

def test_analysis_step_serialization_includes_round_number():
    step = AnalysisStep(step_number=1, step_type="execute_code", round_number=2)
    d = step.model_dump()
    assert "round_number" in d
    assert d["round_number"] == 2

def test_analysis_step_round_number_set_from_current_round():
    state = AnalysisState(id="test", name="test", user_id="test")
    step = AnalysisStep(step_number=1, step_type="execute_code", round_number=max(state.current_round, 1))
    assert step.round_number == 1

    state.current_round = 3
    step2 = AnalysisStep(step_number=2, step_type="execute_code", round_number=max(state.current_round, 1))
    assert step2.round_number == 3

def test_build_full_analysis_log_includes_round_number():
    from backend.services.report.analysis_log import AnalysisLogMixin
    
    state = AnalysisState(id="test", name="test", user_id="test")
    state.analysis_log = [
        AnalysisStep(step_number=1, step_type="execute_code", round_number=1),
        AnalysisStep(step_number=2, step_type="text_response", round_number=2),
    ]
    
    builder = AnalysisLogMixin()
    log = builder._build_full_analysis_log(state)
    assert len(log) == 2
    assert log[0].get("round_number") == 1
    assert log[1].get("round_number") == 2