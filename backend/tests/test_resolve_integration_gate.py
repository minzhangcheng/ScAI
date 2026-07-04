from backend.models.schemas import AnalysisState, AnalysisPhase
from backend.services.interaction_record_capture import resolve_integration_gate


def _state_at_gate():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.current_round = 5
    s.integration_cycle = 2
    s.integration_gate_candidates = [{"direction": "x"}]
    s.integration_gate_literature_evidence = [{"pmid": "1"}]
    return s


def test_resolve_populates_record_and_clears_gate():
    s = _state_at_gate()
    rec = resolve_integration_gate(s, event_type="finalize")
    assert rec.event_type == "finalize"
    assert rec.round_number == 5
    assert rec.phase == "integration"
    assert rec.integration_cycle == 2
    assert rec.candidates == [{"direction": "x"}]
    assert rec.literature_evidence == [{"pmid": "1"}]
    assert s.integration_gate_candidates == []
    assert s.integration_gate_literature_evidence == []


def test_capture_helper_uses_top_level_analysis_phase_import():
    import inspect
    from backend.services import interaction_record_capture

    src = inspect.getsource(interaction_record_capture.populate_direction_record_from_state)
    assert "from ..models.schemas import AnalysisPhase" not in src
    assert "state.analysis_phase" in src
