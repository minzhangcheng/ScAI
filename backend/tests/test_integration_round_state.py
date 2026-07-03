from backend.models.schemas import AnalysisState, AnalysisPhase
import backend.models.schemas as schemas


def test_integration_phase_member_exists():
    assert AnalysisPhase.INTEGRATION.value == "integration"


def test_analysis_phase_type_literal_includes_integration():
    assert "integration" in schemas.AnalysisPhaseType.__args__


def test_snapshot_phase_literal_has_gate_and_keeps_existing():
    args = set(schemas.SnapshotPhase.__args__)
    assert "pre_integration_gate" in args
    assert {"pre_major_revision", "manual", "pre_round_wait"}.issubset(args)


def test_integration_state_field_defaults():
    s = AnalysisState(name="t", user_id="u", initial_instruction="obj")
    assert s.integration_cycle == 0
    assert s.max_integration_cycles == 3
    assert s.pending_integration is False
    assert s.active_direction_batch == []
    assert s.last_integrated_round == 0
    assert s.integration_gate_candidates == []
    assert s.integration_reports == []
    assert s.final_report_requested is False


def test_integration_list_fields_are_independent_instances():
    a = AnalysisState(name="a", user_id="u", initial_instruction="o")
    b = AnalysisState(name="b", user_id="u", initial_instruction="o")
    a.active_direction_batch.append({"direction": "x", "notes": ""})
    a.integration_reports.append("r")
    assert b.active_direction_batch == []
    assert b.integration_reports == []
