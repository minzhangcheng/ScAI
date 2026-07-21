from backend.models.schemas import AnalysisState, AnalysisPhase, RoundResult
from backend.workflow.direction_coverage import (
    at_integration_gate,
    active_batch_incomplete,
    active_batch_complete,
    active_batch_has_unattempted_directions,
)


def _state(**kw):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_at_integration_gate_true_only_when_phase_and_candidates():
    s = _state(analysis_phase=AnalysisPhase.INTEGRATION,
               integration_gate_candidates=[{"direction": "x"}])
    assert at_integration_gate(s) is True


def test_at_integration_gate_false_without_candidates():
    s = _state(analysis_phase=AnalysisPhase.INTEGRATION, integration_gate_candidates=[])
    assert at_integration_gate(s) is False


def test_at_integration_gate_false_wrong_phase():
    s = _state(analysis_phase=AnalysisPhase.EXPLORATION,
               integration_gate_candidates=[{"direction": "x"}])
    assert at_integration_gate(s) is False


def test_active_batch_incomplete_empty_batch_is_false():
    s = _state(active_direction_batch=[])
    assert active_batch_incomplete(s) is False
    assert active_batch_complete(s) is False


def test_batch_predicates_are_complementary_when_batch_present():
    s = _state(active_direction_batch=[{"direction": "dir-a"}])
    # With no completed results, an existing batch is incomplete, not complete.
    assert active_batch_incomplete(s) is True
    assert active_batch_complete(s) is False
    assert active_batch_has_unattempted_directions(s) is True


def test_partial_attempt_is_incomplete_but_not_unattempted():
    direction = "dir-a"
    s = _state(
        active_direction_batch=[{"direction": direction}],
        round_results=[RoundResult(
            round_number=2,
            phase="innovation",
            direction=direction,
            direction_key=direction,
            pi_summary="Partial result",
            bioinformatics_findings=["finding"],
            round_status="partial",
            evidence_status="substantive",
            direction_coverage_eligible=False,
        )],
    )
    assert active_batch_incomplete(s) is True
    assert active_batch_complete(s) is False
    assert active_batch_has_unattempted_directions(s) is False


def test_optional_degraded_attempt_can_complete_active_batch():
    direction = "dir-a"
    s = _state(
        active_direction_batch=[{"direction": direction}],
        round_results=[RoundResult(
            round_number=2,
            phase="innovation",
            direction=direction,
            direction_key=direction,
            pi_summary="Core evidence with degraded optional modality",
            bioinformatics_findings=["finding"],
            round_status="partial",
            evidence_status="substantive",
            direction_coverage_eligible=True,
        )],
    )
    assert active_batch_incomplete(s) is False
    assert active_batch_complete(s) is True
    assert active_batch_has_unattempted_directions(s) is False


def test_synthesis_uses_active_batch_complete_predicate():
    import inspect
    from backend.workflow import synthesis

    assert "active_batch_complete" in inspect.getsource(synthesis._multi_direction_analysis_completed)
    assert "direction_coverage_status_for_batch" not in inspect.getsource(
        synthesis._multi_direction_analysis_completed
    )
