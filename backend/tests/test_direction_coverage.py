from backend.models.schemas import AnalysisPhase, AnalysisState, RoundResult
from backend.workflow.direction_coverage import (
    completed_innovation_direction_keys,
    direction_coverage_status,
    normalize_direction_key,
)


def _state(results):
    return AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[
            {"direction": "Direction A", "notes": ""},
            {"direction": "Direction B", "notes": ""},
            {"direction": "Direction C", "notes": ""},
        ],
        round_results=results,
    )


def _result(direction, *, lineage_id="main", summary="summary", round_status="completed", evidence_status="substantive"):
    return RoundResult(
        round_number=2,
        phase="innovation",
        direction=direction,
        lineage_id=lineage_id,
        direction_key=normalize_direction_key(direction),
        result_identity_key=f"{lineage_id}:innovation:2:{normalize_direction_key(direction)}",
        pi_summary=summary,
        bioinformatics_findings=["substantive finding"],
        round_status=round_status,
        evidence_status=evidence_status,
    )


def test_selected_a_b_c_with_completed_a_b_b_is_incomplete():
    coverage = direction_coverage_status(_state([_result("Direction A"), _result("Direction B"), _result("Direction B")]))
    assert coverage.complete is False
    assert coverage.missing_keys == {"direction c"}


def test_empty_direction_duplicate_does_not_complete_coverage():
    coverage = direction_coverage_status(_state([_result("Direction A"), _result("Direction B"), _result("")]))
    assert coverage.complete is False
    assert coverage.missing_keys == {"direction c"}


def test_failed_non_empty_summary_does_not_complete_coverage():
    coverage = direction_coverage_status(_state([
        _result("Direction A"),
        _result("Direction B"),
        _result("Direction C", summary="Analysis failed after retries", round_status="failed", evidence_status="failed"),
    ]))
    assert coverage.complete is False
    assert coverage.missing_keys == {"direction c"}


def test_completed_a_b_c_is_complete():
    coverage = direction_coverage_status(_state([_result("Direction A"), _result("Direction B"), _result("Direction C")]))
    assert coverage.complete is True
    assert coverage.missing_keys == set()


def test_optional_degraded_round_can_be_explicitly_coverage_eligible():
    degraded = _result("Direction C", round_status="partial")
    degraded.direction_coverage_eligible = True
    coverage = direction_coverage_status(_state([
        _result("Direction A"),
        _result("Direction B"),
        degraded,
    ]))
    assert coverage.complete is True
    assert coverage.missing_keys == set()


def test_coverage_filters_by_lineage():
    state = _state([_result("Direction A"), _result("Direction B"), _result("Direction C", lineage_id="branch-1")])
    coverage = direction_coverage_status(state, lineage_id="main")
    assert coverage.complete is False
    assert coverage.missing_keys == {"direction c"}
    assert completed_innovation_direction_keys(state, lineage_id="branch-1") == {"direction c"}


def test_next_steps_without_core_findings_do_not_count_as_evidence():
    state = _state([
        _result("Direction A"),
        _result("Direction B"),
        RoundResult(
            round_number=4,
            phase="innovation",
            direction="Direction C",
            pi_summary="Could not complete analysis; retry recommended.",
            next_steps=["retry analysis"],
            round_status="completed",
            evidence_status="substantive",
        ),
    ])
    coverage = direction_coverage_status(state)
    assert coverage.complete is False
    assert coverage.missing_keys == {"direction c"}
