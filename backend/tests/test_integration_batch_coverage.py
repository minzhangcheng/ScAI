from backend.models.schemas import AnalysisState, RoundResult
from backend.workflow.direction_coverage import direction_coverage_status_for_batch


def _completed_innovation_round(direction: str, rnd: int) -> RoundResult:
    rr = RoundResult(round_number=rnd)
    rr.phase = "innovation"
    rr.direction = direction
    rr.pi_summary = "done"
    rr.round_status = "completed"
    rr.evidence_status = "substantive"
    rr.bioinformatics_findings = ["finding"]
    return rr


def test_batch_coverage_only_considers_the_given_batch():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.user_chosen_directions = [{"direction": "A"}, {"direction": "B"}, {"direction": "C"}]
    s.round_results = [_completed_innovation_round("C", 5)]
    cov = direction_coverage_status_for_batch(s, [{"direction": "C", "notes": ""}])
    assert cov.complete is True


def test_batch_coverage_incomplete_when_batch_direction_missing():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.round_results = [_completed_innovation_round("C", 5)]
    cov = direction_coverage_status_for_batch(s, [{"direction": "C"}, {"direction": "D"}])
    assert cov.complete is False
    assert "d" in cov.missing_keys


def test_empty_batch_is_not_complete():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    assert direction_coverage_status_for_batch(s, []).complete is False
