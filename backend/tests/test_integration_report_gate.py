from backend.models.schemas import AnalysisState
from backend.workflow.reporting import report_coverage_observation


def test_uncovered_new_batch_is_reported_to_pi():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.allow_partial_report = False
    s.active_direction_batch = [{"direction": "C"}]
    s.round_results = []
    reason = report_coverage_observation(s)
    assert "direction" in reason.lower() or "batch" in reason.lower()
    assert "PI must decide" in reason


def test_partial_flag_does_not_erase_coverage_observation():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.allow_partial_report = True
    s.active_direction_batch = [{"direction": "C"}]
    assert report_coverage_observation(s)
