from backend.models.schemas import AnalysisState
from backend.workflow.reporting import validate_complete_report_allowed


def test_uncovered_new_batch_blocks_complete_report():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.allow_partial_report = False
    s.active_direction_batch = [{"direction": "C"}]
    s.round_results = []
    allowed, reason = validate_complete_report_allowed(s)
    assert allowed is False
    assert "direction" in reason.lower() or "batch" in reason.lower()


def test_partial_flag_allows_report():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.allow_partial_report = True
    s.active_direction_batch = [{"direction": "C"}]
    allowed, _ = validate_complete_report_allowed(s)
    assert allowed is True
