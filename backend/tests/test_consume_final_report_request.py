from backend.models.schemas import AnalysisState


def test_consume_reads_and_clears():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.final_report_requested = True
    assert s.consume_final_report_request() is True
    assert s.final_report_requested is False
    assert s.consume_final_report_request() is False


def test_consume_false_when_not_requested():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    assert s.consume_final_report_request() is False
    assert s.final_report_requested is False
