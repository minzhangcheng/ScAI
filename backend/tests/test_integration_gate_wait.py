import inspect
from backend.workflow.synthesis import SynthesisMixin


def test_wait_recognizes_integration_gate_and_finalize():
    src = inspect.getsource(SynthesisMixin._wait_for_user_input)
    assert "INTEGRATION" in src or "is_integration_gate" in src
    assert "final_report_requested" in src
    assert "should_continue" in src
