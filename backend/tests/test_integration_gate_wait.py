import inspect
from backend.workflow.synthesis import SynthesisMixin
from backend.workflow.synthesis_parts import phases


def test_wait_recognizes_integration_gate_and_finalize():
    src = inspect.getsource(phases._wait_for_user_input)
    assert "INTEGRATION" in src or "is_integration_gate" in src
    assert "final_report_requested" in src
    assert "should_continue" in src
