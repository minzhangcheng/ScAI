"""F4: exploration_failed must surface a STABLE reason code, never a raw internal
status string. The UI localizes off this code."""
from backend.workflow.synthesis import classify_exploration_failure


def test_error_present_is_analysis_error():
    assert classify_exploration_failure({"error": "vision boom"}) == "analysis_error"


def test_status_only_is_did_not_converge():
    # Internal status strings (e.g. max_steps_reached) must NOT leak through.
    assert classify_exploration_failure({"status": "max_steps_reached"}) == "did_not_converge"


def test_empty_is_did_not_converge():
    assert classify_exploration_failure({}) == "did_not_converge"
    assert classify_exploration_failure(None) == "did_not_converge"


def test_returned_code_is_from_stable_set():
    assert classify_exploration_failure({"status": "anything"}) in {
        "did_not_converge", "analysis_error",
    }
