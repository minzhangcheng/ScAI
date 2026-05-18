def test_consume_feedback_returns_value_and_clears():
    from backend.workflow.planning import _consume_user_plan_feedback

    state = {"user_plan_feedback": "please add trajectory analysis"}
    value = _consume_user_plan_feedback(state)
    assert value == "please add trajectory analysis"
    assert state.get("user_plan_feedback") in (None, "")


def test_consume_feedback_empty_when_absent():
    from backend.workflow.planning import _consume_user_plan_feedback

    state = {}
    value = _consume_user_plan_feedback(state)
    assert value == ""
    assert not state.get("user_plan_feedback")


def test_consume_feedback_idempotent():
    from backend.workflow.planning import _consume_user_plan_feedback

    state = {"user_plan_feedback": "x"}
    assert _consume_user_plan_feedback(state) == "x"
    assert _consume_user_plan_feedback(state) == ""