"""Unit tests for RoutingMixin pure routing functions."""
from backend.workflow.routing import RoutingMixin


class _Stub(RoutingMixin):
    def __init__(self):
        self.session_id = "test"
        self.config = type("Cfg", (), {"max_rounds": 5, "convergence_rounds": 3})()


def test_should_plan_returns_do_planning_when_no_plan():
    s = _Stub()
    state = {"analysis_state": type("AS", (), {"current_plan": None})()}
    assert s._should_plan(state) == "do_planning"


def test_should_plan_returns_skip_when_approved():
    s = _Stub()
    plan = type("P", (), {"status": "approved"})()
    state = {"analysis_state": type("AS", (), {"current_plan": plan})()}
    assert s._should_plan(state) == "skip_planning"


def test_should_plan_returns_do_planning_when_plan_not_approved():
    s = _Stub()
    plan = type("P", (), {"status": "draft"})()
    state = {"analysis_state": type("AS", (), {"current_plan": plan})()}
    assert s._should_plan(state) == "do_planning"


def test_after_review_approved():
    s = _Stub()
    state = {"plan_approved": True, "planning_stage": "full", "plan_revision_resume": False}
    assert s._after_review(state) == "approved"


def test_after_review_approved_stage2():
    s = _Stub()
    state = {"plan_approved": True, "planning_stage": "stage2"}
    assert s._after_review(state) == "approved_stage2"


def test_after_review_revision_resume():
    s = _Stub()
    state = {"plan_approved": True, "planning_stage": "full", "plan_revision_resume": True}
    assert s._after_review(state) == "approved_revision"


def test_after_review_revise():
    s = _Stub()
    state = {"plan_approved": False}
    assert s._after_review(state) == "revise"


def test_after_bioinformatics_plan_revision():
    s = _Stub()
    state = {"plan_revision_needed": True, "planning_stage": "full"}
    assert s._after_bioinformatics(state) == "plan_revision"


def test_after_bioinformatics_stage2_approved():
    s = _Stub()
    state = {"plan_revision_needed": False, "planning_stage": "stage2", "plan_approved": True}
    assert s._after_bioinformatics(state) == "continue_to_synthesis"


def test_after_bioinformatics_stage2_not_approved():
    s = _Stub()
    state = {"plan_revision_needed": False, "planning_stage": "stage2", "plan_approved": False}
    assert s._after_bioinformatics(state) == "stage2_planning"


def test_after_bioinformatics_full():
    s = _Stub()
    state = {"plan_revision_needed": False, "planning_stage": "full"}
    assert s._after_bioinformatics(state) == "continue_to_synthesis"


def test_should_continue_after_init_stop_requested():
    s = _Stub()
    state = {"human_stop_requested": True}
    assert s._should_continue_after_init(state) == "stop"


def test_should_continue_after_init_should_continue_false():
    s = _Stub()
    state = {"human_stop_requested": False, "should_continue": False}
    assert s._should_continue_after_init(state) == "stop"


def test_should_continue_after_init_continue():
    s = _Stub()
    state = {"human_stop_requested": False, "should_continue": True}
    assert s._should_continue_after_init(state) == "continue"


def test_should_continue_human_stop():
    s = _Stub()
    state = {"human_stop_requested": True, "should_continue": True,
             "analysis_state": type("AS", (), {"max_rounds": 0, "current_round": 0, "status": "running"})()}
    assert s._should_continue(state) == "stop"


def test_should_continue_max_rounds_reached():
    s = _Stub()
    state = {"human_stop_requested": False, "should_continue": True,
             "analysis_state": type("AS", (), {"max_rounds": 0, "current_round": 5, "status": "running"})()}
    assert s._should_continue(state) == "stop"


def test_should_continue_normal():
    s = _Stub()
    state = {"human_stop_requested": False, "should_continue": True,
             "analysis_state": type("AS", (), {"max_rounds": 0, "current_round": 2, "status": "running"})()}
    assert s._should_continue(state) == "continue"


def test_check_convergence_condition_continue():
    s = _Stub()
    state = {"should_continue": True}
    assert s._check_convergence_condition(state) == "continue"


def test_check_convergence_condition_stop():
    s = _Stub()
    state = {"should_continue": False}
    assert s._check_convergence_condition(state) == "stop"


def test_check_convergence_condition_missing_key_defaults_stop():
    s = _Stub()
    state = {}
    assert s._check_convergence_condition(state) == "stop"
