"""Unit tests for RoutingMixin pure routing functions."""
from backend.models.schemas import AnalysisPhase
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


def test_after_review_abort_when_should_continue_false_and_no_approval():
    """Planning aborted (e.g. bioinfo LLM exhausted with no existing plan, user
    skipped) — must route to terminal sink, not loop back to planning_dialogue."""
    s = _Stub()
    state = {"plan_approved": False, "should_continue": False}
    assert s._after_review(state) == "abort"


def test_after_planning_dialogue_failure_does_not_generate_report():
    s = _Stub()

    assert s._after_planning_dialogue({"failed_node": "planning_dialogue"}) == "failed"
    assert s._after_planning_dialogue({"failed_node": None}) == "review"


def test_after_review_planning_failure_does_not_generate_report():
    s = _Stub()
    state = {
        "failed_node": "planning_dialogue",
        "plan_approved": False,
        "should_continue": False,
    }

    assert s._after_review(state) == "failed"


def test_after_review_approval_wins_over_abort_flag():
    """Defensive: if plan was approved before someone flipped should_continue,
    honour the approval — the planning subgraph already finished its job."""
    s = _Stub()
    state = {"plan_approved": True, "should_continue": False, "planning_stage": "full"}
    assert s._after_review(state) == "approved"


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


def test_should_continue_stops_failed_exploration_bioinformatics_without_mutating_context():
    s = _Stub()
    plan = object()
    analysis_state = type(
        "AS",
        (),
        {
            "analysis_phase": AnalysisPhase.EXPLORATION,
            "max_rounds": 5,
            "current_round": 1,
            "status": "running",
            "current_plan": plan,
        },
    )()
    state = {
        "human_stop_requested": False,
        "should_continue": True,
        "analysis_state": analysis_state,
        "bioinformatics_result": {"status": "failed"},
    }

    assert s._should_continue(state) == "stop"
    assert analysis_state.current_round == 1
    assert analysis_state.current_plan is plan


def test_should_continue_keeps_failed_innovation_direction_routing():
    s = _Stub()
    analysis_state = type(
        "AS",
        (),
        {
            "analysis_phase": AnalysisPhase.INNOVATION,
            "max_rounds": 5,
            "current_round": 1,
            "status": "running",
            "active_direction_batch": [{"direction": "target"}],
            "user_chosen_directions": [{"direction": "target"}],
            "round_results": [],
            "restore_branch_id": "",
            "last_integrated_round": 0,
        },
    )()
    state = {
        "human_stop_requested": False,
        "should_continue": True,
        "analysis_state": analysis_state,
        "bioinformatics_result": {"status": "failed"},
    }

    assert s._should_continue(state) == "continue"


def test_should_continue_keeps_completed_bioinformatics_normal_routing():
    s = _Stub()
    analysis_state = type(
        "AS",
        (),
        {
            "analysis_phase": AnalysisPhase.EXPLORATION,
            "max_rounds": 5,
            "current_round": 1,
            "status": "running",
        },
    )()
    state = {
        "human_stop_requested": False,
        "should_continue": True,
        "analysis_state": analysis_state,
        "bioinformatics_result": {"status": "completed"},
    }

    assert s._should_continue(state) == "continue"


def test_check_convergence_condition_continue():
    s = _Stub()
    state = {"should_continue": True,
             "analysis_state": type("AS", (), {"max_rounds": 0, "current_round": 0, "status": "running"})()}
    assert s._check_convergence_condition(state) == "continue"


def test_check_convergence_condition_stop():
    s = _Stub()
    state = {"should_continue": False,
             "analysis_state": type("AS", (), {"max_rounds": 0, "current_round": 0, "status": "running"})()}
    assert s._check_convergence_condition(state) == "stop"


def test_check_convergence_condition_missing_key_defaults_continue():
    """Missing should_continue key now defaults to 'continue' to avoid
    silently stopping a live run (consistent with _should_continue)."""
    s = _Stub()
    state = {"analysis_state": type("AS", (), {"max_rounds": 0, "current_round": 0, "status": "running"})()}
    assert s._check_convergence_condition(state) == "continue"


def test_after_review_clears_plan_revision_resume_after_routing():
    s = _Stub()
    state = {"plan_approved": True, "planning_stage": "full", "plan_revision_resume": True}
    route = s._after_review(state)
    assert route == "approved_revision"
    assert state.get("plan_revision_resume") is False, "Flag must be cleared after routing"


def test_after_review_clears_plan_revision_resume_even_on_stage2():
    s = _Stub()
    state = {"plan_approved": True, "planning_stage": "stage2", "plan_revision_resume": True}
    route = s._after_review(state)
    assert route == "approved_stage2"
    assert state.get("plan_revision_resume") is False


def test_convergence_routes_stop_on_human_stop_requested():
    s = _Stub()
    state = {"analysis_state": type("AS", (), {"max_rounds": 0, "current_round": 0, "status": "running"})(),
             "human_stop_requested": True, "should_continue": True}
    assert s._check_convergence_condition(state) == "stop"


def test_convergence_routes_stop_at_max_rounds():
    s = _Stub()
    as_ = type("AS", (), {"max_rounds": 3, "current_round": 3, "status": "running"})()
    state = {"analysis_state": as_, "should_continue": True}
    assert s._check_convergence_condition(state) == "stop"


def test_convergence_routes_stop_when_paused():
    from backend.models.schemas import TaskStatus
    s = _Stub()
    as_ = type("AS", (), {"max_rounds": 0, "current_round": 0, "status": TaskStatus.PAUSED})()
    state = {"analysis_state": as_, "should_continue": True}
    assert s._check_convergence_condition(state) == "stop"
