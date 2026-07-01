from backend.models.schemas import AnalysisState
from backend.workflow.planning import resolve_plan_review_wait


def test_round_wait_zero_does_not_make_plan_review_zero_by_default():
    state = AnalysisState(id="s", user_id="u", round_wait_seconds=0)
    assert resolve_plan_review_wait(state) == -1


def test_explicit_plan_review_wait_without_auto_approve_keeps_manual_wait():
    state = AnalysisState(
        id="s",
        user_id="u",
        round_wait_seconds=0,
        plan_review_wait_seconds=30,
        auto_approve_plans=False,
    )
    assert resolve_plan_review_wait(state) == -1


def test_auto_approve_requires_explicit_flag():
    state = AnalysisState(id="s", user_id="u", plan_review_wait_seconds=0, auto_approve_plans=False)
    assert resolve_plan_review_wait(state) == -1


def test_finite_timeout_without_auto_approve_keeps_manual_wait():
    state = AnalysisState(id="s", user_id="u", plan_review_wait_seconds=30, auto_approve_plans=False)
    assert resolve_plan_review_wait(state) == -1


def test_explicit_plan_review_wait_with_auto_approve_is_used():
    state = AnalysisState(
        id="s",
        user_id="u",
        round_wait_seconds=0,
        plan_review_wait_seconds=30,
        auto_approve_plans=True,
    )
    assert resolve_plan_review_wait(state) == 30
