from backend.models.schemas import AnalysisState
from backend.workflow.planning import resolve_plan_review_wait


def test_round_wait_zero_auto_skips_plan_review():
    """round_wait_seconds=0 → plan review auto-skips immediately."""
    state = AnalysisState(id="s", user_id="u", round_wait_seconds=0)
    assert resolve_plan_review_wait(state) == 0


def test_round_wait_positive_countdown_applies_to_plan_review():
    """round_wait_seconds>0 → plan review uses the same countdown."""
    state = AnalysisState(id="s", user_id="u", round_wait_seconds=300)
    assert resolve_plan_review_wait(state) == 300


def test_round_wait_manual_blocks_plan_review():
    """round_wait_seconds=-1 → plan review waits indefinitely (manual)."""
    state = AnalysisState(id="s", user_id="u", round_wait_seconds=-1)
    assert resolve_plan_review_wait(state) == -1


def test_auto_approve_uses_dedicated_plan_review_wait():
    """auto_approve_plans=True → uses plan_review_wait_seconds, ignoring round_wait."""
    state = AnalysisState(
        id="s",
        user_id="u",
        round_wait_seconds=0,
        plan_review_wait_seconds=30,
        auto_approve_plans=True,
    )
    assert resolve_plan_review_wait(state) == 30


def test_auto_approve_with_plan_review_wait_zero():
    """auto_approve_plans=True + plan_review_wait_seconds=0 → auto-skip."""
    state = AnalysisState(
        id="s",
        user_id="u",
        round_wait_seconds=-1,
        plan_review_wait_seconds=0,
        auto_approve_plans=True,
    )
    assert resolve_plan_review_wait(state) == 0
