"""Unit tests for the post-execution stage decision used by
_bioinformatics_analysis. A mid-execution plan-revision pause must NOT be
treated as a stage completion.

Two properties are pinned here beyond the original stage logic, because this
decision is the only one in the system that can send a whole round back through
planning, and it used to do so with no memory that it had already done it:

1. The replan is BOUNDED. The function's own docstring always promised "a
   bounded plan revision"; nothing implemented the bound, so an execution that
   failed the same way every attempt re-planned until LangGraph's recursion
   limit raised `GraphRecursionError` — which nothing in backend/ catches, so
   the session ended FAILED with no report.
2. A figure gap alone no longer sends the round back. The executor's copy of
   that same check was demoted to a non-blocking observation in 0.39.39 with
   the reasoning "A refusal there cannot be worked off, so it is not a gate --
   it is a trap"; re-planning does not make an abandoned figure eligible
   either, so the workflow copy is an observation too.
"""
import inspect

import pytest

from backend.workflow.helpers import (
    MAX_EXECUTION_REVISION_ATTEMPTS,
    decide_bioinfo_post_execution,
)


@pytest.mark.parametrize("stage", ["stage1", "stage2", "full"])
def test_revision_pause_never_advances_stage(stage):
    """Revision requested → 'revision' regardless of stage."""
    assert decide_bioinfo_post_execution(stage, plan_revision_requested=True) == "revision"


def test_stage1_genuine_completion_advances():
    """stage1 finished without revision → advance to stage2."""
    assert decide_bioinfo_post_execution("stage1", plan_revision_requested=False) == "stage1_complete"


def test_stage2_genuine_completion():
    """stage2 finished without revision → normal completion."""
    assert decide_bioinfo_post_execution("stage2", plan_revision_requested=False) == "complete"


def test_full_genuine_completion():
    """full mode finished without revision → normal completion."""
    assert decide_bioinfo_post_execution("full", plan_revision_requested=False) == "complete"


@pytest.mark.parametrize("status", ["error", "forced_stop", "max_steps_reached", "completed_with_fallback"])
@pytest.mark.parametrize("attempts", [0, MAX_EXECUTION_REVISION_ATTEMPTS - 1])
def test_incomplete_execution_replans_while_attempts_remain(status, attempts):
    """An execution that gave up gets a bounded number of fresh attempts."""
    assert decide_bioinfo_post_execution(
        "stage1",
        plan_revision_requested=False,
        result_status=status,
        incomplete_attempts=attempts,
    ) == "incomplete"


@pytest.mark.parametrize("status", ["error", "forced_stop", "max_steps_reached", "completed_with_fallback"])
@pytest.mark.parametrize(
    ("stage", "released"),
    [("stage1", "stage1_complete"), ("stage2", "complete"), ("full", "complete")],
)
def test_the_replan_is_bounded_and_releases_at_the_ceiling(status, stage, released):
    """At the ceiling the round settles as degraded instead of re-planning.

    This is the whole point: the same failure repeating must cost a bounded
    number of cycles, not the graph's recursion budget.
    """
    assert decide_bioinfo_post_execution(
        stage,
        plan_revision_requested=False,
        result_status=status,
        incomplete_attempts=MAX_EXECUTION_REVISION_ATTEMPTS,
    ) == released


def test_the_ceiling_is_small_enough_to_be_a_bound():
    """A ceiling nobody can reach is not a bound. One cycle is a full planning
    dialogue plus a review wait plus a fresh executor attempt."""
    assert 1 <= MAX_EXECUTION_REVISION_ATTEMPTS <= 3


def test_explicit_partial_result_can_advance_with_visible_degradation():
    assert decide_bioinfo_post_execution(
        "stage1", plan_revision_requested=False, result_status="partial"
    ) == "stage1_complete"


def test_a_figure_gap_alone_no_longer_sends_the_round_back_to_planning():
    """A clean execution settles even when required figures are missing.

    The gap is still measured and still reaches the PI - as an observation on
    the round, not as a refusal here. Re-planning cannot make a figure the
    review loop abandoned ("requires upstream reanalysis, skipping revision
    loop") become report-eligible, so refusing on it can never be worked off.
    """
    assert decide_bioinfo_post_execution(
        "stage2", plan_revision_requested=False, result_status="completed"
    ) == "complete"


def test_the_figure_contract_is_not_an_argument_of_this_decision():
    """Pin the removal, so the clause cannot quietly come back.

    A future caller passing figure_contract_passed=False must fail loudly
    rather than silently re-acquire a gate that cannot be worked off.
    """
    assert "figure_contract_passed" not in inspect.signature(
        decide_bioinfo_post_execution
    ).parameters
