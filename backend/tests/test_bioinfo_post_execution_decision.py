"""Repeated executor failure must spend a bounded number of planning cycles."""

import pytest

from backend.workflow.helpers import (
    MAX_EXECUTION_REVISION_ATTEMPTS,
    decide_bioinfo_post_execution,
)


@pytest.mark.parametrize("stage,settled", [
    ("stage1", "stage1_complete"), ("stage2", "complete"), ("full", "complete"),
])
@pytest.mark.parametrize("status", [
    "error", "forced_stop", "max_steps_reached", "completed_with_fallback",
])
def test_execution_replans_only_below_the_ceiling(stage, settled, status):
    assert 1 <= MAX_EXECUTION_REVISION_ATTEMPTS <= 3
    for attempts in (0, MAX_EXECUTION_REVISION_ATTEMPTS - 1, MAX_EXECUTION_REVISION_ATTEMPTS):
        expected = "incomplete" if attempts < MAX_EXECUTION_REVISION_ATTEMPTS else settled
        assert decide_bioinfo_post_execution(stage, False, status, attempts) == expected
        assert decide_bioinfo_post_execution(stage, True, status, attempts) == "revision"


@pytest.mark.parametrize("stage,settled", [
    ("stage1", "stage1_complete"), ("stage2", "complete"), ("full", "complete"),
])
@pytest.mark.parametrize("status", ["completed", "partial"])
def test_completed_or_partial_execution_advances_without_replanning(stage, settled, status):
    assert decide_bioinfo_post_execution(stage, False, status) == settled
