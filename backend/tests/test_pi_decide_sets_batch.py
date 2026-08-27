import inspect
from backend.workflow.planning import PlanningMixin
from backend.tests.split_facade import split_source


def test_pi_decide_branch_sets_active_direction_batch():
    src = split_source("backend.workflow.planning")
    assert "active_direction_batch = selected_dirs" in src or \
           "active_direction_batch = list(selected_dirs)" in src
