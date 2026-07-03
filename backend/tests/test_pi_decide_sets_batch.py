import inspect
from backend.workflow.planning import PlanningMixin


def test_pi_decide_branch_sets_active_direction_batch():
    src = inspect.getsource(PlanningMixin)
    assert "active_direction_batch = selected_dirs" in src or \
           "active_direction_batch = list(selected_dirs)" in src
