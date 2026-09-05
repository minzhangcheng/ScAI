"""The graph superstep ceiling has to cover the rounds a run may execute.

A fixed ceiling of 100 supersteps aborts a long analysis at its very end: a
ten-round run reached its graphic abstract at superstep 100 and raised
GraphRecursionError before the detailed-report node, so the run was recorded as
failed with no report. Size the ceiling from the run's round budget instead.
"""

from types import SimpleNamespace


def test_graph_recursion_limit_scales_with_max_rounds():
    from backend.workflow.core import GRAPH_RECURSION_LIMIT, graph_recursion_limit

    assert graph_recursion_limit(10) > 250
    assert graph_recursion_limit(10) > graph_recursion_limit(3)
    # Short and unset round budgets never lose the historical headroom.
    assert graph_recursion_limit(1) >= GRAPH_RECURSION_LIMIT
    assert graph_recursion_limit(0) >= GRAPH_RECURSION_LIMIT
    assert graph_recursion_limit(None) >= GRAPH_RECURSION_LIMIT


def test_graph_invocations_size_their_ceiling_from_the_round_budget():
    """The graph entry points resolve max_rounds the way routing does."""
    from backend.workflow.core import graph_recursion_limit
    from backend.workflow.core_parts import execution

    workflow = SimpleNamespace(config=SimpleNamespace(max_rounds=3))
    assert execution._recursion_limit_for(
        workflow, SimpleNamespace(max_rounds=10)
    ) == graph_recursion_limit(10)

    # max_rounds <= 0 means "use the configured default", as in routing.
    assert execution._recursion_limit_for(
        workflow, SimpleNamespace(max_rounds=0)
    ) == graph_recursion_limit(3)
    assert execution._recursion_limit_for(workflow, None) == graph_recursion_limit(3)

    # Restore dispatch builds the workflow with __new__, so the ceiling must not
    # depend on __init__ having installed a config.
    assert execution._recursion_limit_for(
        SimpleNamespace(), None
    ) == graph_recursion_limit(0)
