"""Registry-drift guards for the LangGraph workflow construction."""


def _build_test_workflow():
    """Minimal workflow instance with a compiled graph (no agents/DB)."""
    from backend.workflow.core import SingleCellAnalysisWorkflow

    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.checkpointer = None
    wf._build_graph()
    return wf


def test_graph_nodes_match_registry():
    from backend.constants import WORKFLOW_GRAPH_NODES

    wf = _build_test_workflow()
    registered = set(wf.graph.nodes.keys())
    assert registered == set(WORKFLOW_GRAPH_NODES), (
        f"drift: registered-only={registered - set(WORKFLOW_GRAPH_NODES)}, "
        f"registry-only={set(WORKFLOW_GRAPH_NODES) - registered}"
    )


def test_every_restorable_node_has_a_restore_route():
    """The third sync site, and the only one nothing guarded.

    A graph-node restore is performed by _route_from_restore_entry, which
    returns f"route_{node}". A node absent from that map does not raise:
    LangGraph takes the conditional edge's default and the session silently
    resumes at the wrong place.

    restore_entry is excluded because it is the graph's own entry point --
    nothing routes back to it, so route_restore_entry cannot exist.
    """
    from backend.constants import WORKFLOW_GRAPH_NODES

    wf = _build_test_workflow()
    routes = set()
    for branch in (wf.graph.branches.get("restore_entry") or {}).values():
        routes.update((branch.ends or {}).keys())
    expected = {
        f"route_{node}" for node in WORKFLOW_GRAPH_NODES if node != "restore_entry"
    }
    assert expected <= routes, (
        f"restore route map is missing: {sorted(expected - routes)}"
    )


def test_data_profiling_is_wired_into_the_pipeline():
    """The two edges that actually put data_profiling in the pipeline are
    otherwise unguarded.

    test_graph_nodes_match_registry pins node *registration*, and
    test_every_restorable_node_has_a_restore_route pins the *restore* map --
    neither notices if the node were wired out of the live execution path.
    Reverting initialize_round's "continue" route back to its pre-detection
    target ("pi_planning") would silently disable the whole feature: every
    session would fall back to the declared sample count, and every other
    test in this codebase would stay green, because they all call
    _data_profiling or should_profile directly rather than reaching the node
    by walking the graph. That is the exact failure mode this branch exists
    to eliminate, one level up -- a measurement that quietly stops happening
    and nothing notices.
    """
    wf = _build_test_workflow()

    continue_targets = set()
    for branch in (wf.graph.branches.get("initialize_round") or {}).values():
        target = (branch.ends or {}).get("continue")
        if target:
            continue_targets.add(target)
    assert continue_targets == {"data_profiling"}, (
        f'the initialize_round "continue" route targets {continue_targets}, '
        f"not data_profiling -- the profiling pass would never run"
    )

    # add_edge (a plain, unconditional edge) surfaces on the compiled
    # StateGraph as a set of (start, end) tuples -- confirmed by introspecting
    # a real built graph rather than assumed, since this is a different
    # accessor from the conditional-edge `branches` map used above.
    assert ("data_profiling", "pi_planning") in wf.graph.edges, (
        "data_profiling has no outgoing edge to pi_planning -- the graph "
        "would dead-end after profiling"
    )
