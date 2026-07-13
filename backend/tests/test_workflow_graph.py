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
