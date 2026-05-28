"""C4 — NODE_TO_AGENT and NODE_TO_RESULT_KEY drift guard."""
import pytest


def test_result_key_nodes_subset_of_agent_nodes():
    """Every node with a result key must also have an agent assignment."""
    from backend.constants import NODE_TO_AGENT, NODE_TO_RESULT_KEY
    missing = set(NODE_TO_RESULT_KEY) - set(NODE_TO_AGENT)
    assert not missing, f"Nodes in NODE_TO_RESULT_KEY but not NODE_TO_AGENT: {missing}"


def test_agent_nodes_without_result_key_are_documented():
    """The set of agent-nodes without an independent result key is finite & known.

    pi_synthesis writes into round_result, not a node-named key.
    user_review_gate is a gating node, not an analysis step.
    """
    from backend.constants import NODE_TO_AGENT, NODE_TO_RESULT_KEY, RESULT_KEYLESS_NODES
    actual = set(NODE_TO_AGENT) - set(NODE_TO_RESULT_KEY)
    assert actual == RESULT_KEYLESS_NODES, (
        f"Agent-nodes without a result key changed: {actual} vs declared {RESULT_KEYLESS_NODES}. "
        "Update RESULT_KEYLESS_NODES in constants.py if the new node intentionally writes "
        "elsewhere, or add an entry to NODE_TO_RESULT_KEY."
    )


def test_import_time_assertion_does_not_raise():
    """constants.py must import without AssertionError when maps are consistent."""
    import importlib
    import backend.constants as c
    importlib.reload(c)