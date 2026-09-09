"""The entry plan has to survive the hop from ``restore_entry`` to its router.

Found on a real run, not in a test: a session restored from a round-1
annotation snapshot was routed to ``initialize_round`` even though the entry
node had just logged ``intent=graph_node target=bioinformatics_analysis``. The
plan was computed correctly and then thrown away one step later.

The mechanism is the point of this module. ``add_conditional_edges`` hands the
router the node's raw return only when the router's state parameter carries no
type annotation. Annotate it -- and every router in ``routing.py`` is annotated
-- and LangGraph reads the *declared channels* instead. A key that
``WorkflowState`` never declares is therefore invisible to the router that was
written to read it, and ``plan.get("intent", "normal")`` quietly answers with
its default.

Nothing raises. The graph runs, the session progresses, and the only evidence is
two adjacent log lines that disagree. Every test that existed asked
``build_restore_entry_plan`` for a plan and checked the plan; none asked the
graph where it actually went.
"""
from typing import TypedDict

import pytest

from langgraph.graph import END, StateGraph

from backend.constants import SEMANTIC_GATE_ACTION_NEXT_NODES
from backend.workflow.restore_entry import RestoreEntryPlan
from backend.workflow.routing import RoutingMixin
from backend.workflow.state import WorkflowState


class _Router(RoutingMixin):
    """The production routing method, with the one attribute it logs through."""

    session_id = "restore-entry-plan-channel"


def _graph_that_reports_where_it_went(plan: RestoreEntryPlan, arrivals: list[str]):
    """A two-node graph wired exactly like the real one at ``restore_entry``.

    The entry node returns the plan the way ``_restore_entry`` does, the branch
    is the production router, and every target it can name records its arrival.
    """
    def entry(state):
        return {**state, "_restore_entry_plan": plan.model_dump(mode="json")}

    def make_target(name):
        def target(state):
            arrivals.append(name)
            return state
        return target

    targets = ["initialize_round", *sorted(set(SEMANTIC_GATE_ACTION_NEXT_NODES.values()))]
    graph = StateGraph(WorkflowState)
    graph.add_node("restore_entry", entry)
    for name in targets:
        graph.add_node(name, make_target(name))
        graph.add_edge(name, END)
    graph.set_entry_point("restore_entry")
    graph.add_conditional_edges(
        "restore_entry",
        _Router()._route_from_restore_entry,
        {**{f"route_{name}": name for name in targets}, "hold": END},
    )
    return graph.compile()


@pytest.mark.parametrize("gate,target", sorted(SEMANTIC_GATE_ACTION_NEXT_NODES.items()))
def test_a_graph_node_plan_reaches_the_node_it_names(gate, target):
    """Every gate's declared continuation, taken by a compiled graph.

    Parametrised over the gates rather than asserted once, because the value
    that was being dropped was the same for all of them: a single example would
    not have distinguished "the plan is read" from "the default happens to
    match".
    """
    arrivals: list[str] = []
    plan = RestoreEntryPlan(intent="graph_node", target_node=target)
    _graph_that_reports_where_it_went(plan, arrivals).invoke({"current_round": 1})

    assert arrivals == [target], (
        f"gate {gate} names {target}; the graph went to {arrivals}"
    )


class _Declared(TypedDict, total=False):
    """A schema of this module's own, so the mechanism check below cannot start
    passing for an unrelated reason when ``WorkflowState`` changes."""

    kept: str


_SEEN: dict[str, str] = {}


def _annotated_router(state: "_Declared") -> str:
    """Annotated exactly as every router in ``routing.py`` is.

    The annotation must be resolvable at module scope: LangGraph evaluates it
    there, which is also why this pair cannot live inside the test function.
    """
    _SEEN["undeclared"] = state.get("undeclared", "MISSING")
    _SEEN["kept"] = state.get("kept", "MISSING")
    return "end"


def test_an_annotated_router_is_handed_only_the_declared_channels():
    """The mechanism, stated so the declaration is not deleted as redundant.

    ``_restore_entry_plan`` looks like scratch state that one node writes and
    one branch reads inside a single step, which is precisely the shape
    somebody removes from a TypedDict during a tidy-up. This is what happens
    when it is not declared.
    """
    _SEEN.clear()
    graph = StateGraph(_Declared)
    graph.add_node("n", lambda state: {**state, "kept": "yes", "undeclared": "yes"})
    graph.set_entry_point("n")
    graph.add_conditional_edges("n", _annotated_router, {"end": END})
    graph.compile().invoke({})

    assert _SEEN["kept"] == "yes"
    assert _SEEN["undeclared"] == "MISSING", (
        "An annotated router now sees undeclared keys; the declaration of "
        "_restore_entry_plan may no longer be load-bearing, but check before "
        "removing it -- an unannotated router still receives the raw return."
    )
    assert "_restore_entry_plan" in WorkflowState.__annotations__
