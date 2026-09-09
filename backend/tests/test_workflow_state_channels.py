"""Every workflow-state key a node writes must be one the graph carries.

This guard existed once. It was written on 2026-08-13 after `_restore_entry_plan`
shipped undeclared and collapsed every restore continuation into
`initialize_round`; the 0.48.1 rebaseline replaced the code tree and took the
fix and the guard with it, and the same defect was found again on 2026-09-09 by
watching a real session restore into the wrong node. Four weeks, no failing
test, no exception, no log line that looked wrong on its own.

The mechanism, because it is not guessable from the code:
``add_conditional_edges`` builds its branch with schema inference on, and
LangGraph merges the producing node's raw return into the router's input *only
when the inferred schema is None*. Every router in ``routing.py`` annotates
``state: "WorkflowState"``, so a schema is inferred, the passthrough is off, and
a key the TypedDict does not declare arrives absent. ``plan.get("intent",
"normal")`` then answers with its default. Nodes are handed the channel state
too, so an undeclared key is equally invisible to them.

Declaring a key is not free either: declared channels persist in the
checkpoint, so a one-shot value has to be consumed by whoever reads it or every
later round sees the old one. That is why two keys below are deliberately
*not* declared, and why this file names them rather than leaving the next
reader to wonder.
"""
import ast
from pathlib import Path

import pytest

from backend.workflow.state import WorkflowState

WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"

#: Names that hold a WorkflowState in this package. A subscript or ``.get`` on
#: one of these with an underscore key is a channel access.
_STATE_NAMES = frozenset({
    "state", "result", "returned", "workflow_state", "graph_state",
})

#: Underscore keys that are deliberately undeclared, with the reason. Adding to
#: this list is a decision about the graph, not a way to quiet the test.
_DELIBERATELY_UNDECLARED = {
    "_restore_resume_node": (
        "Re-emitted by initialize_round and read by _should_continue_after_init, "
        "both inert because the key is undeclared. Declaring it would make the "
        "branch live, and a one-shot target that initialize_round does not "
        "consume would re-route every later round to it. The graph-node jump is "
        "performed by _route_from_restore_entry off _restore_entry_plan instead."
    ),
    "_preserve_round_on_initialize": (
        "Set by restore_entry for explicit_start and read by _initialize_round, "
        "which cannot see it. Harmless: preserve_restore in that node also tests "
        "last_restored_snapshot_id, which every restored session carries, so the "
        "round is preserved by the other half of the same condition."
    ),
}


def _underscore_keys() -> dict[str, set[str]]:
    """Every underscore key this package reads or writes on a workflow state."""
    found: dict[str, set[str]] = {}
    for path in sorted(WORKFLOW.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            owner = key = None
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if isinstance(node.value, ast.Name):
                    owner, key = node.value.id, node.slice.value
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"get", "pop", "setdefault"} and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(
                        node.func.value, ast.Name
                    ):
                        owner, key = node.func.value.id, argument.value
            if (
                owner in _STATE_NAMES
                and isinstance(key, str)
                and key.startswith("_")
            ):
                found.setdefault(key, set()).add(str(path.relative_to(WORKFLOW)))
    return found


def test_every_underscore_channel_is_declared_or_knowingly_not():
    """The whole class of defect, not the one instance that was found twice."""
    declared = set(WorkflowState.__annotations__)
    undeclared = {
        key: sorted(files)
        for key, files in _underscore_keys().items()
        if key not in declared and key not in _DELIBERATELY_UNDECLARED
    }
    assert undeclared == {}, (
        "These keys are written or read on a workflow state and are not "
        "channels, so a router or node reading one gets nothing and takes its "
        "default branch without raising: "
        f"{undeclared}. Declare them in WorkflowState, or add them to "
        "_DELIBERATELY_UNDECLARED with the reason."
    )


def test_the_restore_entry_plan_is_declared():
    """Named on its own because it is the one that has now shipped broken twice."""
    assert "_restore_entry_plan" in WorkflowState.__annotations__


@pytest.mark.parametrize("key", sorted(_DELIBERATELY_UNDECLARED))
def test_a_key_left_undeclared_on_purpose_is_still_used(key):
    """An exemption for a key nobody touches any more is a stale exemption."""
    assert key in _underscore_keys(), (
        f"{key} is exempted from the channel rule but no longer appears in "
        "backend/workflow/. Remove it from _DELIBERATELY_UNDECLARED."
    )


@pytest.mark.parametrize("key", sorted(_DELIBERATELY_UNDECLARED))
def test_a_key_left_undeclared_on_purpose_is_not_also_declared(key):
    """The exemption and the declaration cannot both be right."""
    assert key not in WorkflowState.__annotations__, (
        f"{key} is now a declared channel, so the reason recorded in "
        "_DELIBERATELY_UNDECLARED no longer applies. Check that whoever reads "
        "it also consumes it, then remove the exemption."
    )
