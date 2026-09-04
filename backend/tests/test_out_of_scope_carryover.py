"""Work a previous round ruled out must not be silently re-planned.

`out_of_scope` is where the PI records a follow-up that needs data or
experiments this session does not have. Nothing read it, so the next round's
planner never saw it, while the *findings* it did see said the work was
unfinished — and the same provenance and manifest steps were planned again
every round.
"""

import ast
import inspect

from backend.agents.pi.instructions import build_out_of_scope_context
from backend.models.schemas import AnalysisState


def _state_with_reviews(*out_of_scope_lists):
    state = AnalysisState(name="s", user_id="u")
    state.post_execution_reviews = [
        {
            "round_number": str(index + 1),
            "review_plan": {"out_of_scope": list(items), "verdict": "supplement"},
        }
        for index, items in enumerate(out_of_scope_lists)
    ]
    return state


def test_context_names_the_ruled_out_work_and_forbids_re_planning():
    state = _state_with_reviews(
        ["Cross-modal sample correspondence requires animal identifiers "
         "not present in the live kernel."],
        ["A versioned animal and aliquot manifest is external metadata."],
    )

    context = build_out_of_scope_context(state)

    assert "Cross-modal sample correspondence" in context
    assert "versioned animal and aliquot manifest" in context
    assert "already ruled out" in context.lower()


def test_repeated_findings_collapse_to_one_line():
    """The same limit recurs every round; it is one fact, not four."""
    repeated = "Animal-level replication requires new biological specimens."
    state = _state_with_reviews([repeated], [repeated], [repeated, "Another limit."])

    context = build_out_of_scope_context(state)

    assert context.count(repeated) == 1
    assert "Another limit." in context


def test_the_cap_keeps_the_most_recent_rulings():
    """A long session accumulates more of these than the cap admits.

    Measured on a real 8-round session: 41 distinct entries. Truncating from the
    front would show the planner round 1's rulings and hide the ones the most
    recent review just made, which are the ones bearing on the round about to be
    planned.
    """
    early = [f"Early limit {i}." for i in range(20)]
    late = ["The most recent limit."]
    state = _state_with_reviews(early, late)

    context = build_out_of_scope_context(state)

    assert "The most recent limit." in context
    assert "Early limit 0." not in context
    assert "Early limit 19." in context


def test_no_reviews_yields_no_section():
    assert build_out_of_scope_context(AnalysisState(name="s", user_id="u")) == ""
    assert build_out_of_scope_context(_state_with_reviews([], [])) == ""


def test_malformed_review_records_do_not_raise():
    state = AnalysisState(name="s", user_id="u")
    state.post_execution_reviews = [
        {}, {"review_plan": None}, {"review_plan": {"out_of_scope": None}},
        {"review_plan": {"out_of_scope": ["kept"]}}, "not a dict",
    ]

    context = build_out_of_scope_context(state)

    assert "kept" in context


def test_innovation_instruction_actually_uses_the_context():
    """A builder nothing calls would leave the loop exactly as it was."""
    from backend.agents.pi import instructions as instructions_module

    source = inspect.getsource(instructions_module)
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_out_of_scope_context" in called

    innovation = inspect.getsource(
        instructions_module.InstructionsMixin.innovation_instruction
    )
    assert "out_of_scope" in innovation
