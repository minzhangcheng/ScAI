"""A per-gate direction cap, off by default, that cannot strand a prerequisite.

``max_rounds`` is extended by exactly ``len(selected_dirs)`` at both PI-decide
gates, so every extra direction is a full innovation round. How long a run may
grow is the operator's call, so the cap is a per-session setting and its default
is no cap at all.

When a cap IS set, ``select_best_directions`` has already appended the
prerequisites its closure promoted to the END of the selection list. A plain
``selected[:limit]`` therefore removes exactly the direction that has to run
first and commits the ones that cannot run without it -- the inversion
``test_pi_prerequisite_closure.py`` exists to prevent, reintroduced one layer
down where that test cannot see it.
"""
import inspect

from backend.models.schemas import AnalysisState, DEFAULT_MAX_DIRECTIONS_PER_GATE
from backend.workflow.planning_parts.shared import (
    _declined_directions,
    _direction_budget,
    _selected_directions_from_decision,
    _selected_directions_within_budget,
    note_deferred_directions,
)


def _dirs(n):
    return [{"direction": f"Direction {i}", "notes": ""} for i in range(1, n + 1)]


def _state(limit=None):
    state = AnalysisState(name="t", user_id="u", initial_instruction="o")
    if limit is not None:
        state.max_directions_per_gate = limit
    return state


# --- default is no cap --------------------------------------------------------


def test_the_shipped_default_is_no_cap():
    assert DEFAULT_MAX_DIRECTIONS_PER_GATE == 0
    assert _state().max_directions_per_gate == 0


def test_without_a_cap_every_selected_direction_is_committed():
    committed, deferred = _selected_directions_within_budget({"selected_directions": _dirs(4)}, _state())

    assert len(committed) == 4
    assert deferred == []


def test_a_missing_state_is_treated_as_uncapped():
    committed, deferred = _selected_directions_within_budget({"selected_directions": _dirs(4)}, None)

    assert len(committed) == 4
    assert deferred == []


def test_a_nonsense_cap_value_is_treated_as_uncapped():
    state = _state()
    state.__dict__["max_directions_per_gate"] = "two"

    assert _direction_budget(state) == 0


# --- with a cap ---------------------------------------------------------------


def test_a_cap_splits_the_selection_into_now_and_later():
    committed, deferred = _selected_directions_within_budget({"selected_directions": _dirs(4)}, _state(2))

    assert [d["direction"] for d in committed] == ["Direction 1", "Direction 2"]
    assert [d["direction"] for d in deferred] == ["Direction 3", "Direction 4"]


def test_a_selection_within_the_cap_is_untouched():
    committed, deferred = _selected_directions_within_budget({"selected_directions": _dirs(2)}, _state(2))

    assert len(committed) == 2
    assert deferred == []


def test_choosing_nothing_never_becomes_a_cap_event():
    committed, deferred = _selected_directions_within_budget({"selected_directions": []}, _state(2))

    assert committed == []
    assert deferred == []


def test_the_cap_reads_the_same_shapes_as_the_plain_extractor():
    for decision in ({"selected_directions": "bad"}, {}, [], _dirs(1)):
        expected = _selected_directions_from_decision(decision)[:2]
        assert _selected_directions_within_budget(decision, _state(2))[0] == expected


# --- prerequisites are never what gets cut ------------------------------------


def _selection_with_promoted_prerequisite():
    # The shape select_best_directions produces: the closure appends the
    # prerequisite AFTER the directions that depend on it.
    return {"selected_directions": [
        {"direction": "Dependent", "notes": ""},
        {"direction": "Other", "notes": ""},
        {"direction": "Foundation", "notes": "Added to satisfy prerequisite of 'Dependent'.",
         "prerequisite_for": "Dependent"},
    ]}


def test_a_promoted_prerequisite_is_committed_even_past_the_cap():
    committed, deferred = _selected_directions_within_budget(
        _selection_with_promoted_prerequisite(), _state(2)
    )

    names = [d["direction"] for d in committed]
    assert "Foundation" in names, "the prerequisite of a committed direction was deferred"
    assert "Dependent" in names
    assert [d["direction"] for d in deferred] == ["Other"]


def test_a_prerequisite_of_a_deferred_direction_is_deferred_with_it():
    decision = {"selected_directions": [
        {"direction": "First", "notes": ""},
        {"direction": "Dependent", "notes": ""},
        {"direction": "Foundation", "notes": "", "prerequisite_for": "Dependent"},
    ]}

    committed, deferred = _selected_directions_within_budget(decision, _state(1))

    assert [d["direction"] for d in committed] == ["First"]
    assert sorted(d["direction"] for d in deferred) == ["Dependent", "Foundation"]


def test_a_prerequisite_chain_is_pulled_in_whole():
    decision = {"selected_directions": [
        {"direction": "Top", "notes": ""},
        {"direction": "Middle", "notes": "", "prerequisite_for": "Top"},
        {"direction": "Bottom", "notes": "", "prerequisite_for": "Middle"},
    ]}

    committed, deferred = _selected_directions_within_budget(decision, _state(1))

    assert sorted(d["direction"] for d in committed) == ["Bottom", "Middle", "Top"]
    assert deferred == []


def test_the_closure_tags_promoted_prerequisites_so_the_cap_can_see_them():
    from backend.agents.pi import direction_selection

    src = inspect.getsource(direction_selection.DirectionSelectionMixin.select_best_directions)
    assert '"prerequisite_for"' in src


# --- the record has to show what happened ------------------------------------


def test_deferred_directions_are_named_in_the_rationale():
    rationale = note_deferred_directions("Picked the two strongest.", _dirs(4)[2:], 2)

    assert "Picked the two strongest." in rationale
    assert "Direction 3" in rationale and "Direction 4" in rationale
    assert "2" in rationale


def test_rationale_is_unchanged_when_nothing_was_deferred():
    assert note_deferred_directions("Picked both.", [], 2) == "Picked both."


def test_rationale_note_stands_alone_when_the_model_gave_none():
    assert note_deferred_directions("", _dirs(3)[2:], 2).strip() != ""


def test_what_the_gate_declined_is_recorded_with_its_reason():
    decision = {"rejected_directions": [
        {"direction": "Zonation loss", "reason_not_selected": "Not measured yet",
         "follow_up_value": "high", "dependency_note": ""},
        {"not_a_direction": True},
    ]}

    declined = _declined_directions(decision)

    assert len(declined) == 1
    assert declined[0]["direction"] == "Zonation loss"
    assert declined[0]["reason_not_selected"] == "Not measured yet"


def test_declined_reads_junk_without_raising():
    assert _declined_directions(None) == []
    assert _declined_directions([]) == []
    assert _declined_directions({"rejected_directions": ["junk", None]}) == []


# --- wiring -------------------------------------------------------------------


def test_both_pi_decide_gates_go_through_the_budget():
    from backend.workflow.planning_parts import pi

    src = inspect.getsource(pi)
    assert src.count("_selected_directions_within_budget(") == 2
    assert "_selected_directions_from_decision(selection_decision)" not in src


def test_both_gates_record_what_they_declined():
    from backend.workflow.planning_parts import pi

    src = inspect.getsource(pi)
    assert src.count("_declined_directions(selection_decision)") == 2


def test_the_stored_selection_summary_matches_what_was_committed():
    # The summary is copied verbatim into the snapshot gate capsule, so a
    # restored session must not show directions the cap held back.
    from backend.workflow.planning_parts import pi

    src = inspect.getsource(pi)
    assert "direction_selection_summary = selection_decision\n" not in src
    assert '"selected_directions": list(selected_dirs)' in src


def test_round1_gate_extends_max_rounds_only_by_what_it_committed():
    import asyncio
    import types

    from backend.workflow.planning_parts.pi import _apply_round1_direction_selection

    state = _state(2)
    node = types.SimpleNamespace(session_id="s")

    async def _instruction(state, direction, notes=""):
        return {"bioinformatics_tasks": []}

    node.pi_agent = types.SimpleNamespace(innovation_instruction=_instruction)
    committed, _ = _selected_directions_within_budget({"selected_directions": _dirs(4)}, state)

    asyncio.run(_apply_round1_direction_selection(node, state, committed))

    assert len(state.user_chosen_directions) == 2
    assert state.max_rounds == 3


def test_explicit_user_selection_is_not_capped():
    # routes/interaction.py installs a user's picks directly; the cap governs
    # the PI's own decisions, not an instruction the user typed.
    from backend.routes import interaction

    src = inspect.getsource(interaction)
    assert "_selected_directions_within_budget" not in src
    assert "max_directions_per_gate" not in src


def test_the_cap_is_settable_per_session():
    from backend.routes import session_config

    src = inspect.getsource(session_config)
    assert "max-directions-per-gate" in src
    assert "state.max_directions_per_gate = body.max_directions_per_gate" in src
