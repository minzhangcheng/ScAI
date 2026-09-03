"""A candidate an earlier gate passed over has to be able to come back.

Two leaks used to end with a direction disappearing for the rest of the run.

1. The integration gate deduplicated against ``user_chosen_directions``, which
   by design holds only the most recent batch, so every earlier cycle's choices
   were invisible and an already-investigated direction could be re-proposed as
   new.
2. Nothing ever reread the candidates an earlier gate proposed and passed over.
   A direction deferred at round 1 -- typically because a prerequisite had not
   run yet -- could only return if the model happened to reinvent it, and the
   prerequisite finishing changed nothing.

Everything here is domain-neutral: direction text, recorded prerequisite ids and
what the gate itself wrote down, never the subject matter.
"""
import pytest

from backend.models.schema_parts.session import DirectionSelectionRecord
from backend.utils.direction_payload import (
    MAX_CARRIED_CANDIDATES,
    investigated_direction_texts,
    merge_unpursued_candidates,
    unpursued_prior_candidates,
)


class _State:
    """Minimal stand-in carrying only the fields these helpers read."""

    def __init__(self, records=None, chosen=None):
        self.direction_selection_records = records or []
        self.user_chosen_directions = chosen or []


def _round1_record():
    return DirectionSelectionRecord(
        round_number=1,
        phase="exploration",
        event_type="pi_decide",
        candidates=[
            {"direction_id": "dir_001", "direction": "Copper handling",
             "prerequisite_dependencies": []},
            {"direction_id": "dir_002", "direction": "Zonation loss",
             "prerequisite_dependencies": []},
            {"direction_id": "dir_003", "direction": "Death execution mode",
             "prerequisite_dependencies": ["dir_001"]},
        ],
        pi_selected=[{"direction": "Copper handling", "notes": ""}],
    )


def _cycle1_record(**kw):
    base = dict(
        round_number=4,
        phase="integration",
        integration_cycle=1,
        event_type="pi_decide",
        candidates=[{"direction_id": "NEW-01", "direction": "Interferon source",
                     "prerequisite_dependencies": []}],
        pi_selected=[{"direction": "Interferon source", "notes": ""}],
    )
    base.update(kw)
    return DirectionSelectionRecord(**base)


# --- 1. what counts as already investigated ----------------------------------


def test_investigated_directions_span_every_gate_not_just_the_last_batch():
    state = _State(
        records=[_round1_record(), _cycle1_record()],
        chosen=[{"direction": "Interferon source", "notes": ""}],
    )

    assert investigated_direction_texts(state) == ["Copper handling", "Interferon source"]


def test_investigated_directions_fall_back_to_the_legacy_field():
    state = _State(chosen=[{"direction": "Copper handling", "notes": ""}])

    assert investigated_direction_texts(state) == ["Copper handling"]


def test_a_restatement_of_an_investigated_direction_is_not_carried():
    # Exact matching re-offered work that had already run, because a gate
    # restates the same question in its own words.
    record = _round1_record()
    record.candidates.append(
        {"direction_id": "dir_004", "direction": "Copper handling in hepatocytes"}
    )
    state = _State(records=[record])

    directions = [c["direction"] for c in unpursued_prior_candidates(state)]

    assert "Copper handling in hepatocytes" not in directions


def test_a_short_placeholder_direction_does_not_swallow_the_whole_list():
    # Containment matching is what lets a restatement be recognised, but a
    # stored placeholder like "A" is a substring of nearly every candidate, so
    # unguarded containment marks everything as already investigated.
    state = _State(records=[_round1_record()], chosen=[{"direction": "A"}, {"direction": "B"}])

    directions = [c["direction"] for c in unpursued_prior_candidates(state)]

    assert directions == ["Zonation loss", "Death execution mode"]


def test_unpursued_candidates_exclude_the_ones_already_investigated():
    state = _State(records=[_round1_record()])

    directions = [c["direction"] for c in unpursued_prior_candidates(state)]

    assert directions == ["Zonation loss", "Death execution mode"]


# --- 2. prerequisites ---------------------------------------------------------


def test_prerequisite_that_has_run_is_reported_as_satisfied():
    state = _State(records=[_round1_record()])

    by_direction = {c["direction"]: c for c in unpursued_prior_candidates(state)}

    assert by_direction["Death execution mode"]["prerequisites_satisfied"] is True
    assert by_direction["Death execution mode"]["unmet_prerequisites"] == []


def test_prerequisite_that_has_not_run_is_named():
    record = _round1_record()
    record.pi_selected = []
    state = _State(records=[record])

    by_direction = {c["direction"]: c for c in unpursued_prior_candidates(state)}

    assert by_direction["Death execution mode"]["prerequisites_satisfied"] is False
    assert by_direction["Death execution mode"]["unmet_prerequisites"] == ["Copper handling"]


def test_a_prerequisite_id_that_resolves_to_nothing_is_not_named_as_a_direction():
    # A record's candidates are the scored survivors, so a survivor can name a
    # sibling that is not in the list. Writing the raw id token into
    # unmet_prerequisites put "dir_099" into the prompt as a direction name.
    record = DirectionSelectionRecord(
        event_type="pi_decide",
        candidates=[{"direction_id": "dir_005", "direction": "Dependent direction",
                     "prerequisite_dependencies": ["dir_099"]}],
    )
    state = _State(records=[record])

    carried = unpursued_prior_candidates(state)[0]

    assert carried["unmet_prerequisites"] == []
    assert "dir_099" not in str(carried["unmet_prerequisites"])


def test_a_resolved_prerequisite_is_frozen_and_not_re_resolved_later():
    # Carried candidates get re-stored by the gate that saw them, keeping ids
    # minted by the gate that first issued them. Re-resolving those ids against
    # a later gate's id space names an unrelated direction.
    later = DirectionSelectionRecord(
        event_type="pi_decide",
        integration_cycle=1,
        phase="integration",
        candidates=[
            {"direction_id": "dir_001", "direction": "Something else entirely"},
            {"direction_id": "dir_003", "direction": "Death execution mode",
             "prerequisite_dependencies": ["dir_001"],
             "unmet_prerequisites": ["Copper handling"]},
        ],
    )
    state = _State(records=[later])

    by_direction = {c["direction"]: c for c in unpursued_prior_candidates(state)}

    assert by_direction["Death execution mode"]["unmet_prerequisites"] == ["Copper handling"]


# --- 3. which copy, and which gate ------------------------------------------


def test_the_most_recently_scored_copy_of_a_direction_wins():
    # Walking oldest-first handed the gate the round-1 version of a direction
    # that a later gate had already re-proposed and scored.
    later = DirectionSelectionRecord(
        event_type="pi_decide",
        integration_cycle=1,
        phase="integration",
        candidates=[{"direction_id": "NEW-07", "direction": "Zonation loss",
                     "novelty_score": 8, "priority_rank": 2}],
    )
    state = _State(records=[_round1_record(), later])

    by_direction = {c["direction"]: c for c in unpursued_prior_candidates(state)}

    assert by_direction["Zonation loss"]["direction_id"] == "NEW-07"
    assert by_direction["Zonation loss"]["novelty_score"] == 8


def test_candidates_are_tagged_with_the_gate_they_came_from():
    state = _State(records=[_round1_record()])

    carried = unpursued_prior_candidates(state)[0]

    assert carried["carried_from_gate"].startswith("Round 1")


def test_a_declined_direction_is_carried_with_the_reason_attached():
    # Suppressing it forever repeats the original bug in mirror image: the
    # results that arrived since are exactly what could change the answer.
    record = _round1_record()
    record.declined = [{
        "direction": "Zonation loss",
        "reason_not_selected": "No zonation markers had been measured yet.",
        "follow_up_value": "high",
    }]
    state = _State(records=[record])

    by_direction = {c["direction"]: c for c in unpursued_prior_candidates(state)}
    declined = by_direction["Zonation loss"]["previously_declined"]

    assert "zonation markers" in declined["reason"]
    assert declined["follow_up_value"] == "high"
    assert declined["gate"].startswith("Round 1")


def test_reading_the_records_never_modifies_them():
    record = _round1_record()
    unpursued_prior_candidates(_State(records=[record]))

    assert "prerequisites_satisfied" not in record.candidates[2]
    assert "carried_from_gate" not in record.candidates[2]


def test_no_internal_ordering_key_leaks_into_the_gate():
    merged = merge_unpursued_candidates(_State(records=[_round1_record()]), [])

    assert all("_carry_recency" not in c for c in merged)


# --- 4. merging into a fresh batch -------------------------------------------


def test_fresh_candidates_keep_precedence_and_carried_ones_follow():
    state = _State(records=[_round1_record()])
    fresh = [{"direction_id": "NEW-01", "direction": "Interferon source"}]

    merged = merge_unpursued_candidates(state, fresh)

    assert merged[0]["direction_id"] == "NEW-01"
    assert [c["direction"] for c in merged[1:]] == ["Zonation loss", "Death execution mode"]


def test_a_fresh_restatement_of_a_carried_direction_wins():
    state = _State(records=[_round1_record()])
    fresh = [{"direction_id": "NEW-01", "direction": "Death execution mode in hepatocytes"}]

    merged = merge_unpursued_candidates(state, fresh)

    assert [c["direction"] for c in merged].count("Death execution mode") == 0
    assert merged[0]["direction_id"] == "NEW-01"


def test_candidates_with_unmet_prerequisites_sort_behind_ready_ones():
    record = _round1_record()
    record.candidates.append(
        {"direction_id": "dir_004", "direction": "Blocked direction",
         "prerequisite_dependencies": ["dir_002"]},
    )
    state = _State(records=[record])

    carried = [c["direction"] for c in merge_unpursued_candidates(state, [])]

    assert carried.index("Death execution mode") < carried.index("Blocked direction")


def test_the_cap_keeps_the_most_recently_deferred_not_the_oldest_backlog():
    # A stable sort over a chronological list spent every slot on the oldest
    # gate's leftovers and cut what the last gate had just held back.
    old = DirectionSelectionRecord(
        event_type="pi_decide",
        candidates=[{"direction_id": f"old_{i}", "direction": f"Old direction {i}"}
                    for i in range(MAX_CARRIED_CANDIDATES + 3)],
    )
    recent = DirectionSelectionRecord(
        event_type="pi_decide",
        integration_cycle=1,
        phase="integration",
        candidates=[{"direction_id": "recent_1", "direction": "Just deferred"}],
    )
    state = _State(records=[old, recent])

    carried = [c["direction"] for c in merge_unpursued_candidates(state, [])]

    assert len(carried) == MAX_CARRIED_CANDIDATES
    assert "Just deferred" in carried


def test_merge_is_a_no_op_when_no_earlier_gate_left_anything_behind():
    fresh = [{"direction_id": "NEW-01", "direction": "Interferon source"}]

    assert merge_unpursued_candidates(_State(), fresh) == fresh


def test_merge_returns_a_list_even_for_an_empty_batch():
    assert merge_unpursued_candidates(_State(), None) == []
    assert merge_unpursued_candidates(_State(), []) == []


# --- 5. wiring ----------------------------------------------------------------


def _state_with_two_gates():
    from backend.models.schemas import AnalysisState

    state = AnalysisState(name="t", user_id="u", initial_instruction="objective")
    state.direction_selection_records = [_round1_record(), _cycle1_record()]
    state.user_chosen_directions = [{"direction": "Interferon source", "notes": ""}]
    return state


@pytest.mark.asyncio
async def test_candidate_prompt_lists_choices_from_every_earlier_gate(monkeypatch):
    from backend.tests.test_integration_gate_no_silent_drop import _make_pi_agent

    agent = _make_pi_agent()
    captured = {}

    async def capture(prompt, temperature=0.3):
        captured["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(agent, "chat", capture)
    await agent.generate_integration_direction_candidates(
        _state_with_two_gates(), [], "integration report", "new literature"
    )

    assert "Copper handling" in captured["prompt"]
    assert "Interferon source" in captured["prompt"]


@pytest.mark.asyncio
async def test_selection_prompt_lists_choices_from_every_earlier_gate(monkeypatch):
    from backend.tests.test_integration_gate_no_silent_drop import _make_pi_agent

    agent = _make_pi_agent()
    captured = {}

    async def capture(prompt, temperature=0.3):
        captured["prompt"] = prompt
        return '{"selected_directions": [], "rejected_directions": [], "selection_rationale": ""}'

    monkeypatch.setattr(agent, "chat", capture)
    await agent.select_integration_directions(
        _state_with_two_gates(),
        [{"direction_id": "NEW-02", "direction": "Something new"}],
        "integration report",
    )

    assert "Copper handling" in captured["prompt"]


def test_integration_node_carries_an_unpursued_candidate_into_the_gate():
    import asyncio
    from backend.tests.test_integration_synthesis_node import _Harness, _state_with_batch

    harness = _Harness([{"direction_id": "NEW-01", "direction": "Interferon source"}])
    state = _state_with_batch()
    state.direction_selection_records = [_round1_record()]

    asyncio.run(harness._integration_synthesis({"analysis_state": state}))

    gate = [c["direction"] for c in state.integration_gate_candidates]
    assert "Interferon source" in gate
    assert "Death execution mode" in gate


def test_carry_forward_runs_after_the_generation_failure_is_recorded():
    # Recycling first made raw_candidates truthy, which rendered the
    # "generation was unavailable" finalize reason unreachable and presented
    # stale candidates as this cycle's proposal.
    import inspect
    from backend.workflow import integration

    src = inspect.getsource(integration)
    failure_at = src.index('last_candidate_generation_error')
    merge_at = src.index('merge_unpursued_candidates(analysis_state')

    assert failure_at < merge_at
