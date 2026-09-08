"""Scoring ranks the integration candidates; it does not remove them.

The gate used to tell its scorer to "Drop candidates that are NOT novel
(already established in literature) or lack observed supporting data". Both
criteria are unsatisfiable for the thing being judged:

* Prior-art volume measures how well studied an area is, never whether the
  direction is worth running for this project. It is also the only signal left
  when the novelty search fails, which made the best-documented candidate the
  most likely to be removed.
* An integration candidate is by definition a direction no round has executed,
  so it cannot have observed support yet. The criterion made producing the data
  a precondition for being allowed to produce it.

A candidate missing from the scorer's reply is gone for good -- nothing
downstream rereads raw_candidates, and the integration node reads an empty
scored list as the scientific statement that no direction is worth another
round. So the removal power is gone and an omission is repaired, unconditionally:
the earlier design only repaired omissions whose novelty verdict was
untrustworthy, which left the drop standing in precisely the well-documented
case the rule was written for.
"""
import inspect
import json
from types import SimpleNamespace

import pytest

from backend.agents.pi_agent import PIAgent
from backend.agents.pi.shared import _coerce_priority_rank, _restore_unscored_candidates


def _make_pi_agent():
    from unittest.mock import MagicMock
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _raw_pair():
    return [
        {"direction_id": "int_001", "direction": "Kept direction"},
        {"direction_id": "int_002", "direction": "Omitted direction"},
    ]


# --- the omission is repaired whatever the novelty record says ----------------


@pytest.mark.parametrize(
    "novelty_status",
    ["unknown_due_to_timeout", "unknown_due_to_failure", "unknown", "known_area",
     "adjacent_work_only", "genuine_novelty"],
)
def test_omitted_candidate_returns_whatever_the_novelty_verdict_was(novelty_status):
    # A retrieval outcome is not a decision. known_area now requires a direct
    # hit and adjacent_work_only carries "related work only", but neither is
    # grounds to drop a candidate: that would make prior-art volume the verdict,
    # the exact thing this gate must not do.
    scored = [{"direction_id": "int_001", "direction": "Kept direction", "priority_rank": 1}]

    restored = _restore_unscored_candidates(scored, _raw_pair())

    assert [c["direction_id"] for c in restored] == ["int_001", "int_002"]


def test_restored_candidate_is_flagged_as_unscored_not_as_rejected():
    restored = _restore_unscored_candidates(
        [{"direction_id": "int_001", "direction": "Kept direction"}], _raw_pair()
    )

    assert "unscored_by_gate" in restored[-1]["risk_flags"]


def test_guard_is_inert_when_the_scorer_returned_everything():
    raw = _raw_pair()
    scored = [dict(raw[0]), dict(raw[1])]

    restored = _restore_unscored_candidates(scored, raw)

    assert [c["direction_id"] for c in restored] == ["int_001", "int_002"]
    assert "unscored_by_gate" not in (restored[1].get("risk_flags") or [])


def test_match_falls_back_to_direction_text_when_the_scorer_drifts_the_id():
    scored = [{"direction_id": "int_1", "direction": "Kept direction"}]

    restored = _restore_unscored_candidates(scored, _raw_pair())

    assert len(restored) == 2
    assert restored[-1]["direction"] == "Omitted direction"


def test_text_match_ignores_case_and_spacing():
    scored = [{"direction_id": "", "direction": "  kept   DIRECTION "}]

    restored = _restore_unscored_candidates(scored, _raw_pair())

    assert [c["direction"] for c in restored if c.get("direction_id")] == ["Omitted direction"]


# --- restored candidates rank last, in every shape the scorer returns ---------


def test_restored_candidate_ranks_below_a_scored_one():
    scored = [{"direction_id": "int_001", "direction": "Kept direction", "priority_rank": 4}]

    restored = _restore_unscored_candidates(scored, _raw_pair())

    assert restored[-1]["priority_rank"] > restored[0]["priority_rank"]


def test_restored_candidates_do_not_take_top_rank_when_nothing_was_scored():
    # scored == [] made next_rank 1, so the least-verified entries arrived at
    # the selection step presented as the highest-priority ones.
    restored = _restore_unscored_candidates([], _raw_pair())

    assert [c["priority_rank"] for c in restored] == [1, 2]
    assert all("unscored_by_gate" in c["risk_flags"] for c in restored)


def test_string_ranks_are_coerced_so_restored_entries_still_sort_last():
    scored = [
        {"direction_id": "int_001", "direction": "Kept direction", "priority_rank": "3"},
    ]

    restored = _restore_unscored_candidates(scored, _raw_pair())

    assert restored[0]["priority_rank"] == 3
    assert restored[-1]["priority_rank"] == 4


def test_a_boolean_rank_is_not_read_as_rank_one():
    assert _coerce_priority_rank(True) is None
    assert _coerce_priority_rank(False) is None
    assert _coerce_priority_rank(2) == 2
    assert _coerce_priority_rank(" 7 ") == 7
    assert _coerce_priority_rank("high") is None


def test_unusable_rank_is_removed_rather_than_left_mixed():
    scored = [{"direction_id": "int_001", "direction": "Kept direction", "priority_rank": "high"}]

    restored = _restore_unscored_candidates(scored, _raw_pair())

    assert "priority_rank" not in restored[0]
    assert restored[-1]["priority_rank"] == 2


# --- the scorer call itself ---------------------------------------------------


@pytest.mark.asyncio
async def test_scorer_returns_every_candidate_the_model_omitted(monkeypatch):
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps([{"direction_id": "int_001", "direction": "Kept direction", "priority_rank": 1}])

    monkeypatch.setattr(agent, "chat", fake_chat)

    scored = await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="x"), _raw_pair(), []
    )

    assert [c["direction_id"] for c in scored] == ["int_001", "int_002"]
    assert agent.last_candidate_scoring_error is None


@pytest.mark.asyncio
async def test_unreadable_scorer_reply_is_retried(monkeypatch):
    agent = _make_pi_agent()
    calls = []

    async def fake_chat(prompt, temperature=0.3):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json at all"
        return json.dumps([{"direction_id": "int_001", "direction": "Kept direction"}])

    monkeypatch.setattr(agent, "chat", fake_chat)

    scored = await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="x"), _raw_pair(), []
    )

    assert len(calls) == 2
    assert [c["direction_id"] for c in scored] == ["int_001", "int_002"]


@pytest.mark.asyncio
async def test_unreadable_after_retry_passes_candidates_on_and_says_why(monkeypatch):
    # Returning [] here would reach the integration node as "no direction is
    # worth another round" -- a scientific conclusion drawn from a parse error.
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        return "still not json"

    monkeypatch.setattr(agent, "chat", fake_chat)

    scored = await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="x"), _raw_pair(), []
    )

    assert [c["direction_id"] for c in scored] == ["int_001", "int_002"]
    assert "DIRECTION SCORING UNAVAILABLE" in agent.last_candidate_scoring_error


@pytest.mark.asyncio
async def test_a_readable_reply_clears_a_previous_failure(monkeypatch):
    agent = _make_pi_agent()
    agent.last_candidate_scoring_error = "stale failure from an earlier cycle"

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps([{"direction_id": "int_001", "direction": "Kept direction"}])

    monkeypatch.setattr(agent, "chat", fake_chat)
    await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="x"), _raw_pair(), []
    )

    assert agent.last_candidate_scoring_error is None


# --- the removal instructions are gone ---------------------------------------


def test_scorer_prompt_no_longer_authorises_removal():
    src = inspect.getsource(PIAgent.score_integration_directions_with_evidence)
    assert "already established in literature" not in src
    assert "lack observed supporting data" not in src
    assert "Drop candidates" not in src


def test_scorer_prompt_says_prior_art_volume_settles_nothing():
    src = inspect.getsource(PIAgent.score_integration_directions_with_evidence).lower()
    assert "prior-art volume" in src


def test_no_trustworthiness_predicate_gates_the_repair():
    # The repair used to be conditional on the novelty verdict being unreliable.
    from backend.agents.pi import direction_selection
    assert not hasattr(PIAgent, "_novelty_verdict_is_trustworthy")
    assert "_novelty_verdict_is_trustworthy" not in inspect.getsource(direction_selection)


def test_gate_selection_does_not_demand_evidence_only_the_round_could_produce():
    src = inspect.getsource(PIAgent.select_integration_directions)
    assert "Well-supported by concrete evidence from the integration analysis" not in src


def test_gate_selection_separates_unmeasured_from_contradicted():
    src = inspect.getsource(PIAgent.select_integration_directions).lower()
    assert "merely unmeasured" in src
    assert "contradicted by this project" in src or "own results already argue against" in src
