"""End-to-end: a direction proposed at the integration gate has to survive it.

This file deliberately imports nothing that the fix introduced, so it collects
and runs against the unfixed tree and fails there for behavioural reasons
rather than on an ImportError.

The run that motivated it proposed a hepatocyte-death direction twice and lost
it twice. Reconstructed from that session's own logs, the candidates serialised
to 7-10 KB each, the scoring prompt truncated the list at 8000 characters, and
every candidate past the first was therefore absent from the prompt entirely.
Whatever the model then returned, the missing ones could not come back: nothing
downstream rereads raw_candidates, and the integration node reads an empty
scored list as "no direction is worth another round".
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents.pi_agent import PIAgent


def _agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _realistic_candidates(n=3, size=3000):
    """Candidates the size the running system actually produces."""
    return [
        {
            "direction_id": f"NEW-{i:02d}",
            "direction": f"Research direction number {i} stated in full",
            "core_observation": "o" * size,
            "main_hypothesis": "h" * size,
            "rationale": "r" * size,
            "alternative_hypotheses": ["alt " + "x" * 200 for _ in range(5)],
            "artifact_risks": ["risk " + "y" * 200 for _ in range(5)],
        }
        for i in range(1, n + 1)
    ]


def _novelty_records(candidates, status="known_area"):
    return [
        {
            "direction_id": c["direction_id"],
            "direction": c["direction"],
            "novelty_status": status,
            "timed_out": False,
            "prior_art_level": "high",
            "direct_evidence_count": 4,
            "analogous_evidence_count": 26,
            "summary": "s" * 2000,
            "direct_evidence": [{"pmid": str(j), "title": "t" * 400} for j in range(30)],
            "queries": [{"category": "prior_art", "query": "q" * 1500} for _ in range(7)],
        }
        for c in candidates
    ]


@pytest.mark.asyncio
async def test_every_candidate_reaches_the_scoring_prompt(monkeypatch):
    agent = _agent()
    captured = {}

    async def capture(prompt, temperature=0.3):
        captured["prompt"] = prompt
        return json.dumps([])

    monkeypatch.setattr(agent, "chat", capture)
    candidates = _realistic_candidates()

    await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="objective"), candidates, []
    )

    absent = [c["direction_id"] for c in candidates if c["direction_id"] not in captured["prompt"]]
    assert not absent, f"candidates never shown to the scorer: {absent}"


@pytest.mark.asyncio
async def test_every_novelty_record_reaches_the_scoring_prompt(monkeypatch):
    agent = _agent()
    captured = {}

    async def capture(prompt, temperature=0.3):
        captured["prompt"] = prompt
        return json.dumps([])

    monkeypatch.setattr(agent, "chat", capture)
    candidates = _realistic_candidates()
    records = _novelty_records(candidates)

    await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="objective"), candidates, records
    )

    block = captured["prompt"]
    absent = [r["direction_id"] for r in records if r["direction_id"] not in block]
    assert not absent, f"novelty verdicts never shown to the scorer: {absent}"


@pytest.mark.asyncio
async def test_an_omitted_candidate_still_reaches_the_gate(monkeypatch):
    # The novelty verdict here is a successful "known_area" -- the well-published
    # case. Suppressing the candidate on that basis is treating prior-art volume
    # as a verdict on the project's own open question.
    agent = _agent()
    candidates = _realistic_candidates(n=3)

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps([{
            "direction_id": "NEW-01",
            "direction": candidates[0]["direction"],
            "priority_rank": 1,
        }])

    monkeypatch.setattr(agent, "chat", fake_chat)

    scored = await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="objective"),
        candidates,
        _novelty_records(candidates),
    )

    assert [c["direction_id"] for c in scored] == ["NEW-01", "NEW-02", "NEW-03"]


@pytest.mark.asyncio
async def test_an_unreadable_reply_does_not_become_an_empty_candidate_list(monkeypatch):
    # An empty scored list reaches the user as "No sufficiently novel research
    # direction was found" -- a scientific conclusion drawn from a parse error.
    agent = _agent()
    candidates = _realistic_candidates(n=2)

    async def fake_chat(prompt, temperature=0.3):
        return "the model wrote prose instead of JSON"

    monkeypatch.setattr(agent, "chat", fake_chat)

    scored = await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="objective"), candidates, []
    )

    assert [c["direction_id"] for c in scored] == ["NEW-01", "NEW-02"]
