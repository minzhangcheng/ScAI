from types import SimpleNamespace

import pytest

from backend.agents.pi_agent import PIAgent


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


def _stub_inputs():
    state = SimpleNamespace(name="s", initial_instruction="x", species="mouse",
                            annotation_state=None)
    state.build_experimental_design_summary = lambda: ""
    round_result = SimpleNamespace(
        bioinformatics_findings=[], pathology_findings=[], biochemistry_findings=[],
        literature_findings=[], pi_summary="")
    return state, round_result


@pytest.mark.asyncio
async def test_unparseable_proposal_produces_no_fabricated_candidate(monkeypatch):
    """This used to synthesise a placeholder candidate and only guard its enums.

    The placeholder was the bug: it entered literature evaluation and PI scoring
    as though the PI had proposed it. A reply we cannot read yields NO candidate,
    and the failure is carried out-of-band where the report can show it.
    """
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        return "{}"  # parses, but carries no candidates key

    monkeypatch.setattr(agent, "chat", fake_chat)
    state, round_result = _stub_inputs()

    candidates = await agent.generate_raw_direction_candidates(state, round_result)
    assert candidates == []
    assert "could not be read" in (agent.last_candidate_generation_error or "")


@pytest.mark.asyncio
async def test_deliberate_empty_proposal_is_not_a_failure(monkeypatch):
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        return '{"candidates": [], "no_candidate_reason": "round 1 answered the question"}'

    monkeypatch.setattr(agent, "chat", fake_chat)
    state, round_result = _stub_inputs()

    candidates = await agent.generate_raw_direction_candidates(state, round_result)
    assert candidates == []
    assert not getattr(agent, "last_candidate_generation_error", None)
