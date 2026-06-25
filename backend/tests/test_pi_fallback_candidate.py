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


@pytest.mark.asyncio
async def test_empty_candidate_fallback_uses_valid_enum(monkeypatch):
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        return "{}"  # no candidates -> fallback path

    monkeypatch.setattr(agent, "chat", fake_chat)
    state = SimpleNamespace(name="s", initial_instruction="x", species="mouse",
                            annotation_state=None)
    state.build_experimental_design_summary = lambda: ""
    round_result = SimpleNamespace(
        bioinformatics_findings=[], pathology_findings=[], biochemistry_findings=[],
        literature_findings=[], pi_summary="")

    candidates = await agent.generate_raw_direction_candidates(state, round_result)
    assert candidates[0]["biology_expectedness"] in {"expected", "partially_expected", "unexpected", "unknown"}
    assert candidates[0]["biology_expectedness"] == "unknown"
