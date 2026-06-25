import json
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
async def test_cyclic_dependency_graph_raises(monkeypatch):
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({
            "research_direction_candidates": [
                {"direction_id": "dir_001", "direction": "A", "overall_priority": "high"}
            ],
            "direction_dependency_graph": {"dir_001": ["dir_002"], "dir_002": ["dir_001"]},
        })

    monkeypatch.setattr(agent, "chat", fake_chat)
    state = SimpleNamespace(name="s", initial_instruction="x", species="mouse", annotation_state=None)
    round_result = SimpleNamespace(
        bioinformatics_findings=[], pathology_findings=[], biochemistry_findings=[],
        literature_findings=[], pi_summary="")

    with pytest.raises(ValueError):
        await agent.score_directions_with_evidence(
            state, round_result,
            [{"direction_id": "dir_001", "direction": "A"}],
            [{"direction_id": "dir_001", "direction": "A", "status": "completed"}],
        )
