import json
from types import SimpleNamespace

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import ExplorationReport
from backend.tests.split_facade import split_source

SRC = split_source("backend.agents.pi_agent")


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


def test_selection_prompt_does_not_force_exactly_two():
    block = SRC.split("async def select_best_directions", 1)[1].split("async def", 1)[0]
    assert "exactly 2" not in block
    assert "reason_not_selected" in block


@pytest.mark.asyncio
async def test_prerequisite_is_added_when_dependent_selected(monkeypatch):
    agent = _make_pi_agent()

    candidates = [
        {"direction_id": "dir_001", "direction": "Foundation", "prerequisite_dependencies": []},
        {"direction_id": "dir_002", "direction": "Dependent", "prerequisite_dependencies": ["dir_001"]},
    ]
    report = ExplorationReport(
        research_direction_candidates=candidates,
        direction_dependency_graph={"dir_002": ["dir_001"]},
    )
    state = SimpleNamespace(exploration_report=report)

    async def fake_chat(prompt, temperature=0.3):
        # PI selects only the dependent direction, omitting its prerequisite.
        return json.dumps({
            "selected_directions": [{"direction": "Dependent", "notes": "n"}],
            "rejected_directions": [],
            "selection_rationale": "r",
        })

    monkeypatch.setattr(agent, "chat", fake_chat)
    decision = await agent.select_best_directions(state)

    selected_texts = {d["direction"] for d in decision["selected_directions"]}
    assert "Dependent" in selected_texts
    assert "Foundation" in selected_texts  # prerequisite closure added it
