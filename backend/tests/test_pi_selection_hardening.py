import json
from types import SimpleNamespace

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import ExplorationReport


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
async def test_cyclic_graph_during_selection_raises(monkeypatch):
    agent = _make_pi_agent()
    report = ExplorationReport(
        research_direction_candidates=[
            {"direction_id": "dir_001", "direction": "A"},
            {"direction_id": "dir_002", "direction": "B"},
        ],
        direction_dependency_graph={"dir_001": ["dir_002"], "dir_002": ["dir_001"]},
    )
    state = SimpleNamespace(exploration_report=report)

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"selected_directions": [{"direction": "A", "notes": ""}],
                           "rejected_directions": [], "selection_rationale": "r"})

    monkeypatch.setattr(agent, "chat", fake_chat)
    with pytest.raises(ValueError):
        await agent.select_best_directions(state)


@pytest.mark.asyncio
async def test_multi_hop_prerequisites_are_added(monkeypatch):
    agent = _make_pi_agent()
    candidates = [
        {"direction_id": "dir_a", "direction": "A", "prerequisite_dependencies": ["dir_b"]},
        {"direction_id": "dir_b", "direction": "B", "prerequisite_dependencies": ["dir_c"]},
        {"direction_id": "dir_c", "direction": "C", "prerequisite_dependencies": []},
    ]
    report = ExplorationReport(research_direction_candidates=candidates)
    state = SimpleNamespace(exploration_report=report)

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"selected_directions": [{"direction": "A", "notes": ""}],
                           "rejected_directions": [], "selection_rationale": "r"})

    monkeypatch.setattr(agent, "chat", fake_chat)
    decision = await agent.select_best_directions(state)
    selected_texts = {d["direction"] for d in decision["selected_directions"]}
    assert {"A", "B", "C"} <= selected_texts  # full transitive closure


@pytest.mark.asyncio
async def test_prerequisite_removed_from_rejected(monkeypatch):
    agent = _make_pi_agent()
    candidates = [
        {"direction_id": "dir_a", "direction": "A", "prerequisite_dependencies": ["dir_b"]},
        {"direction_id": "dir_b", "direction": "B", "prerequisite_dependencies": []},
    ]
    report = ExplorationReport(research_direction_candidates=candidates)
    state = SimpleNamespace(exploration_report=report)

    async def fake_chat(prompt, temperature=0.3):
        # PI selects A but rejects B; B is A's prerequisite and must move out of rejected.
        return json.dumps({
            "selected_directions": [{"direction": "A", "notes": ""}],
            "rejected_directions": [{"direction": "B", "reason_not_selected": "low",
                                     "follow_up_value": "low", "dependency_note": ""}],
            "selection_rationale": "r"})

    monkeypatch.setattr(agent, "chat", fake_chat)
    decision = await agent.select_best_directions(state)
    selected_texts = {d["direction"] for d in decision["selected_directions"]}
    rejected_texts = {r["direction"] for r in decision["rejected_directions"]}
    assert "B" in selected_texts
    assert "B" not in rejected_texts
