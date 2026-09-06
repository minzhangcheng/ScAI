import inspect
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
async def test_cyclic_graph_during_selection_is_repaired(monkeypatch):
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
    decision = await agent.select_best_directions(state)
    assert decision["selected_directions"]
    agent._validate_direction_dependency_graph(report.direction_dependency_graph)
    assert report.direction_dependency_repairs


@pytest.mark.asyncio
async def test_a_cyclic_graph_during_selection_is_broken_not_refused(monkeypatch):
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
    decision = await agent.select_best_directions(state)

    assert decision["selected_directions"]
    agent._validate_direction_dependency_graph(report.direction_dependency_graph)
    assert report.direction_dependency_repairs


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


# --- thin evidence is not a veto at the exploration gate ----------------------
#
# The gate declined a candidate because "the necessary defining measurements --
# lipid peroxidation, labile iron, GPX4 activity, death morphology, and
# ferroptosis-specific rescue -- are absent". Every one of those is a wet-lab
# endpoint no computational round can produce, so the criterion made producing
# the evidence a precondition for being allowed to produce it. The same decline
# also read a susceptibility program rising alongside its own defence program as
# self-contradiction rather than as a state worth resolving. The integration
# gate was taught both distinctions in 0.52.1; this gate was not.


def test_exploration_gate_does_not_reject_for_evidence_only_wet_lab_could_produce():
    src = inspect.getsource(PIAgent.select_best_directions)
    assert "never as the reason it fails" in src
    assert "never for what has not been measured yet" in src


def test_exploration_gate_treats_opposing_evidence_as_confidence_not_veto():
    src = inspect.getsource(PIAgent.select_best_directions)
    assert "it does not retire the direction" in src


def test_exploration_gate_does_not_discount_novelty_by_data_sufficiency():
    # data_sufficiency conflates "this project's data cannot answer it" with "no
    # round has measured it yet", and only the first bears on whether a direction
    # is worth running. It is not rendered into the candidate block either.
    from backend.utils.direction_payload import CANDIDATE_PROMPT_FIELDS
    assert "data_sufficiency" not in inspect.getsource(PIAgent.select_best_directions)
    assert "data_sufficiency" not in CANDIDATE_PROMPT_FIELDS
