"""Tests for integration gate PI-decide flow."""
import pytest
import asyncio
import types
from unittest.mock import AsyncMock, MagicMock
from backend.agents.pi_agent import PIAgent


@pytest.fixture
def pi_agent():
    agent = PIAgent.__new__(PIAgent)
    agent._model = "test-model"
    agent._temperature = 0.3
    agent._history = []
    return agent


def _make_state(candidates=None, integration_report="", innovation_rounds=None):
    """Build a minimal AnalysisState-like object for testing."""
    state = MagicMock()
    state.name = "Test Session"
    state.initial_instruction = "Analyze scRNA-seq data"
    state.species = "human"
    state.integration_gate_candidates = candidates or []
    state.multi_direction_correlation = integration_report
    state.user_chosen_directions = []
    state.build_experimental_design_summary = MagicMock(return_value="")
    if innovation_rounds:
        state.round_results = innovation_rounds
    else:
        state.round_results = []
    return state


@pytest.mark.asyncio
async def test_select_integration_directions_returns_selected(pi_agent):
    """PI selects 1+ directions from integration candidates."""
    candidates = [
        {"direction": "Direction A", "direction_id": "int_001", "score": 0.9},
        {"direction": "Direction B", "direction_id": "int_002", "score": 0.7},
    ]
    state = _make_state(candidates=candidates, integration_report="Cross-dir analysis")

    pi_agent.chat = AsyncMock(return_value='{"selected_directions": [{"direction": "Direction A", "notes": "Strong evidence"}], "rejected_directions": [{"direction": "Direction B", "reason_not_selected": "Low novelty", "follow_up_value": "low", "dependency_note": ""}], "selection_rationale": "Direction A has stronger evidence"}')
    pi_agent._safe_parse_json = MagicMock(return_value={
        "selected_directions": [{"direction": "Direction A", "notes": "Strong evidence"}],
        "rejected_directions": [{"direction": "Direction B", "reason_not_selected": "Low novelty", "follow_up_value": "low", "dependency_note": ""}],
        "selection_rationale": "Direction A has stronger evidence",
    })

    result = await pi_agent.select_integration_directions(state, candidates, "Cross-dir analysis")
    assert len(result["selected_directions"]) == 1
    assert result["selected_directions"][0]["direction"] == "Direction A"


@pytest.mark.asyncio
async def test_select_integration_directions_returns_empty(pi_agent):
    """PI can select 0 directions when none are novel enough."""
    candidates = [
        {"direction": "Redundant direction", "direction_id": "int_001", "score": 0.3},
    ]
    state = _make_state(candidates=candidates, integration_report="No new patterns found")

    pi_agent.chat = AsyncMock(return_value='{"selected_directions": [], "rejected_directions": [{"direction": "Redundant direction", "reason_not_selected": "Not novel", "follow_up_value": "low", "dependency_note": ""}], "selection_rationale": "None sufficiently novel"}')
    pi_agent._safe_parse_json = MagicMock(return_value={
        "selected_directions": [],
        "rejected_directions": [{"direction": "Redundant direction", "reason_not_selected": "Not novel", "follow_up_value": "low", "dependency_note": ""}],
        "selection_rationale": "None sufficiently novel",
    })

    result = await pi_agent.select_integration_directions(state, candidates, "No new patterns found")
    assert result["selected_directions"] == []
    assert "selection_rationale" in result


@pytest.mark.asyncio
async def test_select_integration_directions_no_candidates(pi_agent):
    """When no candidates are provided, PI returns empty selected_directions."""
    state = _make_state(candidates=[], integration_report="")

    result = await pi_agent.select_integration_directions(state, [], "")
    assert result["selected_directions"] == []
    assert "selection_rationale" in result
