import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import AnalysisState, AnalysisPhase
from backend.agents.pi_agent import PIAgent


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _innovation_state():
    s = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.current_round_direction = "copper collapse"
    return s


@pytest.mark.asyncio
async def test_cross_modal_context_appears_in_prompt():
    agent = _make_pi_agent()
    resp = json.dumps({"summary": "s"})
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=resp) as mock_chat:
        await agent.synthesize_round(
            _innovation_state(),
            cross_modal_context="| MATRIX_MARKER | x |",
        )
    prompt = mock_chat.call_args[0][0]
    assert "Cross-Modal Evidence Audit" in prompt
    assert "MATRIX_MARKER" in prompt


@pytest.mark.asyncio
async def test_softened_direction_lock_present_in_innovation():
    agent = _make_pi_agent()
    resp = json.dumps({"summary": "s"})
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=resp) as mock_chat:
        await agent.synthesize_round(_innovation_state())
    prompt = mock_chat.call_args[0][0]
    assert "additionally surface any cross-modal mechanisms" in prompt
    # Old hard lock phrasing must be gone
    assert "Focus your synthesis on findings relevant to this direction" not in prompt


@pytest.mark.asyncio
async def test_no_cross_modal_section_when_empty():
    agent = _make_pi_agent()
    resp = json.dumps({"summary": "s"})
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=resp) as mock_chat:
        await agent.synthesize_round(_innovation_state())
    prompt = mock_chat.call_args[0][0]
    assert "Cross-Modal Evidence Audit" not in prompt
