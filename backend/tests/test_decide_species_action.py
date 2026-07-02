"""Test that decide_species_action propagates UpstreamExhaustedError."""

import pytest
from unittest.mock import AsyncMock

from backend.agents.pi_agent import PIAgent
from backend.config._config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.llm_client import UpstreamExhaustedError


class _DummyClient:
    pass


def _provider():
    state = AnalysisState(id="species-test", name="Species", user_id="u", initial_instruction="x")
    return AgentSettingsProvider(state=state, config=SystemConfig(), global_client=_DummyClient())


@pytest.mark.asyncio
async def test_species_action_propagates_upstream():
    agent = PIAgent(settings_provider=_provider())

    async def boom(*a, **kw):
        raise UpstreamExhaustedError("dead upstream")

    agent.chat = boom

    with pytest.raises(UpstreamExhaustedError):
        await agent.decide_species_action("human", "Mus musculus", {})
