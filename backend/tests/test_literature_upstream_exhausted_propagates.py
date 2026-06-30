"""LLM-failure HITL: LiteratureAgent must NOT silently degrade to quick_search
when the LLM upstream is genuinely exhausted.

Regression for the case where the PubMed synthesis call raised
UpstreamExhaustedError after exhausting retries, but _do_research's broad
fallback chain swallowed it and degraded to quick_search — so the workflow's
LLM-failure HITL gate (retry/skip/stop dialog) never fired and the step was
silently skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.agents.literature_agent import LiteratureAgent
from backend.utils.llm_client import UpstreamExhaustedError


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_llm_client.return_value = MagicMock()
    provider.get_temperature.return_value = 0.7
    provider.get_system_prompt.return_value = "test prompt"
    return provider


def _exhausted():
    return UpstreamExhaustedError(
        "upstream exhausted", agent_name="literature",
        model="gpt-test", llm_server="openrouter",
    )


@pytest.mark.asyncio
async def test_pubmed_upstream_exhausted_propagates(mock_provider):
    """pubmed_llm synthesis exhausting the LLM must propagate, not degrade."""
    agent = LiteratureAgent(settings_provider=mock_provider, literature_mode="pubmed_llm")
    agent._multi_round_pubmed = AsyncMock(side_effect=_exhausted())
    agent._quick_search = AsyncMock(return_value={"report": "degraded", "findings": ["x"]})

    with pytest.raises(UpstreamExhaustedError):
        await agent._do_research("some topic")

    # Must not have silently degraded to quick_search.
    agent._quick_search.assert_not_called()


@pytest.mark.asyncio
async def test_deep_research_upstream_exhausted_propagates(mock_provider):
    """deep_research exhausting the LLM must propagate even through the
    deep->pubmed fallback branch, not degrade to quick_search."""
    agent = LiteratureAgent(settings_provider=mock_provider, literature_mode="deep_research")
    agent._deep_research_via_chat = AsyncMock(side_effect=_exhausted())
    agent._multi_round_pubmed = AsyncMock(side_effect=_exhausted())
    agent._quick_search = AsyncMock(return_value={"report": "degraded", "findings": ["x"]})

    with pytest.raises(UpstreamExhaustedError):
        await agent._do_research("some topic")

    agent._quick_search.assert_not_called()


@pytest.mark.asyncio
async def test_non_upstream_error_still_degrades(mock_provider):
    """A non-LLM-exhaustion failure should still gracefully degrade to
    quick_search (existing resilience behavior preserved)."""
    agent = LiteratureAgent(settings_provider=mock_provider, literature_mode="pubmed_llm")
    agent._multi_round_pubmed = AsyncMock(side_effect=ValueError("boom"))
    agent._quick_search = AsyncMock(return_value={"report": "degraded", "findings": ["x"]})

    result = await agent._do_research("some topic")

    agent._quick_search.assert_called_once()
    assert result["findings"] == ["x"]
