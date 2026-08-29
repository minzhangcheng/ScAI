import httpx
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.agents.literature_agent import LiteratureAgent


def _make_literature_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = "You are a helpful literature assistant."
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    provider.budget_text_dispatch = AsyncMock(side_effect=(
        lambda _agent, messages, max_tokens=None, **_kwargs: SimpleNamespace(
            messages=messages,
            max_tokens=max_tokens,
            pi_observation_ids=(),
        )
    ))
    provider.acknowledge_text_dispatch = AsyncMock()
    agent = LiteratureAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


@pytest.mark.asyncio
async def test_fallback_sends_system_prompt(monkeypatch):
    agent = _make_literature_agent()

    seen = {}

    async def fake_chat(messages, **k):
        seen["messages"] = messages
        return MagicMock(content="x")

    monkeypatch.setattr(agent, "chat_without_history", _fail)
    monkeypatch.setattr(agent.fallback_llm_client, "chat", fake_chat)

    await agent._chat_with_fallback("user question")

    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][1] == {"role": "user", "content": "user question"}


async def _fail(*a, **k):
    raise httpx.ReadTimeout("main model failed")
