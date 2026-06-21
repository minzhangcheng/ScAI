import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.config._config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.llm_client import LLMResponse, OpenAICompatibleClient
from backend.utils.settings_provider import AgentSettingsProvider


def _state() -> AnalysisState:
    state = AnalysisState(
        id="lit-hot-reload",
        name="Lit Hot Reload",
        user_id="user-1",
        initial_instruction="test",
    )
    state.literature_config = {
        "mode": "deep_research",
        "deep_research_model": "old-dr-model",
        "deep_research_llm_base_url": "https://old.example/v1",
        "deep_research_api_key": "old-key",
        "google_api_key": "",
        "pubmed_email": "old@example.com",
        "max_search_rounds": 3,
    }
    return state


def _provider(state: AnalysisState) -> AgentSettingsProvider:
    return AgentSettingsProvider(
        state=state,
        config=SystemConfig(),
        global_client=MagicMock(spec=OpenAICompatibleClient),
        session_id=None,
    )


@pytest.mark.asyncio
async def test_existing_literature_agent_uses_new_deep_research_model_after_config_change(monkeypatch):
    state = _state()
    provider = _provider(state)
    agent = LiteratureAgent(
        settings_provider=provider,
        literature_mode="deep_research",
        deep_research_model="old-dr-model",
        deep_research_llm_base_url="https://old.example/v1",
        deep_research_api_key="old-key",
        pubmed_email="old@example.com",
    )
    agent.set_session("lit-hot-reload")

    monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
    monkeypatch.setattr(agent, "_extract_findings", AsyncMock(return_value=["finding 1", "finding 2", "finding 3"]))

    first_chat = AsyncMock(return_value=LLMResponse("old report https://old.example/ref", "old-dr-model", {}, None))
    fake_dr_client = MagicMock(spec=OpenAICompatibleClient)
    fake_dr_client.chat = first_chat
    agent._deep_research_llm_client = fake_dr_client

    await agent._deep_research_via_chat("topic", state)
    first_chat.assert_awaited_once()
    assert first_chat.await_args.kwargs["model"] == "old-dr-model"

    state.literature_config = {
        "mode": "deep_research",
        "deep_research_model": "new-dr-model",
        "deep_research_llm_base_url": "https://new.example/v1",
        "deep_research_api_key": "new-key",
        "google_api_key": "",
        "pubmed_email": "new@example.com",
        "max_search_rounds": 7,
    }

    second_chat = AsyncMock(return_value=LLMResponse("new report https://new.example/ref", "new-dr-model", {}, None))
    fake_dr_client2 = MagicMock(spec=OpenAICompatibleClient)
    fake_dr_client2.chat = second_chat
    monkeypatch.setattr(agent, "_make_deep_research_client", lambda: fake_dr_client2)
    agent._sync_literature_config_from_provider()

    await agent._deep_research_via_chat("topic", state)
    second_chat.assert_awaited_once()
    assert second_chat.await_args.kwargs["model"] == "new-dr-model"
    assert agent.deep_research_model == "new-dr-model"
    assert agent.pubmed_email == "new@example.com"
    assert agent.max_search_rounds == 7


def test_deep_research_client_uses_session_and_literature_agent_name():
    state = _state()
    provider = _provider(state)
    agent = LiteratureAgent(
        settings_provider=provider,
        literature_mode="deep_research",
        deep_research_model="dr-model",
        deep_research_llm_base_url="https://dr.example/v1",
        deep_research_api_key="dr-key",
    )
    agent.set_session("session-123")

    client = agent._make_deep_research_client()

    assert client.session_id == "session-123"
    assert client.agent_name == "literature"
