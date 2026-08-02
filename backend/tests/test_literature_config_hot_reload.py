import asyncio
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
    try:
        await agent._deep_research_llm_client.close()
        monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
        monkeypatch.setattr(agent, "_extract_findings", AsyncMock(return_value=["finding 1", "finding 2", "finding 3"]))

        first_chat = AsyncMock(return_value=LLMResponse("old report https://old.example/ref", "old-dr-model", {}, None))
        fake_dr_client = MagicMock(spec=OpenAICompatibleClient)
        fake_dr_client.chat = first_chat
        fake_dr_client.close = AsyncMock()
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
        fake_dr_client2.close = AsyncMock()
        monkeypatch.setattr(agent, "_make_deep_research_client", lambda: fake_dr_client2)
        agent._sync_literature_config_from_provider()

        await agent._deep_research_via_chat("topic", state)
        second_chat.assert_awaited_once()
        assert second_chat.await_args.kwargs["model"] == "new-dr-model"
        assert agent.deep_research_model == "new-dr-model"
        assert agent.pubmed_email == "new@example.com"
        assert agent.max_search_rounds == 7
    finally:
        await agent.close_runtime_clients()
        await provider.cleanup()


@pytest.mark.asyncio
async def test_deep_research_client_uses_session_and_literature_agent_name():
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
    try:
        assert client.session_id == "session-123"
        assert client.agent_name == "literature"
    finally:
        await client.close()
        await agent.close_runtime_clients()
        await provider.cleanup()


@pytest.mark.asyncio
async def test_config_refresh_defers_close_until_inflight_research_finishes(
    monkeypatch,
):
    state = _state()
    provider = _provider(state)
    agent = LiteratureAgent(
        settings_provider=provider,
        literature_mode="deep_research",
        deep_research_model="old-dr-model",
        deep_research_llm_base_url="https://old.example/v1",
        deep_research_api_key="old-key",
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    request_task = None
    try:
        await agent._deep_research_llm_client.close()

        async def blocked_chat(**kwargs):
            entered.set()
            await release.wait()
            return LLMResponse(
                "old report https://old.example/ref",
                "old-dr-model",
                {},
                None,
            )

        old_client = MagicMock(spec=OpenAICompatibleClient)
        old_client.chat = AsyncMock(side_effect=blocked_chat)
        old_client.close = AsyncMock()
        replacement = MagicMock(spec=OpenAICompatibleClient)
        replacement.close = AsyncMock()
        agent._deep_research_llm_client = old_client
        monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
        monkeypatch.setattr(agent, "_select_references", lambda *_: [])
        monkeypatch.setattr(agent, "_finalize_result", AsyncMock(return_value={}))

        request_task = asyncio.create_task(
            agent._deep_research_via_chat("topic", state)
        )
        await entered.wait()

        state.literature_config = {
            **state.literature_config,
            "deep_research_model": "new-dr-model",
            "deep_research_llm_base_url": "https://new.example/v1",
            "deep_research_api_key": "new-key",
        }
        monkeypatch.setattr(
            agent,
            "_make_deep_research_client",
            lambda: replacement,
        )
        agent._sync_literature_config_from_provider()
        await asyncio.sleep(0)

        old_client.close.assert_not_awaited()
        release.set()
        await request_task
        await asyncio.sleep(0)

        old_client.close.assert_awaited_once()
        assert agent._deep_research_llm_client is replacement
    finally:
        release.set()
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        await agent.close_runtime_clients()
        await provider.cleanup()


@pytest.mark.asyncio
async def test_provider_client_refresh_defers_close_until_inflight_research_finishes(
    monkeypatch,
):
    state = _state()
    state.literature_config = {
        **state.literature_config,
        "deep_research_llm_base_url": "",
        "deep_research_api_key": "",
    }
    state.agent_server_configs["literature"] = {
        "llm_base_url": "https://old-agent.example/v1",
        "api_key": "old-agent-key",
    }
    provider = _provider(state)
    agent = LiteratureAgent(
        settings_provider=provider,
        literature_mode="deep_research",
        deep_research_model="old-dr-model",
    )
    agent.refresh_runtime_config_if_needed()
    client = provider.get_llm_client("literature")
    original_close = client.close
    close_finished = asyncio.Event()
    request_entered = asyncio.Event()
    request_release = asyncio.Event()
    request_task = None

    async def tracked_close():
        await original_close()
        close_finished.set()

    async def blocked_chat(**kwargs):
        request_entered.set()
        await request_release.wait()
        return LLMResponse("report", "old-dr-model", {}, None)

    client.close = AsyncMock(side_effect=tracked_close)
    client.chat = AsyncMock(side_effect=blocked_chat)
    monkeypatch.setattr(agent, "_select_references", lambda *_: [])
    monkeypatch.setattr(agent, "_finalize_result", AsyncMock(return_value={}))

    try:
        request_task = asyncio.create_task(
            agent._deep_research_via_chat("topic", state)
        )
        await request_entered.wait()

        state.agent_server_configs["literature"] = {
            "llm_base_url": "https://new-agent.example/v1",
            "api_key": "new-agent-key",
        }
        agent.refresh_runtime_config_if_needed()
        await asyncio.sleep(0)

        client.close.assert_not_awaited()
        assert request_task.done() is False

        request_release.set()
        await request_task
        await asyncio.wait_for(close_finished.wait(), timeout=1)

        client.close.assert_awaited_once()
        assert client._closed is True
    finally:
        request_release.set()
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        await agent.close_runtime_clients()
        await provider.cleanup()


@pytest.mark.asyncio
async def test_inflight_retry_uses_frozen_model_and_system_prompt(monkeypatch):
    state = _state()
    state.custom_prompts["literature"] = "old system prompt"
    provider = _provider(state)
    agent = LiteratureAgent(
        settings_provider=provider,
        literature_mode="deep_research",
        deep_research_model="old-dr-model",
        deep_research_llm_base_url="https://old.example/v1",
        deep_research_api_key="old-key",
    )
    await agent._deep_research_llm_client.close()

    calls = []
    old_client = MagicMock(spec=OpenAICompatibleClient)
    old_client.close = AsyncMock()
    replacement = MagicMock(spec=OpenAICompatibleClient)
    replacement.close = AsyncMock()

    async def retrying_chat(**kwargs):
        calls.append((kwargs["model"], kwargs["messages"][0]["content"]))
        if len(calls) == 1:
            state.custom_prompts["literature"] = "new system prompt"
            state.literature_config = {
                **state.literature_config,
                "deep_research_model": "new-dr-model",
                "deep_research_llm_base_url": "https://new.example/v1",
                "deep_research_api_key": "new-key",
            }
            agent._sync_literature_config_from_provider()
            raise OSError("retry with frozen request config")
        return LLMResponse("report", "old-dr-model", {}, None)

    old_client.chat = AsyncMock(side_effect=retrying_chat)
    agent._deep_research_llm_client = old_client
    monkeypatch.setattr(agent, "_make_deep_research_client", lambda: replacement)
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
    monkeypatch.setattr(agent, "_select_references", lambda *_: [])
    monkeypatch.setattr(agent, "_finalize_result", AsyncMock(return_value={}))
    monkeypatch.setattr(
        "backend.agents.literature.clients.asyncio.sleep",
        AsyncMock(),
    )

    try:
        await agent._deep_research_via_chat("topic", state)

        assert calls == [
            ("old-dr-model", "old system prompt"),
            ("old-dr-model", "old system prompt"),
        ]
        assert agent._deep_research_llm_client is replacement
    finally:
        await agent.close_runtime_clients()
        await provider.cleanup()

    old_client.close.assert_awaited_once()
