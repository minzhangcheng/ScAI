import asyncio
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from backend.agents.base import BaseAgent
from backend.config._config import SystemConfig
from backend.models.schemas import AgentType, AnalysisState
from backend.utils.settings_provider import (
    AgentSettingsProvider,
    MASKED_SECRET_VALUE,
)
from backend.state import sessions


class _DummyClient:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def _state(session_id: str = "rev-test") -> AnalysisState:
    return AnalysisState(
        id=session_id,
        name="Revision Test",
        user_id="user-1",
        initial_instruction="test",
    )


def test_config_revision_changes_for_standard_agent_config_without_exposing_secret():
    state = _state()
    provider = AgentSettingsProvider(
        state=state,
        config=SystemConfig(),
        global_client=_DummyClient(),
        session_id=None,
    )

    initial = provider.get_config_revision("pi")
    state.custom_models["pi"] = "new-model"
    after_model = provider.get_config_revision("pi")
    state.custom_prompts["pi"] = "new prompt"
    after_prompt = provider.get_config_revision("pi")
    state.custom_temperatures["pi"] = 0.2
    after_temp = provider.get_config_revision("pi")
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://example.test/v1",
        "api_key": "secret-api-key",
    }
    after_server = provider.get_config_revision("pi")
    state.agent_vision_settings["pi"] = False
    after_vision = provider.get_config_revision("pi")

    assert len({initial, after_model, after_prompt, after_temp, after_server, after_vision}) == 6
    assert "secret-api-key" not in repr(provider.get_config_signature("pi"))


def test_literature_revision_includes_literature_config_without_raw_keys():
    state = _state()
    provider = AgentSettingsProvider(state=state, config=SystemConfig(), global_client=_DummyClient())

    initial = provider.get_config_revision("literature")
    state.literature_config = {
        "mode": "deep_research",
        "deep_research_model": "o4-mini-deep-research",
        "deep_research_llm_base_url": "https://old.example/v1",
        "deep_research_api_key": "old-secret",
        "google_api_key": "google-secret",
        "pubmed_email": "a@example.com",
        "max_search_rounds": 5,
    }
    after_config = provider.get_config_revision("literature")
    state.literature_config["deep_research_api_key"] = "new-secret"
    after_key = provider.get_config_revision("literature")

    assert initial != after_config
    assert after_config != after_key
    signature_text = repr(provider.get_config_signature("literature"))
    assert "old-secret" not in signature_text
    assert "new-secret" not in signature_text
    assert "google-secret" not in signature_text


def test_agent_extra_signature_is_empty_for_standard_agents_and_includes_literature_config():
    state = _state()
    provider = AgentSettingsProvider(state=state, config=SystemConfig(), global_client=_DummyClient())
    state.literature_config = {"deep_research_api_key": "secret", "mode": "deep_research"}

    assert provider._agent_extra_signature("pi", state) == {}
    assert provider._agent_extra_signature("literature", state) == {
        "literature_config": state.literature_config
    }


@pytest.mark.asyncio
async def test_get_llm_client_cache_is_per_agent_and_cleared_per_agent():
    state = _state()
    state.agent_server_configs["pi"] = {"llm_base_url": "https://same.example/v1", "api_key": "same-key"}
    state.agent_server_configs["literature"] = {"llm_base_url": "https://same.example/v1", "api_key": "same-key"}
    provider = AgentSettingsProvider(state=state, config=SystemConfig(), global_client=None, session_id="rev-test")

    pi_client = provider.get_llm_client("pi")
    lit_client = provider.get_llm_client("literature")
    assert pi_client is not lit_client
    assert pi_client.agent_name == "pi"
    assert lit_client.agent_name == "literature"

    try:
        provider.clear_agent_client_cache("pi")
        replacement = provider.get_llm_client("pi")
        assert replacement is not pi_client
        assert provider.get_llm_client("literature") is lit_client
        await asyncio.sleep(0)
        assert pi_client._closed is True
    finally:
        await provider.cleanup()


def test_provider_reads_fresh_state_from_sessions_for_revision():
    session_id = "rev-freshness"
    state = _state(session_id)
    sessions[session_id] = state
    try:
        provider = AgentSettingsProvider(state=state, config=SystemConfig(), global_client=_DummyClient(), session_id=session_id)
        initial = provider.get_config_revision("pi")
        sessions[session_id].custom_models["pi"] = "fresh-model"
        assert provider.get_config_revision("pi") != initial
    finally:
        sessions.pop(session_id, None)


def test_mask_literature_config_masks_keys_and_preserve_update_keeps_existing_keys():
    existing = {
        "mode": "deep_research",
        "deep_research_model": "old-model",
        "deep_research_llm_base_url": "https://old.example/v1",
        "deep_research_api_key": "old-dr-key",
        "google_api_key": "old-google-key",
        "pubmed_email": "old@example.com",
        "max_search_rounds": 4,
    }
    masked = AgentSettingsProvider.mask_literature_config(existing)
    assert masked["deep_research_api_key"] == MASKED_SECRET_VALUE
    assert masked["google_api_key"] == MASKED_SECRET_VALUE

    incoming = {
        "mode": "deep_research",
        "deep_research_model": "new-model",
        "deep_research_llm_base_url": "https://new.example/v1",
        "deep_research_api_key": MASKED_SECRET_VALUE,
        "google_api_key": "",
        "pubmed_email": "new@example.com",
        "max_search_rounds": 7,
    }
    resolved = AgentSettingsProvider.resolve_literature_config_update(existing, incoming)
    assert resolved["deep_research_api_key"] == "old-dr-key"
    assert resolved["google_api_key"] == ""
    assert resolved["deep_research_model"] == "new-model"
    assert resolved["max_search_rounds"] == 7


class _RefreshProbeAgent(BaseAgent):
    def __init__(self, provider):
        super().__init__(AgentType.PI, provider)
        self.changed_calls = []

    async def process(self, state, instruction, **kwargs):
        return {}

    def on_runtime_config_changed(self, previous_revision, current_revision):
        self.changed_calls.append((previous_revision, current_revision))


def test_base_agent_refresh_hook_fires_once_per_revision_change():
    state = _state()
    provider = AgentSettingsProvider(state=state, config=SystemConfig(), global_client=_DummyClient())
    agent = _RefreshProbeAgent(provider)

    agent.refresh_runtime_config_if_needed()
    assert len(agent.changed_calls) == 1
    first_previous, first_current = agent.changed_calls[0]
    assert first_previous is None
    assert isinstance(first_current, int)

    agent.refresh_runtime_config_if_needed()
    assert len(agent.changed_calls) == 1

    state.custom_models["pi"] = "changed-model"
    agent.refresh_runtime_config_if_needed()
    assert len(agent.changed_calls) == 2
    assert agent.changed_calls[1][0] == first_current
    assert agent.changed_calls[1][1] != first_current


@pytest.mark.asyncio
async def test_base_agent_refresh_clears_agent_client_cache_on_revision_change():
    state = _state()
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://pi.example/v1",
        "api_key": "pi-key",
    }
    provider = AgentSettingsProvider(
        state=state,
        config=SystemConfig(),
        global_client=None,
        session_id=None,
    )
    try:
        old_client = provider.get_llm_client("pi")
        agent = _RefreshProbeAgent(provider)

        agent.refresh_runtime_config_if_needed()
        same_revision_client = provider.get_llm_client("pi")
        assert same_revision_client is not old_client

        stable_client = provider.get_llm_client("pi")
        agent.refresh_runtime_config_if_needed()
        assert provider.get_llm_client("pi") is stable_client
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_base_agent_hot_reload_does_not_close_inflight_client():
    from backend.utils.llm_client import LLMResponse

    state = _state()
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://old.example/v1",
        "api_key": "old-key",
    }
    provider = AgentSettingsProvider(
        state=state,
        config=SystemConfig(),
        global_client=None,
        session_id=None,
    )
    agent = _RefreshProbeAgent(provider)
    agent.refresh_runtime_config_if_needed()
    client = provider.get_llm_client("pi")
    entered = asyncio.Event()
    release = asyncio.Event()
    request_task = None

    async def blocked_dispatch(**kwargs):
        entered.set()
        await release.wait()
        return LLMResponse("ok", "model", {}, None)

    client._dispatch_non_streaming = AsyncMock(side_effect=blocked_dispatch)
    client.client.close = AsyncMock()
    client.http_client.aclose = AsyncMock()

    try:
        request_task = asyncio.create_task(agent.chat("hello"))
        await entered.wait()

        state.agent_server_configs["pi"] = {
            "llm_base_url": "https://new.example/v1",
            "api_key": "new-key",
        }
        agent.refresh_runtime_config_if_needed()
        await asyncio.sleep(0)

        assert client._active_requests == 1
        assert client._closed is False
        client.client.close.assert_not_awaited()

        release.set()
        assert await request_task == "ok"
        await provider.cleanup()

        assert client._closed is True
        client.client.close.assert_awaited_once()
    finally:
        release.set()
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        await provider.cleanup()
