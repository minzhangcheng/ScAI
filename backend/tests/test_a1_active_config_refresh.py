"""A1 active settings replacement and in-flight lease tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import backend.state as app_state
import backend.utils.settings_provider as settings
from backend.config._config import APIConfig, SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.settings_provider import AgentSettingsProvider


class _Client:
    def __init__(self, name, *, api_key="", base_url="", proxy_url=""):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.proxy_url = proxy_url
        self._closing = False
        self._closed = False
        self.close = AsyncMock()
        self.on_retry = None


@pytest.mark.asyncio
async def test_config_rotation_publishes_new_before_inflight_old_closes(monkeypatch):
    state = AnalysisState(id="a1", name="A1", user_id="u", initial_instruction="x")
    old_config = SystemConfig(api=APIConfig(
        api_key="old-key",
        llm_base_url="https://old.example/v1",
    ))
    new_config = SystemConfig(api=APIConfig(
        api_key="new-key",
        llm_base_url="https://new.example/v1",
    ))
    old = _Client(
        "old",
        api_key="old-key",
        base_url="https://old.example/v1",
    )
    new = _Client(
        "new",
        api_key="new-key",
        base_url="https://new.example/v1",
    )
    provider = AgentSettingsProvider(state, old_config, old)
    workflow = SimpleNamespace(
        _settings_provider=provider,
        llm_manager=SimpleNamespace(llm=old, openrouter=old),
    )
    app_state.workflows[state.id] = workflow
    monkeypatch.setattr(
        "backend.utils.config_resolver.resolve_config_for_user",
        AsyncMock(return_value=new_config),
    )
    monkeypatch.setattr(
        "backend.utils.settings_provider.OpenAICompatibleClient",
        lambda **_kwargs: new,
    )

    provider.lease_client(old)
    try:
        refreshed = await settings.refresh_active_providers(user_id="u")
        assert refreshed == 1
        assert provider.get_llm_client("pi") is new
        assert workflow.llm_manager.llm is new
        assert workflow.llm_manager.openrouter is new
        await asyncio.sleep(0)
        old.close.assert_not_awaited()

        provider.release_client(old)
        await asyncio.sleep(0)
        old.close.assert_awaited_once()

        # Deleting the effective credential retires the replacement and makes
        # subsequent business calls fail closed instead of reusing it.
        monkeypatch.setattr(
            "backend.utils.config_resolver.resolve_config_for_user",
            AsyncMock(return_value=None),
        )
        await settings.refresh_active_providers(user_id="u")
        with pytest.raises(ValueError, match="api_key"):
            provider.get_llm_client("pi")
    finally:
        app_state.workflows.pop(state.id, None)
        await provider.cleanup()


@pytest.mark.asyncio
async def test_per_agent_rotation_keeps_distinct_endpoint_credential(monkeypatch):
    state = AnalysisState(id="a1-agent", name="A1", user_id="u", initial_instruction="x")
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://old.example/v1",
        "api_key": "old-key",
    }
    provider = AgentSettingsProvider(state, SystemConfig(), None)
    old = _Client("old")
    new = _Client("new")
    clients = iter((old, new))
    monkeypatch.setattr(
        "backend.utils.settings_provider.OpenAICompatibleClient",
        lambda **_kwargs: next(clients),
    )

    assert provider.get_llm_client("pi") is old
    provider.lease_client(old)
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://new.example/v1",
        "api_key": "new-key",
    }
    provider.clear_agent_client_cache("pi")
    assert provider.get_llm_client("pi") is new
    await asyncio.sleep(0)
    old.close.assert_not_awaited()

    provider.release_client(old)
    await asyncio.sleep(0)
    old.close.assert_awaited_once()
    await provider.cleanup()
