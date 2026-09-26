"""Active credential changes publish replacements without interrupting leases."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.state as app_state
import backend.utils.settings_provider as settings
from backend.config._config import APIConfig, SystemConfig
from backend.models.schemas import AnalysisState


def _client(key):
    return MagicMock(
        api_key=key, base_url=f"https://{key}.example/v1", proxy_url="",
        _closing=False, _closed=False, close=AsyncMock(),
        on_retry=None, before_retry=None,
    )


@pytest.mark.asyncio
async def test_config_rotation_publishes_new_before_inflight_old_closes(monkeypatch):
    state = AnalysisState(id="active-refresh", user_id="owner")
    old, new = _client("old"), _client("new")
    old_config, new_config = [SystemConfig(api=APIConfig(
        api_key=client.api_key, llm_base_url=client.base_url,
    )) for client in (old, new)]
    provider = settings.AgentSettingsProvider(state, old_config, old)
    before_retry = AsyncMock()
    on_retry = lambda *args: None
    provider.set_retry_callback(on_retry, before_retry)
    workflow = SimpleNamespace(
        _settings_provider=provider, llm_manager=SimpleNamespace(llm=old, openrouter=old),
    )
    monkeypatch.setitem(app_state.workflows, state.id, workflow)
    resolver = AsyncMock(return_value=new_config)
    monkeypatch.setattr("backend.utils.config_resolver.resolve_config_for_user", resolver)
    monkeypatch.setattr(settings, "OpenAICompatibleClient", lambda **kwargs: new)
    provider.lease_client(old)
    try:
        assert await settings.refresh_active_providers(user_id="owner") == 1
        assert provider.get_llm_client("pi") is new
        assert workflow.llm_manager.llm is new
        assert workflow.llm_manager.openrouter is new
        for client in (old, new):
            assert client.before_retry is before_retry
            assert client.on_retry is on_retry
        await asyncio.sleep(0)
        old.close.assert_not_awaited()
        provider.release_client(old)
        await asyncio.sleep(0)
        old.close.assert_awaited_once()
        resolver.return_value = None
        await settings.refresh_active_providers(user_id="owner")
        with pytest.raises(ValueError, match="api_key"):
            provider.get_llm_client("pi")
    finally:
        await provider.cleanup()
