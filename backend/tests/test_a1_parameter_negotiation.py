"""Custom endpoint credentials and local-only proxy bypass."""

import asyncio
from unittest.mock import MagicMock

import pytest

import backend.utils.llm_client as llm_mod
from backend.config._config import APIConfig, SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.settings_provider import AgentSettingsProvider


def test_custom_endpoint_never_inherits_installation_key():
    state = AnalysisState(id="a1", name="A1", user_id="u", initial_instruction="x")
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://custom.example/v1",
        "api_key": "",
    }
    config = SystemConfig(
        api=APIConfig(
            api_key="installation-key",
            llm_base_url="https://global.example/v1",
        )
    )
    global_client = MagicMock()
    provider = AgentSettingsProvider(state, config, global_client)

    assert provider.get_llm_client("pi") is global_client
    state.agent_server_configs["pi"]["api_key"] = "installation-key"
    assert provider.get_llm_client("pi") is global_client


@pytest.mark.asyncio
async def test_proxy_bypass_is_local_only(monkeypatch):
    http_clients = []

    class _HTTPClient:
        def __init__(self, **kwargs):
            http_clients.append(kwargs)

        async def aclose(self):
            return None

    class _SDKClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def close(self):
            return None

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _HTTPClient)
    monkeypatch.setattr(llm_mod, "AsyncOpenAI", _SDKClient)

    clients = [
        llm_mod.OpenAICompatibleClient(
            api_key="k", base_url=url, proxy_url="http://proxy.test:8080"
        )
        for url in (
            "https://public.example/v1",
            "http://127.0.0.1:8000/v1",
            "http://localhost:8000/v1",
            "http://internal-service/v1",
            "http://api.localhost/v1",
            "http://service.internal/v1",
            "http://0.0.0.0/v1",
            "http://[::]/v1",
        )
    ]
    try:
        assert http_clients[0]["proxy"] == "http://proxy.test:8080"
        assert all("proxy" not in item for item in http_clients[1:])
    finally:
        await asyncio.gather(*(client.close() for client in clients))
