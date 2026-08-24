"""A1 protocol parameter, endpoint, and proxy negotiation tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.utils.llm_client as llm_mod
from backend.config._config import APIConfig, SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.settings_provider import AgentSettingsProvider


def _client():
    value = llm_mod.OpenAICompatibleClient(
        api_key="installation-key",
        base_url="https://provider.test/v1/chat/completions",
        agent_name="a1",
    )
    return value


def _reply(content="ok"):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content, tool_calls=None))]
    response.model = "model"
    response.usage = MagicMock(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_tokens_details=None,
    )
    return response


def _unsupported_field_error(field):
    return llm_mod.BadRequestError(
        message=f"Error code: 400 - unsupported parameter: {field}",
        response=MagicMock(status_code=400),
        body={"error": {"message": f"unsupported parameter: {field}"}},
    )


@pytest.mark.asyncio
async def test_concurrent_field_discovery_serializes_and_caches_proven_choice():
    client = _client()
    calls = []
    first = True
    first_probe_entered = asyncio.Event()
    release_first_probe = asyncio.Event()

    async def create(**kwargs):
        nonlocal first
        calls.append(kwargs)
        if first and "max_tokens" in kwargs:
            first = False
            first_probe_entered.set()
            await release_first_probe.wait()
            raise _unsupported_field_error("max_tokens")
        return _reply()

    client.client.chat.completions.create = create
    try:
        first_call = asyncio.create_task(
            client.chat("model", [{"role": "user", "content": "a"}], max_tokens=8)
        )
        await first_probe_entered.wait()
        second_call = asyncio.create_task(
            client.chat("model", [{"role": "user", "content": "b"}], max_tokens=8)
        )
        await asyncio.sleep(0)
        release_first_probe.set()
        results = await asyncio.gather(first_call, second_call)
        assert [result.content for result in results] == ["ok", "ok"]
        assert client._max_tokens_field_overrides["model"] == "max_completion_tokens"
        assert any("max_completion_tokens" in call for call in calls)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_failed_or_cancelled_adaptation_cas_rolls_back():
    client = _client()
    client.client.chat.completions.create = AsyncMock(
        side_effect=[_unsupported_field_error("max_tokens"), asyncio.CancelledError()]
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await client.chat(
                "model", [{"role": "user", "content": "a"}], max_tokens=8,
                max_retries=0,
            )
        assert "model" not in client._max_tokens_field_overrides
        assert client.client.chat.completions.create.await_count == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_temperature_and_replacement_wording_adapt_once_each():
    client = _client()
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise llm_mod.BadRequestError(
                message="Error code: 400 - temperature is not supported",
                response=MagicMock(status_code=400),
                body={"error": {"message": "temperature is not supported"}},
            )
        if len(calls) == 2:
            raise llm_mod.BadRequestError(
                message=(
                    "Error code: 400 - unsupported parameter max_tokens; "
                    "use max_completion_tokens instead"
                ),
                response=MagicMock(status_code=400),
                body={
                    "error": {
                        "message": (
                            "unsupported parameter max_tokens; use "
                            "max_completion_tokens instead"
                        )
                    }
                },
            )
        return _reply()

    client.client.chat.completions.create = create
    try:
        result = await client.chat(
            "model", [{"role": "user", "content": "a"}], max_tokens=8
        )
        assert result.content == "ok"
        assert len(calls) == 3
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]
        assert "max_tokens" in calls[1]
        assert "max_completion_tokens" in calls[2]
        assert client._max_tokens_field_overrides["model"] == "max_completion_tokens"
    finally:
        await client.close()


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
