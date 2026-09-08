"""Explicit overflow rebuilds requests without multiplying the retry budget."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

from backend.config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.state import sessions
from backend.utils.context_budget import estimate_tokens
from backend.utils.context_overflow import is_context_overflow
from backend.utils.llm_client import OpenAICompatibleClient, LLMToolResponse
from backend.utils.llm_client_parts.retry import retry_with_backoff
from backend.utils.llm_client_parts.errors import UpstreamExhaustedError
from backend.utils.settings_provider import AgentSettingsProvider


def overflow(message="This model's maximum context length is 200000 tokens. "
             "However, your prompt contains at least 200001 input tokens."):
    return BadRequestError(message, response=httpx.Response(
        400, request=httpx.Request("POST", "https://example.test/v1")),
        body={"message": message})


@pytest.mark.parametrize("message, expected", [
    ("prompt is too long: 210000 tokens > 200000 maximum", True),
    ("Temperature parameter is not supported", False),
    ("service unavailable", False),
    ("max_tokens must be positive", False),
])
def test_overflow_classification(message, expected):
    assert is_context_overflow(overflow(message)) is expected


@pytest.mark.asyncio
async def test_no_recovery_stops_without_identical_retry():
    request = AsyncMock(side_effect=overflow())
    with pytest.raises(UpstreamExhaustedError, match="after 1 attempts"):
        await retry_with_backoff(request)
    assert request.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat/completions", "responses", "messages", ""])
@pytest.mark.parametrize("failures", [2, 5])
async def test_transport_rebuilds_and_persists_shared_budget(monkeypatch, protocol, failures):
    endpoint = f"https://example.test/v1/{protocol}".rstrip("/")
    from backend.utils.llm_client_parts.protocol import _PROBE_CACHE, _NO_RESPONSES_ENDPOINT
    # The facade and dispatch share these mutable caches.
    _PROBE_CACHE.clear()
    _NO_RESPONSES_ENDPOINT.discard(endpoint)
    state = AnalysisState(id="overflow-test", user_id="user")
    state.custom_models["bioinformatics"] = "qwen"
    state.agent_server_configs["bioinformatics"] = {"llm_base_url": endpoint}
    config = SystemConfig()
    monkeypatch.setitem(sessions, state.id, state)
    monkeypatch.setattr("backend.utils.config_resolver.resolve_config_for_user",
                        AsyncMock(return_value=config))
    persisted = []

    async def save(_id, current, **_kwargs):
        persisted.append(current.model_dump(mode="json"))

    monkeypatch.setattr("backend.services.session_store._async_save_session_under_lifecycle", save)
    notification = AsyncMock()
    monkeypatch.setattr("backend.utils.websocket_delivery.send_analysis_json", notification)
    client = OpenAICompatibleClient(api_key="k", base_url=endpoint,
                                   session_id=state.id, agent_name="bioinformatics")
    monkeypatch.setattr(client.llm_logger, "enabled", False)
    requests = []
    messages = [{"role": "system", "content": "Analyze the data."}]
    for i in range(8):
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": f"call-{i}", "type": "function",
                "function": {"name": "execute_code", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"call-{i}", "content": "gene 1.234\n" * 15000},
        ])
    original = deepcopy(messages)

    async def request(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        if len(requests) <= failures:
            if protocol == "messages":
                from anthropic import BadRequestError as AnthropicBadRequestError
                error = overflow("prompt is too long: 210000 tokens > 200000 maximum")
                raise AnthropicBadRequestError(str(error), response=error.response, body=error.body)
            raise overflow()
        if protocol == "chat/completions":
            return SimpleNamespace(model="qwen", usage=None, choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[]))])
        return LLMToolResponse(content="done", tool_calls=[], model="qwen", usage={}, raw_response=None)

    if protocol == "chat/completions":
        monkeypatch.setattr(client.client.chat.completions, "create", request)
    elif protocol in {"responses", ""}:
        client._responses_adapter = SimpleNamespace(complete_with_tools=request)
    else:
        client._anthropic_adapter = SimpleNamespace(complete_with_tools=request)
    try:
        call = client.chat_with_tools(model="qwen", messages=messages, tools=[])
        if failures == 5:
            with pytest.raises(UpstreamExhaustedError, match=(
                "after 1 attempts" if protocol == "" else "after 5 attempts"
            )):
                await call
        else:
            assert (await call).content == "done"
        allowance = 3 if protocol == "" else 5
        assert len(requests) == min(failures + 1, allowance)
        sizes = [estimate_tokens(value) for value in requests]
        assert all(after < before for before, after in zip(sizes, sizes[1:]))
        assert messages == original
        for payload in requests:
            ids = {call["id"] for msg in payload for call in msg.get("tool_calls", [])}
            assert all(msg["tool_call_id"] in ids for msg in payload if msg["role"] == "tool")
        expected = 81920 if failures == 5 and protocol else 128000
        assert notification.await_count == len(persisted)
        notification.assert_awaited_with(state.id, {
            "type": "agent_context_updated", "session_id": state.id,
            "agent": "bioinformatics",
        })
        restored = AnalysisState.model_validate(persisted[-1])
        provider = AgentSettingsProvider(restored, config, global_client=None)
        assert provider.get_context_window("bioinformatics") == expected
        assert provider.get_context_window("pi") == 200000
        restored.custom_models["bioinformatics"] = "other-model"
        assert provider.get_context_window("bioinformatics") == 200000
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recovery_uses_remaining_attempts_after_transient_failure():
    errors = [RuntimeError("temporary"), overflow(), overflow(), overflow(), overflow()]
    request = AsyncMock(side_effect=errors)
    recovery = AsyncMock(return_value=True)
    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(request, retry_delay_base=0, on_context_overflow=recovery)
    assert request.await_count == 5
    assert recovery.await_count == 3


@pytest.mark.asyncio
async def test_underestimated_recent_output_still_shrinks(monkeypatch):
    from backend.utils.context_overflow import recover_context_overflow
    state = AnalysisState(id="small-overflow", user_id="user")
    state.custom_models["pi"] = "qwen"
    state.agent_server_configs["pi"] = {"llm_base_url": "https://example.test/v1"}
    monkeypatch.setitem(sessions, state.id, state)
    monkeypatch.setattr("backend.utils.config_resolver.resolve_config_for_user",
                        AsyncMock(return_value=SystemConfig()))
    monkeypatch.setattr("backend.services.session_store._async_save_session_under_lifecycle",
                        AsyncMock())
    observation = {"role": "user", "content":
                   "Technical context observations (advisory; analysis continued):\n" + "fact " * 500}
    messages = [{"role": "system", "content": "Instructions"}, observation,
                {"role": "user", "content": "x" * 12000}]
    for _ in range(4):
        before = estimate_tokens(messages)
        assert await recover_context_overflow(
            session_id=state.id, agent_name="pi", model="qwen",
            endpoint="https://example.test/v1", messages=messages)
        assert estimate_tokens(messages) < before
        assert messages[1] == observation
    assert state.agent_server_configs["pi"]["adaptive_context_window"] == 81920


@pytest.mark.asyncio
async def test_failed_recovery_preserves_original_overflow():
    request = AsyncMock(side_effect=overflow())
    with pytest.raises(UpstreamExhaustedError, match="maximum context length"):
        await retry_with_backoff(request, on_context_overflow=AsyncMock(side_effect=RuntimeError("save")))
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_zero_retry_budget_does_not_invoke_recovery():
    request = AsyncMock(side_effect=overflow())
    recovery = AsyncMock(return_value=True)
    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(request, max_retries=0, on_context_overflow=recovery)
    assert request.await_count == 1
    recovery.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_error", ["transient", "wrong_protocol"])
async def test_cold_probe_preserves_discovery_and_remaining_attempts(monkeypatch, first_error):
    from backend.utils.llm_client_parts.protocol import _PROBE_CACHE, _NO_RESPONSES_ENDPOINT
    from openai import NotFoundError
    endpoint = "https://cold-probe.test/v1"
    _PROBE_CACHE.clear()
    _NO_RESPONSES_ENDPOINT.discard(endpoint)
    client = OpenAICompatibleClient(api_key="k", base_url=endpoint)
    monkeypatch.setattr("backend.utils.llm_client_parts.client_dispatch.asyncio.sleep", AsyncMock())
    recovery = AsyncMock(return_value=True)
    request = AsyncMock(side_effect=[overflow(), overflow(), overflow()])
    response_calls = 0

    async def responses():
        nonlocal response_calls
        response_calls += 1
        if response_calls == 1:
            if first_error == "wrong_protocol":
                raise NotFoundError("Not Found", response=httpx.Response(
                    404, request=httpx.Request("POST", endpoint + "/responses")), body={})
            raise RuntimeError("temporary")
        return await retry_with_backoff(request, max_retries=0)

    calls = []

    def call_for(protocol, *, probe):
        assert probe
        calls.append(protocol)
        if protocol == "responses":
            return responses
        return lambda: retry_with_backoff(request, max_retries=0)

    try:
        with pytest.raises(UpstreamExhaustedError):
            await client._probe_in_order(
                ["responses", "chat", "anthropic"], call_for, "qwen",
                on_context_overflow=recovery,
            )
        assert calls == (["responses", "chat"] if first_error == "wrong_protocol" else ["responses"])
        assert request.await_count == (3 if first_error == "wrong_protocol" else 2)
        assert recovery.await_count == (2 if first_error == "wrong_protocol" else 1)
        assert not _PROBE_CACHE
    finally:
        _NO_RESPONSES_ENDPOINT.discard(endpoint)
        await client.close()


@pytest.mark.asyncio
async def test_user_budget_edit_wins_over_inflight_overflow(monkeypatch):
    from backend.utils.context_overflow import recover_context_overflow, context_settings
    state = AnalysisState(id="edited-overflow", user_id="user")
    state.custom_models["pi"] = "qwen"
    state.agent_server_configs["pi"] = {"llm_base_url": "https://example.test/v1"}
    expected = context_settings(state.agent_server_configs["pi"])
    state.agent_server_configs["pi"]["context_window"] = 96000
    monkeypatch.setitem(sessions, state.id, state)
    monkeypatch.setattr("backend.utils.config_resolver.resolve_config_for_user",
                        AsyncMock(return_value=SystemConfig()))
    assert not await recover_context_overflow(
        session_id=state.id, agent_name="pi", model="qwen",
        endpoint="https://example.test/v1", expected_context=expected,
        messages=[{"role": "user", "content": "x" * 12000}])
    assert state.agent_server_configs["pi"]["context_window"] == 96000
    assert "adaptive_context_window" not in state.agent_server_configs["pi"]
