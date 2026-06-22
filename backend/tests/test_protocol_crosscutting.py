import pytest
from unittest.mock import AsyncMock, MagicMock
import backend.utils.llm_client as llm_mod
from backend.utils.llm_client import _ResponsesAdapter, _AnthropicAdapter


def _responses_adapter_with_capture():
    captured = {}
    resp = MagicMock()
    resp.model_dump.return_value = {"output": [], "usage": {}}
    resp.model = "gpt-x"
    client = MagicMock()
    client.responses = MagicMock()
    async def _create(**kwargs):
        captured.update(kwargs)
        return resp
    client.responses.create = AsyncMock(side_effect=_create)
    return _ResponsesAdapter(client, agent_name="t"), captured


@pytest.mark.asyncio
async def test_responses_skip_temperature_omits_temp():
    adapter, captured = _responses_adapter_with_capture()
    await adapter.complete(model="gpt-x", messages=[{"role": "user", "content": "hi"}],
                           temperature=0.5, max_tokens=10, skip_temperature=True)
    assert "temperature" not in captured


@pytest.mark.asyncio
async def test_responses_default_keeps_temperature():
    adapter, captured = _responses_adapter_with_capture()
    await adapter.complete(model="gpt-x", messages=[{"role": "user", "content": "hi"}],
                           temperature=0.5, max_tokens=10)
    assert captured.get("temperature") == 0.5


from openai import APITimeoutError, BadRequestError
from backend.utils.llm_client import OpenAICompatibleClient, LLMResponse, UpstreamExhaustedError, NotFoundError, ENGLISH_ENFORCEMENT


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.asyncio, "sleep", AsyncMock())


@pytest.fixture(autouse=True)
def _clear_caches():
    llm_mod._NO_RESPONSES_ENDPOINT.clear()
    llm_mod._PROBE_CACHE.clear()
    llm_mod._PROBE_LOCKS.clear()


def _client():
    return OpenAICompatibleClient(api_key="k", base_url="https://x/v1", agent_name="t")


@pytest.mark.asyncio
async def test_invoke_adapter_retries_transient_then_succeeds():
    c = _client()
    c.llm_logger = MagicMock()
    calls = {"n": 0}
    async def _maybe():
        calls["n"] += 1
        if calls["n"] < 3:
            raise APITimeoutError(request=MagicMock())
        return LLMResponse("ok", "m", {"total_tokens": 1}, None)
    result = await c._invoke_adapter(
        interaction_type="chat", model="gpt-4o", log_messages=[{"role": "user", "content": "x"}],
        make_coro=_maybe, make_coro_no_temp=_maybe)
    assert result.content == "ok"
    assert calls["n"] == 3
    assert c.llm_logger.log_request.called and c.llm_logger.log_response.called


@pytest.mark.asyncio
async def test_invoke_adapter_raises_upstream_exhausted():
    c = _client()
    c.llm_logger = MagicMock()
    async def _always_timeout():
        raise APITimeoutError(request=MagicMock())
    with pytest.raises(UpstreamExhaustedError):
        await c._invoke_adapter(
            interaction_type="chat", model="gpt-4o", log_messages=[{"role": "user", "content": "x"}],
            make_coro=_always_timeout, make_coro_no_temp=_always_timeout)
    assert c.llm_logger.log_response.called


@pytest.mark.asyncio
async def test_invoke_adapter_temp_fallback():
    c = _client()
    c.llm_logger = MagicMock()
    async def _with_temp():
        raise BadRequestError(message="invalid temperature value",
                              response=MagicMock(status_code=400), body=None)
    async def _no_temp():
        return LLMResponse("notemp", "m", {}, None)
    result = await c._invoke_adapter(
        interaction_type="chat", model="gpt-4o", log_messages=[{"role": "user", "content": "x"}],
        make_coro=_with_temp, make_coro_no_temp=_no_temp)
    assert result.content == "notemp"


@pytest.mark.asyncio
async def test_adapter_temp_call_pair_passes_skip_temperature_values():
    seen = []

    async def call(skip_temperature):
        seen.append(skip_temperature)
        return LLMResponse("ok", "m", {}, None)

    make_coro, make_coro_no_temp = OpenAICompatibleClient._adapter_temp_call_pair(call)

    first = await make_coro()
    second = await make_coro_no_temp()

    assert first.content == "ok"
    assert second.content == "ok"
    assert seen == [False, True]


@pytest.mark.asyncio
async def test_chat_responses_path_retries_and_logs():
    c = _client()
    c.llm_logger = MagicMock()
    calls = {"n": 0}
    fake = MagicMock()
    async def _complete(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise APITimeoutError(request=MagicMock())
        return LLMResponse("R", "gpt", {"total_tokens": 1}, None)
    fake.complete = AsyncMock(side_effect=_complete)
    c._responses_adapter = fake
    r = await c.chat(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert r.content == "R"
    assert calls["n"] == 2  # retried once
    assert c.llm_logger.log_response.called  # adapter path now logged
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-4o")] == "responses"


@pytest.mark.asyncio
async def test_chat_responses_path_passes_explicit_tools_to_adapter():
    c = _client()
    c.llm_logger = MagicMock()
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=LLMResponse("R", "gpt", {"total_tokens": 1}, None))
    c._responses_adapter = fake

    result = await c.chat(
        model="o4-mini-deep-research",
        messages=[{"role": "user", "content": "research copper transport"}],
        temperature=0.3,
        timeout=1800.0,
        tools=[{"type": "web_search_preview"}],
    )

    assert result.content == "R"
    fake.complete.assert_awaited_once()
    kwargs = fake.complete.call_args.kwargs
    assert kwargs["tools"] == [{"type": "web_search_preview"}]
    assert kwargs["timeout"] == 1800.0


@pytest.mark.asyncio
async def test_chat_responses_endpoint_absent_still_falls_back():
    c = _client()
    c.llm_logger = MagicMock()
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=NotFoundError(
        message="Cannot POST /v1/responses", response=MagicMock(status_code=404), body=None))
    c._responses_adapter = fake
    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT", tool_calls=None))]
    inline_resp.model = "gpt-4o"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)
    r = await c.chat(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert r.content == "CHAT"
    assert "https://x/v1" in llm_mod._NO_RESPONSES_ENDPOINT


@pytest.mark.asyncio
async def test_probe_lock_serializes_protocol_decision_but_all_calls_succeed():
    import asyncio
    c = _client()
    c.llm_logger = MagicMock()
    response_calls = {"n": 0}
    fake = MagicMock()
    async def _slow(**kwargs):
        response_calls["n"] += 1
        await asyncio.sleep(0.01)  # real sleep to create concurrent window
        return LLMResponse("R", "gpt", {"total_tokens": 1}, None)
    fake.complete = AsyncMock(side_effect=_slow)
    c._responses_adapter = fake
    await asyncio.gather(*[
        c.chat(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        for _ in range(10)
    ])
    # The first concurrent caller acquires the lock and probes once, writing
    # _PROBE_CACHE["responses"]. Subsequent callers skip the lock (fast path:
    # cache hit) and each executes their own business request via the adapter.
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-4o")] == "responses"
    assert response_calls["n"] == 10


@pytest.mark.asyncio
async def test_responses_vision_includes_enforcement_instructions():
    adapter, captured = _responses_adapter_with_capture()
    await adapter.complete_vision(model="gpt-x", text_prompt="describe", images=[],
                                  temperature=0.5, max_tokens=10)
    assert captured.get("instructions") and ENGLISH_ENFORCEMENT.split(".")[0] in captured["instructions"]


@pytest.mark.asyncio
async def test_dispatch_non_streaming_uses_responses_then_caches():
    c = _client()
    c.llm_logger = MagicMock()
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=LLMResponse("R", "gpt", {"total_tokens": 1}, None))
    c._responses_adapter = fake

    result = await c._dispatch_non_streaming(
        model="gpt-4o",
        interaction_type="chat",
        log_messages=[{"role": "user", "content": "hi"}],
        make_anthropic_adapter_call=None,
        make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            temperature=0.7, max_tokens=None, timeout=None, skip_temperature=skip_temperature),
        make_inline_call=lambda: c._chat_inline("gpt-4o", [{"role": "user", "content": "hi"}]),
        temperature=0.7,
    )

    assert result.content == "R"
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-4o")] == "responses"


@pytest.mark.asyncio
async def test_dispatch_non_streaming_reactive_responses_only_fallback():
    c = OpenAICompatibleClient(api_key="k", base_url="https://x/v1/chat/completions", agent_name="t")
    c.llm_logger = MagicMock()
    err = BadRequestError(message="This model is only supported in v1/responses",
                          response=MagicMock(status_code=400), body=None)
    inline = AsyncMock(side_effect=err)
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=LLMResponse("R", "m", {}, None))
    c._responses_adapter = fake

    result = await c._dispatch_non_streaming(
        model="responses-only",
        interaction_type="chat",
        log_messages=[{"role": "user", "content": "hi"}],
        make_anthropic_adapter_call=None,
        make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete(
            model="responses-only", messages=[{"role": "user", "content": "hi"}],
            temperature=0.7, max_tokens=None, timeout=None, skip_temperature=skip_temperature),
        make_inline_call=inline,
        temperature=0.7,
    )

    assert result.content == "R"
    inline.assert_awaited_once()
    fake.complete.assert_awaited_once()
    assert llm_mod._PROBE_CACHE[("https://x/v1/chat/completions", "responses-only")] == "responses"


def test_responses_probe_decision_uses_base_url_no_endpoint_cache():
    c = _client()
    llm_mod._NO_RESPONSES_ENDPOINT.add("https://x/v1")

    assert c._responses_probe_decision("gpt-4o") == "chat"


def test_record_responses_probe_failure_caches_chat_only_model():
    c = _client()
    err = BadRequestError(message="This model is only supported in v1/chat/completions",
                          response=MagicMock(status_code=400), body=None)

    assert c._record_responses_probe_failure("gpt-chat-only", err) == "chat"
    assert "https://x/v1" not in llm_mod._NO_RESPONSES_ENDPOINT
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-chat-only")] == "chat"
