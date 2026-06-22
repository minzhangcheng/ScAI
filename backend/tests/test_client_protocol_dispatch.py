import pytest
import json
from unittest.mock import AsyncMock, MagicMock
import backend.utils.llm_client as llm_mod
from backend.utils.llm_client import OpenAICompatibleClient, LLMResponse, LLMToolResponse, NotFoundError, BadRequestError, InternalServerError, RateLimitError


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    llm_mod._NO_RESPONSES_ENDPOINT.clear()
    llm_mod._PROBE_CACHE.clear()
    monkeypatch.setattr(llm_mod.asyncio, "sleep", AsyncMock())


def _client(base_url):
    c = OpenAICompatibleClient(api_key="k", base_url=base_url, agent_name="t")
    return c


@pytest.mark.asyncio
async def test_anthropic_route_uses_anthropic_adapter():
    c = _client("https://api.anthropic.com")  # no /v1 -> anthropic
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=LLMResponse("A", "claude", {}, None))
    c._anthropic_adapter = fake
    r = await c.chat(model="claude-3", messages=[{"role": "user", "content": "hi"}])
    assert r.content == "A"
    fake.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_success_caches_responses():
    c = _client("https://x/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=LLMResponse("R", "gpt", {}, None))
    c._responses_adapter = fake
    r = await c.chat(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert r.content == "R"
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-4o")] == "responses"


@pytest.mark.asyncio
async def test_responses_endpoint_absent_falls_back_to_chat(monkeypatch):
    c = _client("https://x/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=NotFoundError(
        message="Cannot POST /v1/responses", response=MagicMock(status_code=404), body=None))
    c._responses_adapter = fake
    # 现有内联 chat 路径用 self.client.chat.completions.create —— mock 之
    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT", tool_calls=None))]
    inline_resp.model = "gpt-4o"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)
    r = await c.chat(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert r.content == "CHAT"
    assert "https://x/v1" in llm_mod._NO_RESPONSES_ENDPOINT
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-4o")] == "chat"


@pytest.mark.asyncio
async def test_responses_not_implemented_500_falls_back_to_chat_without_retry():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=InternalServerError(
        message="Error code: 500 - {'error': {'message': 'not implemented (request id: x)', 'type': 'new_api_error', 'param': '', 'code': 'local:convert_request_failed'}}",
        response=MagicMock(status_code=500),
        body={"error": {"message": "not implemented (request id: x)", "code": "local:convert_request_failed"}},
    ))
    c._responses_adapter = fake

    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT", tool_calls=None))]
    inline_resp.model = "gemini-3.1-pro-preview"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT"
    fake.complete.assert_awaited_once()
    c.client.chat.completions.create.assert_awaited_once()
    llm_mod.asyncio.sleep.assert_not_awaited()
    assert "https://api.n1n.ai/v1" in llm_mod._NO_RESPONSES_ENDPOINT
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "gemini-3.1-pro-preview")] == "chat"


@pytest.mark.asyncio
async def test_vision_responses_not_implemented_500_falls_back_to_chat():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete_vision = AsyncMock(side_effect=InternalServerError(
        message="Error code: 500 - {'error': {'message': 'not implemented', 'code': 'local:convert_request_failed'}}",
        response=MagicMock(status_code=500),
        body={"error": {"message": "not implemented", "code": "local:convert_request_failed"}},
    ))
    c._responses_adapter = fake

    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="VISION_CHAT", tool_calls=None))]
    inline_resp.model = "gemini-3.1-pro-preview"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat_with_vision(
        model="gemini-3.1-pro-preview",
        text_prompt="describe",
        images=[],
        temperature=0.5,
        max_tokens=10,
    )

    assert r.content == "VISION_CHAT"
    fake.complete_vision.assert_awaited_once()
    c.client.chat.completions.create.assert_awaited_once()
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "gemini-3.1-pro-preview")] == "chat"


@pytest.mark.asyncio
async def test_tools_responses_not_implemented_500_falls_back_to_chat():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete_with_tools = AsyncMock(side_effect=InternalServerError(
        message="Error code: 500 - {'error': {'message': 'not implemented', 'code': 'local:convert_request_failed'}}",
        response=MagicMock(status_code=500),
        body={"error": {"message": "not implemented", "code": "local:convert_request_failed"}},
    ))
    c._responses_adapter = fake

    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="TOOLS_CHAT", tool_calls=None))]
    inline_resp.model = "gemini-3.1-pro-preview"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat_with_tools(
        model="gemini-3.1-pro-preview",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
    )

    assert r.content == "TOOLS_CHAT"
    fake.complete_with_tools.assert_awaited_once()
    c.client.chat.completions.create.assert_awaited_once()
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "gemini-3.1-pro-preview")] == "chat"


@pytest.mark.asyncio
async def test_stream_responses_not_implemented_500_falls_back_to_chat():
    c = _client("https://api.n1n.ai/v1")

    async def _raise_at_first_chunk(**kwargs):
        raise InternalServerError(
            message="Error code: 500 - {'error': {'message': 'not implemented', 'code': 'local:convert_request_failed'}}",
            response=MagicMock(status_code=500),
            body={"error": {"message": "not implemented", "code": "local:convert_request_failed"}},
        )
        yield  # pragma: no cover

    fake = MagicMock()
    fake.stream = MagicMock(side_effect=lambda **kwargs: _raise_at_first_chunk(**kwargs))
    c._responses_adapter = fake

    class _Chunk:
        def __init__(self, t): self.choices = [MagicMock(delta=MagicMock(content=t))]

    async def _chat_stream(*a, **k):
        for t in ["Hel", "lo"]:
            yield _Chunk(t)

    c.client.chat.completions.create = AsyncMock(side_effect=lambda **kw: _chat_stream())

    chunks = []
    async for ch in c.chat_stream(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}]):
        chunks.append(ch)

    assert "".join(chunks) == "Hello"
    assert fake.stream.call_count == 1
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "gemini-3.1-pro-preview")] == "chat"


@pytest.mark.asyncio
async def test_protocol_probe_persistent_500_does_not_cache_protocol():
    c = _client("https://x/v1")
    boom = InternalServerError(
        message="Error code: 500 - upstream overloaded",
        response=MagicMock(status_code=500),
        body={"error": {"message": "upstream overloaded"}},
    )
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=boom)
    c._responses_adapter = fake
    c.client.chat.completions.create = AsyncMock(side_effect=boom)
    anth = MagicMock()
    anth.complete = AsyncMock(side_effect=boom)
    c._anthropic_adapter = anth

    with pytest.raises(llm_mod.UpstreamExhaustedError):
        await c.chat(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert "https://x/v1" not in llm_mod._NO_RESPONSES_ENDPOINT
    assert ("https://x/v1", "gpt-4o") not in llm_mod._PROBE_CACHE
    assert fake.complete.await_count == 3


@pytest.mark.asyncio
async def test_responses_transient_then_chat_success_retries_then_caches_chat():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=InternalServerError(
        message="Error code: 500 - transient responses failure",
        response=MagicMock(status_code=500),
        body={"error": {"message": "transient responses failure"}},
    ))
    c._responses_adapter = fake

    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT_OK", tool_calls=None))]
    inline_resp.model = "gemini-3.1-pro-preview"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT_OK"
    assert fake.complete.await_count == 3
    c.client.chat.completions.create.assert_awaited_once()
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "gemini-3.1-pro-preview")] == "chat"


@pytest.mark.asyncio
async def test_protocol_probe_responses_recovers_before_touching_chat():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        InternalServerError(
            message="Error code: 500 - responses down once",
            response=MagicMock(status_code=500),
            body={"error": {"message": "responses down once"}},
        ),
        LLMResponse("RESPONSES_OK", "gemini-3.1-pro-preview", {}, None),
    ])
    c._responses_adapter = fake
    c.client.chat.completions.create = AsyncMock()

    r = await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "RESPONSES_OK"
    assert fake.complete.await_count == 2
    c.client.chat.completions.create.assert_not_awaited()
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "gemini-3.1-pro-preview")] == "responses"


@pytest.mark.asyncio
async def test_protocol_probe_exhausts_all_protocols_on_persistent_transient():
    c = _client("https://api.n1n.ai/v1")
    boom = InternalServerError(
        message="Error code: 500 - unavailable",
        response=MagicMock(status_code=500),
        body={"error": {"message": "unavailable"}},
    )
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=boom)
    c._responses_adapter = fake
    c.client.chat.completions.create = AsyncMock(side_effect=boom)
    anth = MagicMock()
    anth.complete = AsyncMock(side_effect=boom)
    c._anthropic_adapter = anth

    with pytest.raises(llm_mod.UpstreamExhaustedError) as exc_info:
        await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert "Protocol probe exhausted" in str(exc_info.value)
    assert fake.complete.await_count == 3
    assert c.client.chat.completions.create.await_count == 1
    assert anth.complete.await_count == 3
    assert ("https://api.n1n.ai/v1", "gemini-3.1-pro-preview") not in llm_mod._PROBE_CACHE


@pytest.mark.asyncio
async def test_protocol_probe_rate_limit_retries_then_switches_to_chat():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=RateLimitError(
        message="rate limited on responses",
        response=MagicMock(status_code=429),
        body={"error": {"message": "rate limited on responses"}},
    ))
    c._responses_adapter = fake

    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT_AFTER_429", tool_calls=None))]
    inline_resp.model = "gemini-3.1-pro-preview"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT_AFTER_429"
    assert fake.complete.await_count == 3
    c.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_protocol_probe_json_error_alternates_without_internal_sleep():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=json.JSONDecodeError("bad json", "{", 0))
    c._responses_adapter = fake

    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT_AFTER_JSON", tool_calls=None))]
    inline_resp.model = "gemini-3.1-pro-preview"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT_AFTER_JSON"
    fake.complete.assert_awaited_once()
    c.client.chat.completions.create.assert_awaited_once()
    llm_mod.asyncio.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_endpoint_absent_records_no_responses_but_no_protocol_cache_on_exhaustion():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=NotFoundError(
        message="Cannot POST /v1/responses",
        response=MagicMock(status_code=404),
        body=None,
    ))
    c._responses_adapter = fake
    boom = InternalServerError(
        message="Error code: 500 - chat unavailable",
        response=MagicMock(status_code=500),
        body={"error": {"message": "chat unavailable"}},
    )
    c.client.chat.completions.create = AsyncMock(side_effect=boom)
    anth = MagicMock()
    anth.complete = AsyncMock(side_effect=boom)
    c._anthropic_adapter = anth

    with pytest.raises(llm_mod.UpstreamExhaustedError):
        await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert "https://api.n1n.ai/v1" in llm_mod._NO_RESPONSES_ENDPOINT
    assert ("https://api.n1n.ai/v1", "gemini-3.1-pro-preview") not in llm_mod._PROBE_CACHE


@pytest.mark.asyncio
async def test_tools_responses_route():
    c = _client("https://x/v1")
    fake = MagicMock()
    fake.complete_with_tools = AsyncMock(
        return_value=LLMToolResponse(content="T", tool_calls=[], model="gpt",
                                     usage={}, raw_response=None))
    c._responses_adapter = fake
    r = await c.chat_with_tools(model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
                                tools=[{"type": "function", "function": {"name": "f"}}])
    assert isinstance(r, LLMToolResponse)
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-4o")] == "responses"


@pytest.mark.asyncio
async def test_tools_reactive_chat_to_responses_fallback():
    # base_url forces chat path (explicit suffix), but model is responses-only
    c = _client("https://x/v1/chat/completions")
    # 现有内联 chat_with_tools 第一发抛 responses-only BadRequestError
    err = BadRequestError(message="This model is only supported in v1/responses",
                          response=MagicMock(status_code=400), body=None)
    c.client.chat.completions.create = AsyncMock(side_effect=err)
    fake = MagicMock()
    fake.complete_with_tools = AsyncMock(
        return_value=LLMToolResponse(content="R", tool_calls=[], model="m",
                                     usage={}, raw_response=None))
    c._responses_adapter = fake
    r = await c.chat_with_tools(model="some-model", messages=[{"role": "user", "content": "hi"}],
                                tools=[{"type": "function", "function": {"name": "f"}}])
    assert r.content == "R"
    assert llm_mod._PROBE_CACHE[("https://x/v1/chat/completions", "some-model")] == "responses"


@pytest.mark.asyncio
async def test_stream_responses_endpoint_absent_falls_back_before_first_chunk():
    c = _client("https://x/v1")

    # responses 适配器：建立连接即抛端点缺失（尚未 yield）
    async def _boom(*a, **k):
        raise NotFoundError(message="Cannot POST /v1/responses",
                            response=MagicMock(status_code=404), body=None)
        yield  # pragma: no cover
    fake = MagicMock()
    fake.stream = MagicMock(side_effect=lambda **kw: _boom())
    c._responses_adapter = fake

    # 内联 chat_stream 用 self.client.chat.completions.create(stream=True)
    class _Chunk:
        def __init__(self, t): self.choices = [MagicMock(delta=MagicMock(content=t))]
    async def _chat_stream(*a, **k):
        for t in ["He", "llo"]:
            yield _Chunk(t)
    c.client.chat.completions.create = AsyncMock(side_effect=lambda **kw: _chat_stream())

    chunks = []
    async for ch in c.chat_stream(model="gpt-4o", messages=[{"role": "user", "content": "hi"}]):
        chunks.append(ch)
    assert "".join(chunks) == "Hello"
    assert "https://x/v1" in llm_mod._NO_RESPONSES_ENDPOINT


from backend.utils.llm_client import ENGLISH_ENFORCEMENT


@pytest.mark.asyncio
async def test_chat_native_path_injects_english_enforcement():
    # 显式 /chat/completions 后缀 -> 直接走原生 chat 分支（protocol == "chat"）
    c = _client("https://x/v1/chat/completions")
    captured = {}
    async def _fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None))]
        resp.model = "gpt-4o"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                               prompt_tokens_details=None)
        return resp
    c.client.chat.completions.create = AsyncMock(side_effect=_fake_create)
    await c.chat(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    # enforcement 必须出现在发往 SDK 的消息里
    joined = " ".join(str(m.get("content")) for m in captured["messages"])
    assert ENGLISH_ENFORCEMENT.split(".")[0] in joined


@pytest.mark.asyncio
async def test_chat_stream_true_rejected_before_protocol_dispatch():
    c = _client("https://x/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=LLMResponse("R", "gpt", {}, None))
    c._responses_adapter = fake

    with pytest.raises(ValueError, match="use chat_stream\(\) for streaming responses"):
        await c.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

    fake.complete.assert_not_called()
    assert ("https://x/v1", "gpt-4o") not in llm_mod._PROBE_CACHE


@pytest.mark.asyncio
async def test_plain_chat_reactive_chat_to_responses_fallback():
    c = _client("https://x/v1/chat/completions")
    err = BadRequestError(message="This model is only supported in v1/responses",
                          response=MagicMock(status_code=400), body=None)
    c.client.chat.completions.create = AsyncMock(side_effect=err)
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=LLMResponse("R", "m", {}, None))
    c._responses_adapter = fake

    r = await c.chat(model="responses-only", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "R"
    assert llm_mod._PROBE_CACHE[("https://x/v1/chat/completions", "responses-only")] == "responses"
    fake.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_chat_reactive_chat_to_responses_fallback_before_first_chunk():
    c = _client("https://x/v1/chat/completions")
    err = BadRequestError(message="This model is only supported in v1/responses",
                          response=MagicMock(status_code=400), body=None)
    c.client.chat.completions.create = AsyncMock(side_effect=err)

    async def _responses_stream(**kwargs):
        yield "Re"
        yield "sp"

    fake = MagicMock()
    fake.stream = MagicMock(side_effect=lambda **kwargs: _responses_stream(**kwargs))
    c._responses_adapter = fake

    chunks = []
    async for ch in c.chat_stream(model="responses-only", messages=[{"role": "user", "content": "hi"}]):
        chunks.append(ch)

    assert "".join(chunks) == "Resp"
    assert llm_mod._PROBE_CACHE[("https://x/v1/chat/completions", "responses-only")] == "responses"


@pytest.mark.asyncio
async def test_vision_chat_reactive_chat_to_responses_fallback():
    c = _client("https://x/v1/chat/completions")
    err = BadRequestError(message="This model is only supported in v1/responses",
                          response=MagicMock(status_code=400), body=None)
    c.client.chat.completions.create = AsyncMock(side_effect=err)
    fake = MagicMock()
    fake.complete_vision = AsyncMock(return_value=LLMResponse("VR", "m", {}, None))
    c._responses_adapter = fake

    r = await c.chat_with_vision(model="responses-only", text_prompt="describe",
                                 images=[], temperature=0.5, max_tokens=10)

    assert r.content == "VR"
    assert llm_mod._PROBE_CACHE[("https://x/v1/chat/completions", "responses-only")] == "responses"
    fake.complete_vision.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_responses_chat_only_does_not_mark_base_url_no_responses():
    c = _client("https://x/v1")

    # Simulate a Responses adapter.stream that raises BadRequestError at first
    # __anext__() (connection establishment). The mock must return an async
    # generator, not a coroutine, since adapter.stream is async def with yield.
    async def _raise_at_first_chunk(**kwargs):
        raise BadRequestError(message="This model is only supported in v1/chat/completions",
                              response=MagicMock(status_code=400), body=None)
        yield  # pragma: no cover

    fake = MagicMock()
    fake.stream = MagicMock(side_effect=lambda **kwargs: _raise_at_first_chunk(**kwargs))
    c._responses_adapter = fake

    class _Chunk:
        def __init__(self, t): self.choices = [MagicMock(delta=MagicMock(content=t))]
    async def _chat_stream(*a, **k):
        for t in ["Hel", "lo"]:
            yield _Chunk(t)
    c.client.chat.completions.create = AsyncMock(side_effect=lambda **kw: _chat_stream())

    chunks = []
    async for ch in c.chat_stream(model="gpt-chat-only", messages=[{"role": "user", "content": "hi"}]):
        chunks.append(ch)
    assert "".join(chunks) == "Hello"
    assert "https://x/v1" not in llm_mod._NO_RESPONSES_ENDPOINT
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-chat-only")] == "chat"


@pytest.mark.asyncio
async def test_probe_responses_transient_retries_3x_then_switches_to_chat():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=InternalServerError(
        message="Error code: 500 - transient blip",
        response=MagicMock(status_code=500),
        body={"error": {"message": "transient blip"}},
    ))
    c._responses_adapter = fake
    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT_OK", tool_calls=None))]
    inline_resp.model = "m"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT_OK"
    assert fake.complete.await_count == 3
    sleeps = [call.args[0] for call in llm_mod.asyncio.sleep.await_args_list]
    assert sleeps == [2.0, 15.0, 10.0]
    c.client.chat.completions.create.assert_awaited_once()
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "m")] == "chat"


@pytest.mark.asyncio
async def test_probe_responses_recovers_on_second_attempt_stays_responses():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        InternalServerError(message="Error code: 500 - blip",
                            response=MagicMock(status_code=500),
                            body={"error": {"message": "blip"}}),
        LLMResponse("RESP_OK", "m", {}, None),
    ])
    c._responses_adapter = fake
    c.client.chat.completions.create = AsyncMock()

    r = await c.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "RESP_OK"
    assert fake.complete.await_count == 2
    c.client.chat.completions.create.assert_not_awaited()
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "m")] == "responses"


@pytest.mark.asyncio
async def test_probe_endpoint_absent_switches_immediately_no_retry():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=NotFoundError(
        message="Cannot POST /v1/responses",
        response=MagicMock(status_code=404), body=None))
    c._responses_adapter = fake
    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT", tool_calls=None))]
    inline_resp.model = "m"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT"
    fake.complete.assert_awaited_once()
    llm_mod.asyncio.sleep.assert_not_awaited()
    assert "https://api.n1n.ai/v1" in llm_mod._NO_RESPONSES_ENDPOINT
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "m")] == "chat"


@pytest.mark.asyncio
async def test_probe_anthropic_third_in_order_succeeds_caches_anthropic():
    c = _client("https://gateway.example.com/v1")
    resp_fake = MagicMock()
    resp_fake.complete = AsyncMock(side_effect=NotFoundError(
        message="Cannot POST /v1/responses",
        response=MagicMock(status_code=404), body=None))
    c._responses_adapter = resp_fake
    c.client.chat.completions.create = AsyncMock(side_effect=NotFoundError(
        message="Cannot POST /v1/chat/completions",
        response=MagicMock(status_code=404), body=None))
    anth_fake = MagicMock()
    anth_fake.complete = AsyncMock(return_value=LLMResponse("ANTH", "claude", {}, None))
    c._anthropic_adapter = anth_fake

    r = await c.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "ANTH"
    assert llm_mod._PROBE_CACHE[("https://gateway.example.com/v1", "m")] == "anthropic"


@pytest.mark.asyncio
async def test_cached_responses_uses_full_retry_on_transient():
    c = _client("https://x/v1")
    llm_mod._PROBE_CACHE[("https://x/v1", "m")] = "responses"
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        InternalServerError(message="Error code: 500 - blip1",
                            response=MagicMock(status_code=500),
                            body={"error": {"message": "blip1"}}),
        InternalServerError(message="Error code: 500 - blip2",
                            response=MagicMock(status_code=500),
                            body={"error": {"message": "blip2"}}),
        LLMResponse("RESP_OK", "m", {}, None),
    ])
    c._responses_adapter = fake

    r = await c.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "RESP_OK"
    assert fake.complete.await_count == 3
    assert llm_mod._PROBE_CACHE[("https://x/v1", "m")] == "responses"
