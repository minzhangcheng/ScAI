import asyncio
import httpx
import pytest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from openai import AsyncOpenAI
import backend.utils.llm_client as llm_mod
from backend.utils.llm_client import OpenAICompatibleClient, LLMResponse, LLMToolResponse, NotFoundError, BadRequestError, InternalServerError, RateLimitError
from backend.utils.llm_client_parts.media import normalize_image_input


_OPEN_CLIENTS = []


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    llm_mod._NO_RESPONSES_ENDPOINT.clear()
    llm_mod._NO_RESPONSES_VISION_STREAM.clear()
    llm_mod._NO_TEXT_STREAM.clear()
    llm_mod._NO_CHAT_STREAM_OPTIONS.clear()
    llm_mod._PROBE_CACHE.clear()
    monkeypatch.setattr(llm_mod.asyncio, "sleep", AsyncMock())
    # These protocol unit tests exercise dispatch, not the production JSONL
    # writer.  Keeping it enabled also makes the tests depend on the host's
    # default executor and data-directory permissions.
    llm_logger = llm_mod.get_llm_logger()
    logging_was_enabled = llm_logger.enabled
    llm_logger.enabled = False
    _OPEN_CLIENTS.clear()
    try:
        yield
    finally:
        async def close_all():
            for client in _OPEN_CLIENTS:
                await client.close()

        asyncio.run(close_all())
        _OPEN_CLIENTS.clear()
        llm_logger.enabled = logging_was_enabled


def _client(base_url):
    c = OpenAICompatibleClient(api_key="k", base_url=base_url, agent_name="t")
    _OPEN_CLIENTS.append(c)
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
async def test_plain_chat_completions_prefers_stream_and_aggregates_text():
    c = _client("https://x/v1/chat/completions")

    async def chunks():
        yield SimpleNamespace(
            model="chat-model",
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Hel"), finish_reason=None
            )],
        )
        yield SimpleNamespace(
            model="chat-model",
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="lo"), finish_reason="stop"
            )],
        )
        yield SimpleNamespace(
            model="chat-model",
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                prompt_tokens_details=None,
            ),
            choices=[],
        )

    c.client.chat.completions.create = AsyncMock(return_value=chunks())
    result = await c.chat(
        model="chat-model",
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )

    assert result.content == "Hello"
    assert result.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    request = c.client.chat.completions.create.call_args.kwargs
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_plain_chat_stream_5xx_retries_stream_instead_of_falling_back():
    client = _client("https://x/v1/chat/completions")

    async def chunks():
        yield SimpleNamespace(
            model="chat-model",
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="recovered"), finish_reason="stop"
            )],
        )

    client.client.chat.completions.create = AsyncMock(side_effect=[
        InternalServerError(
            message="temporary upstream error",
            response=MagicMock(status_code=500),
            body={"error": {"message": "temporary upstream error"}},
        ),
        chunks(),
    ])

    result = await client.chat(
        model="chat-model",
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )

    assert result.content == "recovered"
    assert [
        call.kwargs["stream"]
        for call in client.client.chat.completions.create.await_args_list
    ] == [True, True]
    llm_mod.asyncio.sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_plain_chat_midstream_eof_is_not_replayed_nonstreaming():
    async def chunks():
        yield SimpleNamespace(
            model="chat-model",
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="partial"), finish_reason=None
            )],
        )

    client = _client("https://x/v1/chat/completions")
    client.client.chat.completions.create = AsyncMock(return_value=chunks())

    with pytest.raises(
        llm_mod.UpstreamExhaustedError, match="finish_reason"
    ) as exc_info:
        await client.chat(
            model="chat-model",
            messages=[{"role": "user", "content": "hi"}],
            prefer_stream=True,
        )

    # The classified failure keeps the original error reachable and its
    # message intact, so the wire shape that truncated stays diagnosable.
    assert exc_info.value.failure_kind == "stream_interrupted"
    assert isinstance(exc_info.value.__cause__, llm_mod.StreamInterruptedError)
    client.client.chat.completions.create.assert_awaited_once()
    assert client.client.chat.completions.create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_plain_text_empty_stream_falls_back_once_without_caching():
    base_url = "https://x/v1"
    model = "gpt-4o"
    llm_mod._PROBE_CACHE[(base_url, model)] = "responses"

    async def empty_events():
        if False:
            yield None

    completed = MagicMock()
    completed.model = model
    completed.status = "completed"
    completed.model_dump.return_value = {
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "fallback"}],
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    client = _client(base_url)
    client.client.responses.create = AsyncMock(
        side_effect=[empty_events(), completed]
    )

    result = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )

    assert result.content == "fallback"
    assert [
        call.kwargs.get("stream", False)
        for call in client.client.responses.create.await_args_list
    ] == [True, False]
    assert (base_url, model, "text", "responses") not in llm_mod._NO_TEXT_STREAM


@pytest.mark.asyncio
async def test_chat_stream_refusal_is_surfaced_as_the_public_answer():
    """A refusal-only stream must reach callers as the non-streaming path
    renders it, not die as an empty stream with no fallback."""
    base_url = "https://x/v1"
    model = "chat-model"
    llm_mod._PROBE_CACHE[(base_url, model)] = "chat"

    async def chunks():
        yield SimpleNamespace(
            model=model, usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None,
                                      refusal="I cannot help with that"),
                finish_reason="stop")],
        )

    client = _client(base_url)
    client.client.chat.completions.create = AsyncMock(return_value=chunks())

    result = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )

    assert result.content == "[Refusal]: I cannot help with that"
    client.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_protocol_vision_prefers_stream_and_aggregates_text():
    """Vision over Chat Completions streams too, so a long tile batch is held
    open by output rather than by one total-request deadline."""
    base_url = "https://x/v1"
    model = "vision-model"
    llm_mod._PROBE_CACHE[(base_url, model, "vision")] = "chat"

    async def chunks():
        yield SimpleNamespace(
            model=model, usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content='{"ok":'),
                                     finish_reason=None)])
        yield SimpleNamespace(
            model=model,
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3,
                                  total_tokens=10, prompt_tokens_details=None),
            choices=[SimpleNamespace(delta=SimpleNamespace(content=' true}'),
                                     finish_reason="stop")])

    client = _client(base_url)
    client.client.chat.completions.create = AsyncMock(return_value=chunks())

    result = await client.chat_with_vision(
        model=model,
        text_prompt="describe",
        images=["aW1hZ2U="],
        prefer_stream=True,
    )

    assert result.content == '{"ok": true}'
    assert result.usage["prompt_tokens"] == 7
    kwargs = client.client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_chat_protocol_vision_stream_unsupported_is_cached_and_falls_back():
    base_url = "https://x/v1"
    model = "vision-model"
    llm_mod._PROBE_CACHE[(base_url, model, "vision")] = "chat"
    stream_flags = []

    fallback = MagicMock()
    fallback.choices = [MagicMock(message=MagicMock(content="non-stream",
                                                    tool_calls=None))]
    fallback.model = model
    fallback.usage = MagicMock(prompt_tokens=1, completion_tokens=1,
                               total_tokens=2, prompt_tokens_details=None)

    async def create(**kwargs):
        stream_flags.append(kwargs.get("stream", False))
        if kwargs.get("stream"):
            raise BadRequestError(
                message="Error code: 400 - stream is not supported",
                response=MagicMock(status_code=400),
                body={"error": {"message": "stream is not supported",
                                "param": "stream"}},
            )
        return fallback

    first = _client(base_url)
    first.client.chat.completions.create = AsyncMock(side_effect=create)
    assert (await first.chat_with_vision(
        model=model, text_prompt="describe", images=["aW1hZ2U="],
        prefer_stream=True)).content == "non-stream"

    second = _client(base_url)
    second.client.chat.completions.create = AsyncMock(side_effect=create)
    assert (await second.chat_with_vision(
        model=model, text_prompt="again", images=["aW1hZ2U="],
        prefer_stream=True)).content == "non-stream"

    assert stream_flags == [True, False, False]
    assert (base_url, model, "vision", "chat") in llm_mod._NO_TEXT_STREAM
    # The Responses-vision cache is a separate key space and stays untouched.
    assert (base_url, model, "vision") not in llm_mod._NO_RESPONSES_VISION_STREAM


@pytest.mark.asyncio
async def test_anthropic_protocol_vision_requests_a_deferred_stream():
    base_url = "https://api.anthropic.com"  # no /v1 -> anthropic
    model = "claude-3"
    seen = {}

    async def complete_vision(**kwargs):
        seen.update(kwargs)
        return LLMResponse("Red", model, {"prompt_tokens": 6}, None)

    client = _client(base_url)
    adapter = MagicMock()
    adapter.complete_vision = AsyncMock(side_effect=complete_vision)
    client._anthropic_adapter = adapter

    result = await client.chat_with_vision(
        model=model, text_prompt="describe", images=["aW1hZ2U="],
        prefer_stream=True,
    )

    assert result.content == "Red"
    assert seen["stream"] is True and seen["defer_stream"] is True
    adapter.complete_vision.assert_awaited_once()


@pytest.mark.asyncio
async def test_vision_empty_stream_falls_back_once_without_caching():
    base_url = "https://x/v1"
    model = "gemini-3.1-pro-preview"
    llm_mod._PROBE_CACHE[(base_url, model, "vision")] = "responses"

    async def empty_events():
        if False:
            yield None

    completed = MagicMock()
    completed.model = model
    completed.status = "completed"
    completed.model_dump.return_value = {
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": '{"ok": true}'}],
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    client = _client(base_url)
    client.client.responses.create = AsyncMock(
        side_effect=[empty_events(), completed]
    )

    result = await client.chat_with_vision(
        model=model,
        text_prompt="describe",
        images=["aW1hZ2U="],
        prefer_stream=True,
    )

    assert result.content == '{"ok": true}'
    assert [
        call.kwargs.get("stream", False)
        for call in client.client.responses.create.await_args_list
    ] == [True, False]
    assert (base_url, model, "vision") not in llm_mod._NO_RESPONSES_VISION_STREAM


@pytest.mark.asyncio
async def test_tools_empty_stream_falls_back_once_without_caching():
    base_url = "https://x/v1"
    model = "gpt-4o"
    llm_mod._PROBE_CACHE[(base_url, model)] = "responses"

    async def empty_events():
        if False:
            yield None

    completed = MagicMock()
    completed.model = model
    completed.status = "completed"
    completed.model_dump.return_value = {
        "status": "completed",
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "execute_code",
            "arguments": '{"code":"print(1)"}',
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    client = _client(base_url)
    client.client.responses.create = AsyncMock(
        side_effect=[empty_events(), completed]
    )

    result = await client.chat_with_tools(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {
            "name": "execute_code",
            "description": "run code",
            "parameters": {"type": "object", "properties": {}},
        }}],
        prefer_stream=True,
    )

    assert [tc.name for tc in result.tool_calls] == ["execute_code"]
    assert [
        call.kwargs.get("stream", False)
        for call in client.client.responses.create.await_args_list
    ] == [True, False]
    assert (base_url, model, "tools", "responses") not in llm_mod._NO_TEXT_STREAM


@pytest.mark.asyncio
async def test_plain_text_stream_unsupported_is_cached_and_falls_back():
    base_url = "https://x/v1"
    model = "gpt-4o"
    llm_mod._PROBE_CACHE[(base_url, model)] = "responses"
    stream_flags = []

    async def complete(**kwargs):
        stream_flags.append(kwargs.get("stream", False))
        if kwargs.get("stream"):
            raise BadRequestError(
                message="Error code: 400 - stream is not supported",
                response=MagicMock(status_code=400),
                body={"error": {
                    "message": "stream is not supported",
                    "param": "stream",
                }},
            )
        return LLMResponse("fallback", model, {}, None)

    first = _client(base_url)
    first._responses_adapter = MagicMock()
    first._responses_adapter.complete = AsyncMock(side_effect=complete)
    assert (await first.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )).content == "fallback"

    second = _client(base_url)
    second._responses_adapter = MagicMock()
    second._responses_adapter.complete = AsyncMock(side_effect=complete)
    assert (await second.chat(
        model=model,
        messages=[{"role": "user", "content": "again"}],
        prefer_stream=True,
    )).content == "fallback"

    assert stream_flags == [True, False, False]
    assert (base_url, model, "text", "responses") in llm_mod._NO_TEXT_STREAM


@pytest.mark.asyncio
async def test_deep_research_keeps_nonstreaming_background_lifecycle():
    client = _client("https://x/v1/responses")
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value=LLMResponse("research", "o3-deep-research", {}, None)
    )
    client._responses_adapter = adapter

    result = await client.chat(
        model="o3-deep-research",
        messages=[{"role": "user", "content": "research"}],
        prefer_stream=True,
    )

    assert result.content == "research"
    assert adapter.complete.call_args.kwargs.get("stream", False) is False


@pytest.mark.asyncio
async def test_plain_text_stream_transient_failure_retries_stream_only():
    base_url = "https://x/v1"
    model = "gpt-4o"
    llm_mod._PROBE_CACHE[(base_url, model)] = "responses"
    stream_flags = []

    async def complete(**kwargs):
        stream_flags.append(kwargs.get("stream", False))
        if len(stream_flags) <= 2:
            raise InternalServerError(
                message="temporary upstream error",
                response=MagicMock(status_code=500),
                body={"error": {"message": "temporary upstream error"}},
            )
        return LLMResponse("recovered", model, {}, None)

    client = _client(base_url)
    client._responses_adapter = MagicMock()
    client._responses_adapter.complete = AsyncMock(side_effect=complete)

    result = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )

    assert result.content == "recovered"
    assert stream_flags == [True, True, True]
    assert (base_url, model, "text", "responses") not in llm_mod._NO_TEXT_STREAM
    assert llm_mod.asyncio.sleep.await_count == 2


@pytest.mark.asyncio
async def test_responses_stream_remainder_is_consumed_after_retry_boundary(monkeypatch):
    import backend.utils.llm_client_parts.client_params as params_mod

    retry_returned = asyncio.Event()

    async def tracing_retry(func, **_kwargs):
        result = await func()
        assert isinstance(result, llm_mod._StartedStream)
        retry_returned.set()
        return result

    monkeypatch.setattr(params_mod, "retry_with_backoff", tracing_retry)

    release = asyncio.Event()
    completed = MagicMock()
    completed.model = "gpt-x"
    completed.model_dump.return_value = {
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "Hello"}],
        }],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    }

    class BlockingStream:
        def __init__(self):
            self.index = 0
            self.close = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return SimpleNamespace(type="response.output_text.delta", delta="Hel")
            if self.index == 2:
                await release.wait()
                return SimpleNamespace(type="response.completed", response=completed)
            raise StopAsyncIteration

    stream = BlockingStream()
    client = _client("https://x/v1/responses")
    client.client.responses.create = AsyncMock(return_value=stream)

    task = asyncio.create_task(client.chat(
        model="gpt-x",
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    ))
    await asyncio.wait_for(retry_returned.wait(), timeout=1)

    # The retry layer has already returned while the logical response is still
    # streaming. A total request deadline therefore cannot cancel this phase.
    assert task.done() is False
    assert client.client.responses.create.await_count == 1

    release.set()
    result = await task
    assert result.content == "Hello"
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_stream_interruption_is_classified_as_upstream_exhausted():
    """A stream that dies after its first event must leave the client as the
    HITL-gated failure type. Workflow nodes gate on UpstreamExhaustedError; a
    bare RuntimeError reaches their generic handler and kills the run."""

    class TruncatedStream:
        def __init__(self):
            self.index = 0
            self.close = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return SimpleNamespace(
                    type="response.output_text.delta", delta="partial"
                )
            raise StopAsyncIteration

    stream = TruncatedStream()
    client = _client("https://x/v1")
    client.client.responses.create = AsyncMock(return_value=stream)
    client.client.chat.completions.create = AsyncMock()

    with pytest.raises(llm_mod.UpstreamExhaustedError) as exc_info:
        await client.chat(
            model="gpt-x",
            messages=[{"role": "user", "content": "hi"}],
            prefer_stream=True,
        )

    assert exc_info.value.failure_kind == "stream_interrupted"
    assert exc_info.value.agent_name == "t"
    assert exc_info.value.model == "gpt-x"
    assert exc_info.value.llm_server == "https://x/v1"
    # The original error stays reachable: it names which wire shape truncated.
    assert isinstance(exc_info.value.__cause__, llm_mod.StreamInterruptedError)
    # Classification must not weaken the replay invariant: a partially emitted
    # generation is never re-issued on another protocol.
    client.client.chat.completions.create.assert_not_awaited()
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_stream_interruption_is_classified_as_upstream_exhausted():
    """Classification is a property of the client, not of one wire protocol."""

    async def chunks():
        yield SimpleNamespace(
            model="chat-model",
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="partial"), finish_reason=None
            )],
        )

    client = _client("https://x/v1/chat/completions")
    client.client.chat.completions.create = AsyncMock(return_value=chunks())

    with pytest.raises(llm_mod.UpstreamExhaustedError) as exc_info:
        await client.chat(
            model="chat-model",
            messages=[{"role": "user", "content": "hi"}],
            prefer_stream=True,
        )

    assert exc_info.value.failure_kind == "stream_interrupted"
    assert isinstance(exc_info.value.__cause__, llm_mod.StreamInterruptedError)
    client.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_midstream_failure_is_closed_and_not_replayed():
    error = InternalServerError(
        message="connection lost",
        response=MagicMock(status_code=500),
        body={"error": {"message": "connection lost"}},
    )

    class FailingStream:
        def __init__(self):
            self.index = 0
            self.close = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return SimpleNamespace(type="response.output_text.delta", delta="partial")
            raise error

    stream = FailingStream()
    # Leave protocol uncached so this also proves the probe loop does not replay
    # a stream that already emitted an event or switch it to another protocol.
    client = _client("https://x/v1")
    client.client.responses.create = AsyncMock(return_value=stream)
    client.client.chat.completions.create = AsyncMock()

    with pytest.raises(
        llm_mod.UpstreamExhaustedError, match="after the first event"
    ) as exc_info:
        await client.chat(
            model="gpt-x",
            messages=[{"role": "user", "content": "hi"}],
            prefer_stream=True,
        )

    # The classified failure keeps the original error reachable and its
    # message intact, so the wire shape that truncated stays diagnosable.
    assert exc_info.value.failure_kind == "stream_interrupted"
    assert isinstance(exc_info.value.__cause__, llm_mod.StreamInterruptedError)
    client.client.responses.create.assert_awaited_once()
    client.client.chat.completions.create.assert_not_awaited()
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_stream_idle_timeout_closes_without_replay():
    never = asyncio.Event()

    class IdleStream:
        def __init__(self):
            self.index = 0
            self.close = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return SimpleNamespace(type="response.output_text.delta", delta="partial")
            await never.wait()

    stream = IdleStream()
    client = _client("https://x/v1/responses")
    client.client.responses.create = AsyncMock(return_value=stream)

    with pytest.raises(
        llm_mod.UpstreamExhaustedError, match="TimeoutError"
    ) as exc_info:
        await client.chat(
            model="gpt-x",
            messages=[{"role": "user", "content": "hi"}],
            timeout=0.01,
            prefer_stream=True,
        )

    # The classified failure keeps the original error reachable and its
    # message intact, so the wire shape that truncated stays diagnosable.
    assert exc_info.value.failure_kind == "stream_interrupted"
    assert isinstance(exc_info.value.__cause__, llm_mod.StreamInterruptedError)
    client.client.responses.create.assert_awaited_once()
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_responses_stream_closes_and_releases_request_lease():
    entered_remainder = asyncio.Event()
    never = asyncio.Event()

    class BlockingStream:
        def __init__(self):
            self.index = 0
            self.close = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return SimpleNamespace(type="response.output_text.delta", delta="partial")
            entered_remainder.set()
            await never.wait()

    stream = BlockingStream()
    client = _client("https://x/v1/responses")
    client.client.responses.create = AsyncMock(return_value=stream)

    task = asyncio.create_task(client.chat(
        model="gpt-x",
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    ))
    await asyncio.wait_for(entered_remainder.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    stream.close.assert_awaited_once()
    assert client._active_requests == 0
    assert client._requests_idle.is_set()


@pytest.mark.asyncio
async def test_responses_incomplete_reasoning_only_stream_enters_upstream_failure():
    incomplete = MagicMock()
    incomplete.model = "gpt-x"
    incomplete.model_dump.return_value = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "error": None,
        "output": [{"type": "reasoning"}],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 32768,
            "total_tokens": 32773,
        },
    }

    async def events():
        yield SimpleNamespace(type="response.incomplete", response=incomplete)

    client = _client("https://x/v1")
    client.client.responses.create = AsyncMock(return_value=events())
    client.client.chat.completions.create = AsyncMock()

    with pytest.raises(llm_mod.UpstreamExhaustedError) as excinfo:
        await client.chat(
            model="gpt-x",
            messages=[{"role": "user", "content": "answer"}],
            max_tokens=32768,
            prefer_stream=True,
        )

    assert "status=incomplete" in str(excinfo.value)
    assert "max_output_tokens" in str(excinfo.value)
    assert excinfo.value.failure_kind == "incomplete_response"
    client.client.responses.create.assert_awaited_once()
    client.client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_responses_incomplete_text_is_preserved_with_metadata():
    base_url = "https://x/v1"
    model = "gpt-x"
    llm_mod._PROBE_CACHE[(base_url, model)] = "responses"
    result = LLMResponse(
        content="Usable but truncated answer",
        model=model,
        usage={"completion_tokens": 32768},
        raw_response=None,
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )

    client = _client(base_url)
    client._responses_adapter = MagicMock()
    client._responses_adapter.complete = AsyncMock(return_value=result)

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "answer"}],
        prefer_stream=False,
    )

    assert response.content == "Usable but truncated answer"
    assert response.status == "incomplete"
    assert response.incomplete_details == {"reason": "max_output_tokens"}


@pytest.mark.asyncio
async def test_plain_chat_stream_reasoning_only_retries_for_public_answer():
    async def reasoning_only_chunks():
        yield SimpleNamespace(
            model="chat-model",
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    reasoning_content="private chain of thought",
                ),
                finish_reason="stop",
            )],
        )

    async def public_answer_chunks():
        yield SimpleNamespace(
            model="chat-model",
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Public final answer"),
                finish_reason="stop",
            )],
        )

    client = _client("https://x/v1/chat/completions")
    client.client.chat.completions.create = AsyncMock(side_effect=[
        reasoning_only_chunks(),
        public_answer_chunks(),
    ])

    result = await client.chat(
        model="chat-model",
        messages=[{"role": "user", "content": "answer"}],
        prefer_stream=True,
    )

    assert result.content == "Public final answer"
    calls = client.client.chat.completions.create.await_args_list
    assert [call.kwargs["stream"] for call in calls] == [True, True]
    assert calls[1].kwargs["messages"][-1]["content"].startswith(
        "Return the final answer in the public response content field."
    )


def test_stream_unsupported_classifier_does_not_cache_other_bad_parameters():
    error = BadRequestError(
        message="Streaming request has unsupported parameter temperature",
        response=MagicMock(status_code=400),
        body={"error": {
            "message": "Unsupported parameter for streaming request",
            "param": "temperature",
        }},
    )

    assert llm_mod.is_stream_unsupported_error(error) is False


def test_stream_unsupported_classifier_rejects_temperature_without_param_field():
    error = BadRequestError(
        message="Unsupported parameter: temperature is not supported with streaming",
        response=MagicMock(status_code=400),
        body={"error": {
            "message": "Unsupported parameter: temperature is not supported with streaming",
        }},
    )

    assert llm_mod.is_stream_unsupported_error(error) is False

    direct_stream_error = BadRequestError(
        message="Unsupported parameter: stream",
        response=MagicMock(status_code=400),
        body={"error": {"message": "Unsupported parameter: stream"}},
    )
    assert llm_mod.is_stream_unsupported_error(direct_stream_error) is True


@pytest.mark.asyncio
async def test_chat_stream_options_rejection_retries_stream_without_usage_option_and_caches():
    base_url = "https://x/v1/chat/completions"
    model = "chat-model"
    error = BadRequestError(
        message="Unsupported parameter: stream_options",
        response=MagicMock(status_code=400),
        body={"error": {
            "message": "Unsupported parameter: stream_options",
            "param": "stream_options",
        }},
    )

    async def chunks(text):
        yield SimpleNamespace(
            model=model,
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=text), finish_reason="stop"
            )],
        )

    client = _client(base_url)
    client.client.chat.completions.create = AsyncMock(side_effect=[
        error,
        chunks("first"),
        chunks("second"),
    ])

    first = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )
    second = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "again"}],
        prefer_stream=True,
    )

    assert first.content == "first"
    assert second.content == "second"
    assert first.usage == {}
    calls = client.client.chat.completions.create.await_args_list
    assert calls[0].kwargs["stream_options"] == {"include_usage": True}
    assert "stream_options" not in calls[1].kwargs
    assert "stream_options" not in calls[2].kwargs
    assert all(call.kwargs["stream"] is True for call in calls)
    assert (base_url, model) in llm_mod._NO_CHAT_STREAM_OPTIONS
    assert (base_url, model, "text", "chat") not in llm_mod._NO_TEXT_STREAM


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
        system_prompt="PATHOLOGY ROLE",
    )

    assert r.content == "VISION_CHAT"
    fake.complete_vision.assert_awaited_once()
    c.client.chat.completions.create.assert_awaited_once()
    messages = c.client.chat.completions.create.call_args.kwargs["messages"]
    assert "PATHOLOGY ROLE" in messages[0]["content"]
    assert "respond exclusively in English" in messages[0]["content"]
    assert llm_mod._PROBE_CACHE[(
        "https://api.n1n.ai/v1", "gemini-3.1-pro-preview", "vision"
    )] == "chat"
    assert llm_mod._PROBE_CACHE.get((
        "https://api.n1n.ai/v1", "gemini-3.1-pro-preview"
    )) is None


@pytest.mark.asyncio
async def test_vision_stream_transient_failure_retries_stream_only():
    c = _client("https://x/v1")
    model = "gemini-3.1-pro-preview"
    llm_mod._PROBE_CACHE[("https://x/v1", model, "vision")] = "responses"
    stream_flags = []

    async def complete_vision(**kwargs):
        stream_flags.append(kwargs.get("stream", False))
        if len(stream_flags) <= 2:
            raise InternalServerError(
                message="temporary upstream error",
                response=MagicMock(status_code=500),
                body={"error": {"message": "temporary upstream error"}},
            )
        return LLMResponse('{"ok": true}', model, {}, None)

    responses = MagicMock()
    responses.complete_vision = AsyncMock(side_effect=complete_vision)
    c._responses_adapter = responses

    result = await c.chat_with_vision(
        model=model,
        text_prompt="describe",
        images=["aW1hZ2U="],
        prefer_stream=True,
    )

    assert result.content == '{"ok": true}'
    assert stream_flags == [True, True, True]
    assert ("https://x/v1", model, "vision") not in llm_mod._NO_RESPONSES_VISION_STREAM
    assert llm_mod.asyncio.sleep.await_count == 2


@pytest.mark.asyncio
async def test_vision_stream_unsupported_is_cached_per_endpoint_and_model():
    base_url = "https://x/v1"
    model = "gemini-3.1-pro-preview"
    calls = []

    async def complete_vision(**kwargs):
        calls.append((kwargs["model"], kwargs.get("stream", False)))
        if kwargs["model"] == model and kwargs.get("stream"):
            raise BadRequestError(
                message="Error code: 400 - stream is not supported",
                response=MagicMock(status_code=400),
                body={"error": {
                    "message": "stream is not supported",
                    "param": "stream",
                }},
            )
        return LLMResponse('{"ok": true}', kwargs["model"], {}, None)

    first = _client(base_url)
    first._responses_adapter = MagicMock()
    first._responses_adapter.complete_vision = AsyncMock(side_effect=complete_vision)
    await first.chat_with_vision(
        model=model,
        text_prompt="describe",
        images=["aW1hZ2U="],
        prefer_stream=True,
    )

    second = _client(base_url)
    second._responses_adapter = MagicMock()
    second._responses_adapter.complete_vision = AsyncMock(side_effect=complete_vision)
    await second.chat_with_vision(
        model=model,
        text_prompt="describe again",
        images=["aW1hZ2U="],
        prefer_stream=True,
    )
    await second.chat_with_vision(
        model="other-model",
        text_prompt="describe other model",
        images=["aW1hZ2U="],
        prefer_stream=True,
    )

    assert calls == [
        (model, True),
        (model, False),
        (model, False),
        ("other-model", True),
    ]
    assert (base_url, model, "vision") in llm_mod._NO_RESPONSES_VISION_STREAM
    assert (base_url, "other-model", "vision") not in llm_mod._NO_RESPONSES_VISION_STREAM


@pytest.mark.asyncio
async def test_vision_shape_400_reroutes_only_vision():
    c = _client("https://x/v1")
    shape_error = BadRequestError(
        message=(
            "Error code: 400 - {'error': {'message': "
            "'image_url should be a valid dictionary', "
            "'param': 'messages[0].content[1].image_url'}}"
        ),
        response=MagicMock(status_code=400),
        body={
            "error": {
                "message": "image_url should be a valid dictionary",
                "param": "messages[0].content[1].image_url",
            }
        },
    )
    responses = MagicMock()
    responses.complete_vision = AsyncMock(side_effect=shape_error)
    c._responses_adapter = responses

    inline = MagicMock()
    inline.choices = [MagicMock(message=MagicMock(content="CHAT_VISION", tool_calls=None))]
    inline.model = "vision-model"
    inline.usage = MagicMock(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_tokens_details=None,
    )
    c.client.chat.completions.create = AsyncMock(return_value=inline)

    result = await c.chat_with_vision(
        model="vision-model",
        text_prompt="describe",
        images=["aW1hZ2U="],
        max_tokens=10,
    )

    assert result.content == "CHAT_VISION"
    responses.complete_vision.assert_awaited_once()
    c.client.chat.completions.create.assert_awaited_once()
    assert llm_mod._PROBE_CACHE[("https://x/v1", "vision-model", "vision")] == "chat"
    assert llm_mod._PROBE_CACHE.get(("https://x/v1", "vision-model")) is None


def _sdk_flat_vision_bad_request():
    detail = {
        "message": " Invalid   input ",
        "type": "invalid_request_error",
        "param": "messages.1.content",
        "code": None,
    }
    request = httpx.Request("POST", "https://x/v1/responses")
    response = httpx.Response(
        400,
        request=request,
        json={"error": detail},
    )
    sdk = AsyncOpenAI(api_key="k", base_url="https://x/v1")
    return sdk, sdk._make_status_error(
        response.text,
        body=response.json(),
        response=response,
    )


@pytest.mark.asyncio
async def test_sdk_flat_bad_request_body_is_vision_context_only():
    sdk, error = _sdk_flat_vision_bad_request()
    try:
        assert isinstance(error, BadRequestError)
        assert error.body == {
            "message": " Invalid   input ",
            "type": "invalid_request_error",
            "param": "messages.1.content",
            "code": None,
        }
        assert llm_mod.is_vision_request_shape_error(error) is False
        assert llm_mod.is_vision_request_shape_error(
            error, vision_request=True
        ) is True
        assert llm_mod.is_transient_probe_error(error) is True
        assert llm_mod.is_transient_probe_error(
            error, vision_request=True
        ) is False
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_flat_generic_vision_400_redirects_once_without_transient_retry():
    base_url = "https://x/v1"
    model = "vision-model"
    c = _client(base_url)
    sdk, shape_error = _sdk_flat_vision_bad_request()
    try:
        responses = MagicMock()
        responses.complete_vision = AsyncMock(side_effect=shape_error)
        c._responses_adapter = responses

        inline = MagicMock()
        inline.choices = [
            MagicMock(message=MagicMock(content="CHAT_VISION", tool_calls=None))
        ]
        inline.model = model
        inline.usage = MagicMock(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            prompt_tokens_details=None,
        )
        c.client.chat.completions.create = AsyncMock(return_value=inline)

        result = await c.chat_with_vision(
            model=model,
            text_prompt="describe",
            images=["aW1hZ2U="],
            max_tokens=10,
        )

        assert result.content == "CHAT_VISION"
        responses.complete_vision.assert_awaited_once()
        c.client.chat.completions.create.assert_awaited_once()
        llm_mod.asyncio.sleep.assert_not_awaited()
        assert llm_mod._PROBE_CACHE[(base_url, model, "vision")] == "chat"
        assert llm_mod._PROBE_CACHE.get((base_url, model)) is None
    finally:
        await sdk.close()


def test_vision_data_uri_is_normalized_without_double_prefix():
    raw, mime = normalize_image_input("data:image/jpeg;base64,aW1hZ2U=")

    assert raw == "aW1hZ2U="
    assert not raw.startswith("data:")
    assert mime in {"image/jpeg", "image/png"}


@pytest.mark.asyncio
async def test_chat_vision_code_1214_reroutes_once_to_responses():
    c = _client("https://x/v1/chat/completions")
    shape_error = BadRequestError(
        message="Error code: 400 - invalid image url",
        response=MagicMock(status_code=400),
        body={"error": {
            "message": "Invalid image_url payload",
            "param": "messages[0].content[1].image_url",
            "code": 1214,
        }},
    )
    c.client.chat.completions.create = AsyncMock(side_effect=shape_error)
    responses = MagicMock()
    responses.complete_vision = AsyncMock(
        return_value=LLMResponse("RESPONSES_VISION", "m", {}, None)
    )
    c._responses_adapter = responses

    result = await c.chat_with_vision(
        model="vision-model",
        text_prompt="describe",
        images=["data:image/png;base64,aW1hZ2U="],
        max_tokens=10,
    )

    assert result.content == "RESPONSES_VISION"
    c.client.chat.completions.create.assert_awaited_once()
    responses.complete_vision.assert_awaited_once()
    assert responses.complete_vision.call_args.kwargs["images"] == ["aW1hZ2U="]


def test_unrelated_400_is_not_vision_shape_mismatch():
    error = BadRequestError(
        message="Error code: 400 - context length exceeded",
        response=MagicMock(status_code=400),
        body={"error": {"message": "context length exceeded", "param": "messages"}},
    )
    classifier = getattr(llm_mod, "is_responses_vision_shape_error", None)
    assert classifier is not None
    assert classifier(error) is False


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
async def test_responses_forced_tool_choice_retries_once_with_auto():
    c = _client("https://x/v1")
    error = BadRequestError(
        message="Thinking mode does not support this tool_choice",
        response=MagicMock(status_code=400),
        body={"error": {"message": "Thinking mode does not support this tool_choice"}},
    )
    responses = MagicMock()
    responses.complete_with_tools = AsyncMock(side_effect=[
        error,
        LLMToolResponse("ok", [], "thinking-model", {}, None),
    ])
    c._responses_adapter = responses
    inline_message = MagicMock(content="wrong protocol", tool_calls=None)
    inline_message.model_dump.return_value = {
        "content": "wrong protocol",
        "tool_calls": None,
    }
    inline_response = MagicMock(
        choices=[MagicMock(message=inline_message)],
        model="thinking-model",
        usage=None,
    )
    c.client.chat.completions.create = AsyncMock(return_value=inline_response)
    forced = {"type": "function", "function": {"name": "submit_result"}}

    result = await c.chat_with_tools(
        model="thinking-model",
        messages=[{"role": "user", "content": "submit"}],
        tools=[{"type": "function", "function": {"name": "submit_result"}}],
        tool_choice=forced,
    )

    assert result.content == "ok"
    assert responses.complete_with_tools.await_count == 2
    assert responses.complete_with_tools.call_args_list[0].kwargs["tool_choice"] == forced
    assert responses.complete_with_tools.call_args_list[1].kwargs["tool_choice"] == "auto"
    c.client.chat.completions.create.assert_not_awaited()
    assert llm_mod._PROBE_CACHE[("https://x/v1", "thinking-model")] == "responses"
    assert "thinking-model" in c._toolchoice_unsupported_models


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
    assert c.client.chat.completions.create.await_count == 3
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
    assert fake.complete.await_count == 3  # 3 probe retries (JSONDecodeError is now transient)
    c.client.chat.completions.create.assert_awaited_once()
    assert llm_mod.asyncio.sleep.await_count == 3  # 3 probe retries, each sleeps before retrying


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
async def test_tools_responses_stream_aggregates_one_complete_tool_call():
    completed = MagicMock()
    completed.model = "gpt-x"
    completed.model_dump.return_value = {
        "status": "completed",
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "execute_code",
            "arguments": '{"code":"print(1)"}',
        }],
        "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
    }

    async def events():
        yield SimpleNamespace(
            type="response.function_call_arguments.delta", delta='{"code":"print'
        )
        yield SimpleNamespace(
            type="response.function_call_arguments.delta", delta='(1)"}'
        )
        yield SimpleNamespace(type="response.completed", response=completed)

    client = _client("https://x/v1/responses")
    client.client.responses.create = AsyncMock(return_value=events())

    result = await client.chat_with_tools(
        model="gpt-x",
        messages=[{"role": "user", "content": "run code"}],
        tools=[{"type": "function", "function": {"name": "execute_code"}}],
        prefer_stream=True,
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "execute_code"
    assert result.tool_calls[0].arguments == {"code": "print(1)"}
    assert result.raw_assistant_message["tool_calls"][0]["id"] == "call_1"
    assert client.client.responses.create.await_count == 1
    assert client.client.responses.create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_tools_stream_unsupported_falls_back_once_and_is_cached():
    base_url = "https://x/v1"
    model = "gpt-x"
    stream_flags = []

    async def complete_with_tools(**kwargs):
        stream_flags.append(kwargs.get("stream", False))
        if kwargs.get("stream"):
            raise BadRequestError(
                message="stream is not supported",
                response=MagicMock(status_code=400),
                body={"error": {
                    "message": "stream is not supported",
                    "param": "stream",
                }},
            )
        return LLMToolResponse(
            content=None,
            tool_calls=[],
            model=model,
            usage={},
            raw_response=None,
        )

    first = _client(base_url)
    first._responses_adapter = MagicMock()
    first._responses_adapter.complete_with_tools = AsyncMock(
        side_effect=complete_with_tools
    )
    await first.chat_with_tools(
        model=model,
        messages=[{"role": "user", "content": "run"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        prefer_stream=True,
    )

    second = _client(base_url)
    second._responses_adapter = MagicMock()
    second._responses_adapter.complete_with_tools = AsyncMock(
        side_effect=complete_with_tools
    )
    await second.chat_with_tools(
        model=model,
        messages=[{"role": "user", "content": "run again"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        prefer_stream=True,
    )

    assert stream_flags == [True, False, False]
    assert (base_url, model, "tools", "responses") in llm_mod._NO_TEXT_STREAM


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

    with pytest.raises(
        ValueError,
        match=r"use chat_stream\(\) for streaming responses",
    ):
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
    assert llm_mod._PROBE_CACHE[(
        "https://x/v1/chat/completions", "responses-only"
    )] == "responses"


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
    assert llm_mod._PROBE_CACHE[(
        "https://x/v1/chat/completions", "responses-only", "vision"
    )] == "responses"
    assert llm_mod._PROBE_CACHE.get((
        "https://x/v1/chat/completions", "responses-only"
    )) is None
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
