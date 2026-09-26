import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import backend.utils.llm_client as llm_mod
from backend.utils.llm_client import OpenAICompatibleClient, LLMResponse, LLMToolResponse, NotFoundError, BadRequestError, InternalServerError
from backend.utils.llm_client_parts.media import normalize_image_input
from backend.utils.llm_client_parts.prompt_cache import (
    PROMPT_CACHE_EPHEMERAL_TAIL_KEY,
    PromptCachePlan,
    responses_prefix_cache_plan,
)


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


async def _events(*items):
    for item in items:
        yield item


def _chat_chunk(content=None, *, finish_reason=None, **delta):
    return SimpleNamespace(
        model="chat-model", usage=None,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, **delta), finish_reason=finish_reason,
        )],
    )


def _chat_completion(text, model):
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=text, tool_calls=None))], model=model,
        usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                        prompt_tokens_details=None),
    )


def _bad_request(message, *, param=None):
    error = {"message": message}
    if param is not None:
        error["param"] = param
    return BadRequestError(
        message=message, response=MagicMock(status_code=400), body={"error": error},
    )


class _PartialResponseStream:
    """Yield once, then fail or block until the request is cancelled."""

    def __init__(self, error=None):
        self.first = True
        self.error = error
        self.entered_remainder = asyncio.Event()
        self.close = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.first:
            self.first = False
            return SimpleNamespace(type="response.output_text.delta", delta="partial")
        self.entered_remainder.set()
        if self.error is not None:
            raise self.error
        await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_pi_prefix_reaches_responses_wire_without_caching_dynamic_tail(vision, stream):
    model = "openai/gpt-5.6-sol"
    client = _client("https://openrouter.ai/api/v1/responses")
    prefix = "Review rules and project background.\n" * 160
    plan = responses_prefix_cache_plan(client.base_url, model, prefix)
    completed = MagicMock()
    completed.model = model
    completed.model_dump.return_value = {
        "status": "completed",
        "output": [{"type": "message", "content": [{
            "type": "output_text", "text": "Reviewed.",
        }]}],
        "usage": {"input_tokens": 1800, "output_tokens": 3, "total_tokens": 1803},
    }

    async def events():
        yield SimpleNamespace(type="response.completed", response=completed)

    create = AsyncMock(side_effect=lambda **_kwargs: events() if stream else completed)
    client.client.responses.create = create
    for suffix, image in [("First question", "aW1hZ2U="), ("Revised feedback", "bmV3LWltYWdl")]:
        kwargs = dict(model=model, prefer_stream=stream, prompt_cache_plan=plan)
        if vision:
            result = await client.chat_with_vision(
                **kwargs, text_prompt=prefix + suffix, images=[image],
                system_prompt="PI system rules",
            )
        else:
            result = await client.chat(**kwargs, messages=[
                {"role": "system", "content": "PI system rules"},
                {"role": "user", "content": prefix + suffix},
            ])
        assert result.content == "Reviewed."
        wire = create.await_args.kwargs
        blocks = wire["input"][-1]["content"]
        assert blocks[0] == {
            "type": "input_text", "text": prefix,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
        assert blocks[1] == {"type": "input_text", "text": suffix}
        assert "PI system rules" in wire["instructions"]
        assert "exclusively in English" in wire["instructions"]
        assert wire["extra_body"]["prompt_cache_options"] == {"mode": "explicit"}
        assert "session_id" in wire["extra_body"]
        assert bool(wire.get("stream")) is stream
        if vision:
            assert blocks[2] == {
                "type": "input_image", "image_url": f"data:image/png;base64,{image}",
            }
        else:
            assert len(blocks) == 2
    assert create.await_count == 2
    assert client._prompt_cache_controller.describe(
        model, "responses", "vision" if vision else "text", "explicit",
    )["capability"] == "supported"


@pytest.mark.asyncio
async def test_cache_plan_respects_explicit_responses_endpoint_and_disables_cache():
    c = _client("https://openrouter.ai/api/v1/responses")
    responses = MagicMock()
    responses.complete_with_tools = AsyncMock(return_value=LLMToolResponse(
        content="done",
        tool_calls=[],
        model="anthropic/claude-sonnet-5",
        usage={},
        raw_response=None,
    ))
    anthropic = MagicMock()
    anthropic.complete_with_tools = AsyncMock()
    c._responses_adapter = responses
    c._anthropic_adapter = anthropic

    await c.chat_with_tools(
        model="anthropic/claude-sonnet-5",
        messages=[{
            "role": "user",
            "content": "ephemeral",
            PROMPT_CACHE_EPHEMERAL_TAIL_KEY: True,
        }],
        tools=[{"type": "function", "function": {"name": "f"}}],
        prompt_cache_plan=PromptCachePlan(),
    )

    assert responses.complete_with_tools.await_args.kwargs["cache_mode"] == "disabled"
    anthropic.complete_with_tools.assert_not_awaited()


def test_explicit_prompt_cache_support_is_scoped_to_compatible_routes():
    generic = _client("https://openrouter.ai/api/v1")
    explicit_responses = _client(
        "https://openrouter.ai/api/v1/responses"
    )
    explicit_chat = _client(
        "https://openrouter.ai/api/v1/chat/completions"
    )
    direct_anthropic = _client("https://api.anthropic.com")
    custom_without_v1 = _client("https://api.deepseek.com")

    assert generic.prompt_cache_plan_for_history(
        "anthropic/claude-sonnet-5"
    ) == PromptCachePlan()
    assert generic.prompt_cache_plan_for_history(
        "google/gemini-3.8-flash"
    ) is None
    disabled = PromptCachePlan(
        mode="disabled", prefer_anthropic_messages=False
    )
    assert explicit_responses.prompt_cache_plan_for_history(
        "anthropic/claude-sonnet-5"
    ) == disabled
    assert explicit_chat.prompt_cache_plan_for_history(
        "anthropic/claude-sonnet-5"
    ) == disabled
    assert direct_anthropic.prompt_cache_plan_for_history(
        "claude-sonnet-5"
    ) == PromptCachePlan()
    assert custom_without_v1.prompt_cache_plan_for_history(
        "deepseek-chat"
    ) is None
    llm_mod._PROBE_CACHE[(
        "https://api.deepseek.com", "deepseek-chat"
    )] = "anthropic"
    assert custom_without_v1.prompt_cache_plan_for_history(
        "deepseek-chat"
    ) == PromptCachePlan()
    assert explicit_responses.supports_explicit_prompt_cache(
        "anthropic/claude-sonnet-5"
    ) is False


@pytest.mark.asyncio
async def test_plain_chat_completions_prefers_stream_and_aggregates_text():
    c = _client("https://x/v1/chat/completions")

    c.client.chat.completions.create = AsyncMock(return_value=_events(
        _chat_chunk("Hel"),
        _chat_chunk("lo", finish_reason="stop"),
        SimpleNamespace(model="chat-model", choices=[], usage=SimpleNamespace(
            prompt_tokens=3, completion_tokens=2, total_tokens=5, prompt_tokens_details=None,
        )),
    ))
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
async def test_plain_chat_midstream_eof_is_not_replayed_nonstreaming():
    client = _client("https://x/v1/chat/completions")
    client.client.chat.completions.create = AsyncMock(return_value=_events(_chat_chunk("partial")))

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

    client = _client(base_url)
    client.client.chat.completions.create = AsyncMock(return_value=_events(
        _chat_chunk(refusal="I cannot help with that", finish_reason="stop"),
    ))

    result = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    )

    assert result.content == "[Refusal]: I cannot help with that"
    client.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_protocol_vision_stream_unsupported_is_cached_and_falls_back():
    base_url = "https://x/v1"
    model = "vision-model"
    llm_mod._PROBE_CACHE[(base_url, model, "vision")] = "chat"
    stream_flags = []

    fallback = _chat_completion("non-stream", model)

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
@pytest.mark.parametrize("failure", ["server_error", "idle_timeout"])
async def test_responses_midstream_failure_is_closed_and_not_replayed(monkeypatch, failure):
    error = InternalServerError(
        message="connection lost", response=MagicMock(status_code=500),
        body={"error": {"message": "connection lost"}},
    ) if failure == "server_error" else None
    stream = _PartialResponseStream(error)
    # The server-error case leaves protocol uncached: probing must not replay
    # a stream that emitted an event. The idle case exercises an explicit route.
    base_url = "https://x/v1" if error else "https://x/v1/responses"
    client = _client(base_url)
    client.client.responses.create = AsyncMock(return_value=stream)
    client.client.chat.completions.create = AsyncMock()
    kwargs = {}
    if failure == "idle_timeout":
        monkeypatch.setattr(
            "backend.utils.llm_client_parts.retry.MIN_LLM_TIMEOUT_SECONDS", 0.001,
        )
        kwargs["timeout"] = 0.01
    match = "after the first event" if error else "TimeoutError"
    with pytest.raises(llm_mod.UpstreamExhaustedError, match=match) as exc_info:
        await client.chat(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}],
            prefer_stream=True, **kwargs,
        )
    # Keep the original failure reachable and the wire shape diagnosable.
    assert exc_info.value.failure_kind == "stream_interrupted"
    assert isinstance(exc_info.value.__cause__, llm_mod.StreamInterruptedError)
    client.client.responses.create.assert_awaited_once()
    client.client.chat.completions.create.assert_not_awaited()
    stream.close.assert_awaited_once()





@pytest.mark.asyncio
async def test_cancelled_responses_stream_closes_and_releases_request_lease():
    stream = _PartialResponseStream()
    client = _client("https://x/v1/responses")
    client.client.responses.create = AsyncMock(return_value=stream)
    task = asyncio.create_task(client.chat(
        model="gpt-x", messages=[{"role": "user", "content": "hi"}],
        prefer_stream=True,
    ))
    await asyncio.wait_for(stream.entered_remainder.wait(), timeout=1)
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
    client = _client("https://x/v1/chat/completions")
    client.client.chat.completions.create = AsyncMock(side_effect=[
        _events(_chat_chunk(reasoning_content="private chain of thought", finish_reason="stop")),
        _events(_chat_chunk("Public final answer", finish_reason="stop")),
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


def test_stream_unsupported_classifier_rejects_temperature_without_param_field():
    error = _bad_request("Unsupported parameter: temperature is not supported with streaming")

    assert llm_mod.is_stream_unsupported_error(error) is False

    direct_stream_error = _bad_request("Unsupported parameter: stream")
    assert llm_mod.is_stream_unsupported_error(direct_stream_error) is True


@pytest.mark.asyncio
async def test_chat_stream_options_rejection_retries_stream_without_usage_option_and_caches():
    base_url = "https://x/v1/chat/completions"
    model = "chat-model"
    error = _bad_request("Unsupported parameter: stream_options", param="stream_options")

    client = _client(base_url)
    client.client.chat.completions.create = AsyncMock(side_effect=[
        error,
        _events(_chat_chunk("first", finish_reason="stop")),
        _events(_chat_chunk("second", finish_reason="stop")),
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
async def test_responses_not_implemented_500_falls_back_to_chat_without_retry():
    c = _client("https://api.n1n.ai/v1")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=InternalServerError(
        message="Error code: 500 - {'error': {'message': 'not implemented (request id: x)', 'type': 'new_api_error', 'param': '', 'code': 'local:convert_request_failed'}}",
        response=MagicMock(status_code=500),
        body={"error": {"message": "not implemented (request id: x)", "code": "local:convert_request_failed"}},
    ))
    c._responses_adapter = fake

    c.client.chat.completions.create = AsyncMock(
        return_value=_chat_completion("CHAT", "gemini-3.1-pro-preview"),
    )

    r = await c.chat(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT"
    fake.complete.assert_awaited_once()
    c.client.chat.completions.create.assert_awaited_once()
    llm_mod.asyncio.sleep.assert_not_awaited()
    assert "https://api.n1n.ai/v1" in llm_mod._NO_RESPONSES_ENDPOINT
    assert llm_mod._PROBE_CACHE[("https://api.n1n.ai/v1", "gemini-3.1-pro-preview")] == "chat"


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

    c.client.chat.completions.create = AsyncMock(
        return_value=_chat_completion("CHAT_VISION", "vision-model"),
    )

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


def test_vision_data_uri_is_normalized_without_double_prefix():
    raw, mime = normalize_image_input("data:image/jpeg;base64,aW1hZ2U=")

    assert raw == "aW1hZ2U="
    assert not raw.startswith("data:")
    assert mime in {"image/jpeg", "image/png"}


@pytest.mark.asyncio
async def test_responses_forced_tool_choice_retries_once_with_auto():
    c = _client("https://x/v1")
    error = _bad_request("Thinking mode does not support this tool_choice")
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
            raise _bad_request("stream is not supported", param="stream")
        return LLMToolResponse(
            content=None,
            tool_calls=[],
            model=model,
            usage={},
            raw_response=None,
        )

    for prompt in ("run", "run again"):
        client = _client(base_url)
        client._responses_adapter = MagicMock()
        client._responses_adapter.complete_with_tools = AsyncMock(side_effect=complete_with_tools)
        await client.chat_with_tools(
            model=model, messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            prefer_stream=True,
        )

    assert stream_flags == [True, False, False]
    assert (base_url, model, "tools", "responses") in llm_mod._NO_TEXT_STREAM




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
