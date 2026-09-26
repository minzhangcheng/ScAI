"""Provider/model output-budget negotiation at the Anthropic wire boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import APIStatusError

from backend.utils.llm_client_parts import anthropic_output_tokens as policy
from backend.utils.llm_client_parts.adapters import _AnthropicAdapter


@pytest.fixture(autouse=True)
def isolated_cache():
    policy._accepted.clear()
    policy._locks.clear()
    yield
    policy._accepted.clear()
    policy._locks.clear()


def _error(message, status=400):
    return APIStatusError(
        message,
        response=httpx.Response(status, request=httpx.Request("POST", "https://provider.test/v1/messages")),
        body={"error": {"message": message}},
    )


def _rejection(value, limit=32_768):
    return _error(f"max_tokens: {value} > {limit}, which is the maximum allowed number of output tokens")


def _adapter(create, provider="https://provider.test/v1/"):
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = SimpleNamespace(
        base_url=provider, messages=SimpleNamespace(create=create),
    )
    adapter.agent_name = "test"
    return adapter


def _reply():
    return SimpleNamespace(
        model="claude-test",
        model_dump=lambda: {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "usage": {},
        },
    )


async def _stream():
    yield SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="ok"),
    )
    yield SimpleNamespace(type="message_stop")


async def _call(adapter, kind="text", model="claude-test"):
    kwargs = dict(model=model, temperature=0.2, max_tokens=None)
    messages = [{"role": "user", "content": "hi"}]
    if kind in {"vision", "vision_stream"}:
        result = await adapter.complete_vision(
            **kwargs, text_prompt="hi", images=[], stream=kind == "vision_stream",
        )
    elif kind == "tools":
        result = await adapter.complete_with_tools(
            **kwargs, messages=messages, tools=[], tool_choice="auto",
        )
    elif kind == "stream":
        return "".join([chunk async for chunk in adapter.stream(**kwargs, messages=messages)])
    else:
        result = await adapter.complete(**kwargs, messages=messages, stream=kind == "text_stream")
    return result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "tools", "vision", "stream", "text_stream", "vision_stream"])
async def test_all_anthropic_paths_negotiate_and_reuse_across_clients(kind):
    calls = []

    async def create(**params):
        value = params["max_tokens"]
        calls.append(value)
        if value > 32_768:
            raise _rejection(value)
        return _stream() if params.get("stream") else _reply()

    assert await _call(_adapter(create), kind) == "ok"
    assert await _call(_adapter(create, "https://provider.test/v1"), kind) == "ok"
    assert calls == [65_536, 32_768, 32_768]


@pytest.mark.asyncio
async def test_concurrent_clients_only_negotiate_once():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def create(**params):
        value = params["max_tokens"]
        calls.append(value)
        if value == 65_536:
            entered.set()
            await release.wait()
            raise _rejection(value)
        return _reply()

    first = asyncio.create_task(_call(_adapter(create)))
    await entered.wait()
    second = asyncio.create_task(_call(_adapter(create), "tools"))
    await asyncio.sleep(0)
    release.set()
    assert await asyncio.gather(first, second) == ["ok", "ok"]
    assert calls == [65_536, 32_768, 32_768]


@pytest.mark.asyncio
@pytest.mark.parametrize("message,status", [
    ("max_tokens must be less than or equal to 32768", 429),
    ("max_tokens exceeds the maximum", 500),
    ("Invalid API key", 401),
    ("Input tokens + max_tokens exceed the context window", 400),
    ("max_tokens must be greater than budget_tokens for thinking", 400),
    ("max_tokens must be greater than 0", 400),
    ("Unsupported parameter temperature", 400),
])
async def test_unrelated_errors_do_not_shrink_or_cache(message, status):
    error = _error(message, status)
    create = AsyncMock(side_effect=error)
    with pytest.raises(APIStatusError) as caught:
        await _call(_adapter(create))
    assert caught.value is error
    assert create.await_count == 1
    assert policy._accepted == {}


@pytest.mark.asyncio
async def test_cancelled_probe_does_not_cache_and_releases_lock():
    create = AsyncMock(side_effect=[_rejection(65_536), asyncio.CancelledError(), _reply()])
    with pytest.raises(asyncio.CancelledError):
        await _call(_adapter(create))
    assert policy._accepted == {}
    assert await _call(_adapter(create)) == "ok"
    assert [c.kwargs["max_tokens"] for c in create.await_args_list] == [65_536, 32_768, 65_536]
