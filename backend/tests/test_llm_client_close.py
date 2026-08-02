"""Tests: OpenAICompatibleClient.close() idempotency and lifecycle."""
import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_client_close_idempotent():
    from backend.utils.llm_client import OpenAICompatibleClient
    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    assert hasattr(c, "_closed")
    assert c._closed is False
    await c.close()
    assert c._closed is True


@pytest.mark.asyncio
async def test_client_close_failure_is_retryable():
    from backend.utils.llm_client import OpenAICompatibleClient

    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    c.client.close = AsyncMock(
        side_effect=[RuntimeError("close failed"), None]
    )
    c.http_client.aclose = AsyncMock()

    with pytest.raises(RuntimeError, match="close failed"):
        await c.close()
    assert c._closed is False

    await c.close()
    assert c._closed is True
    assert c.client.close.await_count == 2
    await c.close()
    assert c._closed is True


@pytest.mark.asyncio
async def test_client_close_waits_for_active_chat_and_rejects_new_calls():
    from backend.utils.llm_client import LLMResponse, OpenAICompatibleClient

    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_dispatch(**kwargs):
        entered.set()
        await release.wait()
        return LLMResponse("ok", "model", {}, None)

    c._dispatch_non_streaming = AsyncMock(side_effect=blocked_dispatch)
    c.client.close = AsyncMock()
    c.http_client.aclose = AsyncMock()

    request_task = asyncio.create_task(
        c.chat(model="model", messages=[{"role": "user", "content": "hi"}])
    )
    await entered.wait()
    close_task = asyncio.create_task(c.close())
    await asyncio.sleep(0)

    assert c._active_requests == 1
    assert c._closing is True
    assert close_task.done() is False
    c.client.close.assert_not_awaited()
    with pytest.raises(RuntimeError, match="client is closing"):
        await c.chat(
            model="model",
            messages=[{"role": "user", "content": "late"}],
        )

    release.set()
    assert (await request_task).content == "ok"
    await close_task

    assert c._active_requests == 0
    assert c._closed is True
    c.client.close.assert_awaited_once()
    c.http_client.aclose.assert_awaited_once()


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        (
            "chat_with_tools",
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}],
            },
        ),
        (
            "chat_with_vision",
            {
                "model": "model",
                "text_prompt": "inspect",
                "images": ["aGVsbG8="],
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_client_close_waits_for_other_guarded_entrypoints(
    method_name,
    kwargs,
):
    from backend.utils.llm_client import OpenAICompatibleClient

    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    entered = asyncio.Event()
    release = asyncio.Event()
    result = MagicMock()

    async def blocked_dispatch(**dispatch_kwargs):
        entered.set()
        await release.wait()
        return result

    c._dispatch_non_streaming = AsyncMock(side_effect=blocked_dispatch)
    c.client.close = AsyncMock()
    c.http_client.aclose = AsyncMock()

    request_task = asyncio.create_task(getattr(c, method_name)(**kwargs))
    await entered.wait()
    close_task = asyncio.create_task(c.close())
    await asyncio.sleep(0)

    assert c._active_requests == 1
    assert close_task.done() is False

    release.set()
    assert await request_task is result
    await close_task
    assert c._active_requests == 0
    assert c._closed is True


@pytest.mark.asyncio
async def test_cancelled_public_request_releases_active_count():
    from backend.utils.llm_client import OpenAICompatibleClient

    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    entered = asyncio.Event()

    async def blocked_dispatch(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    c._dispatch_non_streaming = AsyncMock(side_effect=blocked_dispatch)
    request_task = asyncio.create_task(
        c.chat(model="model", messages=[{"role": "user", "content": "hi"}])
    )
    await entered.wait()
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert c._active_requests == 0
    assert c._requests_idle.is_set()
    await c.close()


@pytest.mark.asyncio
async def test_client_close_waits_until_stream_generator_is_closed():
    from backend.utils.llm_client import OpenAICompatibleClient

    c = OpenAICompatibleClient(
        api_key="test-key",
        base_url="http://x/v1/chat/completions",
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_stream(*args, **kwargs):
        entered.set()
        await release.wait()
        yield "chunk"

    c._chat_stream_inline = blocked_stream
    c.client.close = AsyncMock()
    c.http_client.aclose = AsyncMock()
    stream = c.chat_stream(
        model="model",
        messages=[{"role": "user", "content": "hi"}],
    )
    first_chunk = asyncio.create_task(stream.__anext__())
    await entered.wait()
    close_task = asyncio.create_task(c.close())
    await asyncio.sleep(0)

    assert c._active_requests == 1
    assert close_task.done() is False

    release.set()
    assert await first_chunk == "chunk"
    assert close_task.done() is False
    await stream.aclose()
    await close_task

    assert c._active_requests == 0
    assert c._closed is True


@pytest.mark.asyncio
async def test_stream_producer_error_releases_active_count():
    from backend.utils.llm_client import OpenAICompatibleClient

    c = OpenAICompatibleClient(
        api_key="test-key",
        base_url="http://x/v1/chat/completions",
    )

    async def failing_stream(*args, **kwargs):
        yield "first"
        raise RuntimeError("stream failed")

    c._chat_stream_inline = failing_stream
    stream = c.chat_stream(
        model="model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert await stream.__anext__() == "first"
    with pytest.raises(RuntimeError, match="stream failed"):
        await stream.__anext__()

    assert c._active_requests == 0
    assert c._requests_idle.is_set()
    await c.close()


@pytest.mark.asyncio
async def test_client_close_idempotent_google_deep_research():
    from backend.utils.llm_client import GoogleDeepResearchClient
    c = GoogleDeepResearchClient(api_key="test-key")
    assert hasattr(c, "_closed")
    assert c._closed is False
    await c.close()
    assert c._closed is True
    await c.close()
    assert c._closed is True


@pytest.mark.asyncio
async def test_google_deep_research_close_failure_is_retryable():
    from backend.utils.llm_client import GoogleDeepResearchClient

    c = GoogleDeepResearchClient(api_key="test-key")
    c.client.aclose = AsyncMock(
        side_effect=[RuntimeError("close failed"), None]
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await c.close()
    assert c._closed is False

    await c.close()
    assert c._closed is True
    assert c.client.aclose.await_count == 2


@pytest.mark.asyncio
async def test_google_deep_research_close_cancellation_is_retryable():
    from backend.utils.llm_client import GoogleDeepResearchClient

    c = GoogleDeepResearchClient(api_key="test-key")
    c.client.aclose = AsyncMock(
        side_effect=[asyncio.CancelledError(), None]
    )

    with pytest.raises(asyncio.CancelledError):
        await c.close()
    assert c._closed is False

    await c.close()
    assert c._closed is True
    assert c.client.aclose.await_count == 2


@pytest.mark.asyncio
async def test_google_deep_research_concurrent_close_waits_for_owner():
    from backend.utils.llm_client import GoogleDeepResearchClient

    c = GoogleDeepResearchClient(api_key="test-key")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_close():
        entered.set()
        await release.wait()

    c.client.aclose = AsyncMock(side_effect=blocked_close)
    first = asyncio.create_task(c.close())
    await entered.wait()
    second = asyncio.create_task(c.close())
    await asyncio.sleep(0)

    assert second.done() is False
    release.set()
    await asyncio.gather(first, second)
    assert c._closed is True
    c.client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_manager_close_idempotent():
    from backend.utils.llm_client import LLMManager
    m = LLMManager(api_key="test-key", base_url="http://x")
    assert m.llm._closed is False
    await m.close()
    assert m.llm._closed is True
    await m.close()
    assert m.llm._closed is True


def test_client_del_warns_if_not_closed():
    import backend.utils.llm_client as llm_mod
    from backend.utils.llm_client import OpenAICompatibleClient
    with patch.object(llm_mod.logger, "warning") as mock_warning:
        c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
        c.__del__()
        mock_warning.assert_called_once()
        args, _ = mock_warning.call_args
        assert "leaked" in args[0]
    asyncio.run(c.close())


def test_client_del_no_warning_if_closed():
    import backend.utils.llm_client as llm_mod
    from backend.utils.llm_client import OpenAICompatibleClient
    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    asyncio.run(c.close())
    with patch.object(llm_mod.logger, "warning") as mock_warning:
        c.__del__()
        mock_warning.assert_not_called()
