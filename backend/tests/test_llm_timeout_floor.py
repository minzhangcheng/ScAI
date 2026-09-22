"""Transport and business deadlines must not silently shorten model requests."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.utils.llm_client_parts import retry
from backend.utils.llm_client_parts.adapters import _ResponsesAdapter
from backend.utils.llm_client_parts.client_params import ClientParamsMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("requested, effective", [(90.0, 600.0), (900.0, 900.0)])
async def test_retry_driver_gives_each_attempt_the_full_timeout(monkeypatch, requested, effective):
    seen = []

    async def wait(awaitable, timeout):
        seen.append(timeout)
        return await awaitable

    monkeypatch.setattr(retry.asyncio, "wait_for", wait)
    request = AsyncMock(return_value="done")
    assert await retry.retry_with_backoff(request, request_timeout=requested) == "done"
    assert seen == [effective]


@pytest.mark.asyncio
@pytest.mark.parametrize("requested, effective", [(90.0, 600.0), (1800.0, 1800.0)])
async def test_chat_http_and_temperature_fallback_keep_the_timeout_floor(requested, effective):
    create = AsyncMock(return_value="done")
    client = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        _apply_model_params=lambda *args, **kwargs: None,
    )
    calls = ClientParamsMixin._build_request_pair(
        client, {"model": "test"}, "test", 0.2, None, request_timeout=requested,
    )
    for call in calls:
        assert await call() == "done"
    assert [call.kwargs["timeout"] for call in create.await_args_list] == [effective, effective]


@pytest.mark.parametrize("requested, effective", [(90.0, 600.0), (1800.0, 1800.0)])
def test_responses_http_keeps_the_timeout_floor(requested, effective):
    adapter = _ResponsesAdapter(SimpleNamespace())
    params = adapter._build_params(
        "test", [{"role": "user", "content": "test"}], None, None, timeout=requested,
    )
    assert params["timeout"] == effective


@pytest.mark.asyncio
async def test_painter_leftover_budget_does_not_shorten_a_started_generation(monkeypatch):
    from backend.agents.painter_agent import PainterAgent

    seen = []

    async def wait(awaitable, timeout):
        seen.append(timeout)
        return await awaitable

    monkeypatch.setattr(retry.asyncio, "wait_for", wait)
    result = await PainterAgent._run_protocol_cascade(
        SimpleNamespace(session_id="timeout-test"),
        [{"name": "images", "run": AsyncMock(return_value="image"), "timeout": 180.0}],
        total_timeout=90.0,
    )
    assert result == "image"
    assert seen == [600.0]
