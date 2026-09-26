import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.utils.llm_client import (
    retry_with_backoff,
    BadRequestError,
    InternalServerError,
    UpstreamExhaustedError,
)


def _err(cls, msg, status):
    return cls(message=msg, response=MagicMock(status_code=status), body=None)


@pytest.mark.asyncio
async def test_5xx_budget_does_not_stick_to_deterministic_failure(monkeypatch):
    calls = 0
    first = _err(
        InternalServerError,
        "temporary upstream failure",
        500,
    )
    later = _err(BadRequestError, "invalid request", 400)

    async def failing_func():
        nonlocal calls
        calls += 1
        raise first if calls == 1 else later

    monkeypatch.setattr("backend.utils.llm_client_parts.retry.asyncio.sleep", AsyncMock())
    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(
            failing_func,
            max_retries=1,
            retry_delay_base=0,
            agent_name="test",
        )

    assert calls == 2


@pytest.mark.asyncio
async def test_responses_input_shape_400_is_wrapped_for_hitl_without_retry():
    call_count = 0

    async def invalid_input_func():
        nonlocal call_count
        call_count += 1
        raise _err(
            BadRequestError,
            "Unknown parameter: 'input[19].tool_calls'",
            400,
        )

    with pytest.raises(UpstreamExhaustedError) as exc_info:
        await retry_with_backoff(
            lambda: invalid_input_func(),
            max_retries=4,
            retry_delay_base=0.01,
            agent_name="bioinformatics",
        )

    assert call_count == 1
    assert "input[19].tool_calls" in str(exc_info.value)


@pytest.mark.asyncio
async def test_before_retry_gate_blocks_until_resume_and_retries_once():
    pause = asyncio.Event()
    gate_entered = asyncio.Event()
    calls = 0

    async def before_retry():
        gate_entered.set()
        await pause.wait()

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError()
        return "ok"

    task = asyncio.create_task(retry_with_backoff(
        request,
        max_retries=1,
        retry_delay_base=0,
        agent_name="test",
        before_retry=before_retry,
    ))
    await gate_entered.wait()
    await asyncio.sleep(0)
    assert calls == 1

    pause.set()
    assert await task == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_before_retry_gate_cancellation_propagates_without_retry():
    pause = asyncio.Event()
    gate_entered = asyncio.Event()
    calls = 0

    async def before_retry():
        gate_entered.set()
        await pause.wait()

    async def request():
        nonlocal calls
        calls += 1
        raise asyncio.TimeoutError()

    task = asyncio.create_task(retry_with_backoff(
        request,
        max_retries=1,
        retry_delay_base=0,
        agent_name="test",
        before_retry=before_retry,
    ))
    await gate_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == 1
