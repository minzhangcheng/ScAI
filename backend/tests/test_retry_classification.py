import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.utils.llm_client import (
    retry_with_backoff,
    BadRequestError,
    InternalServerError,
    UpstreamExhaustedError,
    MAX_RETRIES,
    MAX_RETRIES_UPSTREAM_5XX,
)


def _err(cls, msg, status):
    return cls(message=msg, response=MagicMock(status_code=status), body=None)


@pytest.mark.asyncio
async def test_service_unavailable_400_is_retried():
    call_count = 0

    async def failing_func():
        nonlocal call_count
        call_count += 1
        raise _err(BadRequestError, "service unavailable", 400)

    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(
            lambda: failing_func(),
            max_retries=2,
            retry_delay_base=0.01,
            agent_name="test",
        )

    # Called max_retries + 1 times (3 total) before exhaustion
    assert call_count == 3


@pytest.mark.asyncio
async def test_temperature_400_not_retried_propagates():
    call_count = 0

    async def temp_err_func():
        nonlocal call_count
        call_count += 1
        raise _err(BadRequestError, "Temperature parameter is not supported", 400)

    with pytest.raises(BadRequestError):
        await retry_with_backoff(
            lambda: temp_err_func(),
            max_retries=2,
            retry_delay_base=0.01,
            agent_name="test",
        )

    # Called only once — temperature signal is not retried
    assert call_count == 1


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
async def test_5xx_retry_notice_uses_expanded_budget(monkeypatch):
    notices = []
    calls = 0

    async def record(*args):
        notices.append(args)

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _err(InternalServerError, "temporary upstream failure", 524)
        return "ok"

    monkeypatch.setattr("backend.utils.llm_client_parts.retry.asyncio.sleep", AsyncMock())

    assert await retry_with_backoff(
        request,
        max_retries=1,
        retry_delay_base=0,
        agent_name="bioinformatics",
        on_retry=record,
    ) == "ok"

    assert notices == [
        (0, MAX_RETRIES_UPSTREAM_5XX, 0, "InternalServerError(5xx)", "bioinformatics")
    ]

@pytest.mark.asyncio
async def test_tool_choice_400_not_retried_propagates():
    call_count = 0

    async def tool_choice_err_func():
        nonlocal call_count
        call_count += 1
        raise _err(BadRequestError, "tool_choice parameter not supported", 400)

    with pytest.raises(BadRequestError):
        await retry_with_backoff(
            lambda: tool_choice_err_func(),
            max_retries=2,
            retry_delay_base=0.01,
            agent_name="test",
        )

    assert call_count == 1


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
async def test_on_retry_receives_the_agent_name_from_the_driver():
    """The retry notice's agent label is only correct if the driver forwards it.

    workflow/core.py maps agent_name onto an AgentType to label the WebSocket
    notice. That mapping is exercised by its own test, but the two modules are
    coupled by positional arity across `_notify_retry`'s unconditional
    `except Exception: pass` — so dropping the argument here degrades silently:
    a 4-arg call leaves agent_name at its default and every notice ships
    unlabelled, and a 5-arg call into a 4-arg callback raises a TypeError that
    is swallowed, emitting no notice at all during a multi-minute stall.
    Neither shows up in a log line, so it has to be asserted here.
    """
    calls = []

    async def record(*args):
        calls.append(args)

    async def always_timeout():
        raise asyncio.TimeoutError()

    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(
            lambda: always_timeout(),
            max_retries=1,
            retry_delay_base=0.01,
            agent_name="pathology",
            on_retry=record,
        )

    assert len(calls) == 1
    assert calls[0] == (0, 1, 0.01, "TimeoutError", "pathology")


@pytest.mark.asyncio
async def test_on_retry_is_called_with_the_arity_core_declares():
    """Bind a callback with no defaults so a shape drift raises instead of hiding."""
    seen = {}

    async def strict(attempt, max_retries, delay, error_type, agent_name):
        seen.update(
            attempt=attempt,
            max_retries=max_retries,
            delay=delay,
            error_type=error_type,
            agent_name=agent_name,
        )

    async def always_timeout():
        raise asyncio.TimeoutError()

    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(
            lambda: always_timeout(),
            max_retries=1,
            retry_delay_base=0.01,
            agent_name="biochemistry",
            on_retry=strict,
        )

    assert seen == {
        "attempt": 0,
        "max_retries": 1,
        "delay": 0.01,
        "error_type": "TimeoutError",
        "agent_name": "biochemistry",
    }


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


@pytest.mark.asyncio
async def test_set_before_retry_gate_preserves_notification_and_backoff_order(monkeypatch):
    import backend.utils.llm_client_parts.retry as retry_module

    real_sleep = asyncio.sleep
    sleeps = []
    events = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)

    calls = 0

    async def on_retry(*_args):
        events.append("notify")

    async def before_retry():
        events.append("gate")

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError()
        return "ok"

    assert await retry_with_backoff(
        request,
        max_retries=1,
        retry_delay_base=0.25,
        agent_name="test",
        on_retry=on_retry,
        before_retry=before_retry,
    ) == "ok"
    assert events == ["notify", "gate"]
    assert sleeps == [0.25]
    assert calls == 2
