import pytest
from unittest.mock import MagicMock
from backend.utils.llm_client import (
    retry_with_backoff,
    BadRequestError,
    UpstreamExhaustedError,
    MAX_RETRIES,
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
