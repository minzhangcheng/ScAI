"""L2: the 429 retry budget must be independent of other error types.
A preceding 5xx must NOT shift the first 429 off RATE_LIMIT_DELAYS[0] nor
consume the rate-limit budget."""
import httpx
import pytest
from openai import APITimeoutError, RateLimitError, InternalServerError

from backend.utils import llm_client
from backend.utils.llm_client import (
    MAX_RETRIES,
    retry_with_backoff,
    RATE_LIMIT_DELAYS,
    RETRY_DELAY_BASE,
)


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "http://x"))


@pytest.fixture
def captured_sleeps(monkeypatch):
    """Replace asyncio.sleep with a no-op that records the requested delays."""
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)
    return delays


# --- pure function ---

# --- behavioral: mixed error sequence ---

@pytest.mark.asyncio
async def test_first_429_uses_index_zero_after_5xx(captured_sleeps):
    seq = {"i": 0}
    gate_calls = []

    async def before_retry():
        gate_calls.append(seq["i"])

    async def func():
        i = seq["i"]
        seq["i"] += 1
        if i == 0:
            raise InternalServerError("boom", response=_resp(500), body=None)
        if i == 1:
            raise RateLimitError("Rate limit reached, slow down", response=_resp(429), body=None)
        return "ok"

    result = await retry_with_backoff(
        func,
        max_retries=4,
        agent_name="t",
        before_retry=before_retry,
    )
    assert result == "ok"
    # first delay from the 5xx backoff, second delay must be RATE_LIMIT_DELAYS[0]
    assert captured_sleeps == [RETRY_DELAY_BASE * 1, float(RATE_LIMIT_DELAYS[0])]
    assert gate_calls == [1, 2]


@pytest.mark.asyncio
async def test_long_timeout_budget_does_not_replay_timeout(captured_sleeps):
    attempts = 0

    async def func():
        nonlocal attempts
        attempts += 1
        raise APITimeoutError(request=httpx.Request("POST", "http://x"))

    from backend.utils.llm_client import UpstreamExhaustedError

    with pytest.raises(UpstreamExhaustedError, match="after 1 attempts"):
        await retry_with_backoff(
            func,
            max_retries=0,
            connection_retries=MAX_RETRIES,
            agent_name="report-generator",
        )

    assert attempts == 1
    assert captured_sleeps == []
