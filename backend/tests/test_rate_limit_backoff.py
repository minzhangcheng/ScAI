"""L2: the 429 retry budget must be independent of other error types.
A preceding 5xx must NOT shift the first 429 off RATE_LIMIT_DELAYS[0] nor
consume the rate-limit budget."""
import httpx
import pytest
from openai import RateLimitError, InternalServerError

from backend.utils import llm_client
from backend.utils.llm_client import (
    DEFAULT_REQUEST_TIMEOUT,
    OpenAICompatibleClient,
    retry_with_backoff,
    compute_rate_limit_delay,
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

def test_compute_delay_indexes_table():
    assert compute_rate_limit_delay(0, 4) == float(RATE_LIMIT_DELAYS[0])
    assert compute_rate_limit_delay(1, 4) == float(RATE_LIMIT_DELAYS[1])
    assert compute_rate_limit_delay(3, 4) == float(RATE_LIMIT_DELAYS[3])


def test_compute_delay_budget_exhausted_returns_none():
    assert compute_rate_limit_delay(len(RATE_LIMIT_DELAYS), 4) is None


def test_compute_delay_budget_capped_by_max_retries():
    # budget == min(max_retries, len(RATE_LIMIT_DELAYS)) == 2
    assert compute_rate_limit_delay(0, 2) == float(RATE_LIMIT_DELAYS[0])
    assert compute_rate_limit_delay(2, 2) is None


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
async def test_pure_429_stream_walks_table(captured_sleeps):
    async def func():
        raise RateLimitError("Rate limit reached, slow down", response=_resp(429), body=None)

    from backend.utils.llm_client import UpstreamExhaustedError
    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(func, max_retries=4, agent_name="t")
    # budget == min(4, 4) == 4 → walks the full table in order
    assert captured_sleeps == [float(d) for d in RATE_LIMIT_DELAYS]


@pytest.mark.asyncio
async def test_429_budget_survives_zero_ordinary_retries(captured_sleeps):
    attempts = 0

    async def func():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError(
                "Rate limit reached, slow down",
                response=_resp(429),
                body=None,
            )
        return "ok"

    result = await retry_with_backoff(
        func,
        max_retries=0,
        rate_limit_retries=len(RATE_LIMIT_DELAYS),
        agent_name="deep-research",
    )

    assert result == "ok"
    assert attempts == 2
    assert captured_sleeps == [float(RATE_LIMIT_DELAYS[0])]


def test_long_timeout_disables_ordinary_retries_but_keeps_429_budget():
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.agent_name = "literature"
    client.base_url = "https://example.invalid/v1"
    client.on_retry = None
    async def before_retry():
        return None
    client.before_retry = before_retry

    kwargs = client._build_retry_kwargs(
        "deep-research",
        request_timeout=DEFAULT_REQUEST_TIMEOUT + 1,
    )

    assert kwargs["max_retries"] == 0
    assert kwargs["rate_limit_retries"] == len(RATE_LIMIT_DELAYS)
    assert kwargs["before_retry"] is before_retry
