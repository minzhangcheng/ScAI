"""retry_with_backoff must NOT burn retries on non-transient errors:
401 auth and hard monthly-quota 429 surface immediately as
UpstreamExhaustedError (which the workflow turns into a HITL gate)."""
import httpx
import pytest
from unittest.mock import AsyncMock
from openai import AuthenticationError, RateLimitError

from backend.utils import llm_client
from backend.utils.llm_client import retry_with_backoff, UpstreamExhaustedError


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "http://x"))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm_client.asyncio, "sleep", AsyncMock())


@pytest.mark.asyncio
async def test_auth_error_not_retried():
    calls = {"n": 0}

    async def func():
        calls["n"] += 1
        raise AuthenticationError("Invalid proxy server token", response=_resp(401), body=None)

    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(func, max_retries=4, agent_name="t")
    assert calls["n"] == 1  # called once, not retried


@pytest.mark.asyncio
async def test_hard_quota_429_not_retried():
    calls = {"n": 0}

    async def func():
        calls["n"] += 1
        raise RateLimitError("extra usage auto reload monthly max reached",
                             response=_resp(429), body=None)

    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(func, max_retries=4, agent_name="t")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_transient_rate_limit_is_retried():
    calls = {"n": 0}

    async def func():
        calls["n"] += 1
        raise RateLimitError("Rate limit reached, slow down", response=_resp(429), body=None)

    with pytest.raises(UpstreamExhaustedError):
        await retry_with_backoff(func, max_retries=4, agent_name="t")
    assert calls["n"] == 5  # initial + 4 retries (transient → exhausts all attempts)