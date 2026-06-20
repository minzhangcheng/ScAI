"""Tests for GoogleDeepResearchClient.poll_research transient-failure resilience.

Uses httpx.MockTransport so no real network calls happen. The client instance
is built via __new__ to bypass __init__ (no api_key needed); we then attach an
AsyncClient wired to a mock handler and a minimal _default_headers.
"""
import httpx
import pytest

from backend.utils.llm_client import GoogleDeepResearchClient

INTERACTION_ID = "abc123"

COMPLETED_BODY = {
    "status": "completed",
    "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "final report"}]}
    ],
}


def _client_with_handler(handler):
    c = GoogleDeepResearchClient.__new__(GoogleDeepResearchClient)
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    c._default_headers = {}
    return c


@pytest.mark.asyncio
async def test_poll_recovers_after_transient_disconnects():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.",
                request=request,
            )
        return httpx.Response(200, json=COMPLETED_BODY)

    c = _client_with_handler(handler)
    result = await c.poll_research(INTERACTION_ID, poll_interval=0, max_wait=30)

    assert result["report"] == "final report"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_poll_sustained_failure_times_out_not_raises_raw():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    c = _client_with_handler(handler)

    with pytest.raises(TimeoutError):
        await c.poll_research(INTERACTION_ID, poll_interval=0.01, max_wait=0.05)
