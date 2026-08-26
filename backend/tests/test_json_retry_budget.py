"""L3: a persistently-bad-JSON func must be retried MAX_JSON_RETRIES (2) times,
i.e. invoked 3 times total, before retry_with_backoff gives up."""
import json
import pytest
from unittest.mock import AsyncMock

from backend.utils import llm_client
from backend.utils.llm_client import retry_with_backoff, UpstreamExhaustedError, MAX_JSON_RETRIES


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm_client.asyncio, "sleep", AsyncMock())


@pytest.mark.asyncio
async def test_json_error_retried_exactly_max_times():
    calls = {"n": 0}

    async def func():
        calls["n"] += 1
        raise json.JSONDecodeError("Expecting value", "", 0)

    with pytest.raises(json.JSONDecodeError):
        await retry_with_backoff(func, max_retries=4, agent_name="t")
    # MAX_JSON_RETRIES retries == MAX_JSON_RETRIES + 1 total invocations
    assert calls["n"] == MAX_JSON_RETRIES + 1


@pytest.mark.asyncio
async def test_json_error_then_success():
    calls = {"n": 0}
    gate_calls = []

    async def func():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return "ok"

    async def before_retry():
        gate_calls.append(calls["n"])

    result = await retry_with_backoff(
        func,
        max_retries=4,
        agent_name="t",
        before_retry=before_retry,
    )
    assert result == "ok"
    assert calls["n"] == 3  # 2 bad-JSON attempts then success
    assert gate_calls == [1, 2]
