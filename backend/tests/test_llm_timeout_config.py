"""Deployment timeouts reach the transport, stream and supplementary budgets."""
import os
import subprocess
import sys

import pytest

from backend.utils.llm_client_parts.retry import _resolve_llm_timeout_seconds


@pytest.mark.parametrize("raw,expected,warns", [
    (None, 600, False), (" ", 600, False), (" 1200 ", 1200, False),
    ("1", 1, False), ("0", 600, True), ("-1", 600, True),
    ("bad", 600, True), ("1.5", 600, True), ("nan", 600, True),
    ("inf", 600, True),
])
def test_deployment_timeout_parsing(monkeypatch, caplog, raw, expected, warns):
    monkeypatch.delenv("SCAI_LLM_TIMEOUT_SECONDS", raising=False)
    if raw is not None:
        monkeypatch.setenv("SCAI_LLM_TIMEOUT_SECONDS", raw)
    assert _resolve_llm_timeout_seconds() == expected
    assert ("SCAI_LLM_TIMEOUT_SECONDS" in caplog.text) is warns


@pytest.mark.parametrize("raw,expected", [(None, 600), ("1200", 1200)])
def test_startup_timeout_reaches_shared_consumers(raw, expected, tmp_path):
    # A fresh process proves import-time values reach consumers without leaking
    # module reloads or monkeypatched constants into other tests.
    env = dict(os.environ, SCAI_DATA_DIR=str(tmp_path))
    env.pop("SCAI_LLM_TIMEOUT_SECONDS", None)
    if raw is not None:
        env["SCAI_LLM_TIMEOUT_SECONDS"] = raw
    result = subprocess.run([sys.executable, "-c", """
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch
from backend.utils.llm_client import OpenAICompatibleClient
from backend.utils.llm_client_parts import retry
from backend.utils.llm_client_parts.adapters import _start_stream
from backend.workflow.post_execution import PostExecutionReviewMixin

expected = int(sys.argv[1])
assert retry.llm_timeout_seconds() == expected
assert retry.llm_timeout_seconds(600) == expected
assert retry.llm_timeout_seconds(1800) == 1800

async def main():
    client = OpenAICompatibleClient(api_key="test", base_url="http://localhost:9/v1")
    try:
        assert client.http_client.timeout.read == expected
        assert client.client.timeout == expected
        kwargs = client._build_retry_kwargs("test", request_timeout=600)
        assert kwargs["request_timeout"] == expected
        assert "max_retries" not in kwargs
        longer = client._build_retry_kwargs("test", request_timeout=1800)
        assert longer["request_timeout"] == 1800
        assert longer["max_retries"] == 0

        waits = []
        wait_for = asyncio.wait_for
        async def capture(awaitable, *, timeout):
            waits.append(timeout)
            return await wait_for(awaitable, timeout=timeout)
        async def complete():
            return "done"
        async def events():
            yield "first"
        with patch("asyncio.wait_for", capture):
            assert await retry.retry_with_backoff(complete) == "done"
            started = await _start_stream(
                events(), complete, timeout=600, label="vision",
            )
            try:
                assert started.total_timeout == expected
                assert started.idle_timeout == expected
            finally:
                await started.close()
        assert waits == [expected, expected]
    finally:
        await client.close()

asyncio.run(main())
review = PostExecutionReviewMixin()
review.config = SimpleNamespace(sandbox_timeout=5400)
assert review._post_execution_review_limits()[1] == 5400 + expected
assert review._DEFAULT_STEP_TIMEOUT == 5400 + expected
review.config.post_execution_review_timeout_seconds = 1
assert review._post_execution_review_limits()[1] == expected
""", str(expected)], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
