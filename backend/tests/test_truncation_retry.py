"""Reasoning models (e.g. mimo-v2.5) can spend the entire max_tokens budget on
reasoning_content, leaving the visible content empty and the response truncated
(finish_reason == 'length'). _chat_inline must detect this and retry ONCE with a
larger budget instead of handing the EMPTY_CONTENT_PLACEHOLDER to callers (which
silently swallow it — e.g. the supervisor's no-intervention fallback)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.utils.llm_client import OpenAICompatibleClient, EMPTY_CONTENT_PLACEHOLDER


def _resp(content, *, finish_reason, completion_tokens=512, reasoning=None):
    message = MagicMock(spec=["content", "tool_calls", "refusal", "model_dump"])
    message.content = content
    message.tool_calls = None
    message.refusal = None
    dump = {"content": content, "role": "assistant"}
    if reasoning is not None:
        dump["reasoning_content"] = reasoning
    message.model_dump.return_value = dump
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    response.model = "mimo-v2.5"
    response.usage = MagicMock()
    response.usage.completion_tokens = completion_tokens
    response.usage.prompt_tokens = 100
    response.usage.total_tokens = 100 + completion_tokens
    return response


@pytest.fixture
def client():
    c = OpenAICompatibleClient(api_key="test", base_url="http://test")
    c.llm_logger = MagicMock()
    c.llm_logger.log_request.return_value = MagicMock()
    return c


@pytest.mark.asyncio
async def test_length_truncated_empty_content_retries_with_bigger_budget(client):
    truncated = _resp(None, finish_reason="length")          # reasoning ate the budget
    recovered = _resp('{"severity": "none"}', finish_reason="stop")
    call = AsyncMock(side_effect=[truncated, recovered])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == '{"severity": "none"}'
    assert call.await_count == 2  # retried once after the truncated empty response


@pytest.mark.asyncio
async def test_salvaged_truncated_reasoning_still_retries(client):
    # The extractor salvages the (truncated) reasoning_content into `content`, so
    # the result is non-empty and != EMPTY_CONTENT_PLACEHOLDER — but the *answer*
    # field is still empty and the response is length-truncated, so we must retry.
    truncated = _resp(None, finish_reason="length", reasoning="First, the user is asking me to...")
    recovered = _resp('{"severity": "warning"}', finish_reason="stop")
    call = AsyncMock(side_effect=[truncated, recovered])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == '{"severity": "warning"}'
    assert call.await_count == 2


@pytest.mark.asyncio
async def test_non_truncated_empty_content_does_not_retry(client):
    # finish_reason == 'stop' → genuine empty answer, not a budget truncation
    stopped = _resp(None, finish_reason="stop")
    call = AsyncMock(side_effect=[stopped])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == EMPTY_CONTENT_PLACEHOLDER
    assert call.await_count == 1  # no retry


@pytest.mark.asyncio
async def test_truncation_retry_capped_to_single_attempt(client):
    # Even if the retry is also truncated, we must not loop forever.
    truncated = _resp(None, finish_reason="length")
    call = AsyncMock(side_effect=[truncated, truncated])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == EMPTY_CONTENT_PLACEHOLDER
    assert call.await_count == 2  # original + exactly one retry
