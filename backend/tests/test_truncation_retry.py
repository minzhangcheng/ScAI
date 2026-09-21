"""Reasoning models (e.g. mimo-v2.5) can spend the entire max_tokens budget on
reasoning_content, leaving the visible content empty and the response truncated
(finish_reason == 'length'). _chat_inline must detect this and retry ONCE with a
no output cap instead of handing the EMPTY_CONTENT_PLACEHOLDER to callers (which
silently swallow it — e.g. the supervisor's no-intervention fallback)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.tests.split_facade import patch_facade
from backend.utils.llm_client import (
    OpenAICompatibleClient,
    EMPTY_CONTENT_PLACEHOLDER,
    EMPTY_RESPONSE_MAX_RETRIES,
    LLMResponse,
    UpstreamExhaustedError,
    EmptyResponseError,
    is_transient_probe_error,
)


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


@pytest_asyncio.fixture
async def client():
    c = OpenAICompatibleClient(api_key="test", base_url="http://test")
    c.llm_logger = MagicMock()
    c.llm_logger.log_request.return_value = MagicMock()
    try:
        yield c
    finally:
        await c.close()


@pytest.mark.asyncio
async def test_length_truncated_empty_content_retries_without_output_cap(client):
    truncated = _resp(None, finish_reason="length")          # reasoning ate the budget
    recovered = _resp('{"severity": "none"}', finish_reason="stop")
    call = AsyncMock(side_effect=[truncated, recovered])

    async def invoke(make_request, *_args):
        return await make_request()

    with patch.object(client.client.chat.completions, "create", call), \
         patch.object(client, "_call_with_temp_fallback", side_effect=invoke), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == '{"severity": "none"}'
    assert call.await_count == 2  # retried once after the truncated empty response
    retry_params = call.await_args_list[1].kwargs
    assert "max_tokens" not in retry_params
    assert "max_completion_tokens" not in retry_params


@pytest.mark.asyncio
async def test_private_truncated_reasoning_still_retries(client):
    # A private reasoning field is not answer content. The empty public answer
    # plus length truncation must still trigger the uncapped retry.
    truncated = _resp(None, finish_reason="length", reasoning="First, the user is asking me to...")
    recovered = _resp('{"severity": "warning"}', finish_reason="stop")
    call = AsyncMock(side_effect=[truncated, recovered])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == '{"severity": "warning"}'
    assert call.await_count == 2


@pytest.mark.asyncio
async def test_non_truncated_empty_content_does_not_retry(client):
    # finish_reason == 'stop' → genuine empty answer, not a budget truncation
    stopped = _resp(None, finish_reason="stop")
    call = AsyncMock(side_effect=[stopped])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == EMPTY_CONTENT_PLACEHOLDER
    assert call.await_count == 1  # no retry


@pytest.mark.asyncio
async def test_reasoning_only_stop_retries_for_public_answer_without_exposing_reasoning(
    client,
):
    reasoning_only = _resp(
        None,
        finish_reason="stop",
        reasoning="private chain of thought",
    )
    recovered = _resp("Public final answer", finish_reason="stop")
    call = AsyncMock(side_effect=[reasoning_only, recovered])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._chat_inline(
            "mimo-v2.5",
            [{"role": "user", "content": "x"}],
            max_tokens=512,
        )

    assert result.content == "Public final answer"
    assert call.await_count == 2
    assert "private chain of thought" not in result.content


@pytest.mark.asyncio
async def test_repeated_reasoning_only_stop_is_upstream_exhaustion(client):
    reasoning_only = _resp(
        None,
        finish_reason="stop",
        reasoning="private chain of thought",
    )
    call = AsyncMock(side_effect=[reasoning_only, reasoning_only])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch_facade("backend.utils.llm_client", "logger"):
        with pytest.raises(UpstreamExhaustedError, match="without a public answer"):
            await client._chat_inline(
                "mimo-v2.5",
                [{"role": "user", "content": "x"}],
                max_tokens=512,
            )

    assert call.await_count == 2


@pytest.mark.asyncio
async def test_truncation_retry_capped_to_single_attempt(client):
    # Even if the retry is also truncated, we must not loop forever.
    truncated = _resp(None, finish_reason="length")
    call = AsyncMock(side_effect=[truncated, truncated])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._chat_inline("mimo-v2.5", [{"role": "user", "content": "x"}], max_tokens=512)
    assert result.content == EMPTY_CONTENT_PLACEHOLDER
    assert call.await_count == 2  # original + exactly one retry


@pytest.mark.asyncio
async def test_empty_answer_zero_tokens_retries_then_recovers(client):
    dead = _resp(None, finish_reason="stop", completion_tokens=0)
    recovered = _resp("## Report\n\nSubstantive body.", finish_reason="stop")
    call = AsyncMock(side_effect=[dead, recovered])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._chat_inline(
            "gpt-5.5", [{"role": "user", "content": "x"}], max_tokens=512
        )
    assert result.content == "## Report\n\nSubstantive body."
    assert call.await_count == 2  # retried once after the empty zero-token response


@pytest.mark.asyncio
async def test_empty_answer_zero_tokens_raises_after_cap(client):
    dead = _resp(None, finish_reason="stop", completion_tokens=0)
    call = AsyncMock(side_effect=[dead] * (EMPTY_RESPONSE_MAX_RETRIES + 1))
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch_facade("backend.utils.llm_client", "logger"):
        with pytest.raises(UpstreamExhaustedError):
            await client._chat_inline(
                "gpt-5.5", [{"role": "user", "content": "x"}], max_tokens=512
            )
    assert call.await_count == EMPTY_RESPONSE_MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_empty_answer_with_tokens_does_not_retry(client):
    empty_but_billed = _resp(None, finish_reason="stop", completion_tokens=42)
    call = AsyncMock(side_effect=[empty_but_billed])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._chat_inline(
            "gpt-5.5", [{"role": "user", "content": "x"}], max_tokens=512
        )
    assert result.content == EMPTY_CONTENT_PLACEHOLDER
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_probe_path_raises_transient_not_empty(client):
    dead = _resp(None, finish_reason="stop", completion_tokens=0)
    call = AsyncMock(side_effect=[dead])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch_facade("backend.utils.llm_client", "logger"):
        with pytest.raises(EmptyResponseError):
            await client._chat_inline(
                "gpt-5.5", [{"role": "user", "content": "x"}], max_tokens=512,
                _allow_empty_retry=False,
            )
    assert call.await_count == 1


def test_empty_response_error_is_transient_for_probe():
    assert is_transient_probe_error(EmptyResponseError("x")) is True
    assert is_transient_probe_error(UpstreamExhaustedError("y")) is False


@pytest.mark.asyncio
async def test_chat_dispatch_probe_never_returns_empty(client):
    dead = _resp(None, finish_reason="stop", completion_tokens=0)
    call = AsyncMock(return_value=dead)
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch_facade("backend.utils.llm_client", "_route_protocol_order",
                      return_value=["chat", "bogus"]), \
         patch("backend.utils.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch_facade("backend.utils.llm_client", "logger"):
        with pytest.raises(UpstreamExhaustedError):
            await client.chat(
                "gpt-5.5", [{"role": "user", "content": "x"}], max_tokens=512
            )
    assert call.await_count >= 2


@pytest.mark.asyncio
async def test_adapter_empty_zero_tokens_retries_then_recovers(client):
    dead = LLMResponse(
        content="",
        model="gpt-5.5",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        raw_response=None,
    )
    recovered = LLMResponse(
        content="## Report\n\nRecovered adapter body.",
        model="gpt-5.5",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        raw_response=None,
    )
    call = AsyncMock(side_effect=[dead, recovered])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch_facade("backend.utils.llm_client", "logger"):
        result = await client._invoke_adapter(
            interaction_type="chat",
            model="gpt-5.5",
            log_messages=[{"role": "user", "content": "x"}],
            make_coro=lambda: None,
            make_coro_no_temp=lambda: None,
            temperature=0.3,
            max_tokens=512,
        )
    assert result.content == "## Report\n\nRecovered adapter body."
    assert call.await_count == 2


@pytest.mark.asyncio
async def test_adapter_probe_path_raises_transient_not_empty(client):
    dead = LLMResponse(
        content="",
        model="gpt-5.5",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        raw_response=None,
    )
    call = AsyncMock(side_effect=[dead])
    with patch.object(client, "_call_with_temp_fallback", call), \
         patch("backend.utils.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch_facade("backend.utils.llm_client", "logger"):
        with pytest.raises(EmptyResponseError):
            await client._invoke_adapter(
                interaction_type="chat",
                model="gpt-5.5",
                log_messages=[{"role": "user", "content": "x"}],
                make_coro=lambda: None,
                make_coro_no_temp=lambda: None,
                temperature=0.3,
                max_tokens=512,
                _allow_empty_retry=False,
            )
    assert call.await_count == 1
