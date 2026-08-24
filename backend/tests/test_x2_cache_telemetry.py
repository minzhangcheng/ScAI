"""X2 tests for provider cache telemetry normalization."""

from types import SimpleNamespace

from pydantic import BaseModel

from backend.utils.llm_client_parts.client_content import ClientContentMixin
from backend.utils.llm_client_parts.protocol import (
    _parse_anthropic_dump,
    _parse_responses_dump,
)


def _chat_response(*, details=None, **usage_fields):
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        prompt_tokens_details=details,
        **usage_fields,
    )
    return SimpleNamespace(usage=usage)


class _CachedDetailsModel(BaseModel):
    cached_tokens: int = 0


class _UsageModel(BaseModel):
    prompt_tokens: int = 10
    completion_tokens: int = 4
    total_tokens: int = 14
    prompt_tokens_details: _CachedDetailsModel | None = None
    cache_read_input_tokens: int = 0


def test_chat_cache_telemetry_distinguishes_absent_null_zero_and_gateway_fallback():
    absent = ClientContentMixin._extract_usage(_chat_response())
    assert "cached_tokens" not in absent

    null = ClientContentMixin._extract_usage(
        _chat_response(details=SimpleNamespace(cached_tokens=None))
    )
    assert "cached_tokens" not in null

    explicit_zero = ClientContentMixin._extract_usage(
        _chat_response(details=SimpleNamespace(cached_tokens=0))
    )
    assert explicit_zero["cached_tokens"] == 0

    gateway_fallback = ClientContentMixin._extract_usage(
        _chat_response(
            details=SimpleNamespace(cached_tokens=None),
            cache_read_input_tokens=17,
        )
    )
    assert gateway_fallback["cached_tokens"] == 17

    nested_wins = ClientContentMixin._extract_usage(
        _chat_response(
            details=SimpleNamespace(cached_tokens=3),
            cache_read_input_tokens=17,
        )
    )
    assert nested_wins["cached_tokens"] == 3

    # A Pydantic default-zero field is not a provider measurement until it is
    # explicitly set. An explicitly set nested zero remains authoritative over
    # a non-zero gateway fallback.
    default_zero_with_gateway = ClientContentMixin._extract_usage(
        SimpleNamespace(
            usage=_UsageModel(
                prompt_tokens_details=_CachedDetailsModel(),
                cache_read_input_tokens=17,
            )
        )
    )
    assert default_zero_with_gateway["cached_tokens"] == 17

    default_zero_without_gateway = ClientContentMixin._extract_usage(
        SimpleNamespace(
            usage=_UsageModel(prompt_tokens_details=_CachedDetailsModel())
        )
    )
    assert "cached_tokens" not in default_zero_without_gateway

    explicit_nested_zero = ClientContentMixin._extract_usage(
        SimpleNamespace(
            usage=_UsageModel(
                prompt_tokens_details=_CachedDetailsModel(cached_tokens=0),
                cache_read_input_tokens=17,
            )
        )
    )
    assert explicit_nested_zero["cached_tokens"] == 0


def _responses_dump(details):
    return {
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "input_tokens_details": details,
        },
    }


def test_responses_cache_telemetry_distinguishes_absent_null_and_zero():
    for details in ({}, None, {"cached_tokens": None}):
        _, _, _, usage = _parse_responses_dump(_responses_dump(details))
        assert "cached_tokens" not in usage

    _, _, _, usage = _parse_responses_dump(
        _responses_dump({"cached_tokens": 0})
    )
    assert usage["cached_tokens"] == 0


def _anthropic_dump(usage):
    return {
        "content": [{"type": "text", "text": "ok"}],
        "usage": usage,
    }


def test_anthropic_cache_telemetry_uses_full_input_and_cache_read_only():
    _, _, _, usage = _parse_anthropic_dump(_anthropic_dump({
        "input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 3,
        "output_tokens": 4,
    }))
    assert usage["prompt_tokens"] == 18
    assert usage["total_tokens"] == 22
    assert usage["cached_tokens"] == 3

    _, _, _, unavailable = _parse_anthropic_dump(_anthropic_dump({
        "input_tokens": 10,
        "output_tokens": 4,
    }))
    assert unavailable["prompt_tokens"] == 10
    assert "cached_tokens" not in unavailable

    _, _, _, reported_without_reads = _parse_anthropic_dump(_anthropic_dump({
        "input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "output_tokens": 4,
    }))
    assert reported_without_reads["prompt_tokens"] == 15
    assert reported_without_reads["cached_tokens"] == 0

    _, _, _, reported_read_zero = _parse_anthropic_dump(_anthropic_dump({
        "input_tokens": 10,
        "cache_read_input_tokens": 0,
        "output_tokens": 4,
    }))
    assert reported_read_zero["cached_tokens"] == 0

    for null_usage in (
        {"input_tokens": 10, "cache_creation_input_tokens": None},
        {"input_tokens": 10, "cache_read_input_tokens": None},
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
        },
    ):
        _, _, _, unavailable = _parse_anthropic_dump(_anthropic_dump(null_usage))
        assert "cached_tokens" not in unavailable
