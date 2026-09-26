"""Regression coverage for automatic prompt-cache negotiation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import BadRequestError, RateLimitError

from backend.utils.llm_client_parts.adapters import (
    _AnthropicAdapter,
    _ResponsesAdapter,
)
from backend.utils.llm_client_parts.prompt_cache import (
    PromptCacheController,
    is_prompt_cache_rejection,
    responses_prefix_cache_plan,
)


def _api_error(error_type, message, status, *, param=None):
    body = {"error": {"message": message}}
    if param is not None:
        body["error"]["param"] = param
    return error_type(
        message=message,
        response=MagicMock(status_code=status),
        body=body,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", ["prompt_cache_options", "input.0.content.0.prompt_cache_breakpoint", "extra_body"])
async def test_responses_prefix_rejection_restores_full_input_and_is_remembered(rejected):
    base_url = "https://openrouter.ai/api/v1"
    model = "openai/gpt-5.6-sol"
    prefix = "Stable rules.\n"
    plan = responses_prefix_cache_plan(base_url, model, prefix)
    error = (
        TypeError("create() got an unexpected keyword argument 'extra_body'")
        if rejected == "extra_body" else
        _api_error(BadRequestError, f"Unsupported parameter: {rejected}", 400, param=rejected)
    )
    create = AsyncMock(side_effect=[error, "fallback", "next"])
    controller = PromptCacheController(base_url, lambda: ("session-1", "pi"))
    adapter = _ResponsesAdapter(
        SimpleNamespace(responses=SimpleNamespace(create=create)),
        base_url=base_url, prompt_cache_controller=controller,
    )
    params = {
        "model": model, "instructions": "System rules",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prefix + "Dynamic feedback"},
            {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
        ]}],
    }
    for expected in ["fallback", "next"]:
        assert await adapter._create_responses_with_fallback(
            params, modality="vision", cache_mode="explicit", prompt_cache_plan=plan,
        ) == expected
    assert create.await_count == 3
    assert "prompt_cache_breakpoint" in create.await_args_list[0].kwargs["input"][0]["content"][0]
    for call in create.await_args_list[1:]:
        assert call.kwargs["input"] == params["input"]
        assert call.kwargs["instructions"] == "System rules"
        assert "prompt_cache_options" not in call.kwargs.get("extra_body", {})
        if rejected != "extra_body":
            assert "session_id" in call.kwargs["extra_body"]
    assert controller.describe(model, "responses", "vision", "explicit")["capability"] == "unsupported"


def test_openrouter_anthropic_adapter_uses_bearer_auth(monkeypatch):
    import anthropic

    constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(anthropic, "AsyncAnthropic", constructor)

    _AnthropicAdapter(
        api_key="router-secret",
        base_url="https://openrouter.ai/api",
    )

    kwargs = constructor.call_args.kwargs
    assert kwargs["auth_token"] == "router-secret"
    assert kwargs["api_key"] == ""


@pytest.mark.asyncio
async def test_explicit_cache_rejection_rebuilds_marker_free_request_and_stays_disabled():
    rejection = _api_error(
        BadRequestError,
        "Invalid cache_control in messages.1.content.0.cache_control",
        400,
        param="messages.1.content.0.cache_control",
    )
    controller = PromptCacheController(
        "https://openrouter.ai/api/v1",
        context_getter=lambda: ("session-1", "bioinformatics"),
    )
    marked = {
        "model": "anthropic/claude-sonnet-5",
        "messages": [{"role": "user", "content": [{
            "type": "text",
            "text": "stable",
            "cache_control": {"type": "ephemeral"},
        }]}],
    }
    marker_free = {
        "model": marked["model"],
        "messages": [{"role": "user", "content": "stable"}],
    }
    create = AsyncMock(side_effect=[rejection, "fallback", "next"])

    assert await controller.request(
        create,
        marked,
        fallback_params=marker_free,
        model=marked["model"],
        protocol="anthropic",
        modality="tools",
        cache_mode="explicit",
    ) == "fallback"
    assert create.await_args_list[0].kwargs["messages"] == marked["messages"]
    assert create.await_args_list[1].kwargs["messages"] == marker_free["messages"]
    assert "cache_control" not in create.await_args_list[0].kwargs["extra_body"]
    assert "session_id" in create.await_args_list[1].kwargs["extra_body"]

    assert await controller.request(
        create,
        marked,
        fallback_params=marker_free,
        model=marked["model"],
        protocol="anthropic",
        modality="tools",
        cache_mode="explicit",
    ) == "next"
    assert create.await_args_list[2].kwargs["messages"] == marker_free["messages"]


@pytest.mark.asyncio
async def test_rate_limit_and_unrelated_bad_request_do_not_disable_cache_control():
    controller = PromptCacheController("https://api.anthropic.com")
    params = {"model": "claude-sonnet", "messages": []}
    rate_limit = _api_error(
        RateLimitError,
        "cache_control request was rate limited",
        429,
    )
    unrelated = _api_error(
        BadRequestError,
        "Unsupported parameter: temperature",
        400,
        param="temperature",
    )
    misleading = _api_error(
        BadRequestError,
        "Invalid parameter: temperature, request included cache_control metadata",
        400,
        param="temperature",
    )

    for error in (rate_limit, unrelated, misleading):
        create = AsyncMock(side_effect=error)
        with pytest.raises(type(error)):
            await controller.request(
                create,
                params,
                model=params["model"],
                protocol="anthropic",
            )
        assert create.await_count == 1

    assert is_prompt_cache_rejection(rate_limit) is False
    assert is_prompt_cache_rejection(unrelated) is False
    assert is_prompt_cache_rejection(misleading) is False
    assert is_prompt_cache_rejection(_api_error(
        BadRequestError,
        "Unsupported parameter: cache_control",
        400,
    )) is True
    assert is_prompt_cache_rejection(_api_error(
        BadRequestError,
        "cache_control: Extra inputs are not permitted",
        422,
    )) is True
    assert is_prompt_cache_rejection(_api_error(
        BadRequestError,
        "Invalid parameter",
        400,
        param="cache_control",
    )) is True
    assert controller.describe(params["model"], "anthropic")[
        "cache_control_enabled"
    ] is True
