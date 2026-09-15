"""Regression coverage for automatic prompt-cache negotiation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import BadRequestError, RateLimitError

from backend.utils.llm_client import OpenAICompatibleClient
from backend.utils.llm_client_parts.adapters import (
    _AnthropicAdapter,
    _ResponsesAdapter,
)
from backend.utils.llm_client_parts.prompt_cache import (
    PromptCacheController,
    _is_openrouter_url,
    is_prompt_cache_rejection,
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


def test_openrouter_detection_rejects_lookalike_hosts():
    assert _is_openrouter_url("https://openrouter.ai/api/v1") is True
    assert _is_openrouter_url("https://api.openrouter.ai/v1") is True
    assert _is_openrouter_url("https://openrouter.ai.example.com/v1") is False


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
async def test_openrouter_anthropic_uses_cache_control_and_opaque_session_affinity():
    controller = PromptCacheController(
        "https://openrouter.ai/api/v1",
        context_getter=lambda: ("session-secret", "bioinformatics"),
    )
    create = AsyncMock(return_value="ok")

    result = await controller.request(
        create,
        {"model": "anthropic/claude-sonnet-5", "messages": []},
        model="anthropic/claude-sonnet-5",
        protocol="chat",
    )

    assert result == "ok"
    extra_body = create.await_args.kwargs["extra_body"]
    assert extra_body["cache_control"] == {"type": "ephemeral"}
    assert extra_body["session_id"].startswith("scai-")
    assert "session-secret" not in extra_body["session_id"]
    assert controller.describe(
        "anthropic/claude-sonnet-5", "chat"
    )["capability"] == "supported"

    latest_alias = AsyncMock(return_value="latest")
    await controller.request(
        latest_alias,
        {"model": "~anthropic/claude-sonnet-latest", "messages": []},
        model="~anthropic/claude-sonnet-latest",
        protocol="chat",
    )
    assert "cache_control" in latest_alias.await_args.kwargs["extra_body"]


@pytest.mark.asyncio
async def test_cache_parameter_rejection_retries_once_and_is_remembered():
    rejection = _api_error(
        BadRequestError,
        "Unsupported parameter: cache_control",
        400,
        param="cache_control",
    )
    controller = PromptCacheController(
        "https://openrouter.ai/api/v1",
        context_getter=lambda: ("session-1", "pi"),
    )
    create = AsyncMock(side_effect=[rejection, "fallback", "next", "text"])
    params = {"model": "anthropic/claude-sonnet-5", "messages": []}

    assert await controller.request(
        create,
        params,
        model=params["model"],
        protocol="responses",
        modality="tools",
    ) == "fallback"
    assert "cache_control" in create.await_args_list[0].kwargs["extra_body"]
    assert "cache_control" not in create.await_args_list[1].kwargs["extra_body"]
    assert "session_id" in create.await_args_list[1].kwargs["extra_body"]

    assert await controller.request(
        create,
        params,
        model=params["model"],
        protocol="responses",
        modality="tools",
    ) == "next"
    assert len(create.await_args_list) == 3
    assert "cache_control" not in create.await_args_list[2].kwargs["extra_body"]
    state = controller.describe(params["model"], "responses", "tools")
    assert state["capability"] == "unsupported"
    assert state["cache_control_enabled"] is False
    assert state["routing_constraint"] is None

    assert await controller.request(
        create,
        params,
        model=params["model"],
        protocol="responses",
        modality="text",
    ) == "text"
    assert "cache_control" in create.await_args_list[3].kwargs["extra_body"]


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


@pytest.mark.asyncio
async def test_unknown_provider_uses_implicit_cache_without_private_parameters():
    controller = PromptCacheController("https://llm.example.test/v1")
    create = AsyncMock(return_value="ok")

    await controller.request(
        create,
        {"model": "custom-model", "messages": []},
        model="custom-model",
        protocol="chat",
    )

    assert "extra_body" not in create.await_args.kwargs
    state = controller.describe("custom-model", "chat")
    assert state["strategy"] == "provider_managed"
    assert state["session_affinity"] is False


@pytest.mark.asyncio
async def test_old_sdk_extra_body_rejection_falls_back_and_is_remembered():
    controller = PromptCacheController(
        "https://openrouter.ai/api/v1",
        context_getter=lambda: ("session-old-sdk", "pi"),
    )
    create = AsyncMock(side_effect=[
        TypeError("create() got an unexpected keyword argument 'extra_body'"),
        "fallback",
        "next",
        "chat",
    ])
    params = {"model": "anthropic/claude-sonnet-5", "messages": []}

    assert await controller.request(
        create,
        params,
        model=params["model"],
        protocol="anthropic",
    ) == "fallback"
    assert "extra_body" in create.await_args_list[0].kwargs
    assert "extra_body" not in create.await_args_list[1].kwargs

    assert await controller.request(
        create,
        params,
        model=params["model"],
        protocol="anthropic",
    ) == "next"
    assert "extra_body" not in create.await_args_list[2].kwargs
    state = controller.describe(params["model"], "anthropic")
    assert state["capability"] == "unsupported"
    assert state["cache_control_enabled"] is False
    assert state["session_affinity"] is False
    tools_state = controller.describe(params["model"], "anthropic", "tools")
    assert tools_state["capability"] == "sdk_unsupported"
    assert tools_state["cache_control_enabled"] is False
    assert tools_state["routing_constraint"] is None

    assert await controller.request(
        create,
        params,
        model=params["model"],
        protocol="chat",
    ) == "chat"
    assert "cache_control" in create.await_args_list[3].kwargs["extra_body"]
    assert "session_id" in create.await_args_list[3].kwargs["extra_body"]
    chat_state = controller.describe(params["model"], "chat")
    assert chat_state["capability"] == "supported"
    assert chat_state["session_affinity"] is True


@pytest.mark.asyncio
async def test_chat_responses_and_anthropic_calls_share_the_controller():
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        session_id="session-2",
        agent_name="test",
    )
    try:
        client.client.chat.completions.create = AsyncMock(return_value="chat")
        make_chat, _ = client._build_request_pair(
            {"model": "anthropic/claude-sonnet-5", "messages": []},
            "anthropic/claude-sonnet-5",
            0.2,
            100,
        )
        assert await make_chat() == "chat"
        assert "cache_control" in (
            client.client.chat.completions.create.await_args.kwargs["extra_body"]
        )

        response_client = MagicMock()
        response_client.responses.create = AsyncMock(return_value="responses")
        responses = _ResponsesAdapter(
            response_client,
            base_url=client.base_url,
            prompt_cache_controller=client._prompt_cache_controller,
        )
        assert await responses._create_responses_with_fallback({
            "model": "anthropic/claude-sonnet-5",
            "input": [],
        }) == "responses"
        assert "cache_control" in (
            response_client.responses.create.await_args.kwargs["extra_body"]
        )

        anthropic_client = MagicMock()
        anthropic_client.messages.create = AsyncMock(return_value="anthropic")
        anthropic = object.__new__(_AnthropicAdapter)
        anthropic.client = anthropic_client
        anthropic.agent_name = "test"
        anthropic._prompt_cache_controller = client._prompt_cache_controller
        assert await anthropic._create(
            {"model": "claude-sonnet", "messages": []},
            model="claude-sonnet",
        ) == "anthropic"
        assert "cache_control" in (
            anthropic_client.messages.create.await_args.kwargs["extra_body"]
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_vision_negotiates_cache_independently_and_keeps_affinity():
    controller = PromptCacheController(
        "https://openrouter.ai/api/v1",
        context_getter=lambda: ("session-vision", "pathology"),
    )
    create = AsyncMock(return_value=SimpleNamespace())

    await controller.request(
        create,
        {"model": "anthropic/claude-sonnet-5", "messages": []},
        model="anthropic/claude-sonnet-5",
        protocol="chat",
        modality="vision",
    )

    extra_body = create.await_args.kwargs["extra_body"]
    assert "session_id" in extra_body
    assert "cache_control" in extra_body
    assert controller.describe(
        "anthropic/claude-sonnet-5", "chat", "vision"
    )["capability"] == "supported"
    assert controller.describe(
        "anthropic/claude-sonnet-5", "chat", "text"
    )["capability"] == "probing"
