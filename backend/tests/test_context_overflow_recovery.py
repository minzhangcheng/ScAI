"""Explicit overflow rebuilds requests without multiplying the retry budget."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

from backend.config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.state import sessions
from backend.utils.context_budget import estimate_tokens
from backend.utils.llm_client import LLMResponse, OpenAICompatibleClient
from backend.utils.llm_client_parts.prompt_cache import PromptCachePlan
from backend.utils.llm_client_parts.retry import retry_with_backoff
from backend.utils.llm_client_parts.errors import UpstreamExhaustedError


def overflow(message="This model's maximum context length is 200000 tokens. "
             "However, your prompt contains at least 200001 input tokens."):
    return BadRequestError(message, response=httpx.Response(
        400, request=httpx.Request("POST", "https://example.test/v1")),
        body={"message": message})


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["responses", "messages"])
async def test_pi_history_rebuilds_adapter_request_after_overflow(
    monkeypatch, protocol
):
    endpoint = f"https://openrouter.ai/api/v1/{protocol}"
    model = "anthropic/claude-sonnet-5"
    state = AnalysisState(
        id=f"pi-history-overflow-{protocol}", user_id="user"
    )
    state.custom_models["pi"] = model
    state.agent_server_configs["pi"] = {"llm_base_url": endpoint}
    monkeypatch.setitem(sessions, state.id, state)
    monkeypatch.setattr(
        "backend.utils.config_resolver.resolve_config_for_user",
        AsyncMock(return_value=SystemConfig()),
    )
    monkeypatch.setattr(
        "backend.services.session_store._async_save_session_under_lifecycle",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "backend.utils.websocket_delivery.send_analysis_json", AsyncMock()
    )
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", AsyncMock()
    )
    client = OpenAICompatibleClient(
        api_key="k",
        base_url=endpoint,
        session_id=state.id,
        agent_name="pi",
    )
    monkeypatch.setattr(client.llm_logger, "enabled", False)
    requests = []
    messages = [{"role": "system", "content": "Review the analysis."}]
    for index in range(8):
        messages.extend([
            {"role": "user", "content": f"question {index} " * 12000},
            {"role": "assistant", "content": f"answer {index} " * 12000},
        ])
    original = deepcopy(messages)

    async def request(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        if len(requests) <= 2:
            if protocol == "messages":
                from anthropic import BadRequestError as AnthropicBadRequestError

                error = overflow(
                    "prompt is too long: 210000 tokens > 200000 maximum"
                )
                raise AnthropicBadRequestError(
                    str(error), response=error.response, body=error.body
                )
            raise overflow()
        return LLMResponse(
            content="done", model=model, usage={}, raw_response=None
        )

    if protocol == "responses":
        client._responses_adapter = SimpleNamespace(complete=request)
        plan = PromptCachePlan(
            mode="disabled", prefer_anthropic_messages=False
        )
    else:
        client._anthropic_adapter = SimpleNamespace(complete=request)
        plan = PromptCachePlan()
    try:
        result = await client.chat(
            model=model,
            messages=messages,
            prompt_cache_plan=plan,
            prefer_stream=False,
        )
        assert result.content == "done"
        sizes = [estimate_tokens(value) for value in requests]
        assert len(sizes) == 3
        assert all(
            after < before for before, after in zip(sizes, sizes[1:])
        )
        assert messages == original
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_failed_recovery_preserves_original_overflow():
    request = AsyncMock(side_effect=overflow())
    with pytest.raises(UpstreamExhaustedError, match="maximum context length"):
        await retry_with_backoff(request, on_context_overflow=AsyncMock(side_effect=RuntimeError("save")))
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_user_budget_edit_wins_over_inflight_overflow(monkeypatch):
    from backend.utils.context_overflow import recover_context_overflow, context_settings
    state = AnalysisState(id="edited-overflow", user_id="user")
    state.custom_models["pi"] = "qwen"
    state.agent_server_configs["pi"] = {"llm_base_url": "https://example.test/v1"}
    expected = context_settings(state.agent_server_configs["pi"])
    state.agent_server_configs["pi"]["context_window"] = 96000
    monkeypatch.setitem(sessions, state.id, state)
    monkeypatch.setattr("backend.utils.config_resolver.resolve_config_for_user",
                        AsyncMock(return_value=SystemConfig()))
    assert not await recover_context_overflow(
        session_id=state.id, agent_name="pi", model="qwen",
        endpoint="https://example.test/v1", expected_context=expected,
        messages=[{"role": "user", "content": "x" * 12000}])
    assert state.agent_server_configs["pi"]["context_window"] == 96000
    assert "adaptive_context_window" not in state.agent_server_configs["pi"]
