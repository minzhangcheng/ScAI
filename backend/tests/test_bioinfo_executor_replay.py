"""Replay incomplete tool responses before restarting the workflow node."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.agents import bioinformatics_agent as facade
from backend.agents.bioinformatics import executor_loop
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AnalysisState
from backend.tests.test_bioinfo_context_scopes import _bare_agent
from backend.utils.llm_client import LLMToolResponse, ToolCall, UpstreamExhaustedError
from backend.utils.settings_provider import BudgetedTextDispatch


def _failure(kind):
    return UpstreamExhaustedError(
        "600s request deadline" if kind == "stream_interrupted" else "max_output_tokens",
        agent_name="bioinformatics", model="test-model", failure_kind=kind,
    )


@pytest_asyncio.fixture
async def replay_context(monkeypatch, tmp_path):
    agent = _bare_agent(monkeypatch, tmp_path)
    client = MagicMock(base_url="https://example.invalid/v1")
    client.chat_with_tools = AsyncMock()
    monkeypatch.setattr(BioinformaticsAgent, "llm_client", property(lambda self: client))
    monkeypatch.setattr("backend.utils.llm_client_parts.retry.asyncio.sleep", AsyncMock())
    agent._apply_context_budget = AsyncMock(side_effect=lambda messages, **kwargs:
        BudgetedTextDispatch(messages=messages, report=None, max_tokens=None))
    agent._acknowledge_context_budget = AsyncMock()
    executor_loop.bind_facade(facade)
    context = await executor_loop.prepare_loop_context(
        agent, AnalysisState(id="test-session", user_id="test-user"),
        "Run the next step", max_steps=5, resume_from_checkpoint=False,
    )
    response = LLMToolResponse(
        content="Ready", model="test-model", usage={}, raw_response=None,
        tool_calls=[ToolCall(id="call-1", name="execute_code", arguments={"code": "x = 1"})],
    )
    return context, client, response


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["stream_interrupted", "incomplete_response"])
async def test_tool_request_recovers_without_repeating_turn_work(replay_context, kind):
    context, client, response = replay_context
    client.chat_with_tools.side_effect = [_failure(kind), _failure(kind), response]
    original_messages = list(context.messages)
    step = context.step_count

    result = await executor_loop.start_turn(context)

    assert result.response is response
    assert client.chat_with_tools.await_count == 3
    first = client.chat_with_tools.await_args_list[0]
    assert all(call == first for call in client.chat_with_tools.await_args_list)
    assert first.kwargs["prefer_stream"] is True
    assert context.agent._apply_context_budget.await_count == 1
    assert context.agent._acknowledge_context_budget.await_count == 1
    assert context.step_count == step + 1
    assert context.messages == original_messages
    assert context.analysis_log == []
    notices = [call for call in context.agent._notify.await_args_list
               if "retrying request" in str(call)]
    assert len(notices) == 2

    # A later request has its own allowance even in the same workflow node.
    client.chat_with_tools.side_effect = [_failure(kind)] * 3 + [response]
    assert (await executor_loop.start_turn(context)).response is response
    assert client.chat_with_tools.await_count == 7
    assert context.step_count == step + 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["stream_interrupted", "incomplete_response"])
async def test_tool_replay_exhaustion_reaches_existing_intervention(replay_context, kind):
    context, client, _ = replay_context
    client.chat_with_tools.side_effect = _failure(kind)

    with pytest.raises(UpstreamExhaustedError) as caught:
        await executor_loop.start_turn(context)

    assert caught.value.failure_kind == kind
    assert client.chat_with_tools.await_count == 4
    assert context.agent._apply_context_budget.await_count == 1
    context.agent._acknowledge_context_budget.assert_not_awaited()
    assert context.analysis_log == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_tool_replay_preserves_unrelated_failure_and_cancellation(replay_context, cancelled):
    context, client, _ = replay_context
    error = asyncio.CancelledError() if cancelled else _failure("upstream_exhausted")
    client.chat_with_tools.side_effect = error

    with pytest.raises(type(error)) as caught:
        await executor_loop.start_turn(context)

    assert caught.value is error
    assert client.chat_with_tools.await_count == 1
    context.agent._acknowledge_context_budget.assert_not_awaited()
