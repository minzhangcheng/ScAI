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
from backend.constants import LlmFailureAction
from backend.workflow.helpers import WorkflowSupersededError, WorkflowTerminatedError


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


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [LlmFailureAction.RETRY, LlmFailureAction.SKIP])
async def test_annotation_failure_waits_locally_and_preserves_budget(replay_context, action):
    context, client, response = replay_context
    context.step_count = 2
    context.annotation_retry_required = True
    context.annotation_retry_attempts = 4
    context.annotation_failure_reason = "No clustering column named custom_clusters"
    context.save_checkpoint = AsyncMock()
    client.chat_with_tools.return_value = response
    entered, release = asyncio.Event(), asyncio.Event()

    async def decide(error):
        assert error.failure_kind == "annotation_tool_failure"
        assert context.annotation_failure_reason in str(error)
        entered.set()
        await release.wait()
        return action

    context.agent.execution_intervention_callback = AsyncMock(side_effect=decide)
    turn = asyncio.create_task(executor_loop.start_turn(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not turn.done()
        client.chat_with_tools.assert_not_awaited()
        assert context.step_count == 2
        assert context.max_steps == 5
        release.set()
        result = await turn
    finally:
        if not turn.done():
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn

    assert result.response is response
    assert context.step_count == 3
    assert context.max_steps == 5
    assert context.annotation_retry_attempts == 0
    assert not context.annotation_completed
    assert not context.completed
    assert not context.evidence_gate_degraded
    assert all(not call.kwargs.get("budget_exhausted")
               for call in context.save_checkpoint.await_args_list)
    if action == LlmFailureAction.SKIP:
        assert not context.annotation_retry_required
        assert context.annotation_failure_reason in context.findings[-1]
        assert context.evidence_gate_findings[-1]["severity"] == "advise"
        assert context.save_checkpoint.await_args_list[-1].args == (
            "step_2_annotation_skipped",
        )
    else:
        assert context.annotation_retry_required
        assert context.findings == []
        assert any("Annotation recovery was requested" in str(message)
                   for message in context.messages)


@pytest.mark.asyncio
async def test_annotation_failure_terminate_unwinds_without_spending_turn(replay_context):
    context, client, _ = replay_context
    context.annotation_retry_required = True
    context.annotation_retry_attempts = 4
    context.save_checkpoint = AsyncMock()
    context.agent.execution_intervention_callback = AsyncMock(
        return_value=LlmFailureAction.TERMINATE,
    )
    with pytest.raises(WorkflowTerminatedError):
        await executor_loop.start_turn(context)
    client.chat_with_tools.assert_not_awaited()
    assert context.step_count == 0
    assert not context.completed


@pytest.mark.asyncio
async def test_annotation_failure_without_callback_reaches_workflow_gate(replay_context):
    context, client, _ = replay_context
    context.annotation_retry_required = True
    context.annotation_retry_attempts = 4
    context.save_checkpoint = AsyncMock()
    context.agent.execution_intervention_callback = None
    with pytest.raises(UpstreamExhaustedError, match="annotate_clusters failed"):
        await executor_loop.start_turn(context)
    client.chat_with_tools.assert_not_awaited()
    assert context.step_count == 0
    assert context.save_checkpoint.await_args.kwargs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [WorkflowTerminatedError, WorkflowSupersededError])
async def test_workflow_control_unwinds_tool_batch(replay_context, monkeypatch, error_type):
    context, _, response = replay_context
    error = error_type("User control at annotation gate")
    annotation = AsyncMock(side_effect=error)
    next_tool = AsyncMock()
    monkeypatch.setattr(executor_loop, "handle_annotate_clusters", annotation)
    monkeypatch.setattr(executor_loop, "handle_execute_code", next_tool)
    response.tool_calls.insert(0, ToolCall(
        id="annotation", name="annotate_clusters", arguments={},
    ))
    with pytest.raises(error_type) as caught:
        await executor_loop.process_response(context, response, None)
    assert caught.value is error
    annotation.assert_awaited_once()
    next_tool.assert_not_awaited()
    assert not any(message.get("role") == "tool" for message in context.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["pending", "retry", "skip", "legacy"])
async def test_annotation_decision_survives_checkpoint_reload(
    replay_context, monkeypatch, tmp_path, decision,
):
    context, client, response = replay_context
    context.step_count = 2
    context.annotation_retry_required = True
    context.annotation_retry_attempts = 4
    context.annotation_failure_reason = "Clustering column custom_clusters is missing"
    context.agent.sandbox.runtime_revision = 12
    context.agent.execution_intervention_callback = AsyncMock(
        side_effect=asyncio.CancelledError() if decision in {"pending", "legacy"} else None,
        return_value=decision,
    )
    if decision in {"pending", "legacy"}:
        # Simulate process interruption while the decision is still unresolved;
        # the real checkpoint writer must have saved the pending tool failure.
        with pytest.raises(asyncio.CancelledError):
            await executor_loop._resolve_annotation_failure(context)
    else:
        await executor_loop._resolve_annotation_failure(context)
    saved = await context.checkpoint_manager.load()
    assert saved["step_number"] == 2
    assert saved["total_steps"] == 5
    assert saved["budget_exhausted"] is False
    assert saved["annotation_failure_reason"] == context.annotation_failure_reason
    if decision == "legacy":
        for key in (
            "annotation_retry_required", "annotation_retry_attempts", "annotation_failure_reason",
        ):
            saved.pop(key)
        await context.checkpoint_manager.save(saved)

    restored_agent = _bare_agent(monkeypatch, tmp_path)
    restored_agent.sandbox.runtime_revision = 12
    restored_agent._apply_context_budget = context.agent._apply_context_budget
    restored_agent._acknowledge_context_budget = context.agent._acknowledge_context_budget
    restored_agent.execution_intervention_callback = AsyncMock(
        return_value=LlmFailureAction.TERMINATE,
    )
    restored = await executor_loop.prepare_loop_context(
        restored_agent, context.analysis_state, "Discarded on checkpoint resume",
        max_steps=5, resume_from_checkpoint=True,
    )
    assert restored.step_count == 2
    assert restored.max_steps == 5
    assert restored.messages == saved["messages"]
    if decision == "pending":
        assert restored.annotation_retry_required
        assert restored.annotation_retry_attempts == 4
        assert restored.annotation_failure_reason == context.annotation_failure_reason
        with pytest.raises(WorkflowTerminatedError):
            await executor_loop.start_turn(restored)
        restored_agent.execution_intervention_callback.assert_awaited_once()
        client.chat_with_tools.assert_not_awaited()
        assert restored.step_count == 2
    else:
        assert restored.annotation_retry_attempts == 0
        assert restored.annotation_retry_required is (decision == "retry")
        if decision == "skip":
            assert any(context.annotation_failure_reason in finding for finding in restored.findings)
            assert any("user chose Skip" in str(message) for message in restored.messages)
        if decision == "legacy":
            assert restored.annotation_failure_reason == ""
        client.chat_with_tools.return_value = response
        assert (await executor_loop.start_turn(restored)).response is response
        restored_agent.execution_intervention_callback.assert_not_awaited()
        assert restored.step_count == 3
        assert restored.max_steps == 5
