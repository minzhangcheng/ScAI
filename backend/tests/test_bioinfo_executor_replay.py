"""Replay incomplete tool responses before restarting the workflow node."""

import asyncio
from types import SimpleNamespace
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
async def test_each_code_request_receives_latest_working_adata(replay_context):
    context, client, response = replay_context
    client.chat_with_tools.return_value = response
    context.resolved_adata_source_context["active_adata_var"] = "adata"
    original_messages = list(context.messages)

    await executor_loop.start_turn(context)
    first = client.chat_with_tools.await_args.kwargs["messages"]
    assert "Default variable: `adata`" in first[-1]["content"]

    context.resolved_adata_source_context["active_adata_var"] = "adata_hep"
    await executor_loop.start_turn(context)
    second = client.chat_with_tools.await_args.kwargs["messages"]
    assert "Default variable: `adata_hep`" in second[-1]["content"]
    assert context.messages == original_messages


@pytest.mark.asyncio
async def test_chat_uses_pi_selected_live_adata_instead_of_prior_plan(monkeypatch, tmp_path):
    from backend.models.schemas import AdataSourceSelection
    from backend.tests.test_bioinfo_context_scopes import _plan

    agent = _bare_agent(monkeypatch, tmp_path)
    agent.is_live_adata = AsyncMock(return_value=True)
    state = AnalysisState(id="test-session", user_id="test-user")
    state.current_plan = _plan(plan_id="older-plan")
    state.current_plan.adata_source = AdataSourceSelection(
        mode="checkpoint_copy", checkpoint_id="older-checkpoint"
    )
    executor_loop.bind_facade(facade)
    context = await executor_loop.prepare_loop_context(
        agent, state, "Analyze the selected cells", max_steps=5,
        run_kind="chat", run_scope_id="chat-run", resume_from_checkpoint=False,
        adata_source={
            "selection": {"mode": "current"}, "active_adata_var": "selected_cells",
            "owner_id": "chat-review", "rationale": "Matches the request",
        },
    )
    assert context.resolved_adata_source_context["mode"] == "current"
    assert context.resolved_adata_source_context["active_adata_var"] == "selected_cells"
    assert context.resolved_adata_source_context["chat_source_owner_id"] == "chat-review"
    assert "Approved Analysis Plan" not in str(context.messages)
    agent.is_live_adata.assert_awaited_with("selected_cells")


@pytest.mark.asyncio
async def test_chat_passes_pi_checkpoint_and_inherited_preparation(monkeypatch, tmp_path):
    from backend.services import adata_checkpoints

    agent = _bare_agent(monkeypatch, tmp_path)
    agent.is_live_adata = AsyncMock(return_value=True)
    preparation = SimpleNamespace(
        preparation_id="prep-chat", binding_name="scai_work_chat",
        selection_signature="signature", scope_id="scope-chat",
        binding_id="binding-chat", binding_identity_status="exact",
        adata_lineage_id="lineage", loaded_checkpoint_id="checkpoint",
        ancestor_checkpoint_id="checkpoint",
    )
    prepare = AsyncMock(return_value=preparation)
    monkeypatch.setattr(adata_checkpoints, "ensure_adata_source_prepared", prepare)
    executor_loop.bind_facade(facade)
    context = await executor_loop.prepare_loop_context(
        agent, AnalysisState(id="test-session", user_id="test-user"),
        "Analyze a prior state", max_steps=5,
        run_kind="chat", run_scope_id="chat-run", resume_from_checkpoint=False,
        adata_source={
            "selection": {"mode": "checkpoint_copy", "checkpoint_id": "checkpoint"},
            "active_adata_var": None, "owner_id": "chat-review",
            "preparation_id": "prep-chat", "rationale": "Use this stage",
        },
    )
    assert prepare.await_args.kwargs["owner_kind"] == "chat"
    assert prepare.await_args.kwargs["owner_id"] == "chat-review"
    assert prepare.await_args.kwargs["inherited_preparation_id"] == "prep-chat"
    assert context.resolved_adata_source_context["active_adata_var"] == "scai_work_chat"


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
