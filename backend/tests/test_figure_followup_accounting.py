"""Figure attempts stay outside the analytical settlement/evidence ledger."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.schemas import AnalysisPhase, AnalysisState, RoundResult


def _executor_context(state, *, run_kind, run_scope_id, settlement_scope_id):
    return SimpleNamespace(
        llm_error=None,
        plan_revision_needed=False,
        completed=True,
        evidence_gate_degraded=False,
        analysis_state=state,
        planning_stage="full",
        analysis_log=[],
        step_count=1,
        attempt_start_step=0,
        findings=[],
        figures=[],
        attempt_execution_ids=[f"{run_kind}-attempt"],
        executor_run_kind=run_kind,
        executor_run_scope_id=run_scope_id,
        executor_settlement_scope_id=settlement_scope_id,
        accepted_finding_records=[],
        agent=SimpleNamespace(code_history=[]),
        mechanism_programs=[],
        reported_evidence_ids=[],
        evidence_gate_findings=[],
        force_break=False,
        plan_revision_context={},
        supplementary_request_id=("approved-request" if run_kind == "supplementary" else None),
        resolved_adata_source_context={},
        adata_source=None,
        checkpoint_manager=None,
        save_checkpoint=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_finalizer_routes_figure_attempts_to_separate_persisted_ledger(
    monkeypatch,
):
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_auxiliary
    from backend.services import session_store

    executor_auxiliary.bind_facade(facade)
    save_session = AsyncMock()
    monkeypatch.setattr(session_store, "async_save_session", save_session)
    state = AnalysisState(
        user_id="figure-ledger",
        current_round=3,
        bioinformatics_settlement_attempt_ids={
            "ordinary-scope": ["ordinary-attempt"],
        },
    )
    context = _executor_context(
        state,
        run_kind="figure_followup",
        run_scope_id="figure-scope",
        settlement_scope_id="figure-settlement-scope",
    )

    result = await executor_auxiliary.finalize_run(context)

    assert result["figure_attempt_execution_ids"] == ["figure_followup-attempt"]
    assert result["settlement_attempt_execution_ids"] == []
    assert result["executor_settlement_scope_id"] is None
    assert state.figure_attempt_execution_ids == {
        "figure-scope": ["figure_followup-attempt"],
    }
    assert state.bioinformatics_settlement_attempt_ids == {
        "ordinary-scope": ["ordinary-attempt"],
    }
    assert save_session.await_count == 1

    restored = AnalysisState.model_validate_json(state.model_dump_json())
    assert restored.figure_attempt_execution_ids == state.figure_attempt_execution_ids
    assert restored.bioinformatics_settlement_attempt_ids == (
        state.bioinformatics_settlement_attempt_ids
    )


def _figure_tool_dispatch_case(render_handler, *, run_scope_id="render-scope"):
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_collaboration
    from backend.agents.bioinformatics.executor_dispatch import (
        ToolDispatchContext,
        dispatch_tool as dispatch,
    )
    from backend.models.schemas import AnalysisPlan
    from backend.services.figure_contract import FigureContractAudit, MissingFigure, RequiredFigure
    from backend.services.figure_followup import (
        build_figure_followup_scope,
        persist_figure_followup_scope,
    )
    from backend.services.figure_revision_tasks import materialize_missing_figure_tasks
    from backend.tests.test_figure_followup import _add_eligible_source

    executor_collaboration.bind_facade(facade)
    state = AnalysisState(user_id="figure-attempt-ledger", current_round=3)
    state.current_plan = AnalysisPlan(
        id="figure-attempt-plan", phase="innovation", status="approved", steps=[],
    )
    receipt = _add_eligible_source(state, execution_id="figure-source")
    requirement = RequiredFigure(
        key="plan.2.1", figure_type="volcano", description="Plot the contrast",
        plan_id=state.current_plan.id, plan_step_id=2,
    )
    task = materialize_missing_figure_tasks(
        state, FigureContractAudit(missing=[MissingFigure(requirement)])
    )[0]
    scope = build_figure_followup_scope(
        state, [task], run_scope_id=run_scope_id,
    )
    persist_figure_followup_scope(state, scope)
    context = SimpleNamespace(
        state=state,
        agent=SimpleNamespace(
            code_history=[receipt],
            _notify=AsyncMock(),
            _handle_generate_publication_figure=render_handler,
        ),
        review_callback=None,
        executor_run_kind="figure_followup",
        executor_run_scope_id=run_scope_id,
        executor_settlement_scope_id="must-not-settle",
        figure_followup_scope=scope,
        attempt_execution_ids=[],
        messages=[],
        analysis_log=[],
        step_count=1,
        agent_reasoning="Render an existing result.",
        has_figure_work=False,
        figures_rendered=0,
        step_figures=[],
    )
    async def invoke(tool_call):
        return await executor_collaboration.handle_publication_figure(
            context, tool_call,
        )

    dispatch_context = ToolDispatchContext(
        messages=context.messages,
        known_tool_names=frozenset({"generate_publication_figure"}),
        logger=MagicMock(),
        handlers={"generate_publication_figure": invoke},
        allowed_tool_names=frozenset({"generate_publication_figure"}),
    )
    return state, task, context, dispatch_context, dispatch


def _figure_tool_call(call_id, task):
    from backend.utils.llm_client_parts.responses import ToolCall

    return ToolCall(
        id=call_id,
        name="generate_publication_figure",
        arguments={
            "code": "ax.scatter([0, 1], [1, 0])",
            "figure_name": "contrast_plot",
            "description": "Plot the existing contrast result.",
            "figure_type": "volcano",
            "evidence_execution_id": "figure-source",
            "plan_step_id": task.plan_step_id,
            "logical_figure_id": task.logical_figure_id,
            "figure_task_id": task.id,
        },
    )


@pytest.mark.asyncio
async def test_real_figure_handler_persists_attempts_before_render_and_keeps_them_out_of_settlement():
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_auxiliary
    from backend.services import session_store

    saved_before_render = []

    async def render(**_kwargs):
        persisted = session_store.load_session_from_disk(state.id)
        saved_before_render.append(persisted)
        ids = persisted.figure_attempt_execution_ids["render-scope"]
        assert ids
        return next(outcomes)

    outcomes = iter((
        "figure rendered", "renderer reported an error", "retry rendered",
        "replay rendered",
    ))
    render_handler = AsyncMock(side_effect=render)
    state, task, context, dispatch_context, dispatch = _figure_tool_dispatch_case(
        render_handler,
    )
    await session_store.async_save_session(str(state.id), state, force=True)

    for call_id in ("tool-success", "tool-failure", "tool-retry"):
        result = await dispatch(
            dispatch_context, _figure_tool_call(call_id, task),
        )
        assert result.status.value == "handled"

    attempt_ids = context.attempt_execution_ids
    assert len(attempt_ids) == 3
    assert len(set(attempt_ids)) == 3
    assert all(item.startswith("figure_attempt_") and len(item) == 47 for item in attempt_ids)
    assert len(saved_before_render) == 3
    assert [item.figure_attempt_execution_ids["render-scope"] for item in saved_before_render] == [
        attempt_ids[:1], attempt_ids[:2], attempt_ids[:3],
    ]

    # A replay with the same provider tool-call identity resolves to the same audit ID.
    replay = await dispatch(
        dispatch_context, _figure_tool_call("tool-retry", task),
    )
    assert replay.status.value == "handled"
    assert context.attempt_execution_ids == attempt_ids
    assert session_store.load_session_from_disk(str(state.id)).figure_attempt_execution_ids[
        "render-scope"
    ] == attempt_ids

    executor_auxiliary.bind_facade(facade)
    finalizer_context = _executor_context(
        state,
        run_kind="figure_followup",
        run_scope_id="render-scope",
        settlement_scope_id="must-not-settle",
    )
    finalizer_context.attempt_execution_ids = list(attempt_ids)
    result = await executor_auxiliary.finalize_run(finalizer_context)
    ordered_attempt_ids = sorted(set(attempt_ids))

    assert result["figure_attempt_execution_ids"] == ordered_attempt_ids
    assert result["settlement_attempt_execution_ids"] == []
    assert state.figure_attempt_execution_ids["render-scope"] == ordered_attempt_ids
    assert state.bioinformatics_settlement_attempt_ids == {}
    restored = session_store.load_session_from_disk(str(state.id))
    assert restored.figure_attempt_execution_ids["render-scope"] == ordered_attempt_ids
    assert restored.bioinformatics_settlement_attempt_ids == {}


@pytest.mark.asyncio
async def test_cancelled_render_keeps_pre_dispatch_attempt_durable():
    import asyncio
    from backend.services import session_store

    async def cancel_render(**_kwargs):
        persisted = session_store.load_session_from_disk(str(state.id))
        assert len(persisted.figure_attempt_execution_ids["render-scope"]) == 1
        raise asyncio.CancelledError()

    state, task, context, dispatch_context, dispatch = _figure_tool_dispatch_case(
        AsyncMock(side_effect=cancel_render),
    )
    await session_store.async_save_session(str(state.id), state, force=True)

    with pytest.raises(asyncio.CancelledError):
        await dispatch(dispatch_context, _figure_tool_call("tool-cancel", task))

    restored = session_store.load_session_from_disk(str(state.id))
    assert restored.figure_attempt_execution_ids["render-scope"] == (
        context.attempt_execution_ids
    )
    assert restored.bioinformatics_settlement_attempt_ids == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledgment", [False, None])
async def test_figure_render_fails_closed_when_attempt_save_is_not_acknowledged(
    monkeypatch, acknowledgment,
):
    from backend.agents.bioinformatics.executor_dispatch import DispatchStatus
    from backend.services import session_store

    renderer = AsyncMock(return_value="should not be dispatched")
    state, task, context, dispatch_context, dispatch = _figure_tool_dispatch_case(renderer)
    monkeypatch.setattr(
        session_store, "async_save_session", AsyncMock(return_value=acknowledgment),
    )

    result = await dispatch(dispatch_context, _figure_tool_call("tool-unpersisted", task))

    assert result.status is DispatchStatus.HANDLED
    assert "was not started" in context.messages[-1]["content"]
    renderer.assert_not_awaited()
    assert context.attempt_execution_ids
    assert state.figure_attempt_execution_ids["render-scope"] == context.attempt_execution_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("run_kind", ["ordinary", "supplementary"])
async def test_ordinary_and_supplementary_attempts_keep_settlement_accounting(
    monkeypatch, run_kind,
):
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_auxiliary
    from backend.services import session_store

    executor_auxiliary.bind_facade(facade)
    monkeypatch.setattr(session_store, "async_save_session", AsyncMock())
    state = AnalysisState(user_id=f"{run_kind}-ledger", current_round=3)
    context = _executor_context(
        state,
        run_kind=run_kind,
        run_scope_id=f"{run_kind}-scope",
        settlement_scope_id=f"{run_kind}-settlement",
    )

    result = await executor_auxiliary.finalize_run(context)

    assert result["settlement_attempt_execution_ids"] == [f"{run_kind}-attempt"]
    assert result["figure_attempt_execution_ids"] == []
    assert state.bioinformatics_settlement_attempt_ids == {
        f"{run_kind}-settlement": [f"{run_kind}-attempt"],
    }
    assert state.figure_attempt_execution_ids == {}


@pytest.mark.asyncio
async def test_round_boundary_excludes_figure_attempts_from_evidence_and_checkpoint(
    monkeypatch,
):
    from backend.workflow import adata_checkpoint_hooks
    from backend.workflow.synthesis_parts import phases

    capture = {}
    outcome = SimpleNamespace(available=True, disclosure=None, record=None)

    async def capture_boundary(_self, _analysis_state, **kwargs):
        capture.update(kwargs)
        kwargs["before_state_save"](outcome)
        return outcome

    monkeypatch.setattr(
        adata_checkpoint_hooks, "capture_and_persist_boundary", capture_boundary
    )
    monkeypatch.setattr(
        adata_checkpoint_hooks,
        "clear_settlement_scope_callback",
        lambda _state, _scope: lambda _outcome: False,
    )
    monkeypatch.setattr(phases, "logger", MagicMock(), raising=False)
    monkeypatch.setattr(phases, "AnalysisPhase", AnalysisPhase, raising=False)
    monkeypatch.setattr(
        phases, "derive_round_completion_status", lambda *_args: None, raising=False
    )
    monkeypatch.setattr(
        phases, "current_lineage_id", lambda _state: "main", raising=False
    )
    monkeypatch.setattr(
        phases, "restore_provenance_id", lambda _state: None, raising=False
    )
    monkeypatch.setattr(
        phases, "consume_restore_provenance", lambda _state: None, raising=False
    )
    monkeypatch.setattr(
        phases,
        "_upsert_round_result",
        lambda entries, result, **_kwargs: entries.append(result) or result,
        raising=False,
    )
    monkeypatch.setattr(
        phases, "_should_build_innovation_segment", lambda *_args: False, raising=False
    )
    state = AnalysisState(
        user_id="round-boundary-ledger",
        current_round=2,
        analysis_phase=AnalysisPhase.EXPLORATION,
        figure_attempt_execution_ids={"figure-scope": ["figure-attempt"]},
        bioinformatics_settlement_attempt_ids={
            "ordinary-scope": ["ordinary-attempt"],
        },
    )
    workflow = SimpleNamespace(
        session_id=state.id,
        state_callback=None,
        _compact_checkpoints=AsyncMock(),
    )
    round_result = RoundResult(round_number=2)
    context = {
        "state": {},
        "analysis_state": state,
        "round_result": round_result,
        "bioinformatics_result": {
            "evidence_execution_ids": [],
            "settlement_attempt_execution_ids": [],
            "figure_attempt_execution_ids": ["figure-attempt"],
            "executor_settlement_scope_id": "figure-scope",
        },
    }

    await phases.persist_round_result(workflow, context)

    assert round_result.evidence_execution_ids == []
    assert capture["execution_ids_at_capture"] == ()
    assert capture["attempt_execution_ids_at_capture"] == ()
    assert state.bioinformatics_settlement_attempt_ids == {
        "ordinary-scope": ["ordinary-attempt"],
    }
