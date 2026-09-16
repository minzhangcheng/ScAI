"""Regression cases reproduced during the figure follow-up review."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi.figures import FiguresMixin
from backend.models.schemas import AnalysisState, AnalysisPlan
from backend.services.figure_contract import FigureContractAudit, MissingFigure, RequiredFigure
from backend.services.figure_revision_tasks import materialize_missing_figure_tasks, candidate_figures_for_task
from backend.tests.test_figure_followup import _add_eligible_source, _candidate
from backend.workflow.post_execution import PostExecutionReviewMixin


def setup_task():
    state = AnalysisState(user_id="u", current_round=3)
    state.current_plan = AnalysisPlan(id="plan-1", phase="innovation", steps=[])
    _add_eligible_source(state)
    requirement = RequiredFigure(key="plan.2.1", figure_type="volcano",
                                 description="Plot the chosen contrast", plan_id="plan-1",
                                 plan_step_id=2)
    contract = FigureContractAudit(missing=[MissingFigure(requirement)])
    task = materialize_missing_figure_tasks(state, contract)[0]
    return state, contract, task


def test_candidate_search_excludes_another_plan():
    state, _, task = setup_task()
    candidate = _candidate(logical_id=task.logical_figure_id, requirement_key=task.plan_requirement_key)
    candidate.plan_id = "other-plan"
    state.publication_figures = [candidate]
    assert candidate_figures_for_task(state, task) == []


def test_pi_omission_is_not_recreated_on_node_retry():
    state, contract, task = setup_task()
    task.status = "abandoned"
    task.pi_disposition = "omit"
    task.pi_report_note = "No useful plot can be made from the available result."
    assert materialize_missing_figure_tasks(state, contract) == []


@pytest.mark.asyncio
async def test_visual_fallback_uses_available_record_preview(monkeypatch):
    state, _, task = setup_task()
    candidate = _candidate(logical_id=task.logical_figure_id, requirement_key=task.plan_requirement_key)
    state.publication_figures = [candidate]
    assert candidate_figures_for_task(state, task)

    async def vision(*args, **kwargs):
        if not all(kwargs["images"]):
            raise ValueError("Empty image input")
        return SimpleNamespace(content=json.dumps({"selected_figure_id": candidate.id,
            "rationale": "Best readable plot", "report_note": "Labels remain crowded"}))

    monkeypatch.setattr("backend.agents.pi.figures.replay_safe_vision_call", vision)
    agent = FiguresMixin()
    agent.has_vision = True
    agent.llm_client = object()
    agent.model = "test"
    agent.system_prompt = ""
    agent._set_llm_context = lambda: None
    agent._safe_parse_json = lambda text, default: json.loads(text)
    selected = await agent.select_fallback_figure(state, task, [candidate])
    assert selected.get("selected_figure_id") == candidate.id


@pytest.mark.asyncio
async def test_disposition_retry_keeps_the_task_and_schema():
    state, _, task = setup_task()
    agent = FiguresMixin()
    agent.chat_without_history = AsyncMock(side_effect=["[]", "[]"])
    await agent.decide_figure_followups(state, [task], {task.id: 0})
    retry_prompt = agent.chat_without_history.await_args_list[1].args[0]
    assert "Plot the chosen contrast" in retry_prompt
    assert "targeted_redraw" in retry_prompt


@pytest.mark.asyncio
async def test_blocked_followup_does_not_repeat_execution_on_node_retry():
    state, contract, task = setup_task()
    host = PostExecutionReviewMixin()
    host.session_id = state.id
    host.state_callback = AsyncMock(return_value=True)
    host._figure_review_callback = AsyncMock()
    host._supervisor_callback = AsyncMock()
    host.pi_agent = SimpleNamespace(
        decide_figure_followups=AsyncMock(return_value=[{
            "task_id": task.id, "disposition": "targeted_redraw",
            "instruction": "Plot the chosen contrast", "rationale": "Useful plot",
            "report_note": ""}]),
        select_fallback_figure=AsyncMock(return_value={}),
    )
    host.bioinformatics_agent = SimpleNamespace(run_autonomous_analysis=AsyncMock(
        return_value={"status": "failed", "code_history": [], "analysis_log": []}))
    await host._run_figure_followup(state, contract, "full")
    assert task.status == "blocked"
    await host._run_figure_followup(state, contract, "full")
    assert host.bioinformatics_agent.run_autonomous_analysis.await_count == 1


@pytest.mark.asyncio
async def test_needs_new_analysis_is_durable_context_without_executor_dispatch():
    from backend.services.figure_followup import ANALYSIS_REQUESTS_KEY
    from backend.services.plan_scope import render_innovation_scope_context

    state, contract, task = setup_task()
    host = PostExecutionReviewMixin()
    host.session_id = state.id
    host.state_callback = AsyncMock(return_value=True)
    host._figure_review_callback = AsyncMock()
    host._supervisor_callback = AsyncMock()
    host.pi_agent = SimpleNamespace(
        decide_figure_followups=AsyncMock(return_value=[{
            "task_id": task.id,
            "disposition": "needs_new_analysis",
            "instruction": "Recompute the requested embedding with another parameter.",
            "rationale": "The figure requires a new measurement.",
            "report_note": "Requires separate scientific approval.",
        }]),
        select_fallback_figure=AsyncMock(),
    )
    host.bioinformatics_agent = SimpleNamespace(
        run_autonomous_analysis=AsyncMock()
    )

    result = await host._run_figure_followup(state, contract, "full")

    requests = state.sandbox_state[ANALYSIS_REQUESTS_KEY]
    assert len(requests) == 1
    assert requests[0]["task_id"] == task.id
    assert requests[0]["status"] == "pending_review"
    assert task.status == "blocked"
    assert result["targeted_task_ids"] == []
    host.bioinformatics_agent.run_autonomous_analysis.assert_not_awaited()
    context = render_innovation_scope_context(state)
    assert requests[0]["id"] in context
    assert requests[0]["request"] in context


@pytest.mark.asyncio
async def test_followup_defers_tasks_beyond_one_scope_without_finishing_them():
    state = AnalysisState(user_id="bounded-followup", current_round=3)
    state.current_plan = AnalysisPlan(id="plan-1", phase="innovation", steps=[])
    _add_eligible_source(state)
    missing = [
        MissingFigure(RequiredFigure(
            key=f"plan.2.{index}",
            figure_type="volcano",
            description=f"Render contrast {index}",
            plan_id="plan-1",
            plan_step_id=2,
        ))
        for index in range(1, 14)
    ]
    contract = FigureContractAudit(missing=missing)
    host = PostExecutionReviewMixin()
    host.session_id = state.id
    host.state_callback = AsyncMock(return_value=True)
    host._figure_review_callback = AsyncMock()
    host._supervisor_callback = AsyncMock()

    async def decide_all(_state, tasks, _candidate_counts):
        return [{
            "task_id": task.id,
            "disposition": "targeted_redraw",
            "instruction": "Render the already computed contrast.",
            "rationale": "A bounded redraw is useful.",
            "report_note": "",
        } for task in tasks]

    host.pi_agent = SimpleNamespace(
        decide_figure_followups=AsyncMock(side_effect=decide_all),
        select_fallback_figure=AsyncMock(return_value={}),
    )
    host.bioinformatics_agent = SimpleNamespace(
        run_autonomous_analysis=AsyncMock(return_value={
            "status": "failed", "code_history": [], "analysis_log": [],
        })
    )

    result = await host._run_figure_followup(state, contract, "full")

    assert len(result["targeted_task_ids"]) == 12
    deferred = next(
        task for task in state.figure_revision_tasks
        if task.id not in result["targeted_task_ids"]
    )
    assert deferred.status == "pending"
    assert deferred.followup_started_at is None
    assert deferred.followup_finished_at is None


@pytest.mark.asyncio
async def test_two_missing_outputs_remain_distinct_through_real_figure_dispatch(tmp_path):
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_collaboration
    from backend.models.schemas import PlanStep, FigureRequirement, CodeExecution
    from backend.services.figure_contract import audit_figure_contract
    from backend.services.report.artifacts import eligible_artifacts
    from backend.tests.test_figure_revision_gate import _Agent, _FakeSandbox, _review, CLEAN

    state, _, _ = setup_task()
    state.figure_revision_tasks = []
    state.current_plan.steps = [PlanStep(
        id=2, title="Two contrasts", method="Compare groups", figure_requirements=[
            FigureRequirement(figure_type="volcano", description="Two distinct contrasts", minimum_count=2)
        ], status="completed",
    )]
    state.current_plan.status = "approved"
    state.current_plan.round_number = 3
    agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    state.code_history[0].code = "compute_contrasts()"
    agent.code_history = [state.code_history[0]]
    agent.figure_plan_binding_callback = AsyncMock(return_value={
        "decision": "accept", "plan_requirement_key": "plan.2.1", "reason": "Matches contrast"
    })
    executor_collaboration.bind_facade(facade)
    contract = audit_figure_contract(state, include_stage_baseline=False)
    tasks = materialize_missing_figure_tasks(state, contract)
    assert len(tasks) == 2
    assert len({task.logical_figure_id for task in tasks}) == 2
    context = SimpleNamespace(
        state=state, agent=agent, review_callback=AsyncMock(return_value=_review(approved=True, score=9)),
        figures_rendered=0, step_figures=[], messages=[], analysis_log=[], step_count=1,
        agent_reasoning="Render the requested contrast",
    )
    for index, task in enumerate(tasks):
        await executor_collaboration.handle_publication_figure(context, SimpleNamespace(
            id=f"tool-{index}", arguments={
                "code": "plot()", "figure_name": f"contrast_{index}", "description": "Contrast result",
                "figure_type": "volcano", "plan_step_id": 2, "evidence_execution_id": "source",
                "logical_figure_id": task.logical_figure_id,
            }
        ))
        assert task.status == "resolved"
        if index == 0:
            assert tasks[1].status == "pending"
            assert not audit_figure_contract(state, include_stage_baseline=False).passed
    assert len(eligible_artifacts(state.publication_figures, state=state)) == 2
    assert audit_figure_contract(state, include_stage_baseline=False).passed


def test_new_plan_figures_do_not_hide_previous_plan_figures():
    from backend.services.figure_revision_tasks import apply_pi_fallback_selection
    from backend.services.report.artifacts import eligible_artifacts
    state, contract, first_task = setup_task()
    first = _candidate(logical_id=first_task.logical_figure_id, requirement_key="plan.2.1")
    state.publication_figures.append(first)
    assert apply_pi_fallback_selection(state, first_task, selected_figure_id=first.id,
        rationale="Useful", report_note="Crowded labels") is first
    state.current_plan.id = "plan-2"
    second_req = RequiredFigure(key="plan.2.1", figure_type="volcano",
        description="Another contrast", plan_id="plan-2")
    second_task = materialize_missing_figure_tasks(state,
        FigureContractAudit(missing=[MissingFigure(second_req)]))[0]
    second = _candidate(logical_id=second_task.logical_figure_id, requirement_key="plan.2.1")
    second.id, second.plan_id = "candidate-2", "plan-2"
    state.publication_figures.append(second)
    assert apply_pi_fallback_selection(state, second_task, selected_figure_id=second.id,
        rationale="Useful", report_note="Crowded labels") is second
    assert {fig.id for fig in eligible_artifacts(state.publication_figures)} == {first.id, second.id}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_real_executor_keeps_main_checkpoint_and_receipts(monkeypatch, tmp_path, cancel):
    import asyncio
    from backend.agents.bioinformatics import executor_loop
    from backend.models.schemas import CodeExecution, AnalysisStep
    from backend.tests.test_bioinfo_context_scopes import _bare_agent
    from backend.utils.checkpoint import CheckpointManager
    from backend.services.figure_followup import (
        build_figure_followup_scope,
        persist_figure_followup_scope,
    )

    class _DisposableRuntime:
        def __init__(self, main_sandbox, _state):
            self.main_sandbox = main_sandbox
            self.sandbox = MagicMock()
            self.sandbox.state_dir = tmp_path / "render-states"
            self.sandbox.working_dir = tmp_path / "render-workspace"
            self.sandbox.state_dir.mkdir(exist_ok=True)
            self.sandbox.working_dir.mkdir(exist_ok=True)
            self.sandbox.runtime_revision = 1

        async def __aenter__(self):
            return self.sandbox

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        "backend.services.figure_followup_runtime.FigureFollowupRuntime",
        _DisposableRuntime,
    )

    agent = _bare_agent(monkeypatch, tmp_path)
    agent.sandbox.runtime_revision = 1
    state, _, _ = setup_task()
    parent_receipt = CodeExecution(
        id="main", code="calculate_result()", round_number=3, plan_id="plan-1",
        evidence_schema_version=2, plan_step_ids=[2],
        plan_binding_status="confirmed", executor_run_kind="ordinary",
        execution_status="succeeded",
    )
    state.code_history = [parent_receipt]
    scope = build_figure_followup_scope(
        state, state.figure_revision_tasks, run_scope_id="figure-test-scope"
    )
    persist_figure_followup_scope(state, scope)
    parent_history = agent.code_history = [parent_receipt]
    parent_context = agent.analysis_context = {"computed": "main-result"}
    manager = CheckpointManager(state.id, str(agent.sandbox.state_dir))
    await manager.save({"step_name": "main", "findings": ["original scientific finding"]})
    original_bytes = manager.checkpoint_path.read_bytes()

    async def execute_figure(context):
        assert context.agent.code_history == [parent_receipt]
        assert context.checkpoint_manager.checkpoint_path != manager.checkpoint_path
        context.agent.code_history.append(CodeExecution(id="figure", code="plot_result()"))
        context.analysis_log.append(AnalysisStep(step_number=1, step_type="execute_code"))
        await context.save_checkpoint("figure-only")
        if cancel:
            raise asyncio.CancelledError()
        return {"code_history": list(context.agent.code_history), "status": "completed"}

    monkeypatch.setattr(executor_loop, "run_turn_loop", execute_figure)
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await agent.run_autonomous_analysis(state, "Plot result", figure_followup=True,
                resume_from_checkpoint=False, max_steps=12,
                run_scope_id="figure-test-scope")
    else:
        await agent.run_autonomous_analysis(state, "Plot result", figure_followup=True,
            resume_from_checkpoint=False, max_steps=12,
            run_scope_id="figure-test-scope")
    assert manager.checkpoint_path.read_bytes() == original_bytes
    assert agent.code_history is parent_history
    assert agent.analysis_context is parent_context
    assert {item.id for item in state.code_history} == {"main"}
    assert state.analysis_log == []


@pytest.mark.asyncio
async def test_node_retry_returns_persisted_main_findings_without_execution():
    from backend.workflow import agents as facade
    from backend.workflow.agent_nodes import bioinformatics_node as node
    from backend.workflow.helpers import cache_store
    from backend.models.schemas import CodeExecution

    node.bind_facade(facade)
    state, _, _ = setup_task()
    cache_store(state, node._execution_cache_key(state, "full"), {
        "findings": ["original scientific finding"], "status": "completed",
        "code_history": [CodeExecution(id="main", code="compute()").model_dump(mode="json")],
    })
    state = AnalysisState.model_validate_json(state.model_dump_json())
    host = SimpleNamespace(bioinformatics_agent=SimpleNamespace(run_autonomous_analysis=AsyncMock()))
    ctx = await node.prepare_bioinformatics_instruction(host,
        {"analysis_state": state, "planning_stage": "full"}, state)
    result = await node.execute_bioinformatics_analysis(host, ctx)
    assert result["findings"] == ["original scientific finding"]
    assert result["code_history"][0].id == "main"
    host.bioinformatics_agent.run_autonomous_analysis.assert_not_awaited()


@pytest.mark.asyncio
async def test_followup_prompt_includes_later_pi_decision_and_execution():
    from backend.models.schemas import PIDecisionRecord, CodeExecution, AnalysisStep
    state, _, task = setup_task()
    state.pi_decision_records = [PIDecisionRecord(question="Plot choice?",
        response="Use the paired contrast only", plan_id="plan-1", round_number=3)]
    state.code_history = [CodeExecution(id="computed-contrast", code="contrasts = compute()",
        plan_id="plan-1", round_number=3)]
    state.analysis_log = [AnalysisStep(step_number=3, step_type="report_findings",
        round_number=3, findings_summary="Paired contrast is available")]
    agent = FiguresMixin()
    agent.chat_without_history = AsyncMock(return_value="[]")
    await agent.decide_figure_followups(state, [task], {task.id: 0})
    for call in agent.chat_without_history.await_args_list:
        prompt = call.args[0]
        assert "Use the paired contrast only" in prompt
        assert "computed-contrast" in prompt
        assert "Paired contrast is available" in prompt
