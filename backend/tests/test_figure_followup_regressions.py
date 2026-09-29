"""Regression cases reproduced during the figure follow-up review."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState, AnalysisPlan
from backend.services.figure_contract import FigureContractAudit, MissingFigure, RequiredFigure
from backend.services.figure_revision_tasks import materialize_missing_figure_tasks
from backend.tests.test_figure_followup import _add_eligible_source


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


@pytest.mark.asyncio
async def test_two_missing_outputs_remain_distinct_through_real_figure_dispatch(tmp_path):
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_collaboration
    from backend.models.schemas import PlanStep, FigureRequirement
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


@pytest.mark.asyncio
async def test_followup_revision_lands_on_owner_filesystem_before_publish(
    tmp_path, monkeypatch
):
    from backend.agents.bioinformatics import figures
    from backend.agents.bioinformatics.figure_dependencies import FigureDependencies
    from backend.agents.bioinformatics.figure_revision import execute_revision_candidate
    from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes
    from backend.tests.test_figure_revision_gate import _Agent, _FakeSandbox, CLEAN

    renderer_root = tmp_path / "renderer"
    owner_figures = tmp_path / "owner" / "figures"
    owner_figures.mkdir(parents=True)
    jpg_path = owner_figures / "f.jpg"
    pdf_path = owner_figures / "f.pdf"
    previous_jpg = valid_image_bytes("JPEG", min_bytes=4000)
    previous_pdf = valid_pdf_bytes()
    jpg_path.write_bytes(previous_jpg)
    pdf_path.write_bytes(previous_pdf)

    publish = figures._publish_render_attempt

    def publish_on_owner_filesystem(attempt_dir, staged_jpg, staged_pdf, final_jpg, final_pdf):
        assert attempt_dir.is_relative_to(owner_figures)
        assert staged_jpg.parent == staged_pdf.parent == attempt_dir
        assert final_jpg == jpg_path
        assert final_pdf == pdf_path
        return publish(attempt_dir, staged_jpg, staged_pdf, final_jpg, final_pdf)

    monkeypatch.setattr(figures, "_publish_render_attempt", publish_on_owner_filesystem)
    attempt = SimpleNamespace(
        request=SimpleNamespace(producer_run_kind="figure_followup"),
        agent=_Agent(_FakeSandbox(renderer_root, [CLEAN])),
        deps=FigureDependencies.from_facade(),
        figures_dir=owner_figures,
        safe_name="f",
        jpg_path=jpg_path,
        pdf_path=pdf_path,
    )

    result, error = await execute_revision_candidate(attempt, "plot()")

    assert result.error is None
    assert error is None
    assert jpg_path.read_bytes() != previous_jpg
    assert pdf_path.read_bytes() != previous_pdf
    assert not (renderer_root / "figures" / ".attempts").exists()
    assert not (owner_figures / ".attempts").exists()


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
