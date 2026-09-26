"""Context isolation and plan-revision resume tests for BioinformaticsAgent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents import bioinformatics_agent as bioinformatics_facade
from backend.models.schemas import (
    AdataSourceSelection,
    AgentType,
    AnalysisPlan,
    AnalysisState,
    PlanStep,
)
from backend.tests.split_facade import setattr_facade
from backend.utils.llm_client import ToolCall


def _plan(*, plan_id: str, revision_of: str | None = None) -> AnalysisPlan:
    return AnalysisPlan(
        id=plan_id,
        phase="exploration",
        stage="full",
        overview="Run the bounded revised workflow",
        steps=[
            PlanStep(id=1, title="QC", method="scanpy.pp.filter_cells", status="completed"),
            PlanStep(id=2, title="DE", method="scanpy.tl.rank_genes_groups"),
        ],
        key_decision_points=["Review DE evidence"],
        status="approved",
        revision_of=revision_of,
        revision_reason="The observed data require a different remaining method.",
    )


def _bare_agent(monkeypatch, tmp_path) -> BioinformaticsAgent:
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.session_id = "test-session"
    agent.message_history = []
    agent.code_history = []
    agent.analysis_context = {}
    agent.last_result = None
    agent._recent_sup_feedback = set()
    agent._pause_event = None
    agent.literature_callback = None
    agent.pi_callback = None
    agent.annotate_clusters_callback = None
    agent.message_callback = AsyncMock()
    agent.figure_review_callback = None
    agent.supervisor_callback = None
    agent.technique_agent = None
    agent.species_validation_callback = None
    agent.marker_db_intervention_callback = None
    agent._provider = MagicMock()
    agent._provider.get_config_revision.return_value = 0
    agent._last_config_revision = 0
    agent.agent_type = AgentType.BIOINFORMATICS

    monkeypatch.setattr(BioinformaticsAgent, "model", property(lambda self: "test-model"))
    monkeypatch.setattr(BioinformaticsAgent, "system_prompt", property(lambda self: "system"))
    monkeypatch.setattr(
        BioinformaticsAgent,
        "config",
        property(lambda self: MagicMock(temperature=0.1)),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_static_system_prompt",
        property(lambda self: "static system"),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_build_dynamic_context",
        lambda self, state: "bounded dynamic context",
    )
    monkeypatch.setattr(BioinformaticsAgent, "_set_llm_context", lambda self: None)
    monkeypatch.setattr(BioinformaticsAgent, "_notify", AsyncMock())

    agent.sandbox = MagicMock()
    agent.sandbox.state_dir = tmp_path / "states"
    agent.sandbox.working_dir = tmp_path
    agent.sandbox.km = MagicMock()
    agent.sandbox.start = AsyncMock()
    agent.sandbox.save_session = AsyncMock()
    agent.sandbox.get_code_history.return_value = []
    agent.sandbox.sync_runtime_persistence_state.return_value = {
        "status": "current",
        "degraded_recovery": False,
        "requires_recompute": False,
    }
    return agent


@pytest.mark.asyncio
async def test_resume_preserves_executor_messages_and_persists_new_plan(
    monkeypatch,
    tmp_path,
):
    agent = _bare_agent(monkeypatch, tmp_path)
    original_messages = [
        {"role": "system", "content": "static system"},
        {"role": "user", "content": "[Analysis Context]\nold context"},
        {"role": "user", "content": "Analysis instruction: original"},
    ]
    checkpoint = {
        "step_name": "step_1_plan_revision",
        "session_id": "test-session",
        "round_number": 0,
        "planning_stage": "full",
        "step_number": 1,
        "total_steps": 1,
        "active_plan_id": "old-plan",
        "messages": list(original_messages),
        "findings": ["existing finding"],
        "figures": [],
        "code_history": [],
        "analysis_log": [],
        "analysis_context": {"kept": True},
        "completed": False,
    }
    saves = []

    class _Checkpoint:
        def __init__(self, *args, **kwargs):
            pass

        async def load(self):
            return checkpoint

        async def save(self, data):
            saves.append(data)

        def clear(self):
            pass

        def should_checkpoint(self, step):
            return False

    setattr_facade(
        monkeypatch, "backend.agents.bioinformatics_agent", "CheckpointManager", _Checkpoint
    )
    state = AnalysisState(id="test-session", user_id="test-user")
    state.current_plan = _plan(plan_id="new-plan", revision_of="old-plan")

    result = await agent.run_autonomous_analysis(
        state,
        initial_instruction="This is intentionally discarded on resume.",
        max_steps=1,
        resume_from_checkpoint=True,
    )

    assert result["findings"] == ["existing finding"]
    assert saves, "the updated resume context must be persisted before execution"
    saved = saves[-1]
    assert saved["active_plan_id"] == "new-plan"
    assert saved["attempt_start_step"] == 1
    assert saved["total_steps"] == 2
    assert saved["messages"][:3] == original_messages
    assert "APPROVED PLAN REVISION" in saved["messages"][-1]["content"]
    assert saved["analysis_context"] == {"kept": True}


@pytest.mark.asyncio
async def test_chat_execution_and_report_preserve_links_without_old_plan_credit(monkeypatch, tmp_path):
    from backend.agents.bioinformatics import executor_execute, executor_report, executor_report_gates
    from backend.agents.bioinformatics.executor_dispatch import ToolAction

    for module in (executor_execute, executor_report, executor_report_gates):
        module.bind_facade(bioinformatics_facade)
    agent = _bare_agent(monkeypatch, tmp_path)
    state = AnalysisState(user_id="chat-receipt", current_round=1)
    state.current_plan = _plan(plan_id="historical-plan")
    state.current_plan.adata_source = AdataSourceSelection(mode="checkpoint_copy", checkpoint_id="old-source")
    state.current_plan.adata_source_preparation_id = "historical-preparation"
    context = SimpleNamespace(
        agent=agent, state=state, executor_run_kind="chat", executor_run_scope_id="chat-request",
        executor_settlement_scope_id="chat-request", resolved_adata_source_context={},
        messages=[], analysis_log=[], step_count=1, agent_reasoning="Submit new findings",
        annotation_completed=True, planning_stage="full", completed=False,
        evidence_gate_degraded=False, evidence_gate_findings=[], all_findings=[],
        accepted_finding_records=[], reported_evidence_ids=[], mechanism_programs_registered=True,
    )
    receipt = executor_execute.build_code_execution_receipt(
        state, "print('new result')", SimpleNamespace(output="new result", error=None, figures=[]),
        0.1, "planned", [1], {}, [], "candidate", None, [], executor_context=context,
    )
    assert receipt.executor_run_kind == "chat"
    assert receipt.executor_run_scope_id == "chat-request"
    assert receipt.plan_id is None
    assert receipt.plan_step_ids == []
    assert receipt.adata_requested_checkpoint_id is None
    assert receipt.adata_source_preparation_id is None
    agent.code_history = [receipt]
    call = ToolCall(id="chat-report", name="report_findings", arguments={
        "summary": "New result", "findings": [
            {"text": "New result", "evidence_execution_ids": [receipt.id]},
        ],
    })
    assert await executor_report.handle_report_findings(context, call) == ToolAction.BREAK
    assert context.reported_evidence_ids == [receipt.id]
    finding = context.accepted_finding_records[0]
    assert finding["plan_id"] is None
    assert finding["candidate_evidence_execution_ids"] == [receipt.id]

    # Figure dispatch and preflight must retain the Chat source and must not
    # claim completion of an unrelated ordinary-plan figure requirement.
    from backend.agents.bioinformatics import executor_collaboration
    from backend.agents.bioinformatics.figure_preflight import FigureRequestContext, prepare_figure_request
    from backend.agents.bioinformatics.shared import current_execution_identity_error
    from backend.models.schemas import PublicationFigure

    assert current_execution_identity_error(state, receipt, executor_context=context) is None
    foreign = receipt.model_copy(update={"executor_run_scope_id": "another-chat"})
    assert "another Chat request" in current_execution_identity_error(state, foreign, executor_context=context)
    executor_collaboration.bind_facade(bioinformatics_facade)
    agent._handle_generate_publication_figure = AsyncMock(return_value="render requested")
    agent.figure_plan_binding_callback = AsyncMock(side_effect=AssertionError("Old plan must not bind Chat figures"))
    context.review_callback = None
    figure_call = ToolCall(id="chat-figure", name="generate_publication_figure", arguments={
        "code": "plot_result()", "figure_name": "new-result", "figure_type": "other",
        "description": "New result", "evidence_execution_id": receipt.id, "plan_step_id": 1,
    })
    await executor_collaboration.handle_publication_figure(context, figure_call)
    kwargs = agent._handle_generate_publication_figure.await_args.kwargs
    assert kwargs["source_execution_id"] == receipt.id
    assert kwargs["producer_run_kind"] == "chat"
    assert kwargs["plan_step_id"] is None
    assert kwargs["purpose"] == "supplementary"
    agent.figure_plan_binding_callback.assert_not_awaited()
    agent._validate_code = MagicMock(return_value=(True, ""))
    attempt, error = await prepare_figure_request(FigureRequestContext(
        agent=agent, diagnostic=False,
        dependencies={"logger": MagicMock(), "eligible_artifacts": lambda *a, **kw: []},
        **kwargs,
    ))
    assert error is None
    assert attempt.current_plan_id is None
    figure = PublicationFigure(
        name="chat-result", description="New result", figure_type="other",
        png_path="chat-result.jpg", pdf_path="chat-result.pdf",
        producer_run_kind=kwargs["producer_run_kind"], plan_id=attempt.current_plan_id,
    )
    assert figure.producer_run_kind == "chat"
    assert figure.plan_id is None
