"""An unmet figure contract is a PI-visible observation, not an executor trap.

The executor's only lever against this contract is another render, and the
deliverables it would be asked for are routinely ones the review/layout loop
has already abandoned -- ``requires upstream reanalysis`` skips the revision
loop outright, and a figure that never becomes report-eligible can never
satisfy the requirement.  A refusal there cannot be worked off.

Measured on session ``a7cbb0b4``: ``report_findings`` was refused seven times,
the missing count sat at 49 for four and a half hours, the whole 200-step
budget was spent, and the run closed with one fallback finding.

This drives the real autonomous dispatcher.  A source-only assertion cannot
prove that the call is accepted on the first attempt rather than re-entered.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import (
    AgentType,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    CodeExecution,
    PlanStep,
)


def _plan_promising_an_unrendered_figure() -> AnalysisPlan:
    return AnalysisPlan(
        phase="innovation",
        stage="full",
        status="approved",
        steps=[PlanStep(
            id=1,
            title="Score the response program",
            method="scanpy.tl.score_genes",
            figure_requirements=[
                {"figure_type": "umap", "description": "Score on the embedding"},
            ],
        )],
    )


@pytest.mark.asyncio
async def test_unmet_figure_contract_is_observed_and_the_report_is_accepted(
    monkeypatch, tmp_path,
):
    from backend.tests.split_facade import setattr_facade
    from backend.utils.llm_client import LLMToolResponse, ToolCall

    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.session_id = "figure-contract-release"
    agent.code_history = []
    agent.analysis_context = {}
    agent.last_result = None
    agent._recent_sup_feedback = set()
    agent._pause_event = None
    agent.literature_callback = None
    agent.pi_callback = None
    agent.annotate_clusters_callback = None
    agent.message_callback = None
    agent.figure_review_callback = None
    agent.supervisor_callback = None
    agent.species_validation_callback = None
    agent.marker_db_intervention_callback = None
    agent.technique_agent = None
    agent._provider = MagicMock()
    agent._provider.get_config_revision.return_value = 0
    agent._last_config_revision = 0
    agent.agent_type = AgentType.BIOINFORMATICS

    monkeypatch.setattr(
        BioinformaticsAgent, "model", property(lambda _self: "test-model"),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "system_prompt",
        property(lambda _self: "Test bioinformatics agent."),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "config",
        property(lambda _self: MagicMock(temperature=0.1)),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_static_system_prompt",
        property(lambda _self: "Static bioinformatics prompt."),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_build_dynamic_context",
        lambda _self, _state: "Dynamic test context.",
    )
    monkeypatch.setattr(BioinformaticsAgent, "_set_llm_context", lambda _self: None)
    monkeypatch.setattr(BioinformaticsAgent, "_notify", AsyncMock())

    evidence = CodeExecution(
        id="exec-figure-contract",
        code="print('descriptive result')",
        output="descriptive result",
        round_number=2,
    )
    # One turn only.  Under the refusing gate the loop re-enters and burns
    # every remaining step, so the await count is the assertion that matters.
    responses = [LLMToolResponse(
        content="",
        tool_calls=[ToolCall(
            id="report-once",
            name="report_findings",
            arguments={
                "findings": ["A descriptive signal was retained."],
                "summary": "Closed with the visual deliverable still open.",
                "suggestions": [],
                "evidence_execution_ids": [evidence.id],
            },
        )],
        model="test-model",
        usage={},
        raw_response=None,
        raw_assistant_message=None,
    )]
    fake_client = MagicMock(base_url="https://fake.example/v1")
    fake_client.chat_with_tools = AsyncMock(side_effect=responses)
    monkeypatch.setattr(
        BioinformaticsAgent, "llm_client", property(lambda _self: fake_client),
    )

    agent.sandbox = MagicMock()
    agent.sandbox.state_dir = tmp_path / "states"
    agent.sandbox.state_dir.mkdir()
    agent.sandbox.working_dir = tmp_path
    agent.sandbox.km = MagicMock()
    agent.sandbox.start = AsyncMock()
    agent.sandbox.save_session = AsyncMock()
    agent.sandbox.get_code_history.return_value = []
    agent.sandbox.sync_runtime_persistence_state.return_value = {}

    class _NullCheckpoint:
        def __init__(self, *_args, **_kwargs):
            pass

        async def load(self):
            return None

        async def save(self, _data):
            return None

        def clear(self):
            return None

        def should_checkpoint(self, _step):
            return False

    setattr_facade(
        monkeypatch,
        "backend.agents.bioinformatics_agent",
        "CheckpointManager",
        _NullCheckpoint,
    )
    setattr_facade(
        monkeypatch,
        "backend.agents.bioinformatics_agent",
        "successful_evidence_executions",
        lambda *_args, **_kwargs: [evidence],
    )

    state = AnalysisState(
        id=agent.session_id,
        user_id="owner",
        current_round=2,
        analysis_phase=AnalysisPhase.INNOVATION,
    )
    state.current_plan = _plan_promising_an_unrendered_figure()
    assert not state.publication_figures

    result = await agent.run_autonomous_analysis(
        state,
        initial_instruction="Close the round with the figure still missing.",
        max_steps=5,
        resume_from_checkpoint=False,
    )

    # Accepted on the first attempt: the loop never re-entered to argue.
    assert fake_client.chat_with_tools.await_count == 1

    observations = [
        item for item in (result.get("evidence_gate_findings") or [])
        if item.get("gate") == "required-figure-observation"
    ]
    assert len(observations) == 1, result.get("evidence_gate_findings")
    observation = observations[0]
    assert observation["severity"] == "advise"
    assert observation["forced_release"] is False
    assert "umap" in observation["detail"].lower()

    # The gap reaches the PI as a note the model cannot suppress.
    findings = result.get("findings") or []
    assert any(
        "[PI review observation; non-blocking]" in str(item) for item in findings
    ), findings
