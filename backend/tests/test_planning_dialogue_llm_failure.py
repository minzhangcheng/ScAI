"""Tests for planning_dialogue LLM failure intervention (UpstreamExhaustedError handling)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.utils.llm_client import UpstreamExhaustedError
from backend.constants import LlmFailureAction
from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep, TaskStatus
from backend.agents.bioinformatics import planning as bioinformatics_planning


def _make_workflow():
    """Create a minimal workflow instance with _planning_dialogue dependencies mocked."""
    from backend.workflow import SingleCellAnalysisWorkflow
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-session"
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = MagicMock()
    wf._build_planning_context = MagicMock(return_value=None)
    wf._llm_failure_lock = MagicMock()
    wf._llm_failure_lock.__aenter__ = AsyncMock()
    wf._llm_failure_lock.__aexit__ = AsyncMock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generations = {}
    wf._settings_provider = None
    return wf


def _make_state(analysis_state=None, pi_instructions=None, planning_stage="full"):
    """Create a minimal WorkflowState dict for _planning_dialogue."""
    if analysis_state is None:
        analysis_state = AnalysisState(user_id="test")
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"
    if pi_instructions is None:
        pi_instructions = {"bioinformatics_tasks": ["task1"], "overview": "test"}
    return {
        "analysis_state": analysis_state,
        "pi_instructions": pi_instructions,
        "planning_stage": planning_stage,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_failure", [False, True])
@pytest.mark.parametrize("resolution", [LlmFailureAction.RETRY, LlmFailureAction.SKIP, LlmFailureAction.TERMINATE])
async def test_finalizer_failure_uses_intervention_and_preserves_draft(schema_failure, resolution):
    from backend.workflow.planning_parts import dialogue
    from backend.workflow.helpers import WorkflowTerminatedError
    import backend.workflow.planning as facade

    dialogue.bind_facade(facade)
    wf = _make_workflow()
    draft = AnalysisPlan(phase="exploration", steps=[PlanStep(id=1, title="Original", method="m")])
    original = draft.model_dump()
    analysis_state = AnalysisState(user_id="test", current_plan=draft)
    state = _make_state(analysis_state)
    final = draft.model_copy(deep=True, update={"id": "final", "status": "pi_reviewing"})
    failure = (bioinformatics_planning.PlanGenerationFailure("invalid JSON") if schema_failure
               else UpstreamExhaustedError("timeout", agent_name="pi"))
    wf.pi_agent = MagicMock()
    wf.pi_agent.model = "test-model"
    wf.pi_agent.generate_final_plan_from_dialogue = AsyncMock(side_effect=[failure, final])
    wf._wait_for_llm_intervention = AsyncMock(return_value=resolution)
    ctx = dict(analysis_state=analysis_state, stage="full", dialogue_history=[],
               existing_plan=draft, round_num=2, finalization_reason="no_material_plan_change")

    if resolution == LlmFailureAction.TERMINATE:
        with pytest.raises(WorkflowTerminatedError):
            await dialogue.finalize_planning_dialogue(wf, ctx, state)
        assert analysis_state.status != TaskStatus.FAILED
    else:
        result = await dialogue.finalize_planning_dialogue(wf, ctx, state)
        assert result.get("plan_approved", False) is (resolution == LlmFailureAction.RETRY)
        if resolution == LlmFailureAction.SKIP:
            assert analysis_state.current_plan is draft
            assert result["planning_result"]["status"] == "failed"
    assert draft.model_dump() == original
    wf._wait_for_llm_intervention.assert_awaited_once()
    assert wf.pi_agent.generate_final_plan_from_dialogue.await_count == (2 if resolution == LlmFailureAction.RETRY else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"steps":[{"id":1,"title":"Incomplete', "not json", '{"steps":[]}'])
async def test_incomplete_revision_uses_intervention_without_reviewing_retained_plan(response):
    from backend.agents.bioinformatics_agent import BioinformaticsAgent

    wf = _make_workflow()
    analysis_state = AnalysisState(user_id="test")
    original_plan = AnalysisPlan(phase="exploration", steps=[
        PlanStep(id=1, title="Complete original step", method="m"),
    ])
    analysis_state.current_plan = original_plan
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_vision_setting.return_value = False
    wf.bioinformatics_agent = BioinformaticsAgent(settings_provider=provider, sandbox=None)
    wf.bioinformatics_agent.chat_without_history = AsyncMock(return_value=response)
    wf.pi_agent = MagicMock()
    wf.pi_agent.review_bioinfo_plan = AsyncMock()
    wf.pi_agent.generate_final_plan_from_dialogue = AsyncMock()
    wf._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.SKIP)
    state = _make_state(analysis_state=analysis_state)
    state["user_plan_feedback"] = "Keep the requested comparison"

    result = await wf._planning_dialogue(state)

    assert wf.bioinformatics_agent.chat_without_history.await_count == 2
    wf._wait_for_llm_intervention.assert_awaited_once()
    assert wf._wait_for_llm_intervention.await_args.args[0].failure_kind == "parse_error"
    wf.pi_agent.review_bioinfo_plan.assert_not_awaited()
    wf.pi_agent.generate_final_plan_from_dialogue.assert_not_awaited()
    assert result["user_plan_feedback"] == "Keep the requested comparison"
    assert analysis_state.current_plan is original_plan
    assert original_plan.steps[0].title == "Complete original step"


@pytest.mark.asyncio
async def test_client_stream_interruption_reaches_the_llm_failure_gate():
    """Composition test: the failure is produced by the real client, not a mock.

    An interrupted stream must reach _wait_for_llm_intervention. If it arrives
    as a bare RuntimeError it lands in the generic handler instead, which ends
    the workflow with no way for the user to retry.
    """
    from types import SimpleNamespace
    from backend.utils.llm_client import OpenAICompatibleClient

    class TruncatedStream:
        def __init__(self):
            self.index = 0
            self.close = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return SimpleNamespace(
                    type="response.output_text.delta", delta="partial"
                )
            raise StopAsyncIteration

    client = OpenAICompatibleClient(
        api_key="k", base_url="https://gate.example/v1", agent_name="bioinformatics"
    )
    client.client.responses.create = AsyncMock(return_value=TruncatedStream())

    async def generate_via_real_client(*args, **kwargs):
        return await client.chat(
            model="gpt-x",
            messages=[{"role": "user", "content": "plan"}],
            prefer_stream=True,
        )

    wf = _make_workflow()
    wf._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.SKIP)
    analysis_state = AnalysisState(user_id="test")
    analysis_state.current_plan = None
    analysis_state.analysis_phase = MagicMock()
    analysis_state.analysis_phase.value = "exploration"
    wf.bioinformatics_agent = MagicMock()
    wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(
        side_effect=generate_via_real_client
    )
    wf.pi_agent = MagicMock()
    wf.pi_agent.review_bioinfo_plan = AsyncMock()

    try:
        result = await wf._planning_dialogue(
            _make_state(analysis_state=analysis_state)
        )
    finally:
        await client.close()

    wf._wait_for_llm_intervention.assert_awaited_once()
    wf.pi_agent.review_bioinfo_plan.assert_not_awaited()
    assert result["planning_result"]["status"] == "failed"
    # The upstream diagnosis survives classification and reaches the user.
    assert "after the first event" in result["planning_result"]["error"]


class TestPlanningDialogueBioinfoFailure:
    """Test bioinformatics agent UpstreamExhaustedError handling in _planning_dialogue."""

    @pytest.mark.asyncio
    async def test_bioinfo_upstream_exhausted_skip_no_existing_plan(self):
        """When bioinfo fails on round 1 with no existing plan and user SKIPs,
        workflow terminates gracefully (should_continue=False, status=FAILED)."""
        wf = _make_workflow()
        wf._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.SKIP)

        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = None
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        err = UpstreamExhaustedError("test error", agent_name="bioinformatics", model="gpt-4o")
        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(side_effect=err)

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        assert result["failed_node"] == "planning_dialogue"
        assert result["planning_result"]["status"] == "failed"
        assert "test error" in result["planning_result"]["error"]
        assert result["should_continue"] is False
        assert analysis_state.status == TaskStatus.FAILED
        wf._send_state_sync.assert_called()

class TestPlanningDialoguePIReviewFailure:
    """Test PI review UpstreamExhaustedError handling in _planning_dialogue."""

    @pytest.mark.asyncio
    async def test_pi_review_upstream_exhausted_skip_does_not_approve_plan(self):
        """A plan without a completed PI review cannot be auto-approved."""
        wf = _make_workflow()
        wf._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.SKIP)

        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = None
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        plan = AnalysisPlan(
            phase="exploration", stage="full", overview="Bioinfo plan",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="draft",
        )
        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(return_value=plan)

        err = UpstreamExhaustedError("test error", agent_name="pi", model="gpt-4o")
        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=err)

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        assert plan.status == "draft"
        assert analysis_state.current_plan is None
        assert result["failed_node"] == "planning_dialogue"
        assert result["planning_result"]["status"] == "failed"
        assert result["should_continue"] is False
        assert analysis_state.status == TaskStatus.FAILED
        wf._send_state_sync.assert_awaited_once_with(
            analysis_state, "planning_aborted"
        )
