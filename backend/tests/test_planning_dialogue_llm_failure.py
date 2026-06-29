"""Tests for planning_dialogue LLM failure intervention (UpstreamExhaustedError handling)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.utils.llm_client import UpstreamExhaustedError
from backend.constants import LlmFailureAction
from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep, TaskStatus


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
    wf._llm_failure_generation = 0
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


class TestPlanningDialogueBioinfoFailure:
    """Test bioinformatics agent UpstreamExhaustedError handling in _planning_dialogue."""

    @pytest.mark.asyncio
    async def test_bioinfo_upstream_exhausted_skip_with_existing_plan(self):
        """When bioinfo fails with UpstreamExhaustedError and user SKIPs,
        existing_plan from previous round is preserved as fallback."""
        wf = _make_workflow()
        wf._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.SKIP)

        existing_plan = AnalysisPlan(
            phase="exploration", stage="full", overview="Round 1 plan",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="rejected",
        )
        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = existing_plan
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        err = UpstreamExhaustedError("test error", agent_name="bioinformatics", model="gpt-4o")
        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(side_effect=err)
        # Also mock revise_plan_incrementally since the else branch may be taken
        wf.bioinformatics_agent.revise_plan_incrementally = AsyncMock(side_effect=err)

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        # existing_plan.status stays "rejected" — we don't reinstate a rejected plan
        assert analysis_state.current_plan.status == "rejected"
        assert result.get("failed_node") is None

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

    @pytest.mark.asyncio
    async def test_bioinfo_upstream_exhausted_retry_then_succeed(self):
        """When bioinfo fails then user RETRYs and it succeeds, dialogue continues."""
        wf = _make_workflow()
        wf._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.RETRY)

        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = None
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        err = UpstreamExhaustedError("test error", agent_name="bioinformatics", model="gpt-4o")
        plan = AnalysisPlan(
            phase="exploration", stage="full", overview="New plan",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="draft",
        )
        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(side_effect=[err, plan])

        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(return_value={
            "decision": "approved", "feedback": "ok", "modifications": [],
        })

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        assert result["planning_result"]["plan"]["id"] == plan.id
        assert result.get("failed_node") is None
        wf._wait_for_llm_intervention.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_exception_in_bioinfo_returns_failed_state(self):
        """Generic Exception in bioinfo (not UpstreamExhaustedError) returns failed
        state without showing intervention dialog."""
        wf = _make_workflow()

        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = None
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(
            side_effect=RuntimeError("unexpected internal error")
        )

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        assert result["failed_node"] == "planning_dialogue"
        assert result["planning_result"]["status"] == "failed"
        assert "unexpected internal error" in result["planning_result"]["error"]


class TestPlanningDialoguePIReviewFailure:
    """Test PI review UpstreamExhaustedError handling in _planning_dialogue."""

    @pytest.mark.asyncio
    async def test_pi_review_upstream_exhausted_skip_auto_accepts_plan(self):
        """When PI review fails with UpstreamExhaustedError and user SKIPs,
        the plan is auto-accepted and forwarded to user_review_gate."""
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

        assert plan.status == "pi_reviewing"
        assert analysis_state.current_plan == plan
        assert result["planning_result"]["plan"] == plan.model_dump(mode="json")
        dialogue = result["planning_result"]["dialogue"]
        assert len(dialogue) == 1
        assert dialogue[0]["pi_decision"] == "approved"
        assert "LLM failure" in dialogue[0]["pi_feedback"]
        assert "auto-accepted" in dialogue[0]["pi_feedback"]

    @pytest.mark.asyncio
    async def test_pi_review_upstream_exhausted_retry_then_succeed(self):
        """When PI review fails then user RETRYs, only PI review is retried
        (bioinfo is not called again)."""
        wf = _make_workflow()
        wf._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.RETRY)

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
        wf.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=[
            err,
            {"decision": "approved", "feedback": "ok", "modifications": []},
        ])

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        assert result["planning_result"]["plan"]["id"] == plan.id
        assert wf.bioinformatics_agent.generate_analysis_plan.call_count == 1
        assert wf.pi_agent.review_bioinfo_plan.call_count == 2

    @pytest.mark.asyncio
    @patch('backend.workflow.core.asyncio.sleep', new_callable=AsyncMock)
    async def test_generic_exception_in_pi_review_retries_then_returns_failed(self, mock_sleep):
        """Generic Exception in PI review retries up to MAX_NODE_RETRIES (3) times,
        matching @with_retry behavior. Returns failed state after all retries fail."""
        wf = _make_workflow()

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

        # PI review fails 4 times (1 initial + 3 retries) with generic Exception
        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=[
            ValueError("parse error"),
            ValueError("parse error"),
            ValueError("parse error"),
            ValueError("parse error"),
        ])

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        # PI review called 4 times (1 initial + 3 retries)
        assert wf.pi_agent.review_bioinfo_plan.call_count == 4
        assert result["failed_node"] == "planning_dialogue"
        assert result["planning_result"]["status"] == "failed"
        assert "parse error" in result["planning_result"]["error"]

    @pytest.mark.asyncio
    @patch('backend.workflow.core.asyncio.sleep', new_callable=AsyncMock)
    async def test_generic_exception_in_pi_review_retry_succeeds(self, mock_sleep):
        """Generic Exception in PI review that succeeds on a retry should continue normally."""
        wf = _make_workflow()

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

        # PI review fails twice, succeeds on third attempt
        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=[
            ConnectionError("timeout"),
            ConnectionError("timeout"),
            {"decision": "approved", "feedback": "Looks good after retry", "modifications": []},
        ])

        state = _make_state(analysis_state=analysis_state)

        result = await wf._planning_dialogue(state)

        assert wf.pi_agent.review_bioinfo_plan.call_count == 3
        assert result["planning_result"]["plan"]["id"] == plan.id
        assert result.get("failed_node") is None
        assert result["planning_result"]["dialogue"][0]["pi_decision"] == "approved"
        assert "Looks good after retry" in result["planning_result"]["dialogue"][0]["pi_feedback"]


class TestPlanningDialogueMaxRoundsFallback:
    """Behavior tests for max-rounds fallback and empty user-review plans."""

    @pytest.mark.asyncio
    async def test_max_rounds_history_picks_last_valid(self):
        wf = _make_workflow()
        analysis_state = AnalysisState(user_id="test")
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        round3 = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="round 3",
            steps=[PlanStep(id=i, title=f"R3 step {i}", method="m") for i in range(1, 44)],
        )
        round4 = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="round 4",
            steps=[PlanStep(id=i, title=f"R4 step {i}", method="m") for i in range(1, 46)],
        )
        empty = AnalysisPlan(phase="exploration", stage="full", overview="empty", steps=[])

        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(return_value=empty)
        wf.bioinformatics_agent.revise_plan_incrementally = AsyncMock(
            side_effect=[round3, round4, empty, empty]
        )
        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=[
            {"decision": "revise", "feedback": "need steps", "modifications": []},
            {"decision": "revise", "feedback": "good but revise", "modifications": []},
            {"decision": "revise", "feedback": "add more", "modifications": []},
            {"decision": "revise", "feedback": "still revise", "modifications": []},
            {"decision": "reject", "feedback": "empty", "modifications": []},
        ])

        state = _make_state(analysis_state=analysis_state)
        result = await wf._planning_dialogue(state)

        plan = result["planning_result"]["plan"]
        assert plan["overview"] == "round 4"
        assert len(plan["steps"]) == 45
        assert analysis_state.current_plan.overview == "round 4"
        assert analysis_state.current_plan.status == "pi_reviewing"

    @pytest.mark.asyncio
    async def test_max_rounds_rejected_existing_plan_uses_history(self):
        wf = _make_workflow()
        analysis_state = AnalysisState(user_id="test")
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        valid_history_plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="valid history",
            steps=[PlanStep(id=1, title="Valid", method="m")],
        )
        rejected_plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="rejected final",
            steps=[PlanStep(id=1, title="Rejected", method="m")],
        )

        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(return_value=valid_history_plan)
        wf.bioinformatics_agent.revise_plan_incrementally = AsyncMock(return_value=rejected_plan)
        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=[
            {"decision": "revise", "feedback": "revise", "modifications": []},
            {"decision": "reject", "feedback": "reject", "modifications": []},
            {"decision": "reject", "feedback": "reject", "modifications": []},
            {"decision": "reject", "feedback": "reject", "modifications": []},
            {"decision": "reject", "feedback": "reject", "modifications": []},
        ])

        state = _make_state(analysis_state=analysis_state)
        result = await wf._planning_dialogue(state)

        plan = result["planning_result"]["plan"]
        assert plan["overview"] == "valid history"
        assert plan["steps"][0]["title"] == "Valid"

    @pytest.mark.asyncio
    async def test_max_rounds_skips_rejected_history_plan(self):
        """A non-empty plan rejected by PI must not be recovered from history."""
        wf = _make_workflow()
        analysis_state = AnalysisState(user_id="test")
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"

        valid_plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="valid history",
            steps=[PlanStep(id=1, title="Valid", method="m")],
        )
        rejected_plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="rejected history",
            steps=[PlanStep(id=1, title="Rejected", method="m")],
        )
        empty_plan = AnalysisPlan(phase="exploration", stage="full", overview="empty", steps=[])

        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(
            side_effect=[valid_plan, empty_plan]
        )
        wf.bioinformatics_agent.revise_plan_incrementally = AsyncMock(
            side_effect=[rejected_plan, empty_plan, empty_plan]
        )
        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=[
            {"decision": "revise", "feedback": "revise valid", "modifications": []},
            {"decision": "reject", "feedback": "reject non-empty", "modifications": []},
            {"decision": "revise", "feedback": "empty", "modifications": []},
            {"decision": "revise", "feedback": "empty", "modifications": []},
            {"decision": "revise", "feedback": "empty", "modifications": []},
        ])

        state = _make_state(analysis_state=analysis_state)
        result = await wf._planning_dialogue(state)

        plan = result["planning_result"]["plan"]
        assert plan["overview"] == "valid history"
        assert plan["steps"][0]["title"] == "Valid"

    @pytest.mark.asyncio
    async def test_max_rounds_no_valid_history_calls_pi_final(self):
        wf = _make_workflow()
        analysis_state = AnalysisState(user_id="test")
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"
        empty = AnalysisPlan(phase="exploration", stage="full", overview="empty", steps=[])
        final = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="pi final",
            steps=[PlanStep(id=1, title="PI final step", method="m")],
        )

        wf.bioinformatics_agent = MagicMock()
        wf.bioinformatics_agent.generate_analysis_plan = AsyncMock(return_value=empty)
        wf.bioinformatics_agent.revise_plan_incrementally = AsyncMock(return_value=empty)
        wf.pi_agent = MagicMock()
        wf.pi_agent.review_bioinfo_plan = AsyncMock(return_value={
            "decision": "revise", "feedback": "empty", "modifications": []
        })
        wf.pi_agent.generate_final_plan_from_dialogue = AsyncMock(return_value=final)

        state = _make_state(analysis_state=analysis_state)
        result = await wf._planning_dialogue(state)

        assert result["planning_result"]["plan"]["overview"] == "pi final"
        wf.pi_agent.generate_final_plan_from_dialogue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_user_review_gate_rejects_empty_plan(self):
        wf = _make_workflow()
        wf._plan_review_action = None
        wf._plan_review_event = MagicMock()
        wf._plan_review_event.clear = MagicMock()
        analysis_state = AnalysisState(user_id="test")
        analysis_state.analysis_phase = MagicMock()
        analysis_state.analysis_phase.value = "exploration"
        analysis_state.current_plan = AnalysisPlan(
            phase="exploration", stage="full", overview="empty", steps=[]
        )
        state = _make_state(analysis_state=analysis_state)

        result = await wf._user_review_gate(state)

        assert result["should_continue"] is False
        sent_types = [call.args[0] for call in wf._send_message.call_args_list]
        assert "plan_review" not in sent_types
