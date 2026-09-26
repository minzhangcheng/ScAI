"""PI consultation failures retain unresolved facts and explicit revision requests."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.constants import LlmFailureAction
from backend.models.schemas import AnalysisPlan, AnalysisState
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow.callbacks import CallbacksMixin
from backend.workflow.helpers import WorkflowSupersededError, WorkflowTerminatedError


def _error():
    return UpstreamExhaustedError("PI unavailable", agent_name="pi", model="test")


def _host():
    host = CallbacksMixin.__new__(CallbacksMixin)
    host.session_id = "consult-test"
    host._current_analysis_state = AnalysisState(id=host.session_id, user_id="user")
    host._current_analysis_state.current_plan = AnalysisPlan(phase="innovation", status="approved", steps=[])
    host._current_graph_state = {"planning_stage": "stage2"}
    host._settings_provider = SimpleNamespace(get_temperature=lambda _: 0.1)
    host._send_message = AsyncMock()
    host._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.SKIP)
    host.pi_agent = SimpleNamespace(
        chat_without_history=AsyncMock(), judge_plan_deviation=AsyncMock(),
    )
    return host


@pytest.mark.asyncio
async def test_pi_consult_skip_records_no_judgment():
    host = _host()
    host.pi_agent.chat_without_history.side_effect = _error()

    response = await host._pi_callback("Change method?", "Measured context")

    assert "No PI judgment" in response
    assert "remains unresolved" in response
    assert "standard bioinformatics practices" not in response
    host._wait_for_llm_intervention.assert_awaited_once()
    assert host._current_analysis_state.pi_decision_records[-1].response == response


@pytest.mark.asyncio
async def test_pi_consult_retry_repeats_only_failed_call():
    host = _host()
    host.pi_agent.chat_without_history.side_effect = [_error(), "Use the approved method."]
    host._wait_for_llm_intervention.return_value = LlmFailureAction.RETRY

    response = await host._pi_callback("Which method?", "Context")

    assert response == "Use the approved method."
    assert host.pi_agent.chat_without_history.await_count == 2
    assert len(host._current_analysis_state.pi_decision_records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["consult", "deviation"])
@pytest.mark.parametrize("control", ["terminate", "superseded"])
async def test_pi_consult_control_exceptions_propagate(stage, control):
    host = _host()
    host.pi_agent.chat_without_history.return_value = '{"plan_revision_needed": true}'
    target = (host.pi_agent.chat_without_history if stage == "consult"
              else host.pi_agent.judge_plan_deviation)
    if control == "terminate":
        target.side_effect = _error()
        host._wait_for_llm_intervention.return_value = LlmFailureAction.TERMINATE
        expected = WorkflowTerminatedError
    else:
        target.side_effect = WorkflowSupersededError("replaced")
        expected = WorkflowSupersededError

    with pytest.raises(expected):
        await host._pi_callback("Question", "Context")
