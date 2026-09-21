"""PI consultation failures retain unresolved facts and explicit revision requests."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.constants import LlmFailureAction, PLAN_REVISION_NEEDED_TAG
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
@pytest.mark.parametrize("action", [LlmFailureAction.RETRY, LlmFailureAction.SKIP])
async def test_deviation_intervention_retains_explicit_revision_request(action):
    host = _host()
    host.pi_agent.chat_without_history.return_value = '{"plan_revision_needed": true}'
    host.pi_agent.judge_plan_deviation.side_effect = [_error(), {"needs_user_popup": True}]
    host._wait_for_llm_intervention.return_value = action

    response = await host._pi_callback("Revise?", "Context")

    assert PLAN_REVISION_NEEDED_TAG in response
    host.pi_agent.chat_without_history.assert_awaited_once()
    assert host.pi_agent.judge_plan_deviation.await_count == (2 if action == LlmFailureAction.RETRY else 1)
    assert host._current_analysis_state.pi_decision_records[-1].response == response
    if action == LlmFailureAction.SKIP:
        assert "remains unresolved" in response


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


@pytest.mark.asyncio
async def test_deviation_bug_does_not_discard_revision_request():
    host = _host()
    host.pi_agent.chat_without_history.return_value = '{"plan_revision_needed": true}'
    host.pi_agent.judge_plan_deviation.side_effect = RuntimeError("bad response")

    response = await host._pi_callback("Revise?", "Context")

    assert PLAN_REVISION_NEEDED_TAG in response
    assert "failed technically" in response


@pytest.mark.asyncio
async def test_truncated_deviation_json_is_a_technical_failure():
    from backend.agents.pi.research_planning import ResearchPlanningMixin
    from backend.utils.formatting import safe_parse_json

    agent = SimpleNamespace(
        chat=AsyncMock(return_value='{"severity": "minor", "reasoning": "unfinished'),
        _safe_parse_json=safe_parse_json,
        model="test",
    )
    with pytest.raises(UpstreamExhaustedError) as caught:
        await ResearchPlanningMixin.judge_plan_deviation(
            agent, "A deviation", AnalysisPlan(phase="innovation", steps=[]), [],
        )
    assert caught.value.failure_kind == "parse_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    '{"reasoning":"assessment unavailable"}',
    '{"severity":"major"}',
    '{"needs_user_popup":true}',
    '{"severity":"minor","needs_user_popup":"false"}',
    '{"severity":"unknown","needs_user_popup":false}',
])
async def test_incomplete_deviation_verdict_retains_pi_revision(response):
    from backend.tests.test_planning_response_boundaries import _make_pi_agent

    host = _host()
    pi = _make_pi_agent()
    pi.chat_without_history = AsyncMock(return_value='{"plan_revision_needed":true}')
    pi.chat = AsyncMock(return_value=response)
    host.pi_agent = pi

    result = await host._pi_callback("Revise the plan?", "Context")

    assert PLAN_REVISION_NEEDED_TAG in result
    assert "remains unresolved" in result
    host._wait_for_llm_intervention.assert_awaited_once()
    error = host._wait_for_llm_intervention.await_args.args[0]
    assert isinstance(error, UpstreamExhaustedError)
    assert error.failure_kind == "parse_error"
    assert host._current_analysis_state.pi_decision_records[-1].response == result


@pytest.mark.asyncio
@pytest.mark.parametrize("severity, popup", [("minor", False), ("major", True)])
async def test_explicit_deviation_verdict_preserves_pi_choice(severity, popup):
    import json
    from backend.tests.test_planning_response_boundaries import _make_pi_agent

    pi = _make_pi_agent()
    pi.chat = AsyncMock(return_value=json.dumps({
        "severity": severity, "needs_user_popup": popup,
    }))

    result = await pi.judge_plan_deviation(
        "Deviation", AnalysisPlan(phase="innovation", steps=[]), [],
    )

    assert result["severity"] == severity
    assert result["needs_user_popup"] is popup
