"""Tests for _generate_final_report integrated_report LLM-failure retry loop."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.constants import LlmFailureAction
from backend.models.schemas import AnalysisState, AnalysisStep
from backend.utils.llm_client import UpstreamExhaustedError
from backend.tests.split_facade import patch_facade
from backend.workflow.reporting import ReportingMixin


class _Harness(ReportingMixin):
    def __init__(self, integrated_responses):
        self.session_id = "t"
        self.pi_agent = MagicMock()
        self.pi_agent.generate_final_summary = AsyncMock(return_value="summary")
        self.pi_agent.generate_integrated_report = AsyncMock(side_effect=integrated_responses)
        self.pi_agent.select_brief_figures = AsyncMock(return_value=[])
        self._send_message = AsyncMock()
        self._send_state_sync = AsyncMock()
        self._set_status = MagicMock()
        self._wait_for_llm_intervention = AsyncMock()


def _state():
    s = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    s.current_round = 1
    # Report short-circuit predicate is now log/history-based — these tests
    # exercise the LLM-failure retry path inside the real report pipeline,
    # so we must record at least one analysis step to bypass the short-circuit.
    s.analysis_log.append(AnalysisStep(step_number=1, step_type="execute_code", code="x=1"))
    return {"analysis_state": s}


@pytest.mark.asyncio
async def test_integrated_report_retries_on_upstream_exhausted():
    err = UpstreamExhaustedError("upstream failed", agent_name="pi", model="m", llm_server="s")
    h = _Harness(integrated_responses=[err, "the integrated report html"])
    h._wait_for_llm_intervention.return_value = LlmFailureAction.RETRY

    with patch_facade("backend.workflow.reporting", "get_session_workspace") as gw:
        gw.return_value.exists.return_value = False
        gw.return_value.__truediv__ = lambda self, other: gw.return_value
        out = await h._generate_final_report(_state())

    assert h.pi_agent.generate_integrated_report.await_count == 2
    assert h._wait_for_llm_intervention.await_count == 1
    assert out["analysis_state"].integrated_report == "the integrated report html"


@pytest.mark.asyncio
async def test_integrated_report_skips_on_user_skip():
    err = UpstreamExhaustedError("upstream failed", agent_name="pi", model="m", llm_server="s")
    h = _Harness(integrated_responses=[err])
    h._wait_for_llm_intervention.return_value = LlmFailureAction.SKIP

    with patch_facade("backend.workflow.reporting", "get_session_workspace") as gw:
        gw.return_value.exists.return_value = False
        gw.return_value.__truediv__ = lambda self, other: gw.return_value
        out = await h._generate_final_report(_state())

    assert h.pi_agent.generate_integrated_report.await_count == 1
    assert h._wait_for_llm_intervention.await_count == 1
    assert not getattr(out["analysis_state"], "integrated_report", None)