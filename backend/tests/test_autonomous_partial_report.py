"""Direction coverage informs PI review and never blocks report generation."""
import logging

import pytest
from unittest.mock import AsyncMock

from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.models.schemas import AnalysisState, TaskStatus
from backend.config._config import SystemConfig
from backend.tests.split_facade import setattr_facade


def _make_wf(monkeypatch):
    wf = SingleCellAnalysisWorkflow(config=SystemConfig(), session_id="test-sid")
    monkeypatch.setattr(wf, "_send_message", AsyncMock())
    monkeypatch.setattr(wf, "_send_state_sync", AsyncMock())
    # Force an incomplete-coverage observation. Patch across the
    # reporting facade *and* its split package (reporting_parts.final_report),
    # where _generate_final_report actually resolves the name.
    setattr_facade(
        monkeypatch,
        "backend.workflow.reporting",
        "report_coverage_observation",
        lambda state: "incomplete evidence for test",
    )
    return wf


def _incomplete_state():
    # Empty analysis_log/code_history so that, once the observation is recorded, the
    # node short-circuits at the "no analysis work recorded" branch instead of
    # invoking the real (LLM-heavy) report pipeline.
    st = AnalysisState(user_id="u1")
    st.current_round = 3
    st.allow_partial_report = False
    st.analysis_log = []
    st.code_history = []
    return st


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_seconds", [-1, 0, 300])
async def test_all_wait_modes_proceed_and_leave_acceptance_to_pi(
    monkeypatch, caplog, wait_seconds
):
    wf = _make_wf(monkeypatch)
    st = _incomplete_state()
    st.round_wait_seconds = wait_seconds

    with caplog.at_level(logging.WARNING):
        result = await wf._generate_final_report({"analysis_state": st})

    # Did not block.
    assert not result.get("report_blocked")
    # Host coverage code did not choose a complete/partial disposition.
    assert st.allow_partial_report is False
    # No report_blocked message was emitted to the client.
    sent_types = [c.args[0] for c in wf._send_message.await_args_list if c.args]
    assert "report_blocked" not in sent_types
    assert st.status == TaskStatus.COMPLETED
    assert any("observation recorded for PI review" in r.message for r in caplog.records)
