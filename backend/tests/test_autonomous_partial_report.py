"""Report gate behavior under auto-advancing vs manual round-wait modes.

When direction coverage is incomplete, the final-report gate must:
  * auto-advance modes (round_wait_seconds >= 0: no-wait or countdown) — NOT block;
    it flips allow_partial_report and produces a (partial) report instead of
    stranding the run at a silent dead-end with no report.
  * manual mode (round_wait_seconds == -1) — keep the block-and-wait behavior so
    an attended user can approve the partial report.
"""
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
    # Force the coverage gate to report "incomplete evidence". Patch across the
    # reporting facade *and* its split package (reporting_parts.final_report),
    # where _generate_final_report actually resolves the name.
    setattr_facade(
        monkeypatch,
        "backend.workflow.reporting",
        "validate_complete_report_allowed",
        lambda state: (False, "incomplete evidence for test"),
    )
    return wf


def _incomplete_state():
    # Empty analysis_log/code_history so that, once the gate falls through, the
    # node short-circuits at the "no analysis work recorded" branch instead of
    # invoking the real (LLM-heavy) report pipeline.
    st = AnalysisState(user_id="u1")
    st.current_round = 3
    st.allow_partial_report = False
    st.analysis_log = []
    st.code_history = []
    return st


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_seconds", [0, 300])
async def test_auto_advancing_mode_produces_partial_instead_of_blocking(
    monkeypatch, caplog, wait_seconds
):
    wf = _make_wf(monkeypatch)
    st = _incomplete_state()
    st.round_wait_seconds = wait_seconds

    with caplog.at_level(logging.WARNING):
        result = await wf._generate_final_report({"analysis_state": st})

    # Did not block.
    assert not result.get("report_blocked")
    # Partial report was authorized so the generator labels output correctly.
    assert st.allow_partial_report is True
    # No report_blocked message was emitted to the client.
    sent_types = [c.args[0] for c in wf._send_message.await_args_list if c.args]
    assert "report_blocked" not in sent_types
    # The gate is no longer silent.
    assert any("partial report instead of blocking" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_manual_mode_still_blocks_and_waits(monkeypatch, caplog):
    wf = _make_wf(monkeypatch)
    st = _incomplete_state()
    st.round_wait_seconds = -1  # manual: attended user expected

    with caplog.at_level(logging.WARNING):
        result = await wf._generate_final_report({"analysis_state": st})

    # Blocked and parked for user approval.
    assert result.get("report_blocked") is True
    assert result.get("should_continue") is False
    assert st.status == TaskStatus.WAITING_INPUT
    assert st.allow_partial_report is False
    sent_types = [c.args[0] for c in wf._send_message.await_args_list if c.args]
    assert "report_blocked" in sent_types
