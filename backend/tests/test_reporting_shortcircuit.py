"""Tests for reporting short-circuit when no analysis rounds have been executed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.models.schemas import AnalysisState, AnalysisStep, TaskStatus


def _make_workflow():
    from backend.workflow import SingleCellAnalysisWorkflow
    from backend.models.schemas import TaskStatus as _TS
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-session"
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()

    def _set_status(state, status):
        state.status = status

    wf._set_status = staticmethod(_set_status).__get__(wf, SingleCellAnalysisWorkflow)
    wf._settings_provider = MagicMock()
    wf._settings_provider.get_vision_setting = MagicMock(return_value=True)
    wf.painter_agent = None
    wf.bioinformatics_agent = None
    wf.supervisor_agent = None
    wf.llm_manager = MagicMock()
    wf.sandbox = None
    return wf


def _make_state(current_round=0):
    analysis_state = AnalysisState(user_id="test")
    analysis_state.current_round = current_round
    return {"analysis_state": analysis_state}


class TestGenerateFinalReportShortCircuit:
    @pytest.mark.asyncio
    async def test_short_circuit_when_no_rounds(self):
        wf = _make_workflow()
        state = _make_state(current_round=0)

        result = await wf._generate_final_report(state)

        assert result["analysis_state"].status == TaskStatus.COMPLETED
        assert result["analysis_state"].current_round == 0
        wf._send_message.assert_any_call("completed", {"reason": "stopped_before_execution"}, "pi")
        wf._send_state_sync.assert_called()

    @pytest.mark.asyncio
    async def test_no_short_circuit_when_rounds_exist(self):
        wf = _make_workflow()
        state = _make_state(current_round=1)
        # Predicate is now log/history-based: must record at least one step
        # for the report node to NOT short-circuit.
        state["analysis_state"].analysis_log.append(
            AnalysisStep(step_number=1, step_type="execute_code", code="print('x')")
        )

        wf.pi_agent = MagicMock()
        wf.pi_agent.generate_final_summary = AsyncMock(return_value="summary")
        wf.pi_agent.generate_integrated_report = AsyncMock(return_value="report text")
        wf.pi_agent.select_brief_figures = AsyncMock(return_value=[])

        result = await wf._generate_final_report(state)

        wf.pi_agent.generate_final_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_circuit_when_round_advanced_but_no_work(self):
        """Stops triggered between initialize_round and any analysis: current_round=1
        but no log/history. Must still short-circuit (the targeted bug)."""
        wf = _make_workflow()
        state = _make_state(current_round=1)

        result = await wf._generate_final_report(state)

        assert result["analysis_state"].status == TaskStatus.COMPLETED
        wf._send_message.assert_any_call("completed", {"reason": "stopped_before_execution"}, "pi")


class TestGenerateGraphicAbstractShortCircuit:
    @pytest.mark.asyncio
    async def test_short_circuit_when_no_rounds(self):
        wf = _make_workflow()
        state = _make_state(current_round=0)

        result = await wf._generate_graphic_abstract(state)

        assert result["analysis_state"].current_round == 0


class TestGenerateDetailedReportShortCircuit:
    @pytest.mark.asyncio
    async def test_short_circuit_when_no_rounds(self):
        wf = _make_workflow()
        state = _make_state(current_round=0)

        result = await wf._generate_detailed_report(state)

        assert result["analysis_state"].status == TaskStatus.COMPLETED
        wf._send_message.assert_any_call("completed", {"reason": "stopped_before_execution"}, "pi")