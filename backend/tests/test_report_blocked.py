"""Tests that generate_report clears stale report_blocked from checkpoints."""
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_generate_report_clears_stale_report_blocked(monkeypatch):
    from backend.workflow.core import SingleCellAnalysisWorkflow
    from backend.models.schemas import AnalysisState
    from backend.config._config import SystemConfig

    cfg = SystemConfig()
    wf = SingleCellAnalysisWorkflow(config=cfg, session_id="test-sid")

    class _Snap:
        values = {"report_blocked": True, "analysis_state": None, "current_round": 1}

    wf.compiled_graph = AsyncMock()
    wf.compiled_graph.aget_state = AsyncMock(return_value=_Snap())

    seen = {}

    async def _final(state):
        seen["final_blocked"] = state.get("report_blocked")
        return {**state, "analysis_state": state["analysis_state"]}

    async def _ga(state):
        seen["ga_blocked"] = state.get("report_blocked")
        return state

    async def _detail(state):
        seen["detail_blocked"] = state.get("report_blocked")
        return state

    monkeypatch.setattr(wf, "_generate_final_report", _final)
    monkeypatch.setattr(wf, "_generate_graphic_abstract", _ga)
    monkeypatch.setattr(wf, "_generate_detailed_report", _detail)
    monkeypatch.setattr(wf, "_send_message", AsyncMock())
    monkeypatch.setattr(wf, "initialize", AsyncMock())

    st = AnalysisState(user_id="u1")
    st.current_round = 1
    await wf.generate_report(st, "tid")

    assert seen["ga_blocked"] in (None, False)
    assert seen["detail_blocked"] in (None, False)
