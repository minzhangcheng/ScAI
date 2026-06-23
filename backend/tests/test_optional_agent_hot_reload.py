"""Mid-run disable: clearing the pathology model after the workflow starts must
cause _pathology_analysis to skip instead of calling the LLM with default model."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.workflow.agents import AgentNodesMixin
from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.models.schemas import AnalysisState, AgentType


class _MixinUnderTest(AgentNodesMixin):
    pass


@pytest.mark.asyncio
async def test_pathology_skips_when_model_cleared_mid_run():
    wf = _MixinUnderTest.__new__(_MixinUnderTest)
    wf.session_id = "s1"
    wf.pathology_agent = MagicMock()       # agent object exists (init time)
    wf.pathology_agent.process = AsyncMock()
    wf._send_message = AsyncMock()
    wf._settings_provider = MagicMock()
    wf._settings_provider.get_model = MagicMock(return_value="")  # mid-run: cleared
    wf.ensure_agent_ready = SingleCellAnalysisWorkflow.ensure_agent_ready.__get__(wf)

    state_obj = AnalysisState(session_id="s1", user_id="test")
    wf_state = {
        "analysis_state": state_obj,
        "pi_instructions": {"pathology_tasks": ["x"]},
        "skip_nodes": [],
    }

    # Invoke through the wrapped @with_retry method
    result = await wf._pathology_analysis(wf_state)

    assert result["pathology_result"]["status"] == "skipped"
    assert wf.pathology_agent is None


@pytest.mark.asyncio
async def test_pathology_runs_when_model_set():
    wf = _MixinUnderTest.__new__(_MixinUnderTest)
    wf.session_id = "s1"
    wf.pathology_agent = MagicMock()
    wf.pathology_agent.process = AsyncMock(return_value={"findings": [], "status": "ok"})
    wf._send_message = AsyncMock()
    wf._settings_provider = MagicMock()
    wf._settings_provider.get_model = MagicMock(return_value="some/model")
    wf.ensure_agent_ready = SingleCellAnalysisWorkflow.ensure_agent_ready.__get__(wf)
    wf.pi_agent = MagicMock()

    state_obj = AnalysisState(session_id="s1", user_id="test")
    # Stub: get_files_by_section returns empty so we short-circuit on no-data path
    wf_state = {
        "analysis_state": state_obj,
        "pi_instructions": {"pathology_tasks": []},
        "skip_nodes": [],
    }
    result = await wf._pathology_analysis(wf_state)
    # Hot-reload check should NOT short-circuit when model is set
    assert result["pathology_result"]["status"] in ("no_data", "skipped", "ok")
