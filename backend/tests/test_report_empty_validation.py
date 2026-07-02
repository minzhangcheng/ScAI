"""When the report model returns empty/placeholder content even after the
client-layer retries, generate_detailed_report must NOT save a completed empty
report. It raises UpstreamExhaustedError so the workflow HITL loop can drive
Retry / change-model / Skip."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.utils.llm_client import EMPTY_CONTENT_PLACEHOLDER, UpstreamExhaustedError


def test_validate_report_content_rejects_empty():
    from backend.agents.supervisor_agent import SupervisorAgent
    assert SupervisorAgent._validate_report_content("") is False
    assert SupervisorAgent._validate_report_content("   \n  ") is False
    assert SupervisorAgent._validate_report_content(EMPTY_CONTENT_PLACEHOLDER) is False
    assert SupervisorAgent._validate_report_content("#") is False  # heading only, no body
    assert SupervisorAgent._validate_report_content(
        "## Report\n\nThe DKO livers showed a clear shift in myeloid populations."
    ) is True


@pytest.mark.asyncio
async def test_generate_detailed_report_raises_on_empty(tmp_path):
    from backend.agents.supervisor_agent import SupervisorAgent

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = "sess-test"
    agent.refresh_runtime_config_if_needed = MagicMock()
    agent.get_supervision_log = MagicMock(return_value=[])
    agent.get_summary_stats = MagicMock(return_value={})
    agent._build_analysis_narrative = MagicMock(return_value="narrative")
    agent._collect_figure_paths = MagicMock(return_value={})
    agent._collect_table_paths = MagicMock(return_value=[])
    agent._build_report_prompt = MagicMock(return_value="prompt")
    agent._get_report_system_prompt = MagicMock(return_value="sys")
    agent._set_llm_context = MagicMock()

    empty_resp = MagicMock()
    empty_resp.content = ""
    client = MagicMock()
    client.set_context = MagicMock()
    client.base_url = "http://test"
    client.chat = AsyncMock(return_value=empty_resp)

    analysis_state = MagicMock()

    with pytest.raises(UpstreamExhaustedError):
        await agent.generate_detailed_report(
            analysis_state=analysis_state,
            report_model="gpt-5.5",
            output_dir=str(tmp_path),
            workspace_dir=str(tmp_path),
            report_llm_client=client,
        )
    assert not (tmp_path / "report.json").exists()
