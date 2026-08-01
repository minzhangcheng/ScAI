"""The degraded fallback report keeps its own shape.

When the report model fails, _fallback_report emits a fixed three-section body
whose ids belong to the single-pass template. Forcing that through a
round-derived outline would drop or reshuffle it, so the outline passes are
skipped and its figures fall through to the supplementary section.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import AnalysisState, RoundResult

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _state_with_rounds():
    st = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    st.name = "proj"
    for n, phase, direction in (
        (1, "exploration", ""),
        (2, "innovation", "direction A"),
        (3, "integration", ""),
    ):
        st.round_results.append(RoundResult(
            round_number=n, phase=phase, direction=direction,
            timestamp=_T0 + timedelta(minutes=n * 10),
        ))
    return st


def _agent(chat_side_effect):
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = "t"
    agent.refresh_runtime_config_if_needed = MagicMock()
    agent.get_supervision_log = MagicMock(return_value=[])
    agent.get_summary_stats = MagicMock(return_value={
        "total_steps_monitored": 3,
        "steps_with_technical_issues": 0,
        "pi_escalations": 0,
    })
    agent._set_llm_context = MagicMock()
    agent._build_appendix = MagicMock(return_value={})
    client = MagicMock()
    client.set_context = MagicMock()
    client.chat = AsyncMock(side_effect=chat_side_effect)
    client.base_url = "http://x"
    # llm_client is a read-only property; the client is injected through the
    # report_llm_client argument instead, which takes precedence over it.
    agent._provider = MagicMock()
    return agent, client


@pytest.mark.asyncio
async def test_fallback_report_keeps_its_own_section_order(tmp_path):
    agent, client = _agent(RuntimeError("model down"))
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    report = await agent.generate_detailed_report(
        _state_with_rounds(), "model-x", str(out), workspace_dir=str(ws),
        report_llm_client=client,
    )

    ids = [s["id"] for s in report["sections"]]
    # abstract and conclusions are lifted out; the middle section survives
    # intact rather than being relocated to the end by outline alignment.
    assert ids == ["analysis_overview"]
    assert report["abstract"]
    assert report["conclusions"]


@pytest.mark.asyncio
async def test_successful_report_is_aligned_to_the_round_outline(tmp_path):
    body = (
        "## [conclusions] Wrap up\nend matter\n\n"
        "## [direction_1] What direction A showed\nbody\n\n"
        "## [abstract] Summary\nsummary text\n\n"
        "## [methods] How it was done\nprocedure\n"
    )
    agent, client = _agent([SimpleNamespace(content=body)])
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    report = await agent.generate_detailed_report(
        _state_with_rounds(), "model-x", str(out), workspace_dir=str(ws),
        report_llm_client=client,
    )

    ids = [s["id"] for s in report["sections"]]
    # Emitted conclusions-first; the outline puts methods before the direction
    # and abstract/conclusions are extracted, so what remains is in plan order.
    assert ids == ["methods", "direction_1"]
