import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import (
    AnalysisState, RoundResult, CrossModalSynthesis, CrossModalMechanismCandidate,
    LiteratureReport,
)
from backend.agents.pi_agent import PIAgent


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


@pytest.mark.asyncio
async def test_primary_injects_persisted_candidates():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=2, pi_summary="s")
    rr.cross_modal_synthesis = CrossModalSynthesis(
        matrix_markdown="| m |",
        candidates=[CrossModalMechanismCandidate(
            theme="THEME_MARKER", mechanism_statement="MECH_MARKER",
            confidence="medium", relation_to_current_direction="uncovered",
            report_recommendation="secondary",
        )],
    )
    state.round_results.append(rr)
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value="final report") as mock_chat:
        await agent.generate_final_summary(state)
    prompt = mock_chat.call_args[0][0]
    assert "THEME_MARKER" in prompt
    assert "MECH_MARKER" in prompt


@pytest.mark.asyncio
async def test_fallback_injects_literature_when_no_synthesis():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    state.round_results.append(RoundResult(round_number=1, pi_summary="s"))  # no cross_modal_synthesis
    state.literature_reports.append(LiteratureReport(topic="t", report="LIT_BODY_MARKER and more"))
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value="final report") as mock_chat:
        await agent.generate_final_summary(state)
    prompt = mock_chat.call_args[0][0]
    assert "Literature Context Summary" in prompt
    assert "LIT_BODY_MARKER" in prompt


@pytest.mark.asyncio
async def test_synthesis_present_but_no_candidates_falls_back():
    # O5: a CrossModalSynthesis with empty candidates must NOT trigger the primary
    # (empty) block — it falls back to the lightweight literature summary.
    agent = _make_pi_agent()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=1, pi_summary="s")
    rr.cross_modal_synthesis = CrossModalSynthesis(matrix_markdown="| m |", candidates=[])
    state.round_results.append(rr)
    state.literature_reports.append(LiteratureReport(topic="t", report="LIT_BODY_MARKER"))
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value="final report") as mock_chat:
        await agent.generate_final_summary(state)
    prompt = mock_chat.call_args[0][0]
    assert "Cross-Modal Mechanism Candidates (persisted audit)" not in prompt
    assert "Literature Context Summary" in prompt
    assert "LIT_BODY_MARKER" in prompt


@pytest.mark.asyncio
async def test_do_not_emphasize_candidates_excluded_from_final_summary():
    # Consistency with planning.py: do_not_emphasize candidates must NOT appear
    # in the final summary narrative (they remain in the appendix for traceability).
    agent = _make_pi_agent()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=2, pi_summary="s")
    rr.cross_modal_synthesis = CrossModalSynthesis(
        matrix_markdown="| m |",
        candidates=[
            CrossModalMechanismCandidate(
                theme="GOOD_CANDIDATE", mechanism_statement="m", confidence="medium",
                relation_to_current_direction="uncovered", report_recommendation="secondary",
            ),
            CrossModalMechanismCandidate(
                theme="SUPPRESSED_CANDIDATE", mechanism_statement="m", confidence="low",
                relation_to_current_direction="uncovered", report_recommendation="do_not_emphasize",
            ),
        ],
    )
    state.round_results.append(rr)
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value="final report") as mock_chat:
        await agent.generate_final_summary(state)
    prompt = mock_chat.call_args[0][0]
    assert "GOOD_CANDIDATE" in prompt
    assert "SUPPRESSED_CANDIDATE" not in prompt
