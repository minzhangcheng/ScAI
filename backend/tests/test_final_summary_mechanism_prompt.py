import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisState, RoundResult
from backend.models.schema_parts.analysis import DataDrivenMechanismProgram
from backend.services.report.sections import _build_direction_report_context
from backend.utils.direction_payload import collect_direction_projections


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


@pytest.mark.asyncio
async def test_early_selected_direction_remains_in_final_summary_context():
    """Direction evidence survives later rounds and remains lineage-scoped."""
    agent = _make_pi_agent()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    state.round_results.append(RoundResult(
        round_number=2,
        phase="innovation",
        direction="EARLY_DIRECTION_MARKER",
        lineage_id="main",
        pi_summary="early result",
        data_driven_mechanism_programs=[DataDrivenMechanismProgram(
            round_number=2,
            phase="innovation",
            label="EARLY_MECHANISM_MARKER",
            triggering_observations=["early observation"],
            supporting_genes=["GeneEarly"],
            missing_evidence=["early limiting evidence"],
        )],
    ))
    for round_number in range(3, 8):
        state.round_results.append(RoundResult(
            round_number=round_number,
            phase="innovation",
            direction=f"Later direction {round_number}",
            lineage_id="main",
            pi_summary=f"later result {round_number}",
        ))
    state.round_results.append(RoundResult(
        round_number=2,
        phase="innovation",
        direction="WRONG_LINEAGE_MARKER",
        lineage_id="branch-a",
        pi_summary="branch-only result",
        data_driven_mechanism_programs=[DataDrivenMechanismProgram(
            round_number=2,
            phase="innovation",
            label="WRONG_LINEAGE_MECHANISM_MARKER",
        )],
    ))

    with patch.object(agent, "chat", new_callable=AsyncMock, return_value="final report") as mock_chat:
        await agent.generate_final_summary(state)

    prompt = mock_chat.call_args[0][0]
    assert "early_direction_marker" in prompt
    assert "EARLY_MECHANISM_MARKER" in prompt
    assert "early limiting evidence" in prompt
    assert "WRONG_LINEAGE_MECHANISM_MARKER" not in prompt


@pytest.mark.asyncio
async def test_thirteenth_selected_direction_is_disclosed_as_context_omission():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    for round_number in range(1, 14):
        state.round_results.append(RoundResult(
            round_number=round_number,
            phase="innovation",
            direction=f"Selected direction {round_number}",
            lineage_id="main",
            pi_summary=f"Result marker {round_number}",
        ))

    projections = collect_direction_projections(state)
    assert len(projections) == 12
    assert "1 selected innovation direction(s) omitted" in " ".join(
        projections[0]["omissions"]
    )
    report_context = _build_direction_report_context(state)
    for round_number in range(1, 13):
        assert f"direction: selected direction {round_number}" in report_context.lower()
    assert "direction: selected direction 13" not in report_context.lower()
    assert "1 selected innovation direction(s) omitted" in report_context

    with patch.object(agent, "chat", new_callable=AsyncMock, return_value="final report") as mock_chat:
        await agent.generate_final_summary(state)
    final_prompt = mock_chat.call_args[0][0]
    for round_number in range(1, 13):
        assert f"direction: selected direction {round_number}" in final_prompt.lower()
    assert "direction: selected direction 13" not in final_prompt.lower()
    assert "1 selected innovation direction(s) omitted" in final_prompt
