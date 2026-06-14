"""Tests for [PI_DECIDE] / [CONTINUE] fast-paths in PIAgent.process_user_input."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.models.schemas import AnalysisState, AnalysisPhase
from backend.agents.pi_agent import PIAgent
from backend.workflow import SingleCellAnalysisWorkflow


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _make_state():
    return AnalysisState(
        user_id="u1",
        name="DKO liver",
        initial_instruction="Analyze DKO mouse liver snRNA-seq",
    )


class TestPiDecideFastPath:
    @pytest.mark.asyncio
    async def test_pi_decide_returns_continue_without_calling_llm(self):
        """[PI_DECIDE] must deterministically continue so pi_planning can
        select directions — it must NOT be sent to the stop/continue LLM."""
        agent = _make_pi_agent()
        state = _make_state()
        with patch.object(agent, "chat", new_callable=AsyncMock) as mock_chat:
            result = await agent.process_user_input(state, "[PI_DECIDE]", [])
        assert result["action"] == "continue"
        mock_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pi_decide_with_surrounding_whitespace(self):
        agent = _make_pi_agent()
        state = _make_state()
        with patch.object(agent, "chat", new_callable=AsyncMock) as mock_chat:
            result = await agent.process_user_input(state, "  [PI_DECIDE]  ", [])
        assert result["action"] == "continue"
        mock_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_continue_marker_still_continues_without_llm(self):
        """Regression: existing [CONTINUE] fast-path unchanged."""
        agent = _make_pi_agent()
        state = _make_state()
        with patch.object(agent, "chat", new_callable=AsyncMock) as mock_chat:
            result = await agent.process_user_input(state, "[CONTINUE]", [])
        assert result["action"] == "continue"
        mock_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_real_message_still_uses_llm(self):
        """Regression: a genuine user message still goes through the LLM
        classifier and can stop."""
        agent = _make_pi_agent()
        state = _make_state()
        response = json.dumps({"action": "stop", "response": "Done."})
        with patch.object(agent, "chat", new_callable=AsyncMock,
                          return_value=response) as mock_chat:
            result = await agent.process_user_input(
                state, "Please stop and write the report", []
            )
        assert result["action"] == "stop"
        mock_chat.assert_awaited_once()


class TestInitializeRoundDoesNotStopOnPiDecide:
    """Workflow-level regression: the no-wait [PI_DECIDE] sentinel reaching
    _initialize_round via pending_human_input must NOT route to the final
    report. This is the end-to-end path that ended session 505ff8cf after
    Round 1 before the fast-path fix."""

    def _make_workflow_mock(self, pi_agent):
        mock = MagicMock()
        mock.session_id = "s"
        mock._pause_event = None
        mock._send_message = AsyncMock()
        mock._send_state_sync = AsyncMock()
        mock.pi_agent = pi_agent
        return mock

    @pytest.mark.asyncio
    async def test_pi_decide_input_continues_into_planning(self):
        # Real PIAgent so the [PI_DECIDE] fast-path actually executes (no LLM).
        agent = _make_pi_agent()
        mock = self._make_workflow_mock(agent)

        s = _make_state()
        s.analysis_phase = AnalysisPhase.INNOVATION
        s.current_round = 1
        s.max_rounds = 3
        s.user_chosen_direction = "[PI_DECIDE]"
        s.pending_human_input = "[PI_DECIDE]"
        state = {"analysis_state": s}

        # Guard: if the fast-path regressed and the LLM classifier ran, fail
        # loudly instead of making a network call.
        with patch.object(agent, "chat", new_callable=AsyncMock,
                          side_effect=AssertionError("LLM must not be called for [PI_DECIDE]")):
            result = await SingleCellAnalysisWorkflow._initialize_round(mock, state)

        # Must NOT route to final report.
        assert result.get("human_stop_requested") is not True
        assert result.get("user_action") == "continue"
        assert result.get("should_continue") is True
        # Direction sentinel preserved for _pi_planning's select_best_directions();
        # only the pending_human_input transport field is cleared.
        assert s.user_chosen_direction == "[PI_DECIDE]"
        assert s.pending_human_input is None
        assert s.current_round == 2

    @pytest.mark.asyncio
    async def test_explicit_stop_still_routes_to_final_report(self):
        """Control: a real stop request must still short-circuit to the
        final report, proving the test above isn't masking that path."""
        agent = MagicMock()
        agent.process_user_input = AsyncMock(return_value={"action": "stop",
                                                           "response": "Done."})
        mock = self._make_workflow_mock(agent)

        s = _make_state()
        s.analysis_phase = AnalysisPhase.INNOVATION
        s.current_round = 1
        s.pending_human_input = "stop please"
        state = {"analysis_state": s}

        result = await SingleCellAnalysisWorkflow._initialize_round(mock, state)

        assert result.get("human_stop_requested") is True
        assert result.get("should_continue") is False
