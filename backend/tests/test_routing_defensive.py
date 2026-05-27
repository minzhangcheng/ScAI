"""Defensive guards: node protection, review-gate fallback, supplementary None."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.workflow import SingleCellAnalysisWorkflow
from backend.models.schemas import AnalysisState


@pytest.mark.asyncio
async def test_initialize_round_is_retry_wrapped():
    """A skip_nodes entry short-circuits the body — proof the node is
    wrapped by @with_retry (which provides exception protection)."""
    mock = MagicMock()
    mock.session_id = "s"
    mock._pause_event = None
    mock._send_message = AsyncMock()
    mock.pi_agent = MagicMock()
    mock.pi_agent.process_user_input = AsyncMock()

    bound = SingleCellAnalysisWorkflow._initialize_round.__get__(mock, type(mock))
    state = {"analysis_state": AnalysisState(user_id="u"),
             "skip_nodes": ["initialize_round"]}
    result = await bound(state)

    # If wrapped, the skip branch returns before touching the PI agent.
    mock.pi_agent.process_user_input.assert_not_awaited()
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_review_gate_sets_should_continue_false_when_no_plan():
    mock = MagicMock()
    mock.session_id = "s"
    mock._send_message = AsyncMock()
    s = AnalysisState(user_id="u")
    s.current_plan = None
    state = {"analysis_state": s}

    bound = SingleCellAnalysisWorkflow._user_review_gate.__get__(mock, type(mock))
    result = await bound(state)
    # No plan ⇒ must mark termination so _after_review aborts (not "revise").
    assert result["should_continue"] is False


def test_check_convergence_missing_key_continues():
    mock = MagicMock()
    mock.session_id = "s"
    fn = SingleCellAnalysisWorkflow._check_convergence_condition.__get__(mock, type(mock))
    # Missing should_continue must not silently stop a live run.
    assert fn({}) == "continue"