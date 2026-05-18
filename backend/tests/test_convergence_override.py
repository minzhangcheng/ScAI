"""Tests for convergence override conditions in workflow synthesis."""
from backend.models.schemas import AnalysisState, RoundResult


def _state_with_converged_rounds() -> AnalysisState:
    state = AnalysisState(user_id="u", initial_instruction="analyze")
    state.current_round = 2
    state.round_results = [
        RoundResult(round_number=1, new_findings=False),
        RoundResult(round_number=2, new_findings=False),
    ]
    return state


def test_user_text_input_forces_next_round_despite_convergence():
    from backend.workflow.synthesis import _should_force_next_round

    state = _state_with_converged_rounds()
    state.pending_human_input = "请重新分析 Kupffer cell 的炎症通路"

    assert _should_force_next_round(state) is True


def test_uploaded_files_force_next_round_despite_convergence():
    from backend.workflow.synthesis import _should_force_next_round

    state = _state_with_converged_rounds()
    state.pending_uploaded_files = [{"filename": "new_bulk_counts.csv"}]

    assert _should_force_next_round(state) is True


def test_remaining_direction_queue_forces_next_round():
    from backend.workflow.synthesis import _should_force_next_round

    state = _state_with_converged_rounds()
    state.user_chosen_directions = [
        {"direction": "A", "notes": ""},
        {"direction": "B", "notes": ""},
    ]
    state.directions_queue = [{"direction": "B", "notes": ""}]

    assert _should_force_next_round(state) is True


def test_empty_signals_do_not_force_next_round():
    from backend.workflow.synthesis import _should_force_next_round

    state = _state_with_converged_rounds()
    state.pending_human_input = "[CONTINUE]"
    state.pending_uploaded_files = []
    state.directions_queue = []

    assert _should_force_next_round(state) is False