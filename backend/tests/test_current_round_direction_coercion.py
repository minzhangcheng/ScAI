from backend.models.schemas import AnalysisState


def test_none_current_round_direction_coerced_to_empty_string():
    state = AnalysisState(user_id="u1", current_round_direction=None)
    assert state.current_round_direction == ""


def test_valid_current_round_direction_preserved():
    state = AnalysisState(user_id="u1", current_round_direction="liver fibrosis")
    assert state.current_round_direction == "liver fibrosis"
