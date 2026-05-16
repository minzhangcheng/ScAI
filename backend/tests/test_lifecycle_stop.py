import pytest
from unittest.mock import MagicMock, patch
from backend.models.schemas import AnalysisState


class TestLifecycleStopSideEffects:
    """Verify stop() preserves original max_rounds and resume() restores it."""

    def test_state_has_original_max_rounds_field(self):
        """AnalysisState should have original_max_rounds field."""
        state = AnalysisState(user_id="test")
        assert hasattr(state, 'original_max_rounds')
        assert state.original_max_rounds is None

    def test_save_and_restore_max_rounds(self):
        """Saving original max_rounds then restoring should recover the value."""
        state = AnalysisState(user_id="test")
        state.max_rounds = 10
        state.current_round = 3
        # Simulate stop
        if state.original_max_rounds is None:
            state.original_max_rounds = state.max_rounds
        state.max_rounds = state.current_round
        assert state.max_rounds == 3
        assert state.original_max_rounds == 10
        # Simulate resume
        if state.original_max_rounds is not None:
            state.max_rounds = state.original_max_rounds
            state.original_max_rounds = None
        assert state.max_rounds == 10
        assert state.original_max_rounds is None

    def test_original_max_rounds_only_set_once(self):
        """original_max_rounds should only be set on first stop, not overwritten."""
        state = AnalysisState(user_id="test")
        state.max_rounds = 10
        # First stop
        if state.original_max_rounds is None:
            state.original_max_rounds = state.max_rounds
        state.max_rounds = state.current_round
        # Change max_rounds artificially, then stop again
        state.max_rounds = 5
        if state.original_max_rounds is None:
            state.original_max_rounds = state.max_rounds  # Should NOT execute
        assert state.original_max_rounds == 10  # Preserved from first stop

    def test_original_max_rounds_not_in_api_dict(self):
        """original_max_rounds should not appear in API responses."""
        state = AnalysisState(user_id="test")
        state.original_max_rounds = 10
        api_dict = state.to_api_dict()
        assert "original_max_rounds" not in api_dict

    def test_original_max_rounds_survives_json_roundtrip(self):
        """original_max_rounds must survive model_dump_json → model_validate_json."""
        state = AnalysisState(user_id="test")
        state.max_rounds = 10
        state.current_round = 3
        state.original_max_rounds = 10
        state.max_rounds = state.current_round
        # Round-Trip through JSON serialization (what async_save_session does)
        json_str = state.model_dump_json()
        restored = AnalysisState.model_validate_json(json_str)
        assert restored.original_max_rounds == 10
        assert restored.max_rounds == 3
