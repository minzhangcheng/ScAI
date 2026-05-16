import pytest
from backend.models.schemas import Snapshot, SnapshotPhase


class TestSnapshotPhaseCompat:
    """Verify SnapshotPhase handles historical phase values."""

    def test_current_phases_are_valid(self):
        """All current SnapshotPhase values should validate."""
        for phase in ["pre_annotation", "pre_direction", "pre_round_wait",
                       "pre_execution_stage1", "pre_execution_stage2",
                       "pre_execution_full", "pre_major_revision", "post_report"]:
            snap = Snapshot(name="test", phase=phase, round_number=1)
            assert snap.phase == phase

    def test_historical_post_annotation_phase_loads(self):
        """Historical snapshot with 'post_annotation' phase should load without error."""
        snap = Snapshot(name="test", phase="post_annotation", round_number=1)
        assert snap.phase == "post_annotation"

    def test_historical_post_direction_phase_loads(self):
        """Historical snapshot with 'post_direction' phase should load without error."""
        snap = Snapshot(name="test", phase="post_direction", round_number=1)
        assert snap.phase == "post_direction"

    def test_historical_manual_phase_loads(self):
        """Historical snapshot with 'manual' phase should load without error."""
        snap = Snapshot(name="test", phase="manual", round_number=1)
        assert snap.phase == "manual"