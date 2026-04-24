"""Verify dead-code removal: BioinformaticsCheckpoint should not exist in schemas."""
import pytest


def test_bioinformatics_checkpoint_removed():
    """BioinformaticsCheckpoint was dead code — confirm it no longer exists."""
    from backend.models import schemas
    assert not hasattr(schemas, "BioinformaticsCheckpoint"), (
        "BioinformaticsCheckpoint should have been removed — it was never used"
    )


def test_checkpoint_manager_still_exists():
    """The active CheckpointManager must remain intact."""
    from backend.utils.checkpoint import CheckpointManager

    assert CheckpointManager is not None
