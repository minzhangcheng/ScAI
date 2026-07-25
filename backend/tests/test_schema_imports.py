"""Verify dead-code removal: BioinformaticsCheckpoint should not exist in schemas."""
import pytest

from backend.tests.split_facade import binds


def test_bioinformatics_checkpoint_removed():
    """BioinformaticsCheckpoint was dead code — confirm it no longer exists."""
    found = binds("backend.models.schemas", "BioinformaticsCheckpoint")
    assert not found, (
        "BioinformaticsCheckpoint should have been removed — it was never used "
        f"(still bound in {[module.__name__ for module in found]})"
    )


def test_checkpoint_manager_still_exists():
    """The active CheckpointManager must remain intact."""
    from backend.utils.checkpoint import CheckpointManager

    assert CheckpointManager is not None
