from pathlib import Path

from backend.services.snapshot_v2.checkpoint_paths import (
    CHECKPOINT_SQLITE_FILENAME,
    checkpoint_sqlite_path,
)


def test_checkpoint_filename_matches_workflow_sqlite_saver_contract():
    assert CHECKPOINT_SQLITE_FILENAME == "checkpoints.sqlite"
    assert checkpoint_sqlite_path(Path("/tmp/states")) == Path(
        "/tmp/states/checkpoints.sqlite"
    )
