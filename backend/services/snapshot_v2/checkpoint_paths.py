"""Deprecated location. Snapshot V2 is deleted in A2-5; import from
``backend.utils.checkpoint_paths`` instead."""

from backend.utils.checkpoint_paths import (  # noqa: F401
    CHECKPOINT_SQLITE_FILENAME,
    SQLITE_COMPANION_SUFFIXES,
    checkpoint_sqlite_path,
    sqlite_companion_paths,
)
