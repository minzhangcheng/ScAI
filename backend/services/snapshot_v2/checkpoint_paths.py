"""Shared Snapshot V2 checkpoint SQLite path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

CHECKPOINT_SQLITE_FILENAME: Final[str] = "checkpoints.sqlite"


def checkpoint_sqlite_path(states_dir: str | Path) -> Path:
    """Return the LangGraph checkpoint SQLite path used by workflow/core.py."""
    return Path(states_dir) / CHECKPOINT_SQLITE_FILENAME
