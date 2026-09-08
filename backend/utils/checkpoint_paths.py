"""LangGraph checkpoint SQLite path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

CHECKPOINT_SQLITE_FILENAME: Final[str] = "checkpoints.sqlite"

#: Suffixes of the WAL-mode companions a SQLite database carries. The stash side
#: (restore) and the reinstate side (recovery) must agree exactly or a rollback
#: replays the wrong frames, so both read the pair from here.
SQLITE_COMPANION_SUFFIXES: Final[tuple[str, ...]] = ("-wal", "-shm")


def checkpoint_sqlite_path(states_dir: str | Path) -> Path:
    """Return the LangGraph checkpoint SQLite path used by workflow/core.py."""
    return Path(states_dir) / CHECKPOINT_SQLITE_FILENAME


def sqlite_companion_paths(db_path: str | Path) -> tuple[Path, ...]:
    """Return the ``-wal``/``-shm`` companion paths of *db_path*."""
    db_path = Path(str(db_path))
    return tuple(
        db_path.with_name(db_path.name + companion)
        for companion in SQLITE_COMPANION_SUFFIXES
    )
