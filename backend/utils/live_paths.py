"""Canonical paths shared by live session runtime consumers."""

from pathlib import Path


def live_notebook_path(states_root: str | Path, session_id: str) -> Path:
    """Return the sole live notebook path for a session."""
    return Path(states_root) / f"{session_id}.ipynb"
