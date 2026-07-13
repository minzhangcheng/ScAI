"""Restore provenance helpers for first restored round recording."""
from __future__ import annotations

from typing import Any, Optional


def restore_provenance_id(state: Any) -> Optional[str]:
    for attr in ("restore_provenance_snapshot_id", "last_restored_snapshot_id"):
        value = getattr(state, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def seed_restore_provenance(state: Any) -> None:
    if state.last_restored_snapshot_id and not state.restore_provenance_snapshot_id:
        state.restore_provenance_snapshot_id = state.last_restored_snapshot_id


def consume_restore_provenance(state: Any) -> None:
    state.restore_provenance_snapshot_id = None
    state.last_restored_snapshot_id = None
