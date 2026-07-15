"""Lineage resolution shared by normal Snapshot V2 producers."""

from __future__ import annotations

from typing import Any, Optional


def _clean_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def capture_lineage_from_state(state: Any) -> dict[str, Optional[str]]:
    """Return the parent and active branch for a forward capture."""
    branch_id = _clean_optional(getattr(state, "restore_branch_id", None)) or "main"
    branch_label = (
        _clean_optional(getattr(state, "restore_branch_label", None)) or branch_id
    )
    return {
        "parent_id": _clean_optional(getattr(state, "last_snapshot_id", None)),
        "branch_id": branch_id,
        "branch_label": branch_label,
    }
