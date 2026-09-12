"""Restore provenance helpers for first restored round recording.

Two lifetimes live here and must not be conflated:

* ``restore_provenance_snapshot_id`` is one-shot: it is attached to the first
  canonical branch result and then cleared.
* ``last_restored_snapshot_id`` is durable for the life of the child: it is the
  parent/child listing and immediate-origin identity.

``restore_provenance_id`` falls back to the durable origin only for legacy
states that never received the one-shot field, and only while no existing
``RoundResult`` already carries that origin. That keeps first-result
attribution single-use instead of repeating it on every future result.
"""
from __future__ import annotations

from typing import Any, Optional


def _round_result_already_attributes(state: Any, snapshot_id: str) -> bool:
    for result in getattr(state, "round_results", None) or []:
        if getattr(result, "restored_from_snapshot_id", None) == snapshot_id:
            return True
    return False


def restore_provenance_id(state: Any) -> Optional[str]:
    """The origin for the next canonical branch result, if one is pending."""
    one_shot = getattr(state, "restore_provenance_snapshot_id", None)
    if isinstance(one_shot, str) and one_shot:
        return one_shot
    durable = getattr(state, "last_restored_snapshot_id", None)
    if (
        isinstance(durable, str)
        and durable
        and not _round_result_already_attributes(state, durable)
    ):
        return durable
    return None


def consume_restore_provenance(state: Any) -> None:
    """Clear only the one-shot provenance, never the durable restore origin."""
    state.restore_provenance_snapshot_id = None
