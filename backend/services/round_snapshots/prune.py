"""Reclaim the least valuable snapshots when the volume fills. Part B spec 4.7.

Off unless the operator sets a reserve. That is the project owner's decision
and it is the right default: a snapshot is the only copy of the decision it
precedes, so giving one up is a judgement about what a user is likely to want
back, not bookkeeping.

When it is armed, the order is not this module's to invent. Each manifest
records the ``retention_rank`` its decision was given at capture, and the
listing turns that into a ``deletion_order`` the browser and the CLI show; this
walks the same order, so what an operator was told would go first is what goes
first.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .gates import in_deletion_order
from .layout import (
    SnapshotsDisabled,
    snapshot_dir,
    snapshot_storage_root,
)
from .listing import read_session_listing

logger = logging.getLogger(__name__)

#: The one decision a prune will not take on a session that is still working.
#: Rank 1 is the cell annotation, the earliest decision an analysis makes and
#: the one every later label depends on; reclaiming it out from under a running
#: session would give up the restore point most likely to be wanted.
_PROTECTED_RANK = 1


def free_space_bytes(path: Path) -> int:
    import os

    try:
        usage = os.statvfs(path)
    except OSError:
        # An unreadable volume is not evidence of pressure. Reporting zero here
        # would make every capture delete something.
        logger.warning("Could not read free space on %s", path, exc_info=True)
        return -1
    return usage.f_bavail * usage.f_frsize


def _candidates(*, keep: str | None, keep_session: str | None) -> list[dict[str, Any]]:
    """Every snapshot on this installation, ordered by which to give up first.

    Across sessions, not within one: pressure is a property of the volume, and
    a session that has finished holds restore points nobody is going to answer
    while a running one holds the decision in front of the user.
    """
    try:
        root = snapshot_storage_root()
    except SnapshotsDisabled:
        return []
    if not root.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        session_id = entry.name
        try:
            listing = read_session_listing(session_id)
        except Exception:
            logger.warning("Could not list snapshots for %s", session_id, exc_info=True)
            continue
        for row in listing["snapshots"]:
            if row["snapshot_id"] == keep and session_id == keep_session:
                continue
            rows.append({**row, "session_id": session_id})
    return in_deletion_order(rows)


def _is_running(session_id: str) -> bool:
    """Whether an analysis still owns this session in this process.

    Read from the live registries rather than from the persisted status: a
    status on disk can say ``running`` for a session whose process is gone,
    and the question here is whether anyone is about to be asked the decision
    this snapshot precedes.
    """
    from backend.state import sessions as live_sessions, workflows

    if session_id in workflows:
        return True
    state = live_sessions.get(session_id)
    status = getattr(getattr(state, "status", None), "value", None)
    return status in {
        "running", "waiting_input", "waiting_annotation",
        "waiting_plan_review", "waiting_species_input", "generating_report",
    }


def _remove(session_id: str, snapshot: str) -> int:
    """Rename aside, then remove. Returns the bytes the manifest claimed."""
    bundle = snapshot_dir(session_id, snapshot)
    if not bundle.is_dir():
        return 0
    doomed = bundle.with_name(f".deleting-{bundle.name}")
    shutil.rmtree(doomed, ignore_errors=True)
    bundle.rename(doomed)
    shutil.rmtree(doomed, ignore_errors=True)
    return 1


def prune_for_free_space(*, keep_session: str | None = None, keep: str | None = None) -> list[str]:
    """Give up snapshots until the volume is back above its reserve.

    Returns the identifiers removed, most valuable last. Never removes the
    snapshot named by *keep* -- the capture that just wrote it is the caller --
    and never a rank-1 snapshot of a session that is still running.

    Runs only from the writer, so an installation with the feature off prunes
    nothing, and one with no reserve set prunes nothing either.
    """
    from backend.config import snapshot_free_space_reserve_bytes

    reserve = snapshot_free_space_reserve_bytes()
    if reserve is None:
        return []
    try:
        root = snapshot_storage_root()
    except SnapshotsDisabled:
        return []

    removed: list[str] = []
    for row in _candidates(keep=keep, keep_session=keep_session):
        free = free_space_bytes(root)
        if free < 0 or free >= reserve:
            break
        if int(row["retention_rank"]) == _PROTECTED_RANK and _is_running(row["session_id"]):
            continue
        try:
            _remove(row["session_id"], row["snapshot_id"])
        except OSError:
            logger.warning(
                "Could not reclaim snapshot %s/%s",
                row["session_id"], row["snapshot_id"], exc_info=True,
            )
            continue
        removed.append(row["snapshot_id"])
        logger.info(
            "Reclaimed snapshot %s/%s (rank %s) to stay above the free-space "
            "reserve", row["session_id"], row["snapshot_id"], row["retention_rank"],
        )
    return removed
