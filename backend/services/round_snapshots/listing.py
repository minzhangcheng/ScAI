"""What one session's snapshot storage holds. Part B spec 4.8.

Reads manifests and failed-capture records off disk and turns them into the
rows the API, the frontend and the CLI all share -- including the deletion
order, computed once here rather than three times downstream.
"""
from __future__ import annotations

import json
import logging
import stat
from pathlib import Path
from typing import Any

from .gates import mark_superseded, with_deletion_order
from .layout import (
    FAILED_SUFFIX,
    MANIFEST_NAME,
    SnapshotsDisabled,
    parse_snapshot_id,
    snapshots_root,
)
from .manifest import SnapshotManifestError, read_snapshot_manifest

logger = logging.getLogger(__name__)


def _row(manifest: dict[str, Any]) -> dict[str, Any]:
    coverage = manifest.get("coverage") or {}
    return {
        "snapshot_id": manifest["snapshot_id"],
        "round": manifest["round"],
        "gate": manifest["gate"],
        "runtime_revision": manifest["runtime_revision"],
        "created_at": manifest["created_at"],
        "bytes_total": manifest["bytes_total"],
        "capture_seconds": manifest["capture_seconds"],
        "retention_rank": manifest["retention_rank"],
        "saved_names": coverage.get("saved_names", 0),
        "skipped_count": len(coverage.get("skipped_names") or []),
        "executor_checkpoint_present": manifest["executor_checkpoint_present"],
        "scai_version": manifest["scai_version"],
    }


def _failed_row(path: Path, snapshot: str, *, reason: str | None = None) -> dict[str, Any]:
    """A decision that has no restore point, and what is known about why.

    Read leniently and never signed: nothing restores from this record, so it
    cannot influence a restore, and refusing to display a truncated one would
    hide the very fact it exists to report.
    """
    payload: dict[str, Any] = {}
    if reason is None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, ValueError):
            logger.warning("Unreadable snapshot failure record: %s", path)
    try:
        round_number, gate, revision = parse_snapshot_id(snapshot)
    except ValueError:
        round_number, gate, revision = 0, "", 0
    return {
        "snapshot_id": snapshot,
        "round": round_number,
        "gate": gate,
        "runtime_revision": revision,
        "reason": reason or str(payload.get("reason") or "unknown"),
        "at": payload.get("at"),
        "bytes_written": payload.get("bytes_written"),
    }


def read_session_listing(session_id: str) -> dict[str, Any]:
    """Every snapshot and every failed capture this session has.

    A bundle whose manifest does not verify is reported as a failed capture
    rather than as a snapshot: something is on disk, it cannot be restored, and
    silently omitting it would leave the space it occupies unexplained.
    """
    try:
        root = snapshots_root(session_id)
    except SnapshotsDisabled:
        return {"snapshots": [], "failed": [], "bytes_total": 0}
    if not root.is_dir():
        return {"snapshots": [], "failed": [], "bytes_total": 0}

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            try:
                parse_snapshot_id(entry.name)
            except ValueError:
                continue
            try:
                rows.append(_row(read_snapshot_manifest(entry / MANIFEST_NAME)))
            except (SnapshotManifestError, KeyError) as exc:
                logger.warning("Unusable snapshot bundle %s: %s", entry, exc)
                failed.append(_failed_row(entry, entry.name, reason="unverified"))
            continue
        if entry.is_file() and entry.name.endswith(FAILED_SUFFIX):
            failed.append(
                _failed_row(entry, entry.name[: -len(FAILED_SUFFIX)])
            )

    # A decision that failed once and was captured on a later attempt is not
    # a gap. The two records can share an identifier because the failure is
    # named before the capture knows which revision it will commit.
    published = {row["snapshot_id"] for row in rows}
    failed = [row for row in failed if row["snapshot_id"] not in published]

    rows = with_deletion_order(mark_superseded(rows))
    rows.sort(key=lambda row: (row["round"], row["retention_rank"], row["runtime_revision"]))
    failed.sort(key=lambda row: (row["round"], row["snapshot_id"]))
    return {
        "snapshots": rows,
        "failed": failed,
        "bytes_total": _occupied_bytes(root),
    }


def _occupied_bytes(root: Path) -> int:
    """What this session's snapshots actually cost, counting shares once.

    Summing each manifest's ``bytes_total`` would double-count a dump two
    snapshots share, and the sharing is a hardlink -- so the filesystem is the
    only thing that knows, and an inode seen twice is counted once.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*"):
        try:
            info = path.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += info.st_size
    return total
