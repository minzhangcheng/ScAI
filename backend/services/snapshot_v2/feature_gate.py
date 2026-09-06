"""Central operator feature gate for Snapshot V2 HTTP operations."""

from fastapi import HTTPException

from backend.config import is_snapshot_v2_enabled


def require_snapshot_v2_enabled() -> None:
    """Reject Snapshot V2 HTTP operations when the subsystem is disabled."""
    if not is_snapshot_v2_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SNAPSHOT_V2_DISABLED",
                "message": "Snapshot V2 is disabled",
            },
        )


# Journal kinds whose unfinished transaction really does leave the session's own
# files — state, notebook, checkpoint, workspace — at different points in the
# run. ``journals/`` is a single directory shared by every kind (``paths.py``),
# so status alone cannot answer the question this predicate asks: a delete
# (``deletion.py``) or an import touches only ``objects/`` and ``committed/``
# and passes through STAGED/APPLYING inside a try whose except deliberately
# leaves the journal behind, and an export protection journal is not a mutation
# at all. Only a restore swaps the four session files against each other.
# Legacy flat restore journals carry no ``kind``, and one whose kind cannot be
# read is counted as a restore for the same fail-closed reason an unparseable
# journal is.
_HALF_SWAP_JOURNAL_KINDS = frozenset({"", "restore", "restore_apply", "restore_job"})


def has_unfinished_restore_journal(session_id: str) -> bool:
    """True while a STAGED/STAGING/APPLYING *restore* journal is still on disk.

    Also drives the startup scan in ``backend/main.py``. Only a restore that
    never reached a terminal state leaves the session's files mixing pre- and
    post-restore content; COMMITTED and FAILED journals are merely waiting to be
    reaped, and a stranded journal of any other kind describes work that never
    touched those files. Anything unreadable counts as unfinished, because a
    truncated write is itself evidence of an interrupted one.
    """
    import json

    from backend.services.snapshot_v2.paths import SnapshotV2Paths

    if not session_id:
        return False
    unfinished = {"STAGED", "STAGING", "APPLYING"}
    try:
        journals = SnapshotV2Paths.for_session(session_id).journals
        if not journals.is_dir():
            return False
        for entry in journals.iterdir():
            if not entry.is_file():
                continue
            try:
                doc = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return True
            if not isinstance(doc, dict):
                return True
            if str(doc.get("status", "")).upper() not in unfinished:
                continue
            kind = doc.get("kind")
            kind = "" if kind is None else str(kind).strip().lower()
            if kind in _HALF_SWAP_JOURNAL_KINDS:
                return True
    except OSError:
        # Unknown means fail closed: the flag exists to stop a half-applied
        # restore from being driven forward.
        return True
    return False
