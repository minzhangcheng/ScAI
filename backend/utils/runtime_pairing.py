"""Control-state identity paired with a committed runtime generation."""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path


def checkpoint_identity(session_id: str, states_dir=None) -> str | None:
    """Hash the latest checkpoints and their pending writes in one read snapshot.

    SQLite file bytes are not identity: WAL checkpointing/compaction changes
    them without changing a graph position. Read logical rows, including pending
    writes, which can advance a task without creating a new checkpoint ID.
    """
    from backend.config import get_session_states

    path = Path(states_dir or get_session_states(session_id)) / "checkpoints.sqlite"
    if not path.exists():
        return None
    digest = hashlib.sha256()
    count = 0
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "checkpoints" not in tables:
            return None
        latest = list(conn.execute(
            "SELECT checkpoint_ns, MAX(checkpoint_id) FROM checkpoints "
            "WHERE thread_id=? GROUP BY checkpoint_ns ORDER BY checkpoint_ns",
            (session_id,),
        ))
        for namespace, checkpoint_id in latest:
            for table, order in (("checkpoints", "checkpoint_id"), ("writes", "task_id, idx")):
                if table not in tables:
                    continue
                rows = conn.execute(
                    f"SELECT * FROM {table} WHERE thread_id=? AND checkpoint_ns=? "
                    f"AND checkpoint_id=? ORDER BY {order}",
                    (session_id, namespace, checkpoint_id),
                )
                for row in rows:
                    digest.update(table.encode())
                    for value in row:
                        payload = value if isinstance(value, bytes) else repr(value).encode()
                        digest.update(len(payload).to_bytes(8, "big"))
                        digest.update(payload)
                    count += 1
    return digest.hexdigest() if count else None


def capture_pairing(session_id: str, state, states_dir=None) -> dict | None:
    if state is None:
        return None  # Standalone/legacy sandbox without a workflow owner.
    return {
        "state_version": int(state.state_version),
        "checkpoint": checkpoint_identity(session_id, states_dir),
    }


def refresh_runtime_availability(state) -> None:
    """Check cold-resume file availability before the graph can advance.

    Absent persistence records alone do not establish loss for old sessions or
    standalone callers. A checkpoint or notebook record plus prior work does.
    Explicit recompute owns a deliberately empty namespace.
    """
    if getattr(state, "runtime_recompute_pending", False):
        return
    if int(getattr(state, "current_round", 0) or 0) <= 0:
        return
    from backend.config import get_session_states
    from backend.utils.runtime_apply import _live_pkl, RUNTIME_H5AD_MANIFEST_FILENAME

    sid = getattr(state, "id", None)
    if not sid:
        return
    states = get_session_states(sid)
    if _live_pkl(sid).exists() or (states / RUNTIME_H5AD_MANIFEST_FILENAME).exists():
        return
    if not ((states / f"{sid}_state.json").exists() or (states / "checkpoints.sqlite").exists()):
        return
    state.runtime_restore_status = "unavailable"
    state.runtime_restore_warning = "No runtime survived. Re-run from the input data to recompute it."
