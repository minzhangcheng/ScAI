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


def is_live_runtime_continuation_safe(session_id: str, state, workflow=None) -> bool:
    """Use a parked graph's intact namespace instead of its older disk dump.

    The epoch is latched on the owner's first kernel start. A restart cannot
    inherit this proof, even if it retains the graph pointers and gate events.
    Cold owners are checked before they have a live graph state.
    """
    from backend.state import workflows

    owner = workflows.get(session_id)
    if owner is None or (workflow is not None and owner is not workflow):
        return False
    if getattr(owner, "session_id", None) != session_id or getattr(state, "id", None) != session_id:
        return False
    graph = getattr(owner, "_current_graph_state", None)
    if not isinstance(graph, dict) or getattr(graph.get("analysis_state"), "id", None) != session_id:
        return False
    if getattr(getattr(owner, "_current_analysis_state", None), "id", None) != session_id:
        return False
    sandbox = getattr(owner, "sandbox", None)
    if sandbox is None or not getattr(sandbox, "_initialized", False):
        return False
    if any(getattr(sandbox, flag, False) for flag in (
        "_kernel_dead", "_session_restore_failed", "_restarting", "_starting",
    )):
        return False
    epoch = getattr(owner, "_continuation_kernel_epoch", None)
    if epoch is None or epoch != getattr(sandbox, "kernel_epoch", None):
        return False
    if getattr(sandbox, "kc", None) is None or getattr(sandbox, "km", None) is None:
        return False
    try:
        if not sandbox.km.is_alive():
            return False
    except Exception:
        return False
    if any(getattr(owner, flag, False) for flag in (
        "_close_complete", "_discard_runtime_on_close", "_restore_superseded",
    )):
        return False
    closing = getattr(owner, "_close_task", None)
    if closing is not None and not closing.done():
        return False
    generation = int(getattr(state, "lifecycle_generation", 0) or 0)
    if generation != getattr(sandbox, "_expected_lifecycle_generation", None):
        return False
    if generation != getattr(sandbox.notebook_state, "lifecycle_generation", None):
        return False
    status = getattr(getattr(state, "status", None), "value", getattr(state, "status", None))
    pause = getattr(owner, "_pause_event", None)
    if pause is None:
        return False
    if status == "paused":
        return not pause.is_set()
    gates = {
        "waiting_input": "_continue_event",
        "waiting_annotation": "_annotation_event",
        "waiting_plan_review": "_plan_review_event",
        "waiting_species_input": "_species_event",
    }
    gate = getattr(owner, gates.get(status, ""), None)
    return pause.is_set() and gate is not None and not gate.is_set()


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
    from backend.utils.runtime_apply import _live_pkl

    sid = getattr(state, "id", None)
    if not sid:
        return
    states = get_session_states(sid)
    if _live_pkl(sid).exists():
        return
    if not ((states / f"{sid}_state.json").exists() or (states / "checkpoints.sqlite").exists()):
        return
    state.runtime_restore_status = "unavailable"
    state.runtime_restore_warning = "No runtime survived. Re-run from the input data to recompute it."
