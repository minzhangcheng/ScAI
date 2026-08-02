"""Global in-memory state for backend sessions, workflows, and tasks.

Centralizes the dicts/locks that multiple modules (main, services, tasks,
routes) must share, to avoid circular imports.
"""
import asyncio
import time
from typing import Any, Dict, Optional


# In-memory session store: session_id -> AnalysisState
sessions: Dict[str, Any] = {}

# Running workflow instances: session_id -> SingleCellAnalysisWorkflow
workflows: Dict[str, Any] = {}

# Standalone partial-report workflow instances. These are intentionally
# separate from ``workflows`` because they must not make a WebSocket reconnect
# look like a graph analysis is running, but deletion still needs an ownership
# handle until every resource has closed.
report_workflows: Dict[str, Any] = {}

# Background run tasks: session_id -> asyncio.Task
_run_tasks: Dict[str, Any] = {}

# Sessions currently being initialized (prevents concurrent workflow creation).
initializing_sessions: set = set()
# Identity owners for producer-created initialization markers. A restore/test
# may still install a set-only marker; producer cleanup must never clear one it
# did not claim.
initializing_session_owners: Dict[str, object] = {}


def claim_initializing_session(session_id: str) -> object:
    """Install and return an identity token for one initialization owner."""
    token = object()
    initializing_sessions.add(session_id)
    initializing_session_owners[session_id] = token
    return token


def release_initializing_session(
    session_id: str,
    token: object | None = None,
) -> bool:
    """Clear an initialization marker unconditionally or by exact ownership."""
    if token is not None and initializing_session_owners.get(session_id) is not token:
        return False
    initializing_sessions.discard(session_id)
    initializing_session_owners.pop(session_id, None)
    return True

# Lock for protecting concurrent access to the dicts above.
state_lock: asyncio.Lock = asyncio.Lock()

# Per-session locks for finer-grained save/evict operations.
session_locks: Dict[str, asyncio.Lock] = {}

# Per-session locks for the snapshot-failure HITL actions (retry/skip/terminate).
# Deliberately separate from session_locks - see get_snapshot_action_lock.
snapshot_action_locks: Dict[str, asyncio.Lock] = {}

# Last time each session was accessed, used by cache eviction.
session_access_times: Dict[str, float] = {}

# Last save time per session (save throttle).
last_save_times: Dict[str, float] = {}


def get_session_lock(session_id: str) -> asyncio.Lock:
    """Return the asyncio.Lock for a session, creating it on first use.

    Used to serialize mutating route handlers (read-modify-save) and the
    WebSocket workflow create-or-observe decision so concurrent calls cannot
    lose updates or both create a duplicate workflow (P1-8/P1-9).

    Safe under asyncio: the dict check + insert below contains no `await`,
    so the operation is atomic w.r.t. other coroutines on the same loop.
    """
    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()
    return session_locks[session_id]


def get_snapshot_action_lock(session_id: str) -> asyncio.Lock:
    """Return the lock serializing a session's snapshot-failure HITL actions.

    Retry / Skip / Terminate all read-modify-persist the *same* cached
    AnalysisState. Retry re-runs the failed capture and can hold the request
    open for minutes with the gate still armed, so an interleaved Terminate
    passes its is_snapshot_failure_blocked() check, closes the kernel the
    capture is still reading, and then has its STOPPED write overwritten by the
    finishing Retry.

    This is NOT get_session_lock: every Retry producer reaches
    SnapshotService.capture(), which acquires that lock itself, and
    asyncio.Lock is not reentrant - taking it at the route would deadlock the
    handler against its own capture. Acquisition order is always
    snapshot-action -> session and never the reverse, so the two cannot
    deadlock against each other.

    Safe under asyncio: the dict check + insert below contains no `await`,
    so the operation is atomic w.r.t. other coroutines on the same loop.
    """
    if session_id not in snapshot_action_locks:
        snapshot_action_locks[session_id] = asyncio.Lock()
    return snapshot_action_locks[session_id]


def release_session_lock_slot(session_id: str) -> None:
    """Drop a session's lock entry once the session is destroyed.

    Best-effort: callers should invoke this only when no coroutine still
    holds the lock (i.e. after session cleanup paths). Missing keys are
    silently ignored.
    """
    session_locks.pop(session_id, None)
    snapshot_action_locks.pop(session_id, None)


def touch_access_time(session_id: str) -> None:
    """Record current time as last access (callers hold state_lock)."""
    session_access_times[session_id] = time.time()


# Current system config (set via /api/config).
# Avoids circular import with backend.config.SystemConfig.
system_config: Dict[str, Optional[Any]] = {"current": None}

# WebSocket connections per session (managed by the websocket endpoint).
# Exposed here to avoid circular imports with routes that need to check/modify.
websockets: Dict[str, Any] = {}

# Export lifecycle records keyed by one-shot token. Records cover building,
# ready, streaming, and cleanup-failed phases so deletion can drain every
# producer before allowing a same-ID generation to be created.
prepared_exports: Dict[str, Dict[str, Any]] = {}
prepared_exports_lock: asyncio.Lock = asyncio.Lock()

# Import jobs for background tar.gz extraction: import_id -> {"status": str, ...}
import_jobs: Dict[str, dict] = {}
import_jobs_lock: asyncio.Lock = asyncio.Lock()

# Chat delegation tracking: session_id → background delegation task
chat_tasks: Dict[str, asyncio.Task] = {}

# Lock for protecting chat task concurrent access
chat_task_lock: asyncio.Lock = asyncio.Lock()

# Chat sandboxes: session_id → sandbox for active chat (delegation/figure gen)
chat_sandboxes: Dict[str, Any] = {}

# WebSocket monitor tasks (for cancellation on reconnect): session_id -> monitor task
_monitor_tasks: Dict[str, asyncio.Task] = {}

# WebSocket connections for report generation: session_id -> WebSocket
report_websockets: Dict[str, Any] = {}

# Lock for protecting concurrent access to report_websockets
report_ws_lock: asyncio.Lock = asyncio.Lock()

# Detailed report generation state: session_id -> report dict
detailed_reports: Dict[str, Any] = {}

# Tracked background detailed-report producers. Unlike the process-local
# pending marker, these task handles let deletion cancel and await the writer
# before removing a session tree or permitting same-ID reuse.
detailed_report_tasks: Dict[str, asyncio.Task] = {}

# Snapshot V2 capture jobs: job_id -> status dict
snapshot_jobs: Dict[str, Any] = {}

# Snapshot V2 restore jobs: transaction_id -> RestoreJobHandle
# Shared process-wide so cancel/status work across independent HTTP requests.
snapshot_restore_jobs: Dict[str, Any] = {}


def has_live_session_task(session_id: str) -> bool:
    """Whether any registry holds an unfinished task for this session.

    These task registries plus an incomplete standalone report workflow are the
    complete set of places a session's work can be executing, so a caller asking
    "is anything running here?" must consult all of them.
    """
    from backend.services.workflow_runner import _report_tasks  # local: cycle

    if session_id in report_workflows:
        return True
    for registry in (
        _run_tasks,
        _report_tasks,
        chat_tasks,
        detailed_report_tasks,
    ):
        task = registry.get(session_id)
        if task is not None and not task.done():
            return True
    return False
