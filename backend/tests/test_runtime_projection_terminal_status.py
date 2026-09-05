"""The close-time runtime projection must not rewind a finished run's status.

``_runtime_persistence_commit_callback`` fires from ``close()``, after the
terminal status has already been persisted, and it published
``self._current_analysis_state``.  That pointer holds a *graph-side*
AnalysisState: ``_restore_entry`` deep-copies the state on entry, and a resume
runs on a checkpoint-hydrated copy while the pointer keeps the pre-resume
object.  Persisting it wrote the pre-terminal status back over COMPLETED, so a
finished analysis stayed ``generating_report`` on disk: the report was never
offered and every WebSocket reconnect resumed the terminated graph again.

The projection carries runtime-persistence facts only, so it must land on
whichever AnalysisState the session store currently treats as authoritative.
"""

import pytest

from backend.constants import StateSyncChangeType
from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.callbacks import CallbacksMixin


class _ProjectionWorkflow(CallbacksMixin):
    """Minimal host for the callback: it only needs the sync seam."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._current_analysis_state = None
        self._state = None
        self.synced = []

    async def _send_state_sync(self, analysis_state, change_type):
        self.synced.append((analysis_state, change_type))


def _state(session_id: str, status: TaskStatus) -> AnalysisState:
    state = AnalysisState(
        id=session_id,
        name="projection",
        user_id="u1",
        initial_instruction="x",
    )
    state.status = status
    return state


def _commit_event(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "lifecycle_generation": 0,
        "current_revision": 7,
        "saved_revision": 7,
        "runtime_persistence": {"status": "current", "saved_revision": 7},
    }


@pytest.fixture
def live_sessions():
    from backend import state as backend_state

    installed = []

    def install(session_id, state):
        backend_state.sessions[session_id] = state
        installed.append(session_id)

    try:
        yield install
    finally:
        for session_id in installed:
            backend_state.sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_runtime_commit_projection_keeps_the_authoritative_status(
    live_sessions,
):
    """A stale graph-side pointer must not reinstate a pre-terminal status."""
    session_id = "projection-terminal"
    authoritative = _state(session_id, TaskStatus.COMPLETED)
    stale = _state(session_id, TaskStatus.GENERATING_REPORT)

    workflow = _ProjectionWorkflow(session_id)
    workflow._current_analysis_state = stale
    live_sessions(session_id, authoritative)

    await workflow._runtime_persistence_commit_callback(_commit_event(session_id))

    assert workflow.synced, "the committed generation must be published"
    published, change_type = workflow.synced[-1]
    assert change_type == StateSyncChangeType.RUNTIME_PERSISTENCE_COMMITTED.value
    assert published is authoritative, (
        "the projection must land on the session's authoritative state, not on "
        "the workflow's graph-side pointer"
    )
    assert published.status is TaskStatus.COMPLETED
    assert published.runtime_persistence == {
        "status": "current",
        "saved_revision": 7,
    }
    assert published.sandbox_state["saved_runtime_revision"] == 7


@pytest.mark.asyncio
async def test_runtime_commit_projection_falls_back_to_the_workflow_state():
    """With no cached session the workflow's own state stays the target."""
    session_id = "projection-uncached"
    workflow = _ProjectionWorkflow(session_id)
    workflow._current_analysis_state = _state(session_id, TaskStatus.RUNNING)

    await workflow._runtime_persistence_commit_callback(_commit_event(session_id))

    published, _change_type = workflow.synced[-1]
    assert published is workflow._current_analysis_state
    assert published.runtime_persistence["saved_revision"] == 7
