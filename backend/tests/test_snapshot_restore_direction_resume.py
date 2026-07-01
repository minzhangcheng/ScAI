"""Snapshot restore must preserve the in-flight innovation direction."""
from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisPhase, AnalysisState, TaskStatus
from backend.workflow.initialize import InitializeMixin


class DummyInitializeWorkflow(InitializeMixin):
    def __init__(self):
        self.session_id = "snapshot-direction-resume"
        self._current_analysis_state = None
        self._current_graph_state = None
        self.sent_messages = []
        self.statuses = []

    async def _send_message(self, message_type, content, agent=None):
        self.sent_messages.append((message_type, content, agent))

    async def _send_state_sync(self, analysis_state, change_type):
        self.sent_messages.append(("state_sync", change_type, None))

    def _set_status(self, analysis_state, status):
        analysis_state.status = status
        self.statuses.append(status)


@pytest.mark.asyncio
async def test_snapshot_restore_keeps_current_direction_and_queue_for_same_round():
    """Restoring a pre-execution snapshot resumes the same round, not the next one."""
    direction = {"direction": "Direction three", "notes": ""}
    analysis_state = AnalysisState(
        session_id="s",
        user_id="test",
        current_round=4,
        max_rounds=4,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_round_direction="Direction three",
        user_chosen_directions=[
            {"direction": "Direction one", "notes": ""},
            {"direction": "Direction two", "notes": ""},
            direction,
        ],
        directions_queue=[direction.copy()],
        last_restored_snapshot_id="snapshot-pre-execution-full",
        status=TaskStatus.RUNNING,
    )

    workflow = DummyInitializeWorkflow()
    result = await workflow._initialize_round({"analysis_state": analysis_state})

    restored = result["analysis_state"]
    assert restored.current_round == 4
    assert restored.current_round_direction == "Direction three"
    assert restored.directions_queue == [direction]
    assert restored.status == TaskStatus.RUNNING
