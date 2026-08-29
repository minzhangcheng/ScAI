"""Manual-mode _wait_for_user_input signal handling regressions."""
import asyncio
import types

import pytest

from backend.models.schemas import AnalysisPhase, AnalysisState, TaskStatus
from backend.workflow.synthesis import SynthesisMixin


class _Harness(SynthesisMixin):
    def __init__(self):
        self.session_id = "sess"
        self.config = types.SimpleNamespace(max_rounds=5)
        self._continue_event = asyncio.Event()
        self._supplementary_queue = []
        self.sent = []

    async def _send_state_sync(self, state, change_type):
        self.sent.append((state.status, change_type))

    def _set_status(self, state, status):
        state.status = status


@pytest.mark.asyncio
async def test_signal_during_supplementary_not_lost():
    from backend.workflow.synthesis import _consume_signal_then_clear

    event = asyncio.Event()
    event.set()
    await event.wait()
    _consume_signal_then_clear(event)
    assert not event.is_set()

    event.set()

    try:
        await asyncio.wait_for(event.wait(), timeout=0.2)
    except asyncio.TimeoutError:
        pytest.fail("Signal received during handling was lost — wait() blocked")


@pytest.mark.asyncio
async def test_consumed_continue_signal_immediately_leaves_waiting_status():
    workflow = _Harness()
    state = AnalysisState(
        name="t",
        user_id="u",
        initial_instruction="o",
        current_round=1,
        analysis_phase=AnalysisPhase.INNOVATION,
        status=TaskStatus.WAITING_INPUT,
        round_wait_seconds=-1,
    )
    workflow._continue_event.set()

    await workflow._wait_for_user_input({"analysis_state": state})

    assert state.status == TaskStatus.RUNNING
    assert workflow.sent == [(TaskStatus.RUNNING, "continue")]
