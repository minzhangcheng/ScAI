"""Tests for species inference integration in _pi_planning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from backend.models.schemas import AnalysisState, TaskStatus
from backend.tests.split_facade import patch_facade
from backend.workflow.planning import PlanningMixin


def _build_mixin(inference_result):
    """Construct a minimal PlanningMixin instance with stubbed collaborators."""
    inst = PlanningMixin.__new__(PlanningMixin)
    inst.session_id = "s1"
    inst._species_event = asyncio.Event()
    inst._species_user_response = None
    inst.pi_agent = SimpleNamespace(
        infer_species=AsyncMock(return_value=inference_result),
        infer_species_from_hint=AsyncMock(return_value=inference_result),
        reset_history=MagicMock(),
    )
    inst._send_message = AsyncMock()
    inst._send_state_sync = AsyncMock()
    inst._set_status = MagicMock(side_effect=lambda s, st: setattr(s, "status", st))
    inst._require_core_agent = AsyncMock(return_value=None)
    return inst


@pytest.mark.asyncio
@pytest.mark.parametrize("round_wait", [0, 30, -1])
async def test_species_state_restored_on_cancellation(monkeypatch, round_wait):
    """If wait_with_skip_time raises CancelledError, species_input_info / wait_input_reason
    must still be reset (try/finally guarantee)."""
    inst = _build_mixin({"confidence": "low", "suggestions": ["Mus musculus"]})
    state = AnalysisState(user_id="u", current_round=1)
    state.status = TaskStatus.RUNNING
    state.round_wait_seconds = round_wait

    async def _raise(event, wait_seconds, *a, **k):
        assert wait_seconds == -1
        raise asyncio.CancelledError()
    wstate = {"analysis_state": state}
    with patch_facade("backend.workflow.planning", "wait_with_skip_time", _raise):
        with pytest.raises(asyncio.CancelledError):
            await PlanningMixin._pi_planning.__wrapped__(inst, wstate)
    assert state.species_input_info is None
    assert state.wait_input_reason is None
