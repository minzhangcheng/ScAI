"""Tests for bioinformatics agent species validation and organism override."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.callbacks import CallbacksMixin


def _build_callbacks_mixin(state):
    inst = CallbacksMixin.__new__(CallbacksMixin)
    inst.session_id = "s1"
    inst._current_analysis_state = state
    inst._species_event = asyncio.Event()
    inst._species_user_response = None
    inst._send_state_sync = AsyncMock()
    inst._send_message = AsyncMock()
    inst._set_status = MagicMock(side_effect=lambda s, st: setattr(s, "status", st))
    inst.pi_agent = SimpleNamespace(decide_species_action=AsyncMock(return_value={"action": "keep"}))
    return inst


@pytest.mark.asyncio
@pytest.mark.parametrize("round_wait", [0, 30, -1])
async def test_gene_validation_cleanup_on_cancel(monkeypatch, round_wait):
    state = AnalysisState(user_id="u", species="Homo sapiens")
    state.status = TaskStatus.RUNNING
    state.round_wait_seconds = round_wait
    inst = _build_callbacks_mixin(state)

    async def _raise(event, wait_seconds, *a, **k):
        assert wait_seconds == -1
        raise asyncio.CancelledError()
    monkeypatch.setattr("backend.workflow.callbacks.wait_with_skip_time", _raise)

    with pytest.raises(asyncio.CancelledError):
        await inst._species_validation_callback(["TP53"], "Homo sapiens",
                                                {"reasoning": "n/a", "likely_species": ""})
    assert state.species_input_info is None
    assert state.wait_input_reason is None
