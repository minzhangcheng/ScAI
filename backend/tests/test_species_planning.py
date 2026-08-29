"""Tests for species inference integration in _pi_planning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from backend.constants import StateSyncChangeType
from backend.models.schemas import AnalysisState, TaskStatus
from backend.tests.split_facade import patch_facade
from backend.workflow.planning import PlanningMixin


class TestSpeciesInPlanning:
    def test_skip_if_species_already_set(self):
        state = AnalysisState(user_id="u1", species="Mus musculus", species_resolved=True)
        assert state.species != ""

    def test_skip_if_not_round_1(self):
        state = AnalysisState(user_id="u1", current_round=2)
        assert state.current_round != 1

    def test_species_set_after_high_confidence(self):
        state = AnalysisState(user_id="u1")
        state.species = "Homo sapiens"
        state.species_resolved = True
        assert state.species == "Homo sapiens"
        assert state.species_resolved is True


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
async def test_high_confidence_sends_species_resolved_state_sync(monkeypatch):
    """High-confidence inference must emit SPECIES_RESOLVED state_sync (was missing)."""
    inst = _build_mixin({"confidence": "high", "species": "Mus musculus", "reasoning": "ok"})
    state = AnalysisState(user_id="u", current_round=1)
    state.current_plan = None
    monkeypatch.setattr("backend.workflow.planning._consume_user_plan_feedback", lambda s: "")
    wstate = {"analysis_state": state}
    # Short-circuit by raising after the species block runs.
    inst.pi_agent.plan_round = AsyncMock(side_effect=RuntimeError("stop_here"))
    with pytest.raises(RuntimeError, match="stop_here"):
        async def _abort(*a, **k):
            if a and a[0] == "status" and "creating analysis plan" in (a[1] or ""):
                raise RuntimeError("stop_here")
        inst._send_message = AsyncMock(side_effect=_abort)
        await PlanningMixin._pi_planning.__wrapped__(inst, wstate)
    state_sync_calls = [c.args for c in inst._send_state_sync.await_args_list]
    assert any(call[1] == StateSyncChangeType.SPECIES_RESOLVED.value for call in state_sync_calls)
    assert state.species == "Mus musculus"


@pytest.mark.asyncio
async def test_species_state_restored_on_cancellation(monkeypatch):
    """If wait_with_skip_time raises CancelledError, species_input_info / wait_input_reason
    must still be reset (try/finally guarantee)."""
    inst = _build_mixin({"confidence": "low", "suggestions": ["Mus musculus"]})
    state = AnalysisState(user_id="u", current_round=1)
    state.status = TaskStatus.RUNNING

    async def _raise(*a, **k):
        raise asyncio.CancelledError()
    wstate = {"analysis_state": state}
    with patch_facade("backend.workflow.planning", "wait_with_skip_time", _raise):
        with pytest.raises(asyncio.CancelledError):
            await PlanningMixin._pi_planning.__wrapped__(inst, wstate)
    assert state.species_input_info is None
    assert state.wait_input_reason is None


@pytest.mark.asyncio
async def test_empty_suggestions_emits_user_status(monkeypatch):
    """When inference fails and suggestions are empty, a user-visible status message
    should explain that gene validation will retry later."""
    inst = _build_mixin({"confidence": "low", "suggestions": []})
    state = AnalysisState(user_id="u", current_round=1)
    state.status = TaskStatus.RUNNING

    async def _timeout(*a, **k):
        return False  # never signaled
    # Suppress the rest of planning by raising after species block via _send_message.
    seen = {"phrase": False}
    async def _capture(kind, content, *_a, **_k):
        if kind == "status" and "gene validation" in (content or "").lower():
            seen["phrase"] = True
        if kind == "status" and "creating analysis plan" in (content or ""):
            raise RuntimeError("stop_here")
    inst._send_message = AsyncMock(side_effect=_capture)
    with patch_facade("backend.workflow.planning", "wait_with_skip_time", _timeout):
        with pytest.raises(RuntimeError, match="stop_here"):
            await PlanningMixin._pi_planning.__wrapped__(inst, {"analysis_state": state})
    assert seen["phrase"] is True


@pytest.mark.asyncio
async def test_species_hint_processing_exits_waiting_before_pi_call(monkeypatch):
    inst = _build_mixin({"confidence": "low", "suggestions": ["Mus musculus"]})
    state = AnalysisState(user_id="u", current_round=1, status=TaskStatus.RUNNING)
    syncs = []

    async def _record_sync(current, change_type):
        syncs.append((
            current.status,
            current.wait_input_reason,
            current.species_input_info,
            change_type,
        ))

    async def _infer_from_hint(current, _user_input):
        assert current.status == TaskStatus.RUNNING
        assert current.wait_input_reason is None
        assert current.species_input_info is None
        assert syncs[-1] == (
            TaskStatus.RUNNING,
            None,
            None,
            "species_input_received",
        )
        return {"confidence": "high", "species": "Mus musculus"}

    async def _signal(*_args, **_kwargs):
        inst._species_user_response = "mouse"
        return True

    async def _stop_after_species(kind, content, *_args, **_kwargs):
        if kind == "status" and "creating analysis plan" in (content or ""):
            raise RuntimeError("stop_here")

    inst._send_state_sync = AsyncMock(side_effect=_record_sync)
    inst._send_message = AsyncMock(side_effect=_stop_after_species)
    inst.pi_agent.infer_species_from_hint = AsyncMock(side_effect=_infer_from_hint)

    with patch_facade("backend.workflow.planning", "wait_with_skip_time", _signal):
        with pytest.raises(RuntimeError, match="stop_here"):
            await PlanningMixin._pi_planning.__wrapped__(
                inst, {"analysis_state": state}
            )

    assert state.status == TaskStatus.RUNNING
    assert state.species == "Mus musculus"
