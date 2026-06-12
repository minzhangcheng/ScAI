"""Tests for species event signal mechanism in workflow core."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import pytest

from backend.models.schemas import AnalysisState, TaskStatus


class TestSpeciesSignal:
    def test_signal_species_input_sets_event(self):
        from backend.workflow.core import SingleCellAnalysisWorkflow
        wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        wf._species_event = asyncio.Event()
        wf._species_user_response = None
        wf.session_id = "test"

        wf.signal_species_input("Homo sapiens")

        assert wf._species_event.is_set()
        assert wf._species_user_response == "Homo sapiens"

    def test_species_event_initialized(self):
        from backend.workflow.core import SingleCellAnalysisWorkflow
        wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        wf._species_event = asyncio.Event()
        wf._species_user_response = None
        assert not wf._species_event.is_set()
        assert wf._species_user_response is None


class _FakeWorkflow:
    def __init__(self):
        self.signal_species_called_with = None
        self.resume_called = False
    def signal_species_input(self, text):
        self.signal_species_called_with = text
    def signal_resume(self):
        self.resume_called = True


@pytest.mark.asyncio
async def test_resume_from_waiting_species_input(monkeypatch):
    """Direct resume from WAITING_SPECIES_INPUT should not 400; should signal workflow."""
    from backend.routes.lifecycle import _resume_analysis_locked
    from backend.state import workflows

    sid = "sess-species-direct"
    st = AnalysisState(user_id="u1", session_id=sid)
    st.status = TaskStatus.WAITING_SPECIES_INPUT

    fake_wf = _FakeWorkflow()
    workflows[sid] = fake_wf

    async def _fake_owner(**kwargs):
        return st
    monkeypatch.setattr("backend.routes.lifecycle.require_session_ownership", _fake_owner)
    monkeypatch.setattr("backend.routes.lifecycle._update_session_cache",
                        lambda *_a, **_k: asyncio.sleep(0))
    monkeypatch.setattr("backend.routes.lifecycle.async_save_session",
                        lambda *_a, **_k: asyncio.sleep(0))

    try:
        resp = await _resume_analysis_locked(sid, "u1")
        assert resp.success is True
        assert fake_wf.signal_species_called_with == ""
        assert st.status == TaskStatus.RUNNING
    finally:
        workflows.pop(sid, None)


@pytest.mark.asyncio
async def test_resume_from_paused_restores_waiting_species_input(monkeypatch):
    """Resume from PAUSED whose pre_pause_status was WAITING_SPECIES_INPUT must restore that
    status (not become RUNNING)."""
    from backend.routes.lifecycle import _resume_analysis_locked
    from backend.state import workflows

    sid = "sess-species-paused"
    st = AnalysisState(user_id="u1", session_id=sid)
    st.status = TaskStatus.PAUSED
    st.pre_pause_status = TaskStatus.WAITING_SPECIES_INPUT

    fake_wf = _FakeWorkflow()
    workflows[sid] = fake_wf

    async def _fake_owner(**kwargs):
        return st
    monkeypatch.setattr("backend.routes.lifecycle.require_session_ownership", _fake_owner)
    monkeypatch.setattr("backend.routes.lifecycle._update_session_cache",
                        lambda *_a, **_k: asyncio.sleep(0))
    monkeypatch.setattr("backend.routes.lifecycle.async_save_session",
                        lambda *_a, **_k: asyncio.sleep(0))

    try:
        resp = await _resume_analysis_locked(sid, "u1")
        assert resp.success is True
        assert st.status == TaskStatus.WAITING_SPECIES_INPUT
        assert st.pre_pause_status is None
        assert fake_wf.resume_called is True
    finally:
        workflows.pop(sid, None)
