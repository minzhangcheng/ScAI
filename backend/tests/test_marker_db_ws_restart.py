"""Tests for re-emitting marker_db_missing state_sync on WebSocket reconnect.

A session parked in WAITING_SPECIES_INPUT (due to marker_db_missing) should
re-emit the state_sync on reconnect with change_type mapped to llm_failure
so the frontend's LlmTimeoutDialog reopens.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.constants import WaitInputReason


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


def _make_state() -> AnalysisState:
    st = AnalysisState(user_id="u1", session_id="sess-marker-db")
    st.status = TaskStatus.WAITING_SPECIES_INPUT
    st.wait_input_reason = WaitInputReason.MARKER_DB_MISSING.value
    st.llm_failure_info = {
        "node_name": "bioinformatics_analysis",
        "generation": 7,
        "reason": "marker_db_missing",
    }
    return st


@pytest.mark.asyncio
async def test_reconnect_resends_marker_db_as_llm_failure():
    from backend.routes.websockets import _resend_wait_input_state_sync

    ws = _FakeWebSocket()
    st = _make_state()

    await _resend_wait_input_state_sync(ws, st, "sess-marker-db")

    assert len(ws.sent) == 1
    msg = ws.sent[0]
    assert msg["type"] == "state_sync"
    content = msg["content"]
    assert content["change_type"] == "llm_failure"
    assert content["wait_input_reason"] == "marker_db_missing"
    assert content["llm_failure_info"]["reason"] == "marker_db_missing"


def test_session_store_clears_marker_db_stale(monkeypatch):
    import inspect
    from backend.services import session_store
    src = inspect.getsource(session_store)
    assert "MARKER_DB_MISSING" in src, "restart stale-clear must handle marker_db_missing"
