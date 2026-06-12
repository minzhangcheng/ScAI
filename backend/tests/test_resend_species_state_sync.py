"""Tests for re-emitting species HITL state_sync on WebSocket reconnect.

A session parked in WAITING_SPECIES_INPUT is driven entirely by state_sync
(the frontend's speciesInputInfo store is only filled from state_sync). On
page reload the reconnect path replays message_history, which does NOT feed
the state_sync handler, so the species dialog never reopens unless the
pending state_sync is re-emitted — exactly like llm_failure / agent_disabled.
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


def _make_state(reason: str) -> AnalysisState:
    st = AnalysisState(user_id="u1", session_id="sess-species")
    st.status = TaskStatus.WAITING_SPECIES_INPUT
    st.wait_input_reason = reason
    st.species_input_info = {
        "source": "pi_inference",
        "suggestions": ["Homo sapiens", "Mus musculus"],
        "context": "Unable to determine species automatically",
    }
    return st


@pytest.mark.asyncio
async def test_resend_species_input_state_sync():
    from backend.routes.websockets import _resend_wait_input_state_sync

    ws = _FakeWebSocket()
    st = _make_state(WaitInputReason.SPECIES_INPUT.value)

    await _resend_wait_input_state_sync(ws, st, "sess-species")

    assert len(ws.sent) == 1
    msg = ws.sent[0]
    assert msg["type"] == "state_sync"
    content = msg["content"]
    assert content["change_type"] == "species_input"
    assert content["wait_input_reason"] == "species_input"
    assert content["species_input_info"]["source"] == "pi_inference"


@pytest.mark.asyncio
async def test_resend_gene_species_mismatch_state_sync():
    from backend.routes.websockets import _resend_wait_input_state_sync

    ws = _FakeWebSocket()
    st = _make_state(WaitInputReason.GENE_SPECIES_MISMATCH.value)

    await _resend_wait_input_state_sync(ws, st, "sess-species")

    assert len(ws.sent) == 1
    content = ws.sent[0]["content"]
    assert content["change_type"] == "gene_species_mismatch"
    assert content["species_input_info"] is not None


@pytest.mark.asyncio
async def test_resend_skips_unrelated_reason():
    """Reasons outside the HITL set must not emit a resend."""
    from backend.routes.websockets import _resend_wait_input_state_sync

    ws = _FakeWebSocket()
    st = AnalysisState(user_id="u1", session_id="sess-x")
    st.wait_input_reason = "round_feedback"

    await _resend_wait_input_state_sync(ws, st, "sess-x")

    assert ws.sent == []
