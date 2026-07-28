"""When _user_review_gate is entered without a plan, it must announce the abort.

Since ver 0.35.1 the abort is terminal rather than advisory: the gate emits a
user-visible message, sets failed_node/should_continue, and emits a
"planning_aborted" state_sync so a missing plan can never drift into a
completed final report. This pins all three, instead of grepping the message
text for the word "abort" — wording is not the contract.
"""
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_user_review_gate_sends_status_when_no_plan():
    from backend.workflow.planning import PlanningMixin

    class _GateStub(PlanningMixin):
        def __init__(self):
            self.session_id = "test"
            self._sent_messages = []
            self._state_syncs = []

        async def _send_message(self, msg_type, content, agent=None):
            self._sent_messages.append({"type": msg_type, "content": content})

        async def _send_state_sync(self, analysis_state, change_type):
            self._state_syncs.append(change_type)

    stub = _GateStub()
    analysis_state = MagicMock()
    analysis_state.current_plan = None
    state = {"analysis_state": analysis_state, "should_continue": False}

    result = await stub._user_review_gate(state)

    # The user must be told on a visible channel — never a silent return that
    # leaves the run looking healthy.
    assert any(
        m["type"] in ("status", "error") for m in stub._sent_messages
    ), f"Expected a user-visible abort message, got: {stub._sent_messages}"

    # The machine-readable signal the frontend keys off.
    assert "planning_aborted" in stub._state_syncs, (
        f"Expected a planning_aborted state_sync, got: {stub._state_syncs}"
    )

    # ...and the abort must be terminal, not a fall-through to reporting.
    assert state["failed_node"] == "planning_dialogue"
    assert state["should_continue"] is False
    assert result is state
