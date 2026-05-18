"""When _user_review_gate is entered without a plan, it must announce the abort."""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest


@pytest.mark.asyncio
async def test_user_review_gate_sends_status_when_no_plan():
    from backend.workflow.planning import PlanningMixin
    from backend.models.schemas import AgentType

    class _GateStub(PlanningMixin):
        def __init__(self):
            self.session_id = "test"
            self._sent_messages = []

        async def _send_message(self, msg_type, content, agent=None):
            self._sent_messages.append({"type": msg_type, "content": content})

        async def _send_state_sync(self, *args, **kwargs):
            pass

    stub = _GateStub()
    analysis_state = MagicMock()
    analysis_state.current_plan = None
    state = {"analysis_state": analysis_state, "should_continue": False}

    result = await stub._user_review_gate(state)

    assert any(
        m["type"] in ("status", "error") and "abort" in str(m["content"]).lower()
        for m in stub._sent_messages
    ), f"Expected abort message, got: {stub._sent_messages}"
    assert result is state