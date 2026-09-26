from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState
from backend.workflow.messaging import MessagingMixin


@pytest.mark.asyncio
async def test_strict_state_sync_rejects_unacknowledged_persistence():
    class Harness(MessagingMixin):
        session_id = "report-test"
        message_callback = AsyncMock()
        state_callback = AsyncMock(return_value=False)

    state = AnalysisState(user_id="user", initial_instruction="test")
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        await Harness()._send_state_sync(
            state, "integrated_report_published", raise_on_save_error=True,
        )

    Harness.message_callback.assert_not_awaited()
