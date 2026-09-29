from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState
from backend.services.report.integrated_report import sanitize_integrated_report
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


@pytest.mark.parametrize("heading", ["## Reporting Limitations", "## Limitations", "## Study Limitations"])
def test_integrated_report_omits_standalone_limitation_section(heading):
    markdown = (
        "## Discussion\n\nThe n=1 comparison needs cautious interpretation.\n\n"
        f"{heading}\n\n- plan.1: redraw needed\n\n"
        "## Conclusions\n\nThe finding remains exploratory."
    )

    report = sanitize_integrated_report(
        markdown, [], reporting_limitations=["plan.2: omitted figure"],
    )

    assert "The n=1 comparison needs cautious interpretation." in report
    assert "## Conclusions" in report
    assert "Reporting Limitations" not in report
    assert "## Limitations" not in report
    assert "## Study Limitations" not in report
    assert "plan.1" not in report
    assert "plan.2" not in report
