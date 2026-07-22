from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow import SingleCellAnalysisWorkflow


@pytest.mark.asyncio
async def test_resume_at_graph_end_preserves_failed_status():
    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "failed-session"
    workflow.await_restore_safe_point = AsyncMock()
    workflow._set_status = MagicMock()

    failed = AnalysisState(
        id="failed-session",
        user_id="user",
        status=TaskStatus.FAILED,
    )
    workflow.compiled_graph = MagicMock()
    workflow.compiled_graph.aget_state = AsyncMock(return_value=SimpleNamespace(
        values={"analysis_state": failed},
        next=(),
    ))

    result = await workflow.resume("failed-session")

    assert result is failed
    assert result.status == TaskStatus.FAILED
    workflow._set_status.assert_not_called()
