import pytest
from unittest.mock import AsyncMock
from backend.workflow.helpers import WorkflowTerminatedError


@pytest.fixture
def workflow():
    from backend.workflow.core import SingleCellAnalysisWorkflow
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-snapshot-terminate"
    wf.sandbox = None
    wf._send_message = AsyncMock()
    return wf


@pytest.mark.asyncio
async def test_auto_snapshot_propagates_terminate(workflow, monkeypatch):
    from backend.utils.snapshot import SnapshotManager
    async def boom(*a, **k):
        raise WorkflowTerminatedError("stop")
    monkeypatch.setattr(SnapshotManager, "create_and_register_snapshot", boom)
    with pytest.raises(WorkflowTerminatedError):
        await workflow._create_auto_snapshot(None, "test")
