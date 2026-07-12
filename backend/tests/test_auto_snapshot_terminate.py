from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.models.schemas import AnalysisState
from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
from backend.services.snapshot_v2.models import ObjectRef, SemanticGateCapsule
from backend.workflow.core import SingleCellAnalysisWorkflow


@pytest.fixture
def workflow():
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-snapshot-terminate"
    wf.sandbox = None
    wf._send_message = AsyncMock()
    wf._current_graph_state = {}
    return wf


def valid_state():
    return AnalysisState(session_id="test-snapshot-terminate", user_id="u1")


@pytest.mark.asyncio
async def test_auto_snapshot_propagates_required_snapshot_block(workflow):
    workflow.snapshot_service = MagicMock()
    workflow.snapshot_service.object_store = MagicMock()
    workflow.snapshot_service.capture_required_snapshot = AsyncMock(
        side_effect=RequiredSnapshotBlocked("quota", "snapshot full"),
    )
    stub_ref = ObjectRef(sha256="c" * 64, size=1, media_type="application/json")
    dummy_capsule = SemanticGateCapsule(
        graph_version="2.0",
        gate="round_wait",
        payload_ref=stub_ref,
    )
    with patch(
        "backend.services.snapshot_v2.continuation.ContinuationRegistry.freeze",
        return_value=dummy_capsule,
    ):
        with patch(
            "backend.config.get_session_workspace",
            return_value=Path("/tmp"),
        ):
            with pytest.raises(RequiredSnapshotBlocked):
                await workflow._create_auto_snapshot(valid_state(), "pre_round_wait")
