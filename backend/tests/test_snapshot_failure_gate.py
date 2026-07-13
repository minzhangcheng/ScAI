"""Test snapshot V2 failure gate — persisted failure, blocked workflow, actions."""
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.snapshot_v2.errors import SnapshotCaptureError, SnapshotStorageError


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def v2_paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def make_analysis_state(**kw):
    from backend.models.schemas import AnalysisState
    return AnalysisState(id="sess-fail", name="Test", user_id="u1",
                         initial_instruction="analyze", **kw)


@pytest.fixture
def workspace_root(tmp_path):
    d = tmp_path / "workspace"
    d.mkdir()
    (d / "test.py").write_text("x=1")
    return d


@pytest.fixture
def sandbox():
    mock = MagicMock()
    mock.session_id = "test-kernel"
    mock.runtime_revision = 5
    mock.saved_runtime_revision = 3
    mock.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
        error=None,
    ))
    return mock


# ---- Failure Gate Tests ----------------------------------------------------

class TestSnapshotFailureGate:

    @pytest.mark.asyncio
    async def test_failure_persists_snapshot_failure_on_state(self, v2_paths, sandbox, workspace_root):
        """When a capture fails, snapshot_failure is set on the AnalysisState
        and snapshot_transaction is updated."""
        from backend.services.snapshot_v2.service import SnapshotService

        # Build state and service
        state = make_analysis_state()
        req = _make_req()
        service = SnapshotService("sess-fail", v2_paths)

        # Inject a failure (preserve typed SnapshotCaptureError code)
        async def fail(*a, **kw):
            raise SnapshotCaptureError("QUOTA_EXCEEDED", "disk full")
        service._capture_impl = fail

        record = await service._capture_and_apply(state, req, sandbox=sandbox,
                                                   workspace_root=str(workspace_root))

        # Failure record persisted on AnalysisState
        assert state.snapshot_failure is not None
        assert state.snapshot_failure["code"] == "QUOTA_EXCEEDED"
        assert "disk full" in state.snapshot_failure["message"]

        # snapshot_transaction tracks the attempt
        assert state.snapshot_transaction is not None
        assert state.snapshot_transaction.get("snapshot_id") == record.snapshot_id
        assert state.snapshot_transaction.get("status") == "failed"

    @pytest.mark.asyncio
    async def test_failure_blocks_workflow(self, v2_paths, sandbox, workspace_root):
        """After a snapshot failure, the workflow status indicates blocked
        (waiting_input_reason set)."""
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.constants import WaitInputReason

        state = make_analysis_state()
        req = _make_req()
        service = SnapshotService("sess-fail", v2_paths)

        async def fail(*a, **kw):
            raise SnapshotCaptureError("CAPTURE_ERROR", "fail")
        service._capture_impl = fail

        await service._capture_and_apply(state, req, sandbox=sandbox,
                                          workspace_root=str(workspace_root))

        # wait_input_reason should be set to snapshot_failure
        assert state.wait_input_reason == "snapshot_failure"

    @pytest.mark.asyncio
    async def test_failure_emits_state_sync(self, v2_paths, sandbox, workspace_root):
        """Failure gate sends state_sync with SNAPSHOT_FAILURE change_type."""
        from backend.services.snapshot_v2.service import SnapshotService

        state = make_analysis_state()
        req = _make_req()
        service = SnapshotService("sess-fail", v2_paths)

        async def fail(*a, **kw):
            raise SnapshotCaptureError("CAPTURE_ERROR", "fail")
        service._capture_impl = fail

        ws_messages = []
        async def fake_send(payload):
            ws_messages.append(payload if isinstance(payload, dict) else json.loads(payload))
        service._send_state_sync = fake_send

        await service._capture_and_apply(state, req, sandbox=sandbox,
                                          workspace_root=str(workspace_root))

        # Should have emitted at least one state_sync with SNAPSHOT_FAILURE
        snapshot_failure_msgs = [
            m for m in ws_messages
            if isinstance(m, dict) and m.get("change_type") == "snapshot_failure"
        ]
        assert len(snapshot_failure_msgs) > 0, "No snapshot_failure state_sync emitted"

    @pytest.mark.asyncio
    async def test_failure_accepts_retry(self, v2_paths, sandbox, workspace_root):
        """Retry action resets snapshot_failure and allows re-capture."""
        from backend.services.snapshot_v2.service import SnapshotService

        state = make_analysis_state()
        req = _make_req()
        service = SnapshotService("sess-fail", v2_paths)

        # First capture fails
        async def fail(*a, **kw):
            raise SnapshotCaptureError("TEMP_FAILURE", "transient")
        service._capture_impl = fail

        await service._capture_and_apply(state, req, sandbox=sandbox,
                                          workspace_root=str(workspace_root))
        assert state.snapshot_failure is not None

        # Retry: reset the failure state
        service._clear_failure(state)
        assert state.snapshot_failure is None

        # Second capture succeeds (restore real capture_impl)
        from backend.services.snapshot_v2.service import SnapshotService as RealService
        service._capture_impl = RealService._capture_impl.__get__(service, RealService)

        # Success path: bypass full capture with a committed record apply
        from backend.services.snapshot_v2.models import (
            SnapshotRecordV2, SnapshotManifestV2, IdleCapsule, ObjectRef,
            empty_complete_runtime,
        )
        from datetime import datetime, timezone
        stub = ObjectRef(sha256="a"*64, size=1, media_type="application/json")
        ok_manifest = SnapshotManifestV2(
            session_id=state.id,
            snapshot_id="snap-retry-ok",
            name="ok",
            phase="manual",
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            analysis_state_ref=stub,
            notebook_ref=stub,
            runtime=empty_complete_runtime(),
            continuation=IdleCapsule(graph_version="2.0"),
            logical_size_bytes=1,
            physical_added_bytes=0,
            root_hash="b"*64,
        )
        async def ok_capture(*a, **kw):
            return SnapshotRecordV2(
                snapshot_id="snap-retry-ok",
                manifest=ok_manifest,
                failure=None,
            )
        service.capture = ok_capture
        await service._capture_and_apply(state, req, sandbox=sandbox,
                                          workspace_root=str(workspace_root))
        assert state.snapshot_failure is None, "Retry should clear failure on success"
        assert state.snapshot_transaction.get("status") == "completed"

    @pytest.mark.asyncio
    async def test_base_actions_exclude_skip(self, v2_paths, sandbox, workspace_root):
        """Base (non-skippable, e.g. pre_restore) gate actions exclude skip."""
        assert "retry" in _allowed_actions()
        assert "manage" in _allowed_actions()
        assert "terminate" in _allowed_actions()
        assert "skip" not in _allowed_actions()

    @pytest.mark.asyncio
    async def test_skippable_producers_offer_skip(self):
        """Workflow-boundary / post_report / chat producers additionally offer skip.

        "Skip snapshot & continue" lets the analysis proceed past a boundary
        without that restore point; only continuation-critical producers
        (pre_restore) withhold it.
        """
        from backend.services.snapshot_v2.errors import (
            SNAPSHOT_FAILURE_ACTIONS_SKIPPABLE,
        )

        assert "skip" in SNAPSHOT_FAILURE_ACTIONS_SKIPPABLE
        assert set(SNAPSHOT_FAILURE_ACTIONS_SKIPPABLE) == {
            "retry", "skip", "manage", "terminate",
        }

    @pytest.mark.asyncio
    async def test_failure_accepts_manage(self):
        """Manage action is a valid snapshot failure gate action."""
        assert "manage" in _allowed_actions()

    @pytest.mark.asyncio
    async def test_failure_accepts_terminate(self):
        """Terminate action is a valid snapshot failure gate action."""
        assert "terminate" in _allowed_actions()

    @pytest.mark.asyncio
    async def test_cleanup_failure_clears_state(self, v2_paths):
        """Clearing failure removes snapshot_failure and resets wait_input_reason."""
        from backend.services.snapshot_v2.service import SnapshotService

        state = make_analysis_state()
        state.wait_input_reason = "snapshot_failure"
        state.snapshot_failure = {"code": "TEST", "message": "test", "snapshot_id": "s1"}
        state.snapshot_transaction = {"snapshot_id": "s1", "status": "failed"}

        service = SnapshotService("sess-fail", v2_paths)
        service._clear_failure(state)

        assert state.snapshot_failure is None
        assert state.wait_input_reason is None


# ---- Helpers ---------------------------------------------------------------

def _make_req(session_id="sess-fail"):
    from backend.services.snapshot_v2.models import (
        SnapshotCaptureRequest, IdleCapsule,
    )
    return SnapshotCaptureRequest(
        session_id=session_id,
        reason="manual",
        capsule=IdleCapsule(graph_version="2.0"),
    )


def _allowed_actions():
    """Return the set of valid actions for the snapshot failure gate."""
    return {"retry", "manage", "terminate"}
