"""Tests: post-publish session save failure and capture idempotency."""
from unittest.mock import AsyncMock, patch

import pytest

from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest, write_manifest_to_store


def _make_analysis_state(id: str = "sess-idy"):
    from backend.models.schemas import AnalysisState

    return AnalysisState(
        id=id,
        name="Idempotency Test",
        user_id="u1",
        initial_instruction="analyze",
        snapshots=[],
    )


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
            for d in [
                self.root, self.objects, self.staging,
                self.journals, self.integrity, self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


@pytest.mark.asyncio
async def test_session_save_failure_after_publish_returns_committed_snapshot(
    tmp_path, v2_paths,
):
    """After manifest publish, session save failure must not un-commit."""
    from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
    from backend.services.snapshot_v2.service import SnapshotService

    session_id = "sess-commit-tail"
    service = SnapshotService(session_id, v2_paths)
    state = _make_analysis_state(id=session_id)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    request = SnapshotCaptureRequest(
        session_id=session_id,
        reason="pre_round_wait",
        phase="pre_round_wait",
        capsule=IdleCapsule(graph_version="2.0"),
        idempotency_key="round-7-wait",
    )

    await service.safe_point.acquire(
        session_id,
        {"kernel_present": False, "status": "completed"},
        eligibility_trusted=True,
        owner_token="test-owner",
    )
    try:
        with patch.object(service, "_get_session_state", new=AsyncMock(return_value=state)):
            with patch.object(
                service,
                "_save_session_state",
                new=AsyncMock(side_effect=OSError("disk full")),
            ):
                with patch.object(service, "_send_state_sync", new=AsyncMock()):
                    result = await service.capture_required_snapshot(
                        state,
                        request,
                        sandbox=None,
                        workflow_state={"kernel_present": False},
                        workspace_root=str(workspace),
                        # This test holds the safe point itself; a caller that
                        # does must hand the token down, or the capture tries to
                        # re-acquire and gets SAFE_POINT_BUSY. Mirrors the one
                        # production caller that pre-acquires
                        # (workflow/reporting.py).
                        safe_point_owner_token="test-owner",
                    )
    finally:
        await service.safe_point.release(session_id, "test-owner")

    assert result.manifest is not None
    assert result.failure is None
    assert result.committed is True
    assert result.session_reconciliation_pending is True
    assert service.manifest_store.load(result.manifest.snapshot_id)


@pytest.mark.asyncio
async def test_idempotent_retry_after_session_save_failure(tmp_path, v2_paths):
    """Same idempotency key returns original committed manifest after save failure."""
    from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
    from backend.services.snapshot_v2.service import SnapshotService

    session_id = "sess-idy-retry"
    service = SnapshotService(session_id, v2_paths)
    state = _make_analysis_state(id=session_id)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def _req():
        return SnapshotCaptureRequest(
            session_id=session_id,
            reason="pre_round_wait",
            phase="pre_round_wait",
            capsule=IdleCapsule(graph_version="2.0"),
            idempotency_key="round-7-wait",
        )

    await service.safe_point.acquire(
        session_id,
        {"kernel_present": False, "status": "completed"},
        eligibility_trusted=True,
        owner_token="test-owner",
    )
    try:
        with patch.object(service, "_get_session_state", new=AsyncMock(return_value=state)):
            with patch.object(service, "_send_state_sync", new=AsyncMock()):
                with patch.object(
                    service,
                    "_save_session_state",
                    new=AsyncMock(side_effect=OSError("disk full")),
                ):
                    first = await service.capture_required_snapshot(
                        state, _req(), sandbox=None,
                        workflow_state={"kernel_present": False},
                        workspace_root=str(workspace),
                        safe_point_owner_token="test-owner",
                    )
                with patch.object(service, "_save_session_state", new=AsyncMock()):
                    second = await service.capture_required_snapshot(
                        state, _req(), sandbox=None,
                        workflow_state={"kernel_present": False},
                        workspace_root=str(workspace),
                        safe_point_owner_token="test-owner",
                    )
    finally:
        await service.safe_point.release(session_id, "test-owner")

    assert first.failure is None
    assert first.session_reconciliation_pending is True
    assert second.failure is None
    assert second.snapshot_id == first.snapshot_id
    assert len([
        e for e in service.list_snapshots() if e.get("status") == "ok"
    ]) == 1


def test_idempotency_lookup_uses_verify_all_committed(v2_paths, monkeypatch):
    from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
    from backend.services.snapshot_v2.object_store import ManifestStore
    from backend.services.snapshot_v2.service import SnapshotService

    calls: list[str] = []
    orig_verify = ManifestStore.verify_all_committed
    orig_diag = ManifestStore.list_committed_with_diagnostics

    def tracking_verify(self):
        calls.append("verify_all_committed")
        return orig_verify(self)

    def tracking_diag(self):
        calls.append("list_committed_with_diagnostics")
        return orig_diag(self)

    monkeypatch.setattr(ManifestStore, "verify_all_committed", tracking_verify)
    monkeypatch.setattr(
        ManifestStore, "list_committed_with_diagnostics", tracking_diag,
    )

    m = make_snapshot_v2_manifest(
        snapshot_id="snap-idy",
        session_id="s1",
        phase="pre_round_wait",
        idempotency_key="k1",
    )
    write_manifest_to_store(m, v2_paths.committed, v2_paths.integrity)

    service = SnapshotService("s1", v2_paths)
    req = SnapshotCaptureRequest(
        session_id="s1",
        reason="pre_round_wait",
        phase="pre_round_wait",
        capsule=IdleCapsule(graph_version="2.0"),
        idempotency_key="k1",
    )
    found = service._find_committed_by_idempotency(req)
    assert found is not None
    assert found.snapshot_id == "snap-idy"
    assert "verify_all_committed" in calls
    assert "list_committed_with_diagnostics" not in calls
