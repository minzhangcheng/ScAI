"""Runtime restore status and auto-snapshot V2 capture regression tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from backend.models.schemas import AnalysisState


def test_runtime_restore_status_defaults_to_complete():
    state = AnalysisState(session_id="s1", user_id="u1")
    assert state.runtime_restore_status == "complete"


def test_runtime_restore_status_roundtrips_through_json():
    state = AnalysisState(session_id="s1", user_id="u1")
    state.runtime_restore_status = "unavailable"
    dumped = state.model_dump(mode="json")
    assert dumped["runtime_restore_status"] == "unavailable"
    restored = AnalysisState.model_validate(dumped)
    assert restored.runtime_restore_status == "unavailable"


@pytest.mark.asyncio
async def test_restore_route_409_when_runtime_unavailable_no_choice(monkeypatch):
    """Batch 4: V1 restore is always HTTP 410 (runtime choice no longer applies)."""
    import backend.routes.snapshots as snaps
    from backend.models.schemas import AnalysisState, Snapshot

    sid = "sess-route-409"
    state = AnalysisState(session_id=sid, user_id="u1")
    snap = Snapshot(name="t", phase="manual", runtime_state_included=False)
    state.snapshots.append(snap)

    async def fake_owner(session_id, user_id):
        return state
    monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)

    class Req:
        state = type("S", (), {"user_id": "u1"})()
        async def json(self): return {}

    with pytest.raises(HTTPException) as ei:
        await snaps.restore_snapshot(sid, snap.id, Req())
    assert ei.value.status_code == 410
    assert "V1" in str(ei.value.detail)


@pytest.mark.asyncio
async def test_rollback_preview_reports_runtime_status(monkeypatch):
    import backend.routes.snapshots as snaps
    from backend.models.schemas import AnalysisState, Snapshot

    sid = "sess-preview-status"
    state = AnalysisState(session_id=sid, user_id="u1")
    # rollback_preview goes through _require_v1_snapshot (ver 0.33.14), which
    # rejects a snapshot with no state_file as "not a valid V1 file-backed
    # snapshot". A real V1 inventory snapshot always has one.
    snap = Snapshot(
        name="t",
        phase="manual",
        runtime_state_included=False,
        state_file="snap-t_state.json",
    )
    state.snapshots.append(snap)

    async def fake_owner(session_id, user_id):
        return state
    monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)

    class Req:
        state = type("S", (), {"user_id": "u1"})()

    resp = await snaps.rollback_preview(sid, snap.id, Req())
    assert resp.data["runtime_status"] == "state_only"


@pytest.mark.asyncio
async def test_resume_blocked_when_runtime_unavailable(monkeypatch):
    import backend.routes.lifecycle as life
    from backend.models.schemas import AnalysisState, TaskStatus

    sid = "sess-resume-block"
    state = AnalysisState(
        session_id=sid, user_id="u1", status=TaskStatus.WAITING_INPUT,
    )
    state.runtime_restore_status = "unavailable"

    async def fake_owner(session_id, user_id):
        return state
    monkeypatch.setattr(life, "require_session_ownership", fake_owner)

    resp = await life._resume_analysis_locked(sid, "u1")
    assert resp.data.get("runtime_unavailable") is True
    assert resp.data.get("needs_reconnect") is False


@pytest.mark.asyncio
async def test_rerun_clears_runtime_unavailable(monkeypatch):
    import backend.routes.lifecycle as life
    from backend.models.schemas import AnalysisState, TaskStatus

    sid = "sess-rerun-clear"
    state = AnalysisState(
        session_id=sid, user_id="u1", status=TaskStatus.COMPLETED, current_round=2,
    )
    state.runtime_restore_status = "unavailable"

    async def fake_owner(session_id, user_id):
        return state
    async def noop(*a, **k):
        return None
    monkeypatch.setattr(life, "require_session_ownership", fake_owner)
    monkeypatch.setattr(life, "teardown_running_workflow", noop)
    monkeypatch.setattr(life, "_update_session_cache", noop)
    monkeypatch.setattr(life, "async_save_session", noop)

    resp = await life._resume_analysis_locked(sid, "u1")
    assert state.runtime_restore_status == "complete"
    assert resp.data.get("needs_reconnect") is True


@pytest.mark.asyncio
async def test_workflow_resume_blocked_when_runtime_unavailable():
    """resume() must not run the graph when the restored runtime is unavailable."""
    from backend.workflow import SingleCellAnalysisWorkflow
    from backend.models.schemas import AnalysisState, TaskStatus

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "s-wf-block"
    workflow.compiled_graph = MagicMock()
    workflow.compiled_graph.aget_state = AsyncMock(return_value=None)
    workflow._current_analysis_state = None
    workflow._send_message = AsyncMock()

    state = AnalysisState(
        session_id="s-wf-block", user_id="u", status=TaskStatus.WAITING_INPUT, current_round=2,
    )
    state.runtime_restore_status = "unavailable"

    result = await workflow.resume("t1", state)

    assert result is state
    workflow.compiled_graph.aget_state.assert_not_called()


@pytest.mark.asyncio
async def test_workflow_resume_proceeds_when_runtime_complete():
    """resume() with a complete runtime still reaches the graph (regression guard)."""
    from backend.workflow import SingleCellAnalysisWorkflow
    from backend.models.schemas import AnalysisState, TaskStatus

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "s-wf-ok"
    workflow.compiled_graph = MagicMock()
    workflow.compiled_graph.aget_state = AsyncMock(return_value=None)
    workflow._current_analysis_state = None
    workflow._send_message = AsyncMock()

    state = AnalysisState(
        session_id="s-wf-ok", user_id="u", status=TaskStatus.WAITING_INPUT, current_round=2,
    )

    await workflow.resume("t1", state)

    workflow.compiled_graph.aget_state.assert_called()


def _v2_record(*, runtime_included: bool, snapshot_id: str = "snap-v2"):
    """Minimal SnapshotRecordV2-like object for auto-snapshot tests."""
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        ObjectRef,
        RuntimeDependency,
        RuntimeDependencyManifest,
        RuntimeManifest,
        SnapshotManifestV2,
        SnapshotRecordV2,
        empty_complete_runtime,
    )

    stub = ObjectRef(sha256="a" * 64, size=1, media_type="application/json")
    if runtime_included:
        residual = ObjectRef(
            sha256="b" * 64, size=10, media_type="application/octet-stream",
        )
        # residual_ref requires non-empty captured_residual_names (model invariant).
        runtime = RuntimeManifest(
            status="complete",
            dependencies=RuntimeDependencyManifest(
                required={
                    "x": RuntimeDependency(
                        name="x", serializer="dill", type_name="int",
                    ),
                },
            ),
            residual_ref=residual,
            captured_residual_names=["x"],
            adata_refs={},
        )
    else:
        runtime = empty_complete_runtime()

    manifest = SnapshotManifestV2(
        session_id="sess",
        snapshot_id=snapshot_id,
        name=snapshot_id,
        description="",
        phase="manual",
        created_at="2026-01-01T00:00:00+00:00",
        analysis_state_ref=stub,
        notebook_ref=stub,
        runtime=runtime,
        continuation=IdleCapsule(graph_version="2.0"),
        logical_size_bytes=1,
        physical_added_bytes=1,
        root_hash="c" * 64,
        objects={stub.sha256: stub},
    )
    return SnapshotRecordV2(snapshot_id=snapshot_id, manifest=manifest, failure=None)


class _SnapshotWorkflowHarness:
    def __init__(self, session_id, sandbox):
        self.session_id = session_id
        self.sandbox = sandbox
        self._send_message = AsyncMock()
        self._current_graph_state = {"current_round": 1, "kernel_present": False}
        self.snapshot_service = MagicMock()


@pytest.mark.asyncio
async def test_auto_snapshot_capture_failure_keeps_runtime_block(monkeypatch):
    """Required capture failure must not clear runtime_restore_status unavailable."""
    import backend.state as state_mod
    from backend.models.schemas import AnalysisState
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.workflow import SingleCellAnalysisWorkflow

    sid = "sess-auto-save-failed"
    state = AnalysisState(session_id=sid, user_id="u1")
    state.runtime_restore_status = "unavailable"
    state_mod.sessions[sid] = state
    sandbox = type("Sandbox", (), {})()
    sandbox.notebook_state = None
    sandbox.km = None
    workflow = _SnapshotWorkflowHarness(sid, sandbox)

    async def _fail(*a, **k):
        raise RequiredSnapshotBlocked("required_runtime_unpersistable", "save failed")

    workflow.snapshot_service.capture_required_snapshot = AsyncMock(side_effect=_fail)
    workflow.snapshot_service.object_store = MagicMock()

    try:
        with pytest.raises(RequiredSnapshotBlocked):
            await SingleCellAnalysisWorkflow._create_auto_snapshot(
                workflow, state, "manual",
            )
        assert state.runtime_restore_status == "unavailable"
        workflow.snapshot_service.capture_required_snapshot.assert_awaited()
    finally:
        state_mod.sessions.pop(sid, None)


@pytest.mark.asyncio
async def test_auto_snapshot_state_only_commit_keeps_runtime_block(monkeypatch):
    """Successful capture without runtime payload must not clear unavailable block."""
    import backend.state as state_mod
    from backend.models.schemas import AnalysisState
    from backend.workflow import SingleCellAnalysisWorkflow

    sid = "sess-auto-state-only"
    state = AnalysisState(session_id=sid, user_id="u1")
    state.runtime_restore_status = "unavailable"
    state_mod.sessions[sid] = state
    sandbox = type("Sandbox", (), {})()
    sandbox.notebook_state = None
    sandbox.km = None
    workflow = _SnapshotWorkflowHarness(sid, sandbox)

    record = _v2_record(runtime_included=False, snapshot_id="state-only")

    async def _capture(st, request, *a, **k):
        # V2 capture_required_snapshot does not clear runtime_restore_status.
        return record

    workflow.snapshot_service.capture_required_snapshot = AsyncMock(side_effect=_capture)
    workflow.snapshot_service.object_store = MagicMock()
    workflow.snapshot_service.rebuild_snapshots_from_manifests = MagicMock(return_value=[])

    try:
        result = await SingleCellAnalysisWorkflow._create_auto_snapshot(
            workflow, state, "manual",
        )
        assert result is record
        assert state.runtime_restore_status == "unavailable"
    finally:
        state_mod.sessions.pop(sid, None)


@pytest.mark.asyncio
async def test_auto_snapshot_complete_commit_does_not_auto_clear_block(monkeypatch):
    """V2 capture success does not auto-clear runtime_restore_status (restore/rerun own that)."""
    import backend.state as state_mod
    from backend.models.schemas import AnalysisState
    from backend.workflow import SingleCellAnalysisWorkflow

    sid = "sess-auto-complete"
    state = AnalysisState(session_id=sid, user_id="u1")
    state.runtime_restore_status = "unavailable"
    state_mod.sessions[sid] = state
    sandbox = type("Sandbox", (), {})()
    sandbox.notebook_state = None
    sandbox.km = None
    workflow = _SnapshotWorkflowHarness(sid, sandbox)

    record = _v2_record(runtime_included=True, snapshot_id="complete")

    async def _capture(st, request, *a, **k):
        return record

    workflow.snapshot_service.capture_required_snapshot = AsyncMock(side_effect=_capture)
    workflow.snapshot_service.object_store = MagicMock()
    workflow.snapshot_service.rebuild_snapshots_from_manifests = MagicMock(return_value=[])

    try:
        result = await SingleCellAnalysisWorkflow._create_auto_snapshot(
            workflow, state, "manual",
        )
        assert result is record
        # Capture success rebuilds inventory but does not flip runtime_restore_status;
        # only restore/rerun paths clear the block.
        assert state.runtime_restore_status == "unavailable"
    finally:
        state_mod.sessions.pop(sid, None)


@pytest.mark.asyncio
async def test_capture_required_snapshot_never_clears_runtime_block():
    """SnapshotService.capture_required_snapshot must not clear runtime_restore_status."""
    from backend.models.schemas import AnalysisState
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        SnapshotCaptureRequest,
    )
    from backend.services.snapshot_v2.service import SnapshotService
    from pathlib import Path
    from unittest.mock import patch

    sid = "sess-register-no-unblock"
    state = AnalysisState(session_id=sid, user_id="u1")
    state.runtime_restore_status = "unavailable"

    import tempfile
    base = Path(tempfile.mkdtemp(prefix="scai-snap-"))
    paths = type("Paths", (), {
        "root": base,
        "objects": base / "objects",
        "staging": base / "staging",
        "journals": base / "journals",
        "integrity": base / "integrity",
        "committed": base / "committed",
    })()
    for d in [paths.root, paths.objects, paths.staging, paths.journals, paths.integrity, paths.committed]:
        d.mkdir(parents=True, exist_ok=True)
    service = SnapshotService(sid, paths)
    record = _v2_record(runtime_included=True)

    async def _capture(*a, **k):
        return record

    service.capture = AsyncMock(side_effect=_capture)
    service.rebuild_snapshots_from_manifests = MagicMock(return_value=[])
    service._find_committed_by_idempotency = MagicMock(return_value=None)

    req = SnapshotCaptureRequest(
        session_id=sid,
        reason="manual",
        phase="manual",
        capsule=IdleCapsule(graph_version="2.0"),
    )
    result = await service.capture_required_snapshot(
        state, req, sandbox=None, workflow_state={"kernel_present": False},
        workspace_root="/tmp",
    )
    assert result is record
    assert state.runtime_restore_status == "unavailable"
