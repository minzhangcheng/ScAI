"""Post-chat snapshot safety: exact safe-point + empty-complete eligibility."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_state(session_id: str = "sess-chat-safe", **kw):
    from backend.models.schemas import AnalysisState

    sid = kw.pop("id", session_id)
    return AnalysisState(
        id=sid,
        name="Chat Snapshot Safety",
        user_id="u1",
        initial_instruction="analyze",
        **kw,
    )


def _complete_record(session_id: str = "sess-chat-safe", snapshot_id: str = "snap-chat-ok"):
    from backend.services.snapshot_v2.models import (
        GraphNodeCapsule,
        IdleCapsule,
        ObjectRef,
        SnapshotManifestV2,
        SnapshotRecordV2,
        empty_complete_runtime,
    )

    stub = ObjectRef(sha256="a" * 64, size=1, media_type="application/json")
    return SnapshotRecordV2(
        snapshot_id=snapshot_id,
        manifest=SnapshotManifestV2(
            session_id=session_id,
            snapshot_id=snapshot_id,
            name=snapshot_id,
            phase="post_chat",
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            analysis_state_ref=stub,
            notebook_ref=stub,
            runtime=empty_complete_runtime(),
            continuation=IdleCapsule(graph_version="2.0"),
            logical_size_bytes=1,
            physical_added_bytes=0,
            root_hash="b" * 64,
        ),
        failure=None,
    )


def _v2_paths(tmp_path: Path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [
                self.root,
                self.objects,
                self.staging,
                self.journals,
                self.integrity,
                self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    paths = FakePaths()
    paths.ensure_dirs()
    return paths


def _globals_result():
    return MagicMock(
        success=True,
        output=(
            "__RUNTIME_GLOBALS_BEGIN__\n"
            '{"results": "dict", "adata": "anndata.AnnData"}\n'
            "__RUNTIME_GLOBALS_END__"
        ),
        error=None,
    )


def _mock_sandbox_with_kernel(session_id: str = "sess-chat-safe"):
    mock = MagicMock()
    mock.session_id = session_id
    mock.runtime_revision = 3
    mock.saved_runtime_revision = 1
    mock.km = MagicMock()
    mock.runtime_dependency_tracker = None
    mock.execute = AsyncMock(return_value=_globals_result())
    mock.await_pending_saves = AsyncMock()
    mock._exec_lock = asyncio.Lock()
    return mock


def _post_chat_request(session_id: str = "sess-chat-safe", capsule=None):
    from backend.services.snapshot_v2.models import (
        GraphNodeCapsule,
        ObjectRef,
        SnapshotCaptureRequest,
    )

    if capsule is None:
        stub = ObjectRef(sha256="c" * 64, size=1, media_type="application/json")
        capsule = GraphNodeCapsule(
            graph_version="2.0",
            next_node="pi_synthesis",
            workflow_state_ref=stub,
            planning_stage="full",
        )
    return SnapshotCaptureRequest(
        session_id=session_id,
        reason="post_chat",
        phase="post_chat",
        capsule=capsule,
    )


@pytest.mark.asyncio
async def test_post_chat_snapshot_is_required_v2_producer(monkeypatch):
    import backend.routes.chat as chat
    from backend.services.snapshot_v2.service import SnapshotService

    state = _make_state()
    capture = AsyncMock(return_value=_complete_record(state.id))
    monkeypatch.setattr(SnapshotService, "capture_required_snapshot", capture)

    await chat._create_chat_snapshot(state.id, state)

    capture.assert_awaited_once()


@pytest.mark.asyncio
async def test_post_chat_snapshot_failure_persists_no_skip_gate(monkeypatch):
    import backend.routes.chat as chat
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.service import SnapshotService

    state = _make_state(id="sess-chat-fail")
    state.wait_input_reason = "snapshot_failure"
    state.snapshot_failure = {
        "code": "LIVE_SANDBOX_REQUIRED",
        "message": "live sandbox required",
        "allowed_actions": ["retry", "manage", "terminate"],
    }

    async def _fail(*_a, **_kw):
        raise RequiredSnapshotBlocked(
            "LIVE_SANDBOX_REQUIRED",
            "A live or unproven session cannot commit empty-complete runtime",
        )

    capture = AsyncMock(side_effect=_fail)
    monkeypatch.setattr(SnapshotService, "capture_required_snapshot", capture)

    update = AsyncMock()
    save = AsyncMock()
    monkeypatch.setattr(
        "backend.services.session_store.update_session_cache",
        update,
    )
    monkeypatch.setattr(
        "backend.services.session_store.async_save_session",
        save,
    )

    with pytest.raises(RequiredSnapshotBlocked, match="LIVE_SANDBOX_REQUIRED"):
        await chat._create_chat_snapshot(state.id, state)

    update.assert_awaited_once()
    save.assert_awaited_once()
    assert state.wait_input_reason == "snapshot_failure"
    assert "skip" not in state.snapshot_failure["allowed_actions"]


@pytest.mark.asyncio
async def test_active_workflow_post_chat_uses_live_sandbox_and_graph_capsule(
    tmp_path, monkeypatch,
):
    from backend.routes.chat import capture_post_chat_snapshot
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import sessions, workflows

    session_id = "sess-post-chat-active"
    state = _make_state(id=session_id)
    sandbox = _mock_sandbox_with_kernel(session_id)

    async def _capture_runtime(self, sb, **kw):
        from backend.services.snapshot_v2.models import (
            RuntimeDependency,
            RuntimeDependencyManifest,
            RuntimeManifest,
        )

        # Prove the live sandbox was used (capture path invokes execute).
        await sb.execute("pass")
        residual = self.object_store.put_bytes(
            b"results-payload", "application/octet-stream",
        ).ref
        return RuntimeManifest(
            status="complete",
            dependencies=RuntimeDependencyManifest(
                required={
                    "results": RuntimeDependency(
                        name="results", serializer="dill", type_name="dict",
                    ),
                },
            ),
            residual_ref=residual,
            captured_residual_names=["results"],
            adata_refs={},
        )

    active_workflow = MagicMock()
    active_workflow.session_id = session_id
    active_workflow.sandbox = sandbox
    active_workflow._current_graph_state = {
        "current_round": 1,
        "next_node": "pi_synthesis",
        "kernel_present": True,
    }

    paths = _v2_paths(tmp_path)
    monkeypatch.setattr(SnapshotService, "_capture_runtime", _capture_runtime)

    sp = get_snapshot_safe_point_manager()
    _orig_request = sp.request_safe_point

    async def _request_and_ack(sid: str, owner_token: str = "test-owner", **kwargs):
        created = await _orig_request(sid, owner_token)
        gen = sp.get_pending_generation(sid)
        if gen is not None:
            await sp.acknowledge_safe_point_request(
                sid,
                gen,
                {
                    "kernel_present": True,
                    "next_node": "pi_synthesis",
                    "current_round": 1,
                },
                role="workflow",
            )
        return created

    monkeypatch.setattr(sp, "request_safe_point", _request_and_ack)

    sessions[session_id] = state
    workflows[session_id] = active_workflow
    try:
        (tmp_path / "ws").mkdir(exist_ok=True)
        with patch(
            "backend.config._config.get_session_workspace",
            return_value=tmp_path / "ws",
        ):
            with patch(
                "backend.services.snapshot_v2.paths.SnapshotV2Paths.for_session",
                return_value=paths,
            ):
                record = await capture_post_chat_snapshot(session_id, state)

        assert record.failure is None, f"unexpected failure: {record.failure}"
        assert record.manifest.continuation.kind == "graph_node"
        assert record.manifest.runtime.dependencies.required
        sandbox.execute.assert_awaited()
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)
        await sp.release(session_id, "test-owner")
        await sp.cancel_safe_point_request(session_id, "test-owner")


@pytest.mark.asyncio
async def test_missing_sandbox_with_live_kernel_descriptor_fails_closed(tmp_path):
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.service import SnapshotService

    session_id = "sess-live-kernel-no-sb"
    state = _make_state(id=session_id)
    paths = _v2_paths(tmp_path)
    service = SnapshotService(session_id, paths)
    request = _post_chat_request(session_id)

    with patch.object(service, "_send_state_sync", new=AsyncMock()):
        with pytest.raises(RequiredSnapshotBlocked, match="LIVE_SANDBOX_REQUIRED"):
            await service.capture_required_snapshot(
                state,
                request,
                sandbox=None,
                workflow_state={"kernel_present": True, "next_node": "pi_synthesis"},
                workspace_root=str(tmp_path / "ws"),
            )


@pytest.mark.asyncio
async def test_idle_empty_complete_requires_acknowledged_no_kernel_descriptor(tmp_path):
    from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
    from backend.services.snapshot_v2.service import SnapshotService

    session_id = "sess-idle-empty"
    state = _make_state(id=session_id)
    paths = _v2_paths(tmp_path)
    service = SnapshotService(session_id, paths)
    (tmp_path / "ws").mkdir(exist_ok=True)

    # Missing descriptor must NOT authorize empty-complete.
    request = SnapshotCaptureRequest(
        session_id=session_id,
        reason="post_chat",
        phase="post_chat",
        capsule=IdleCapsule(graph_version="2.0"),
    )
    with patch.object(service, "_send_state_sync", new=AsyncMock()):
        from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

        with pytest.raises(RequiredSnapshotBlocked, match="LIVE_SANDBOX_REQUIRED"):
            await service.capture_required_snapshot(
                state,
                request,
                sandbox=None,
                workflow_state={},
                workspace_root=str(tmp_path / "ws"),
            )

    # Producer-owned trusted no-kernel + inactive descriptor may empty-complete.
    # Pre-hold makes _create_safe_point a no-op (idempotent) and keeps trust.
    await service.safe_point.acquire(
        session_id,
        {"kernel_present": False, "inactive": True},
        eligibility_trusted=True,
        owner_token="test-owner",
    )
    try:
        with patch.object(service, "_send_state_sync", new=AsyncMock()):
            record = await service.capture_required_snapshot(
                state,
                request,
                sandbox=None,
                workflow_state={"kernel_present": False, "inactive": True},
                workspace_root=str(tmp_path / "ws"),
            )
        assert record.failure is None
        assert record.manifest.runtime.status == "complete"
        assert record.manifest.runtime.dependencies.required == {}
    finally:
        await service.safe_point.release(session_id, "test-owner")


@pytest.mark.asyncio
async def test_terminal_empty_complete_with_no_kernel_descriptor(tmp_path):
    from backend.services.snapshot_v2.models import (
        SnapshotCaptureRequest,
        TerminalCapsule,
    )
    from backend.services.snapshot_v2.service import SnapshotService

    session_id = "sess-terminal-empty"
    state = _make_state(id=session_id)
    paths = _v2_paths(tmp_path)
    service = SnapshotService(session_id, paths)
    (tmp_path / "ws").mkdir(exist_ok=True)
    request = SnapshotCaptureRequest(
        session_id=session_id,
        reason="post_report",
        phase="post_report",
        capsule=TerminalCapsule(graph_version="2.0", status="completed"),
    )
    await service.safe_point.acquire(
        session_id,
        {"kernel_present": False, "status": "completed"},
        eligibility_trusted=True,
        owner_token="test-owner",
    )
    try:
        with patch.object(service, "_send_state_sync", new=AsyncMock()):
            record = await service.capture_required_snapshot(
                state,
                request,
                sandbox=None,
                workflow_state={
                    "kernel_present": False,
                    "status": "completed",
                },
                workspace_root=str(tmp_path / "ws"),
            )
        assert record.failure is None
        assert record.manifest.runtime.dependencies.required == {}
        assert record.manifest.continuation.kind == "terminal"
    finally:
        await service.safe_point.release(session_id, "test-owner")


@pytest.mark.asyncio
async def test_active_pre_restore_never_accepts_empty_complete(tmp_path):
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import sessions, workflows

    session_id = "sess-pre-restore-active"
    state = _make_state(id=session_id)
    paths = _v2_paths(tmp_path)
    service = SnapshotService(session_id, paths)
    (tmp_path / "ws").mkdir(exist_ok=True)

    active_workflow = MagicMock()
    active_workflow.session_id = session_id
    active_workflow.sandbox = _mock_sandbox_with_kernel(session_id)
    sessions[session_id] = state
    workflows[session_id] = active_workflow

    # Contradictory trusted no-kernel descriptor must not authorize empty-complete.
    await service.safe_point.acquire(
        session_id,
        {"kernel_present": False, "inactive": True},
        eligibility_trusted=True,
        owner_token="test-owner",
    )
    request = SnapshotCaptureRequest(
        session_id=session_id,
        reason="pre_restore",
        phase="pre_restore",
        capsule=IdleCapsule(graph_version="2.0"),
    )
    try:
        with patch.object(service, "_send_state_sync", new=AsyncMock()):
            with pytest.raises(
                RequiredSnapshotBlocked, match="LIVE_SANDBOX_REQUIRED",
            ):
                await service.capture_required_snapshot(
                    state,
                    request,
                    sandbox=None,
                    workflow_state={
                        "kernel_present": False,
                        "inactive": True,
                    },
                    workspace_root=str(tmp_path / "ws"),
                )
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)
        await service.safe_point.release(session_id, "test-owner")


@pytest.mark.asyncio
async def test_post_chat_does_not_cancel_foreign_safe_point_hold(tmp_path, monkeypatch):
    """finally must not release a safe point owned by restore/another capture."""
    from backend.routes.chat import capture_post_chat_snapshot
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import sessions, workflows

    session_id = "sess-foreign-hold"
    state = _make_state(id=session_id)
    paths = _v2_paths(tmp_path)
    sp = get_snapshot_safe_point_manager()

    # Pre-existing foreign hold (e.g. restore).
    foreign_desc = {"kernel_present": True, "holder": "restore"}
    assert await sp.acquire(session_id, foreign_desc, owner_token="foreign-owner")
    sessions[session_id] = state
    workflows[session_id] = MagicMock(session_id=session_id, sandbox=None)

    try:
        with patch(
            "backend.config._config.get_session_workspace",
            return_value=tmp_path / "ws",
        ):
            with patch(
                "backend.services.snapshot_v2.paths.SnapshotV2Paths.for_session",
                return_value=paths,
            ):
                with pytest.raises(
                    RequiredSnapshotBlocked, match="LIVE_SANDBOX_REQUIRED",
                ):
                    await capture_post_chat_snapshot(session_id, state)

        # Foreign hold must still be present with original descriptor.
        assert sp.is_paused(session_id)
        assert sp.get_descriptor(session_id) == foreign_desc
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)
        await sp.release(session_id, "foreign-owner")
        await sp.cancel_safe_point_request(session_id, "foreign-owner")


@pytest.mark.asyncio
async def test_active_path_without_ack_fails_closed_no_force_acquire(
    tmp_path, monkeypatch,
):
    """Cancel race / lost ack must not force-acquire."""
    from backend.routes.chat import capture_post_chat_snapshot
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    from backend.state import sessions, workflows

    session_id = "sess-no-ack"
    state = _make_state(id=session_id)
    paths = _v2_paths(tmp_path)
    sp = get_snapshot_safe_point_manager()

    active_workflow = MagicMock()
    active_workflow.session_id = session_id
    active_workflow.sandbox = _mock_sandbox_with_kernel(session_id)
    active_workflow._current_graph_state = {
        "current_round": 1,
        "next_node": "pi_synthesis",
    }

    # request succeeds but is cancelled before workflow acks (simulate race).
    _orig_request = sp.request_safe_point

    async def _request_then_drop(sid: str, owner_token: str = "test-owner", **kwargs):
        created = await _orig_request(sid, owner_token)
        if created:
            # Simulate cancel race: pending request withdrawn, never paused.
            await sp.cancel_safe_point_request(sid, owner_token)
        return created

    monkeypatch.setattr(sp, "request_safe_point", _request_then_drop)

    sessions[session_id] = state
    workflows[session_id] = active_workflow
    try:
        with patch(
            "backend.config._config.get_session_workspace",
            return_value=tmp_path / "ws",
        ):
            with patch(
                "backend.services.snapshot_v2.paths.SnapshotV2Paths.for_session",
                return_value=paths,
            ):
                with pytest.raises(
                    RequiredSnapshotBlocked, match="LIVE_SANDBOX_REQUIRED",
                ):
                    await capture_post_chat_snapshot(session_id, state)

        # Must not leave a force-acquired hold behind.
        assert not sp.is_paused(session_id)
        assert not sp.has_pending_request(session_id)
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)
        await sp.release(session_id, "test-owner")
        await sp.cancel_safe_point_request(session_id, "test-owner")


@pytest.mark.asyncio
async def test_raw_workflow_state_cannot_authorize_empty_complete(tmp_path):
    """Raw workflow_state must not authorize empty-complete on the default path.

    Unpatched production path: ``_create_safe_point`` runs and must not seed
    empty-complete authority from caller-invented idle fields.
    """
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    from backend.services.snapshot_v2.service import SnapshotService

    session_id = "sess-raw-ws"
    state = _make_state(id=session_id)
    paths = _v2_paths(tmp_path)
    service = SnapshotService(session_id, paths)
    sp = get_snapshot_safe_point_manager()
    await sp.release(session_id, "test-owner")
    (tmp_path / "ws").mkdir(exist_ok=True)
    request = SnapshotCaptureRequest(
        session_id=session_id,
        reason="post_chat",
        phase="post_chat",
        capsule=IdleCapsule(graph_version="2.0"),
    )

    # No active workflow, no prior ack — inventing idle workflow_state must fail.
    with patch.object(service, "_send_state_sync", new=AsyncMock()):
        with pytest.raises(
            RequiredSnapshotBlocked, match="LIVE_SANDBOX_REQUIRED",
        ):
            await service.capture_required_snapshot(
                state,
                request,
                sandbox=None,
                workflow_state={
                    "kernel_present": False,
                    "inactive": True,
                },
                workspace_root=str(tmp_path / "ws"),
            )
    assert not sp.is_paused(session_id)


@pytest.mark.asyncio
async def test_trusted_post_chat_idle_empty_complete(tmp_path, monkeypatch):
    """Trusted idle post-chat producer may empty-complete after lifecycle proof."""
    from backend.models.schemas import TaskStatus
    from backend.routes.chat import capture_post_chat_snapshot
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    from backend.state import sessions, workflows

    session_id = "sess-trusted-idle"
    state = _make_state(id=session_id, status=TaskStatus.PAUSED)
    paths = _v2_paths(tmp_path)
    (tmp_path / "ws").mkdir(exist_ok=True)
    sp = get_snapshot_safe_point_manager()
    sessions.pop(session_id, None)
    workflows.pop(session_id, None)

    try:
        with patch(
            "backend.config._config.get_session_workspace",
            return_value=tmp_path / "ws",
        ):
            with patch(
                "backend.services.snapshot_v2.paths.SnapshotV2Paths.for_session",
                return_value=paths,
            ):
                with patch.object(
                    __import__(
                        "backend.services.snapshot_v2.service",
                        fromlist=["SnapshotService"],
                    ).SnapshotService,
                    "_send_state_sync",
                    new=AsyncMock(),
                ):
                    record = await capture_post_chat_snapshot(session_id, state)

        assert record is not None
        assert record.failure is None
        assert record.manifest.runtime.status == "complete"
        assert record.manifest.runtime.dependencies.required == {}
        assert not sp.is_paused(session_id)
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)
        await sp.release(session_id, "test-owner")
        await sp.cancel_safe_point_request(session_id, "test-owner")


@pytest.mark.asyncio
async def test_hitl_wait_statuses_allow_idle_empty_complete(tmp_path, monkeypatch):
    """WAITING_ANNOTATION / PLAN_REVIEW / SPECIES_INPUT authorize idle empty-complete."""
    from backend.models.schemas import TaskStatus
    from backend.routes.chat import capture_post_chat_snapshot
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import sessions, workflows

    hitl_statuses = (
        TaskStatus.WAITING_ANNOTATION,
        TaskStatus.WAITING_PLAN_REVIEW,
        TaskStatus.WAITING_SPECIES_INPUT,
    )
    paths = _v2_paths(tmp_path)
    sp = get_snapshot_safe_point_manager()

    for status in hitl_statuses:
        session_id = f"sess-hitl-{status.value}"
        state = _make_state(id=session_id, status=status)
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)

        capture = AsyncMock(return_value=_complete_record(session_id, f"snap-{status.value}"))
        monkeypatch.setattr(SnapshotService, "capture_required_snapshot", capture)

        try:
            with patch(
                "backend.config._config.get_session_workspace",
                return_value=tmp_path / "ws",
            ):
                with patch(
                    "backend.services.snapshot_v2.paths.SnapshotV2Paths.for_session",
                    return_value=paths,
                ):
                    record = await capture_post_chat_snapshot(session_id, state)

            assert record is not None
            capture.assert_awaited()
            call_kw = capture.await_args.kwargs
            assert call_kw.get("sandbox") is None
            ws = call_kw.get("workflow_state") or {}
            assert ws.get("kernel_present") is False
            assert ws.get("inactive") is True
            # Own hold released in finally.
            assert not sp.is_paused(session_id)
        finally:
            sessions.pop(session_id, None)
            workflows.pop(session_id, None)
            await sp.release(session_id, "test-owner")
            await sp.cancel_safe_point_request(session_id, "test-owner")


@pytest.mark.asyncio
async def test_chat_gate_helper_persists_chat_retry_context():
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.failure_gate import (
        enter_chat_snapshot_failure_gate,
    )

    state = AnalysisState(id="chat-retry", user_id="u1", status=TaskStatus.COMPLETED)
    error = RequiredSnapshotBlocked("NO_SPACE", "disk full")
    enter_chat_snapshot_failure_gate(state, error, TaskStatus.COMPLETED)
    assert state.snapshot_failure["retry_context"]["producer"] == "chat"
    assert state.snapshot_failure["retry_context"]["previous_status"] == "completed"

