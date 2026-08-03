"""Linearization contracts between snapshot reservations and producers."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.requires_snapshot_v2


def _session_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def test_every_production_reservation_uses_the_lifecycle_barrier():
    """New call sites must not recreate a check-then-publish window."""
    from backend.routes.chat_parts.snapshots import capture_post_chat_snapshot
    from backend.services.snapshot_v2.jobs import RestoreJobManager
    from backend.services.snapshot_v2.service_parts.session_state import (
        SessionStateMixin,
    )
    from backend.workflow.reporting import ReportingMixin

    reservers = (
        SessionStateMixin._create_safe_point,
        capture_post_chat_snapshot,
        ReportingMixin._capture_post_report_snapshot,
        RestoreJobManager.enqueue,
    )
    for reserver in reservers:
        source = inspect.getsource(reserver)
        assert "establish_snapshot_operation" in source, reserver.__qualname__

    enqueue_source = inspect.getsource(RestoreJobManager.enqueue)
    assert "request_safe_point_when_available" not in enqueue_source


@pytest.mark.asyncio
async def test_capture_reservation_blocks_late_headless_publication(monkeypatch):
    """A producer cannot publish in capture's manager-mutation window."""
    from backend.models.schemas import AnalysisState
    from backend.services import workflow_runner
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.services.workflow_runner import SessionWorkflowConflictError
    from backend.state import _run_tasks, sessions

    session_id = _session_id("capture-first")
    state = AnalysisState(id=session_id, user_id="owner")
    sessions[session_id] = state
    safe_point = get_snapshot_safe_point_manager()
    service = SnapshotService.__new__(SnapshotService)
    service.session_id = session_id
    service.safe_point = safe_point

    entered_acquire = asyncio.Event()
    release_acquire = asyncio.Event()
    original_acquire = safe_point.acquire

    async def delayed_acquire(*args, **kwargs):
        entered_acquire.set()
        await release_acquire.wait()
        return await original_acquire(*args, **kwargs)

    monkeypatch.setattr(safe_point, "acquire", delayed_acquire)
    monkeypatch.setattr(
        workflow_runner,
        "_headless_workflow_runner",
        AsyncMock(),
    )

    capture = asyncio.create_task(
        service._create_safe_point({}, analysis_state=state),
    )
    producer = None
    try:
        await asyncio.wait_for(entered_acquire.wait(), timeout=1)
        producer = asyncio.create_task(
            workflow_runner.start_workflow_if_needed(
                session_id, expected_state=state,
            )
        )
        await asyncio.sleep(0)
        assert not producer.done(), "producer bypassed the lifecycle barrier"

        release_acquire.set()
        await asyncio.wait_for(capture, timeout=1)
        with pytest.raises(SessionWorkflowConflictError):
            await asyncio.wait_for(producer, timeout=1)
        assert session_id not in _run_tasks
    finally:
        release_acquire.set()
        if not capture.done():
            capture.cancel()
        if producer is not None and not producer.done():
            producer.cancel()
        await asyncio.gather(
            capture,
            *(tuple([producer]) if producer is not None else ()),
            return_exceptions=True,
        )
        await service._release_safe_point()
        sessions.pop(session_id, None)
        _run_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_existing_headless_publication_blocks_idle_chat_hold(
    monkeypatch,
    tmp_path,
):
    """The reverse ordering cannot misclassify a build gap as idle."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes.chat_parts.snapshots import capture_post_chat_snapshot
    from backend.services import workflow_runner
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import _run_tasks, sessions

    session_id = _session_id("producer-first")
    state = AnalysisState(
        id=session_id,
        user_id="owner",
        status=TaskStatus.PAUSED,
    )
    sessions[session_id] = state
    root = tmp_path / session_id
    paths = SnapshotV2Paths(
        root=root / "v2",
        objects=root / "objects",
        staging=root / "staging",
        journals=root / "journals",
        integrity=root / "integrity",
        committed=root / "committed",
    )
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()

    async def held_runner(*_args, **_kwargs):
        runner_started.set()
        await release_runner.wait()

    capture = AsyncMock()
    monkeypatch.setattr(workflow_runner, "_headless_workflow_runner", held_runner)
    monkeypatch.setattr(
        "backend.services.snapshot_v2.paths.SnapshotV2Paths.for_session",
        lambda _sid: paths,
    )
    monkeypatch.setattr(
        "backend.config._config.get_session_workspace",
        lambda _sid: root / "workspace",
    )
    monkeypatch.setattr(SnapshotService, "capture_required_snapshot", capture)

    task = None
    try:
        assert await workflow_runner.start_workflow_if_needed(
            session_id, expected_state=state,
        )
        task = _run_tasks[session_id]
        await asyncio.wait_for(runner_started.wait(), timeout=1)

        with pytest.raises(RequiredSnapshotBlocked, match="active producer"):
            await capture_post_chat_snapshot(session_id, state)
        capture.assert_not_awaited()
    finally:
        release_runner.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        _run_tasks.pop(session_id, None)
        sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_tombstone_refuses_reservation_before_manager_mutation(
    monkeypatch,
    tmp_path,
):
    """Deletion winning the barrier leaves no stale safe-point hold."""
    from backend.models.schemas import AnalysisState
    from backend.services import session_lifecycle
    from backend.services.session_lifecycle import (
        SessionGoneError,
        establish_snapshot_operation,
        register_new_generation,
        tombstone_generation,
    )

    root = tmp_path / "lifecycle"
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", root / "sessions")
    monkeypatch.setattr(session_lifecycle, "UPLOAD_DIR", root / "uploads")
    monkeypatch.setattr(session_lifecycle, "WORKSPACE_DIR", root / "workspace")
    monkeypatch.setattr(session_lifecycle, "REPORTS_DIR", root / "reports")
    monkeypatch.setattr(session_lifecycle, "STATE_DIR", root / "states")
    session_lifecycle.reset_lifecycle_locks()

    session_id = _session_id("deleted-first")
    state = AnalysisState(id=session_id, user_id="owner")
    register_new_generation(state)
    tombstone_generation(session_id, "owner", state.lifecycle_generation)
    reserve = AsyncMock(return_value=True)

    with pytest.raises(SessionGoneError):
        await establish_snapshot_operation(session_id, reserve)
    reserve.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_handle_is_published_before_deletion_can_tombstone(
    monkeypatch,
    tmp_path,
):
    """Reservation ownership and restore registry publication are atomic."""
    from types import SimpleNamespace

    from backend.models.schemas import AnalysisState
    from backend.services import session_deletion, session_lifecycle
    from backend.services.session_lifecycle import (
        bind_import_generation,
        claim_import_generation,
        commit_import_generation,
        complete_tombstone_cleanup,
        get_lifecycle_lock,
        register_new_generation,
        tombstone_generation,
    )
    from backend.services.snapshot_v2.jobs import (
        RestoreJobManager,
        _persist_handle,
    )
    from backend.services.snapshot_v2.models import SnapshotRestoreRequest
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )
    from backend.state import (
        initializing_sessions,
        sessions,
        snapshot_restore_jobs,
    )

    lifecycle_root = tmp_path / "lifecycle"
    session_root = lifecycle_root / "sessions"
    for module in (session_lifecycle, session_deletion):
        monkeypatch.setattr(module, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_lifecycle, "UPLOAD_DIR", lifecycle_root / "uploads")
    monkeypatch.setattr(
        session_lifecycle, "WORKSPACE_DIR", lifecycle_root / "workspace",
    )
    monkeypatch.setattr(
        session_lifecycle, "REPORTS_DIR", lifecycle_root / "reports",
    )
    monkeypatch.setattr(session_lifecycle, "STATE_DIR", lifecycle_root / "states")
    session_lifecycle.reset_lifecycle_locks()

    paths = SimpleNamespace(
        root=tmp_path / "v2",
        objects=tmp_path / "objects",
        staging=tmp_path / "staging",
        journals=tmp_path / "journals",
        integrity=tmp_path / "integrity",
        committed=tmp_path / "committed",
    )
    for path in vars(paths).values():
        path.mkdir(parents=True, exist_ok=True)

    session_id = _session_id("restore-delete-window")
    state = AnalysisState(id=session_id, user_id="owner")
    generation = register_new_generation(state)
    sessions[session_id] = state
    initializing_sessions.add(session_id)

    manager = RestoreJobManager(paths=paths, session_id=session_id)
    safe_point = get_snapshot_safe_point_manager()
    reserved = asyncio.Event()
    let_request_return = asyncio.Event()
    original_request = safe_point.request_safe_point

    async def _hold_after_reservation(*args, **kwargs):
        reservation = await original_request(*args, **kwargs)
        reserved.set()
        await let_request_return.wait()
        return reservation

    monkeypatch.setattr(safe_point, "request_safe_point", _hold_after_reservation)

    observed_handle = None

    async def _delete_after_barrier():
        nonlocal observed_handle
        async with get_lifecycle_lock(session_id):
            candidates = [
                handle
                for handle in snapshot_restore_jobs.values()
                if handle.record.session_id == session_id
            ]
            assert len(candidates) == 1
            observed_handle = candidates[0]
            assert observed_handle.lifecycle_generation == generation
            tombstone_generation(session_id, "owner", generation)
            observed_handle.cancel_event.set()
            if observed_handle.task is not None:
                observed_handle.task.cancel()
            snapshot_restore_jobs.pop(
                observed_handle.record.transaction_id, None,
            )

    enqueue = asyncio.create_task(manager.enqueue(
        session_id,
        MagicMock(status="ready"),
        SnapshotRestoreRequest(
            snapshot_id="snapshot-target",
            runtime_choice="reset",
            current_state_choice="abort",
        ),
    ))
    deletion = None
    try:
        await asyncio.wait_for(reserved.wait(), timeout=1)
        deletion = asyncio.create_task(_delete_after_barrier())
        await asyncio.sleep(0)
        assert not deletion.done(), "deletion bypassed restore publication"

        let_request_return.set()
        record = await asyncio.wait_for(enqueue, timeout=1)
        await asyncio.wait_for(deletion, timeout=1)

        assert observed_handle is not None
        assert observed_handle.lifecycle_generation == generation
        if observed_handle.task is not None:
            await asyncio.gather(observed_handle.task, return_exceptions=True)

        # Reuse the same ID as the next lifecycle generation. The old handle
        # must not publish another journal revision into that generation.
        assert complete_tombstone_cleanup(session_id, "owner", generation)
        imported_dir = tmp_path / "imported-generation"
        assert claim_import_generation(
            session_id, "owner", imported_dir,
        ) == generation + 1
        replacement = AnalysisState(id=session_id, user_id="owner")
        bind_import_generation(replacement)
        commit_import_generation(replacement)
        sessions[session_id] = replacement

        journal = paths.journals / f"{record.transaction_id}.json"
        before = journal.read_bytes()
        observed_handle.record = observed_handle.record.model_copy(
            update={"status": "failed"},
        )
        assert _persist_handle(observed_handle) is False
        assert journal.read_bytes() == before
    finally:
        let_request_return.set()
        if not enqueue.done():
            enqueue.cancel()
        if deletion is not None and not deletion.done():
            deletion.cancel()
        await asyncio.gather(
            enqueue,
            *(tuple([deletion]) if deletion is not None else ()),
            return_exceptions=True,
        )
        if observed_handle is not None and observed_handle.task is not None:
            observed_handle.task.cancel()
            await asyncio.gather(observed_handle.task, return_exceptions=True)
        owner_token = (
            observed_handle.owner_token if observed_handle is not None else ""
        )
        if owner_token:
            await safe_point.cancel_safe_point_request(session_id, owner_token)
        initializing_sessions.discard(session_id)
        sessions.pop(session_id, None)
        for transaction_id, handle in list(snapshot_restore_jobs.items()):
            if handle.record.session_id == session_id:
                snapshot_restore_jobs.pop(transaction_id, None)


@pytest.mark.asyncio
async def test_restore_releases_lifecycle_without_waiting_for_capture_hold(
    monkeypatch,
):
    """A capture hold cannot form SP-to-lifecycle/lifecycle-to-SP deadlock."""
    from backend.services.session_lifecycle import get_lifecycle_lock
    from backend.services.snapshot_v2.errors import SnapshotTransactionError
    from backend.services.snapshot_v2.jobs import RestoreJobManager
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )

    session_id = _session_id("capture-restore-lock-order")
    manager = RestoreJobManager(paths=MagicMock(), session_id=session_id)
    safe_point = get_snapshot_safe_point_manager()
    foreign_owner = f"capture:{session_id}"
    assert await safe_point.acquire(
        session_id,
        {"kernel_present": False},
        owner_token=foreign_owner,
    )

    entered_request = asyncio.Event()
    release_request = asyncio.Event()
    persistence_entered = asyncio.Event()
    original_request = safe_point.request_safe_point

    async def observed_nonblocking_request(*args, **kwargs):
        # Instrument the exact manager mutation window. A capture's session
        # persistence queues behind it, then must acquire immediately after the
        # one non-blocking request returns even though the foreign SP remains.
        assert get_lifecycle_lock(session_id).locked()
        entered_request.set()
        await release_request.wait()
        return await original_request(*args, **kwargs)

    forbidden_wait = AsyncMock(
        side_effect=AssertionError("restore waited for safe-point availability"),
    )
    monkeypatch.setattr(safe_point, "request_safe_point", observed_nonblocking_request)
    monkeypatch.setattr(
        safe_point,
        "request_safe_point_when_available",
        forbidden_wait,
    )

    async def capture_persistence() -> None:
        async with get_lifecycle_lock(session_id):
            persistence_entered.set()

    enqueue = asyncio.create_task(
        manager.enqueue(
            session_id,
            MagicMock(),
            MagicMock(snapshot_id="snapshot-target"),
        )
    )
    persistence = None
    try:
        await asyncio.wait_for(entered_request.wait(), timeout=1)
        persistence = asyncio.create_task(capture_persistence())
        await asyncio.sleep(0)
        assert not persistence_entered.is_set()

        release_request.set()
        with pytest.raises(SnapshotTransactionError) as exc_info:
            await asyncio.wait_for(enqueue, timeout=1)
        assert exc_info.value.code == "SAFE_POINT_BUSY"
        await asyncio.wait_for(persistence_entered.wait(), timeout=1)
        await persistence
        forbidden_wait.assert_not_awaited()
        assert safe_point.is_owned_by(session_id, foreign_owner)
    finally:
        release_request.set()
        if not enqueue.done():
            enqueue.cancel()
        if persistence is not None and not persistence.done():
            persistence.cancel()
        await asyncio.gather(
            enqueue,
            *(tuple([persistence]) if persistence is not None else ()),
            return_exceptions=True,
        )
        await safe_point.release(session_id, foreign_owner)
