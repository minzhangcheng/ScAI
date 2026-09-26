import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState
from backend.services import session_deletion, session_lifecycle
from backend.services.session_deletion import SessionProducerStopError
from backend.tests.split_facade import setattr_facade
from backend.utils.asyncio_tasks import run_in_owned_thread
from backend.services.session_lifecycle import (
    abandon_import_generation,
    claim_import_generation,
    complete_tombstone_cleanup,
    read_lifecycle_record,
    register_new_generation,
)
from backend.state import (
    detailed_report_tasks,
    prepared_exports,
    session_locks,
    sessions,
    workflows,
)


async def _wait_thread_event(event: threading.Event, timeout: float = 2.0) -> bool:
    """Wait without installing a cancellable default-executor waiter."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _prepare_delete_case(monkeypatch, tmp_path, session_id):
    session_root = tmp_path / "sessions"
    session_dir = session_root / session_id
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_lifecycle, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(session_lifecycle, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(session_lifecycle, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(session_lifecycle, "STATE_DIR", tmp_path / "states")
    session_lifecycle.reset_lifecycle_locks()
    monkeypatch.setattr(session_deletion, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_deletion, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(session_deletion, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(session_deletion, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(session_deletion, "STATE_DIR", tmp_path / "states")
    monkeypatch.setattr(session_deletion, "INTEGRITY_DIR", tmp_path / "integrity")
    monkeypatch.setattr(session_deletion, "get_session_dir", lambda _sid: session_dir)
    monkeypatch.setattr(session_deletion, "_PRODUCER_STOP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        "backend.utils.websocket_delivery.close_websocket", AsyncMock(),
    )
    monkeypatch.setattr(
        "backend.routes.reports.clear_detailed_report_cache", lambda _sid: None,
    )
    setattr_facade(
        monkeypatch,
        "backend.routes.reports",
        "clear_generation_pending",
        lambda _sid, _token=None: None,
    )
    monkeypatch.setattr(
        "backend.services.upload.cleanup_session_uploads", AsyncMock(),
    )
    state = AnalysisState(id=session_id, user_id="owner")
    register_new_generation(state)
    session_dir.mkdir(parents=True)
    marker = session_dir / "live.txt"
    marker.write_text("producer-owned", encoding="utf-8")
    sessions[session_id] = state
    return state, session_dir, marker


@pytest.mark.asyncio
async def test_export_unlink_failure_keeps_tombstone_incomplete_until_retry(
    monkeypatch, tmp_path,
):
    from backend.services.export_lifecycle import (
        bind_export_path,
        publish_export_ready,
        reserve_export_record,
    )

    session_id = "delete-export-unlink-failure"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    integrity_root = tmp_path / "integrity" / session_id
    integrity_root.mkdir(parents=True)
    archive = tmp_path / "prepared_export_delete-fail.g1.tar.gz"
    archive.write_bytes(b"archive")
    info = await reserve_export_record(
        "delete-export-fail-token",
        session_id=session_id,
        user_id="owner",
        state=state,
        kind="prepared",
    )
    info["path"] = str(archive)
    info["path_owned"] = True
    await bind_export_path(info, archive)
    await publish_export_ready(info, filename="x.tar.gz", size=7)
    original_unlink = type(archive).unlink
    fail_archive_unlink = True

    def controlled_unlink(path, *args, **kwargs):
        if fail_archive_unlink and path == archive:
            raise OSError("archive busy")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(archive), "unlink", controlled_unlink)

    with pytest.raises(OSError, match="export archive"):
        await session_deletion.delete_session_generation(
            session_id, state, "owner",
        )

    record = read_lifecycle_record(session_id)
    assert record["status"] == "deleted"
    assert record["cleanup_complete"] is False
    assert archive.exists()
    assert session_dir.exists()
    assert integrity_root.exists()
    assert prepared_exports["delete-export-fail-token"]["phase"] == "cleanup_failed"

    fail_archive_unlink = False
    result = await session_deletion.cleanup_tombstoned_generation(
        session_id, "owner",
    )
    assert result["pending_producers"] == 0
    assert not archive.exists()
    assert not session_dir.exists()
    assert not integrity_root.exists()
    assert "delete-export-fail-token" not in prepared_exports
    assert read_lifecycle_record(session_id)["cleanup_complete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_id", "ambiguous_name"),
    [
        ("abc_state", "abc_state.json"),
        ("abc_session", "abc_session.pkl"),
    ],
)
async def test_delete_preserves_ambiguous_legacy_file_owned_by_shorter_id(
    monkeypatch, tmp_path, session_id, ambiguous_name,
):
    state, _session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    ambiguous = tmp_path / "states" / ambiguous_name
    ambiguous.parent.mkdir(parents=True)
    ambiguous.write_bytes(b"shorter-session")

    await session_deletion.delete_session_generation(
        session_id, state, "owner",
    )

    assert ambiguous.read_bytes() == b"shorter-session"


@pytest.mark.asyncio
async def test_stale_tombstone_cleanup_cannot_delete_reclaimed_import_generation(
    monkeypatch, tmp_path,
):
    session_id = "delete-then-import-same-id"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    # Model a completed physical deletion followed by a same-owner import
    # claiming generation 2 while a stale cleanup invocation is still queued.
    session_lifecycle.tombstone_generation(
        session_id, state.user_id, state.lifecycle_generation,
    )
    import shutil

    shutil.rmtree(session_dir)
    assert complete_tombstone_cleanup(
        session_id, state.user_id, state.lifecycle_generation,
    ) is True
    assert claim_import_generation(session_id, state.user_id, session_dir) == 2
    imported_marker = session_dir / "imported.txt"
    imported_marker.write_text("generation-2", encoding="utf-8")
    replacement_lock = asyncio.Lock()
    session_locks[session_id] = replacement_lock

    stats = await session_deletion._delete_tombstoned_files(session_id)

    assert stats == {"deleted_sessions": 0, "deleted_state_files": 0}
    assert imported_marker.read_text(encoding="utf-8") == "generation-2"
    assert read_lifecycle_record(session_id)["status"] == "importing"
    assert session_locks.get(session_id) is replacement_lock
    abandon_import_generation(session_id)
    shutil.rmtree(session_dir)
    sessions.pop(session_id, None)
    session_locks.pop(session_id, None)


@pytest.mark.asyncio
async def test_concurrent_delete_reuses_published_producer_handles(
    monkeypatch, tmp_path,
):
    session_id = "delete-concurrent-handle-reuse"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    class _Workflow:
        calls = 0

        async def close(self, *, save_runtime=True):
            assert save_runtime is False
            self.calls += 1
            started.set()
            await release.wait()

    workflow = _Workflow()
    workflows[session_id] = workflow
    first = asyncio.create_task(
        session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        published = session_deletion._pending_tombstone_cleanup[session_id]
        second = asyncio.create_task(
            session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )
        )
        with pytest.raises(SessionProducerStopError):
            await first
        second_result = await second

        assert second_result["pending_producers"] == 1
        assert session_deletion._pending_tombstone_cleanup[session_id] is published
        assert published.workflow is workflow
        assert workflow.calls == 1
        assert marker.exists()

        release.set()
        await asyncio.wait_for(published.workflow_task, timeout=1)
        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert not session_dir.exists()
    finally:
        release.set()
        if not first.done():
            first.cancel()
        workflows.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


@pytest.mark.asyncio
async def test_closer_that_swallows_cancellation_cannot_hang_delete(
    monkeypatch, tmp_path,
):
    """Deletion observes an independent closer instead of wait_for-cancelling it."""
    session_id = "delete-cancellation-swallowing-closer"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    release = asyncio.Event()

    class _Workflow:
        calls = 0
        cancellations = 0

        async def close(self, *, save_runtime=True):
            self.calls += 1
            try:
                await release.wait()
            except asyncio.CancelledError:
                self.cancellations += 1
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
                await release.wait()

    workflow = _Workflow()
    workflows[session_id] = workflow
    deletion = asyncio.create_task(
        session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )
    )
    try:
        done, _pending = await asyncio.wait({deletion}, timeout=0.5)
        assert deletion in done, "bounded deletion was held by the closer"
        with pytest.raises(SessionProducerStopError):
            await deletion

        retained = session_deletion._pending_tombstone_cleanup[session_id]
        assert retained.workflow_task is not None
        assert not retained.workflow_task.done()
        assert workflow.calls == 1
        # The timeout is observational; it never injects cancellation into a
        # producer whose shutdown proof must remain reusable by the retry.
        assert workflow.cancellations == 0
        assert marker.exists()

        release.set()
        await asyncio.wait_for(retained.workflow_task, timeout=1)
        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert workflow.calls == 1
        assert not session_dir.exists()
    finally:
        release.set()
        if not deletion.done():
            deletion.cancel()
        workflows.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


@pytest.mark.asyncio
async def test_force_delete_cancels_and_awaits_detailed_report_task(
    monkeypatch, tmp_path,
):
    session_id = "delete-detailed-report-task"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def detailed_report_producer():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()

    producer = asyncio.create_task(detailed_report_producer())
    detailed_report_tasks[session_id] = producer
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        result = await session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )

        assert result["deleted_sessions"] == 1
        assert stopped.is_set()
        assert producer.done()
        assert session_id not in detailed_report_tasks
        assert not session_dir.exists()
    finally:
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        detailed_report_tasks.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


@pytest.mark.asyncio
async def test_concurrent_cleanup_waiters_run_side_cleanup_once(
    monkeypatch, tmp_path,
):
    session_id = "delete-cleanup-singleflight"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    session_lifecycle.tombstone_generation(
        session_id, state.user_id, state.lifecycle_generation,
    )
    original = session_deletion._cleanup_tombstone_side_surfaces
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def _blocked_side_cleanup(target_session_id, target_generation):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        await original(target_session_id, target_generation)

    monkeypatch.setattr(
        session_deletion,
        "_cleanup_tombstone_side_surfaces",
        _blocked_side_cleanup,
    )
    first = asyncio.create_task(
        session_deletion.cleanup_tombstoned_generation(session_id, "owner")
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        second = asyncio.create_task(
            session_deletion.cleanup_tombstoned_generation(session_id, "owner")
        )
        await asyncio.sleep(0)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result["pending_producers"] == 0
        assert second_result["pending_producers"] == 0
        assert calls == 1
        assert not session_dir.exists()
        assert read_lifecycle_record(session_id)["cleanup_complete"] is True
    finally:
        release.set()
        if not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_cancel_after_tombstone_defers_until_handles_are_published(
    monkeypatch, tmp_path,
):
    session_id = "delete-publication-cancel"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    publication_gap = asyncio.Event()
    release_publication = asyncio.Event()

    class _TwoStageStateLock:
        enters = 0

        async def __aenter__(self):
            self.enters += 1
            if self.enters == 2:
                publication_gap.set()
                await release_publication.wait()
            return self

        async def __aexit__(self, *_args):
            return False

    class _Workflow:
        calls = 0

        async def close(self, *, save_runtime=True):
            assert save_runtime is False
            self.calls += 1
            return True

    workflow = _Workflow()
    workflows[session_id] = workflow
    monkeypatch.setattr(session_deletion, "state_lock", _TwoStageStateLock())
    deletion = asyncio.create_task(
        session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )
    )
    try:
        await asyncio.wait_for(publication_gap.wait(), timeout=1)
        record = read_lifecycle_record(session_id)
        assert record["status"] == "deleted"
        assert record["cleanup_complete"] is False
        assert session_id not in session_deletion._pending_tombstone_cleanup

        deletion.cancel()
        await asyncio.sleep(0)
        assert not deletion.done()
        assert marker.exists()

        release_publication.set()
        with pytest.raises(asyncio.CancelledError):
            await deletion

        handles = session_deletion._pending_tombstone_cleanup[session_id]
        assert handles.workflow is workflow
        assert handles.lease is not None and not handles.lease.released
        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert workflow.calls == 1
        assert not session_dir.exists()
    finally:
        release_publication.set()
        workflows.pop(session_id, None)
        sessions.pop(session_id, None)
        handles = session_deletion._pending_tombstone_cleanup.pop(
            session_id, None,
        )
        if handles and handles.lease:
            handles.lease.release()


@pytest.mark.asyncio
async def test_stale_chunk_upload_cannot_write_after_delete_tombstone(
    monkeypatch, tmp_path,
):
    from backend.services.upload import upload_chunk
    from backend.services.session_lifecycle import (
        SessionGoneError,
        get_lifecycle_lock,
    )

    session_id = "delete-races-chunk-upload"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    upload_temp = tmp_path / "upload-temp"
    upload_temp.mkdir()
    monkeypatch.setattr("backend.services.upload.UPLOAD_TEMP_DIR", upload_temp)
    db = AsyncMock()
    db.get_upload_session.return_value = {
        "id": "upload-1",
        "session_id": session_id,
        "completed": 0,
        "chunk_size": 1024,
        "total_chunks": 1,
    }
    db.mark_chunk_received.return_value = {
        "received_chunks": 1,
        "total_chunks": 1,
    }
    monkeypatch.setattr("backend.services.upload.get_db", AsyncMock(return_value=db))

    lifecycle_lock = get_lifecycle_lock(session_id)
    await lifecycle_lock.acquire()
    deletion = asyncio.create_task(
        session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )
    )
    for _ in range(100):
        waiters = getattr(lifecycle_lock, "_waiters", None)
        if waiters and len(waiters) >= 1:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("deletion did not queue on the lifecycle barrier")
    upload = asyncio.create_task(
        upload_chunk(session_id, "upload-1", 0, b"old-generation")
    )
    await asyncio.sleep(0)
    lifecycle_lock.release()
    try:
        deleted = await asyncio.wait_for(deletion, timeout=2)
        assert deleted["pending_producers"] == 0
        with pytest.raises(SessionGoneError):
            await upload
        assert not (upload_temp / "upload-1.chunk.0").exists()
        assert not session_dir.exists()
    finally:
        if not deletion.done():
            deletion.cancel()
        if not upload.done():
            upload.cancel()
        await asyncio.gather(deletion, upload, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_import_cleanup_holds_shared_lease_until_completion(
    monkeypatch, tmp_path,
):
    """A sweeper cannot overlap partial cleanup or reach a reused ID."""
    session_id = "failed-import-shared-cleanup-lease"
    session_root = tmp_path / "sessions"
    session_dir = session_root / session_id
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_deletion, "SESSION_DIR", session_root)
    monkeypatch.setattr(
        session_deletion, "get_session_dir", lambda _sid: session_dir,
    )
    session_lifecycle.reset_lifecycle_locks()
    assert claim_import_generation(session_id, "owner", session_dir) == 1
    (session_dir / "partial.txt").write_text("partial", encoding="utf-8")

    entered = threading.Event()
    release = threading.Event()
    original_rmtree = session_deletion.shutil.rmtree

    def delayed_rmtree(path):
        entered.set()
        assert release.wait(timeout=2)
        return original_rmtree(path)

    monkeypatch.setattr(session_deletion.shutil, "rmtree", delayed_rmtree)
    cleanup = asyncio.create_task(
        run_in_owned_thread(
            session_deletion.cleanup_failed_import_generation, session_id,
        )
    )
    try:
        assert await _wait_thread_event(entered)
        record = read_lifecycle_record(session_id)
        assert record is not None
        assert record["status"] == "deleted"
        assert record["cleanup_complete"] is False

        pending = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert pending["pending_producers"] == 1

        release.set()
        assert await asyncio.wait_for(cleanup, timeout=2) is True
        assert read_lifecycle_record(session_id)["cleanup_complete"] is True
        assert claim_import_generation(session_id, "owner", session_dir) == 2
        (session_dir / "generation-2.txt").write_text("current", encoding="utf-8")
        assert (session_dir / "generation-2.txt").read_text() == "current"
    finally:
        release.set()
        await asyncio.gather(cleanup, return_exceptions=True)
        sessions.pop(session_id, None)


def test_completed_tombstone_cannot_be_reclaimed_before_cleanup_lease_release(
    monkeypatch, tmp_path,
):
    session_id = "delete-complete-before-unlock"
    state, session_dir, _marker = _prepare_delete_case(monkeypatch, tmp_path, session_id)
    lease = session_deletion._try_acquire_cleanup_lease_sync(session_id)
    assert lease is not None
    try:
        tombstone = session_lifecycle.tombstone_generation(
            session_id, state.user_id, state.lifecycle_generation,
        )
        session_deletion.shutil.rmtree(session_dir)
        assert complete_tombstone_cleanup(
            session_id, state.user_id, int(tombstone["generation"]),
        ) is True
        assert read_lifecycle_record(session_id)["cleanup_complete"] is True
        assert claim_import_generation(session_id, state.user_id, session_dir) is None
        assert not session_dir.exists()
    finally:
        lease.release()
        sessions.pop(session_id, None)
    assert claim_import_generation(session_id, state.user_id, session_dir) == 2
