import asyncio
import threading
import traceback
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
    request_import_generation_cancellation,
    read_lifecycle_record,
    register_new_generation,
    tombstone_generation,
)
from backend.state import (
    _run_tasks,
    chat_sandboxes,
    chat_tasks,
    detailed_report_tasks,
    prepared_exports,
    report_workflows,
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


class _DeletionTraceRecorder:
    """Record the deletion transaction through its historical collaborators."""

    def __init__(self):
        self.trace = []
        self._drained = False
        self._revalidated = False

    def install(self, monkeypatch):
        original_get_lock = session_deletion._get_cleanup_lock
        original_lease = session_deletion._try_acquire_cleanup_lease
        original_publish = session_deletion._tombstone_detach_and_publish
        original_drain = session_deletion._drain_tombstone_producers
        original_read = session_deletion.read_lifecycle_record
        original_side_cleanup = session_deletion._cleanup_tombstone_side_surfaces
        original_delete = session_deletion._delete_tombstoned_files
        original_release_slot = session_deletion.release_session_lock_slot
        original_complete = session_deletion.complete_tombstone_cleanup
        original_release_handles = session_deletion._release_cleanup_handles

        recorder = self

        class _TracedCleanupLock:
            def __init__(self, lock):
                self._lock = lock

            async def __aenter__(self):
                recorder.trace.append("acquire-session-cleanup-lock")
                await self._lock.acquire()

            async def __aexit__(self, *_exc):
                self._lock.release()

        def traced_get_lock(session_id):
            return _TracedCleanupLock(original_get_lock(session_id))

        async def traced_lease(*args, **kwargs):
            self.trace.append("acquire-process-and-file-lease")
            return await original_lease(*args, **kwargs)

        async def traced_publish(*args, **kwargs):
            self.trace.append("publish-tombstone-and-detach")
            return await original_publish(*args, **kwargs)

        async def traced_drain(*args, **kwargs):
            self.trace.append("drain-producers")
            result = await original_drain(*args, **kwargs)
            self._drained = result == 0
            return result

        def traced_read(*args, **kwargs):
            if self._drained and not self._revalidated:
                self.trace.append("revalidate-tombstone")
                self._revalidated = True
            return original_read(*args, **kwargs)

        async def traced_side_cleanup(*args, **kwargs):
            self.trace.append("cleanup-side-surfaces")
            return await original_side_cleanup(*args, **kwargs)

        async def traced_delete(*args, **kwargs):
            self.trace.append("delete-generation-files")
            return await original_delete(*args, **kwargs)

        def traced_release_slot(*args, **kwargs):
            self.trace.append("release-session-lock-slot")
            return original_release_slot(*args, **kwargs)

        def traced_complete(*args, **kwargs):
            self.trace.append("publish-cleanup-complete")
            return original_complete(*args, **kwargs)

        def traced_release_handles(*args, **kwargs):
            self.trace.append("release-cleanup-handles-and-lease")
            return original_release_handles(*args, **kwargs)

        for name, replacement in (
            ("_get_cleanup_lock", traced_get_lock),
            ("_try_acquire_cleanup_lease", traced_lease),
            ("_tombstone_detach_and_publish", traced_publish),
            ("_drain_tombstone_producers", traced_drain),
            ("read_lifecycle_record", traced_read),
            ("_cleanup_tombstone_side_surfaces", traced_side_cleanup),
            ("_delete_tombstoned_files", traced_delete),
            ("release_session_lock_slot", traced_release_slot),
            ("complete_tombstone_cleanup", traced_complete),
            ("_release_cleanup_handles", traced_release_handles),
        ):
            monkeypatch.setattr(session_deletion, name, replacement)


@pytest.mark.asyncio
async def test_successful_deletion_preserves_transaction_phase_order(
    monkeypatch, tmp_path,
):
    session_id = "delete-phase-order"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    recorder = _DeletionTraceRecorder()
    recorder.install(monkeypatch)

    result = await session_deletion.delete_session_generation(
        session_id, state, "owner",
    )

    assert result["pending_producers"] == 0
    assert not session_dir.exists()
    assert recorder.trace == [
        "acquire-session-cleanup-lock",
        "acquire-process-and-file-lease",
        "publish-tombstone-and-detach",
        "drain-producers",
        "revalidate-tombstone",
        "cleanup-side-surfaces",
        "delete-generation-files",
        "release-session-lock-slot",
        "publish-cleanup-complete",
        "release-cleanup-handles-and-lease",
    ]


@pytest.mark.asyncio
async def test_delete_drains_export_builder_and_removes_integrity_root(
    monkeypatch, tmp_path,
):
    from backend.services.export_lifecycle import (
        abandon_export_record,
        bind_export_path,
        reserve_export_record,
    )

    session_id = "delete-export-builder"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    monkeypatch.setattr(
        session_deletion, "_PRODUCER_STOP_TIMEOUT_SECONDS", 1.0,
    )
    integrity_root = tmp_path / "integrity" / session_id
    integrity_root.mkdir(parents=True)
    (integrity_root / "signature").write_text("old", encoding="utf-8")
    archive = tmp_path / "prepared_export_delete.g1.tar.gz"
    started = asyncio.Event()

    async def export_builder():
        info = await reserve_export_record(
            "delete-export-token",
            session_id=session_id,
            user_id="owner",
            state=state,
            kind="prepared",
        )
        archive.write_bytes(b"archive")
        info["path"] = str(archive)
        info["path_owned"] = True
        await bind_export_path(info, archive)
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await abandon_export_record(info)
            raise

    producer = asyncio.create_task(export_builder())
    await asyncio.wait_for(started.wait(), timeout=1)

    result = await session_deletion.delete_session_generation(
        session_id, state, "owner",
    )

    assert result["pending_producers"] == 0
    assert producer.cancelled()
    assert not archive.exists()
    assert not session_dir.exists()
    assert not integrity_root.exists()
    assert "delete-export-token" not in prepared_exports
    assert read_lifecycle_record(session_id)["cleanup_complete"] is True


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
async def test_delete_removes_only_exact_legacy_state_files(
    monkeypatch, tmp_path,
):
    session_id = "abc"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    state_dir = tmp_path / "states"
    state_dir.mkdir(parents=True)
    owned = [
        state_dir / f"{session_id}.ipynb",
        state_dir / f"{session_id}_state.json",
        state_dir / f"{session_id}_session.pkl",
        state_dir / f"{session_id}_bioinfo_checkpoint.json",
        state_dir / f"{session_id}_report_checkpoint.json",
    ]
    near_prefix = [
        state_dir / "abcd_session.pkl",
        state_dir / "abc-extra_state.json",
        state_dir / "abc_extra_state.json",
    ]
    for path in owned:
        path.write_bytes(b"owned")
    for path in near_prefix:
        path.write_bytes(b"other-session")

    result = await session_deletion.delete_session_generation(
        session_id, state, "owner",
    )

    assert result["deleted_state_files"] == len(owned)
    assert not session_dir.exists()
    assert all(not path.exists() for path in owned)
    assert all(path.read_bytes() == b"other-session" for path in near_prefix)


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
async def test_timeout_keeps_tombstoned_tree_for_safe_retry(monkeypatch, tmp_path):
    session_id = "delete-stubborn-producer"
    session_root = tmp_path / "sessions"
    session_dir = session_root / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "live.txt").write_text("producer-owned", encoding="utf-8")

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
    monkeypatch.setattr(session_deletion, "get_session_dir", lambda _sid: session_dir)
    monkeypatch.setattr(
        session_deletion, "_PRODUCER_STOP_TIMEOUT_SECONDS", 0.01,
    )
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
    # Creation precedes the session tree in production. The test created its
    # marker first for readability, so temporarily move it aside while the
    # collision guard registers the new generation.
    marker = session_dir / "live.txt"
    marker_bytes = marker.read_bytes()
    marker.unlink()
    session_dir.rmdir()
    register_new_generation(state)
    session_dir.mkdir(parents=True)
    marker.write_bytes(marker_bytes)
    sessions[session_id] = state

    async def stubborn_producer():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Ignore the first cancellation long enough to exceed the deletion
            # timeout, then stop normally so a later cleanup retry can finish.
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            await asyncio.sleep(0.05)

    producer = asyncio.create_task(stubborn_producer())
    _run_tasks[session_id] = producer
    recorder = _DeletionTraceRecorder()
    recorder.install(monkeypatch)

    try:
        deletion = asyncio.create_task(
            session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )
        )
        done, _ = await asyncio.wait({deletion}, timeout=1)
        if not done:
            stack = "".join(
                traceback.format_list(traceback.extract_stack(frame))
                for frame in deletion.get_stack()
            )
            deletion.cancel()
            pytest.fail(f"deletion did not reach its bounded timeout:\n{stack}")
        with pytest.raises(SessionProducerStopError, match="did not stop") as raised:
            await deletion

        record = read_lifecycle_record(session_id)
        assert isinstance(raised.value, SessionProducerStopError)
        assert recorder.trace == [
            "acquire-session-cleanup-lock",
            "acquire-process-and-file-lease",
            "publish-tombstone-and-detach",
            "drain-producers",
        ]
        assert record is not None and record["status"] == "deleted"
        assert record["cleanup_complete"] is False
        retained = session_deletion._pending_tombstone_cleanup[session_id]
        assert retained.lease is not None and not retained.lease.released
        assert marker.read_text(encoding="utf-8") == "producer-owned"
        assert session_id not in sessions
        assert session_id not in _run_tasks
    finally:
        await asyncio.wait_for(producer, timeout=1)
        stats = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner"
        )
        assert stats["pending_producers"] == 0
        assert not session_dir.exists()
        # Physical cleanup never removes or reactivates the stable 410 record.
        record = read_lifecycle_record(session_id)
        assert record is not None and record["status"] == "deleted"
        sessions.pop(session_id, None)
        _run_tasks.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)
        prepared_exports.clear()


@pytest.mark.asyncio
async def test_background_sweep_finishes_tombstone_left_by_prior_process(
    monkeypatch, tmp_path,
):
    session_id = "delete-restart-sweep"
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
    monkeypatch.setattr(session_deletion, "get_session_dir", lambda _sid: session_dir)

    state = AnalysisState(id=session_id, user_id="owner")
    register_new_generation(state)
    session_dir.mkdir(parents=True)
    (session_dir / "orphan.txt").write_text("stale", encoding="utf-8")
    session_lifecycle.tombstone_generation(
        session_id, "owner", state.lifecycle_generation
    )
    # Simulate restart: producer handles are gone, tombstone and disk remain.
    session_deletion._pending_tombstone_cleanup.clear()

    result = await session_deletion.sweep_tombstoned_generations()

    assert result == {"completed": 1, "pending": 0, "failed": 0}
    assert not session_dir.exists()
    assert read_lifecycle_record(session_id)["status"] == "deleted"


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
async def test_delayed_closer_is_retained_and_never_invoked_twice(
    monkeypatch, tmp_path,
):
    session_id = "delete-delayed-closer"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    release = asyncio.Event()

    class _Workflow:
        calls = 0

        async def close(self, *, save_runtime=True):
            self.calls += 1
            await release.wait()

    workflow = _Workflow()
    workflows[session_id] = workflow
    try:
        with pytest.raises(SessionProducerStopError):
            await session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )
        retained = session_deletion._pending_tombstone_cleanup[session_id]
        retained_task = retained.workflow_task
        assert retained_task is not None and not retained_task.done()
        assert workflow.calls == 1
        assert marker.exists()

        pending = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert pending["pending_producers"] == 1
        assert session_deletion._pending_tombstone_cleanup[
            session_id
        ].workflow_task is retained_task
        assert workflow.calls == 1
        assert marker.exists()

        release.set()
        await asyncio.wait_for(retained_task, timeout=1)
        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert workflow.calls == 1
        assert not session_dir.exists()
    finally:
        release.set()
        workflows.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


@pytest.mark.asyncio
async def test_workflow_false_result_is_retried_by_tombstone_sweeper(
    monkeypatch, tmp_path,
):
    session_id = "delete-workflow-false-retry"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )

    class _Workflow:
        calls = 0

        async def close(self, *, save_runtime=True):
            assert save_runtime is False
            self.calls += 1
            return self.calls > 1

    workflow = _Workflow()
    workflows[session_id] = workflow
    try:
        with pytest.raises(SessionProducerStopError):
            await session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )
        retained = session_deletion._pending_tombstone_cleanup[session_id]
        first_attempt = retained.workflow_task
        assert first_attempt is not None and first_attempt.done()
        assert workflow.calls == 1
        assert marker.exists()

        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert workflow.calls == 2
        assert retained.workflow_task is not first_attempt
        assert not session_dir.exists()
    finally:
        workflows.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


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
async def test_report_workflow_close_is_part_of_tombstone_shutdown_proof(
    monkeypatch, tmp_path,
):
    session_id = "delete-report-workflow-retry"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )

    class _ReportWorkflow:
        calls = 0

        async def close(self, *, save_runtime=True):
            assert save_runtime is False
            self.calls += 1
            return self.calls > 1

    report_workflow = _ReportWorkflow()
    report_workflows[session_id] = report_workflow
    try:
        with pytest.raises(SessionProducerStopError):
            await session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )
        handles = session_deletion._pending_tombstone_cleanup[session_id]
        assert handles.report_workflow is report_workflow
        assert handles.report_workflow_task is not None
        assert report_workflow.calls == 1
        assert marker.exists()

        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert report_workflow.calls == 2
        assert not session_dir.exists()
    finally:
        report_workflows.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


@pytest.mark.asyncio
async def test_failed_chat_closer_is_retried_before_delete(
    monkeypatch, tmp_path,
):
    session_id = "delete-failed-chat-closer"
    state, _session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )

    class _ChatSandbox:
        calls = 0

        async def stop(self, *, save=True):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("close failed")

    sandbox = _ChatSandbox()
    chat_sandboxes[session_id] = sandbox
    try:
        with pytest.raises(SessionProducerStopError):
            await session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )
        retained = session_deletion._pending_tombstone_cleanup[session_id].chat_task
        assert retained is not None and retained.done()
        assert sandbox.calls == 1
        assert marker.exists()

        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert session_id not in session_deletion._pending_tombstone_cleanup
        assert sandbox.calls == 2
        assert not marker.exists()
    finally:
        chat_sandboxes.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


@pytest.mark.asyncio
async def test_cancelled_delete_keeps_independent_closer_for_cleanup_retry(
    monkeypatch, tmp_path,
):
    session_id = "delete-caller-cancelled"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    class _Workflow:
        calls = 0

        async def close(self, *, save_runtime=True):
            self.calls += 1
            started.set()
            await release.wait()

    workflow = _Workflow()
    workflows[session_id] = workflow
    deletion = asyncio.create_task(
        session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        retained = session_deletion._pending_tombstone_cleanup[session_id]
        retained_task = retained.workflow_task
        deletion.cancel()
        with pytest.raises(asyncio.CancelledError):
            await deletion

        assert retained_task is not None and not retained_task.cancelled()
        assert workflow.calls == 1
        assert marker.exists()
        assert read_lifecycle_record(session_id)["status"] == "deleted"

        release.set()
        await asyncio.wait_for(retained_task, timeout=1)
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
async def test_live_chat_task_exclusively_owns_sandbox_stop(
    monkeypatch, tmp_path,
):
    """A cancelled chat producer and deletion must not both stop its sandbox."""
    session_id = "delete-chat-single-stop"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    started = asyncio.Event()

    class _ChatSandbox:
        calls = 0

        async def stop(self, *, save=True):
            self.calls += 1

    sandbox = _ChatSandbox()

    async def chat_producer():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await sandbox.stop()

    producer = asyncio.create_task(chat_producer())
    chat_tasks[session_id] = producer
    chat_sandboxes[session_id] = sandbox
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        result = await session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )

        assert result["deleted_sessions"] == 1
        assert producer.done()
        assert sandbox.calls == 1
        assert not session_dir.exists()
    finally:
        if not producer.done():
            producer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await producer
        chat_tasks.pop(session_id, None)
        chat_sandboxes.pop(session_id, None)
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
async def test_cancellation_swallowing_chat_task_is_reused_until_single_stop(
    monkeypatch, tmp_path,
):
    session_id = "delete-chat-retained-single-stop"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    started = asyncio.Event()
    release = asyncio.Event()
    cancellations = 0

    class _ChatSandbox:
        calls = 0

        async def stop(self, *, save=True):
            self.calls += 1

    sandbox = _ChatSandbox()

    async def chat_producer():
        nonlocal cancellations
        try:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellations += 1
                    task = asyncio.current_task()
                    if task is not None:
                        task.uncancel()
        finally:
            await sandbox.stop()

    producer = asyncio.create_task(chat_producer())
    chat_tasks[session_id] = producer
    chat_sandboxes[session_id] = sandbox
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(SessionProducerStopError):
            await session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )

        retained = session_deletion._pending_tombstone_cleanup[session_id]
        assert retained.tasks == [producer]
        assert retained.chat_task is None
        assert sandbox.calls == 0
        assert marker.exists()

        pending = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert pending["pending_producers"] == 1
        assert session_deletion._pending_tombstone_cleanup[
            session_id
        ].tasks == [producer]
        assert cancellations >= 2
        assert sandbox.calls == 0

        release.set()
        await asyncio.wait_for(producer, timeout=1)
        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert sandbox.calls == 1
        assert not session_dir.exists()
    finally:
        release.set()
        if not producer.done():
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        chat_tasks.pop(session_id, None)
        chat_sandboxes.pop(session_id, None)
        sessions.pop(session_id, None)
        session_deletion._pending_tombstone_cleanup.pop(session_id, None)


@pytest.mark.asyncio
async def test_side_cleanup_failure_keeps_tombstone_incomplete_and_retriable(
    monkeypatch, tmp_path,
):
    session_id = "delete-side-cleanup-retry"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    upload_cleanup = AsyncMock(side_effect=OSError("upload database busy"))
    monkeypatch.setattr(
        "backend.services.upload.cleanup_session_uploads", upload_cleanup,
    )
    try:
        with pytest.raises(OSError, match="upload database busy"):
            await session_deletion.delete_session_generation(
                session_id, state, "owner", force_stop=True,
            )

        record = read_lifecycle_record(session_id)
        assert record["cleanup_complete"] is False
        assert marker.exists()
        assert claim_import_generation(
            session_id, state.user_id, session_dir,
        ) is None
        handles = session_deletion._pending_tombstone_cleanup[session_id]
        assert handles.lease is not None and not handles.lease.released

        upload_cleanup.side_effect = None
        completed = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert completed["pending_producers"] == 0
        assert not session_dir.exists()
        assert read_lifecycle_record(session_id)["cleanup_complete"] is True
    finally:
        sessions.pop(session_id, None)
        handles = session_deletion._pending_tombstone_cleanup.pop(
            session_id, None,
        )
        if handles and handles.lease:
            handles.lease.release()


@pytest.mark.asyncio
async def test_cleanup_lease_blocks_a_second_process_until_owner_releases(
    monkeypatch, tmp_path,
):
    session_id = "delete-cross-process-lease"
    state, session_dir, marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    session_lifecycle.tombstone_generation(
        session_id, state.user_id, state.lifecycle_generation,
    )
    external_lease = await session_deletion._try_acquire_cleanup_lease(
        session_id
    )
    assert external_lease is not None
    try:
        pending = await session_deletion.cleanup_tombstoned_generation(
            session_id, "owner",
        )
        assert pending["pending_producers"] == 1
        assert marker.exists()
        assert read_lifecycle_record(session_id)["cleanup_complete"] is False
    finally:
        external_lease.release()

    completed = await session_deletion.cleanup_tombstoned_generation(
        session_id, "owner",
    )
    assert completed["pending_producers"] == 0
    assert not session_dir.exists()
    assert read_lifecycle_record(session_id)["cleanup_complete"] is True
    sessions.pop(session_id, None)


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
async def test_cancel_during_cleanup_completion_releases_exact_lease(
    monkeypatch, tmp_path,
):
    """Caller cancellation cannot split completion from lease release."""
    session_id = "delete-finalize-cancel"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    entered = threading.Event()
    release = threading.Event()
    original_complete = session_deletion.complete_tombstone_cleanup

    def delayed_complete(sid, owner_id, generation):
        entered.set()
        assert release.wait(timeout=2)
        return original_complete(sid, owner_id, generation)

    monkeypatch.setattr(
        session_deletion, "complete_tombstone_cleanup", delayed_complete,
    )
    deletion = asyncio.create_task(
        session_deletion.delete_session_generation(
            session_id, state, "owner", force_stop=True,
        )
    )
    try:
        assert await _wait_thread_event(entered)
        deletion.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(deletion, timeout=2)

        record = read_lifecycle_record(session_id)
        assert record is not None and record["cleanup_complete"] is True
        assert not session_dir.exists()
        assert session_id not in session_deletion._pending_tombstone_cleanup
        replacement_lease = await session_deletion._try_acquire_cleanup_lease(
            session_id,
        )
        assert replacement_lease is not None
        replacement_lease.release()
    finally:
        release.set()
        if not deletion.done():
            deletion.cancel()
        await asyncio.gather(deletion, return_exceptions=True)


def test_completed_tombstone_cannot_be_reclaimed_before_cleanup_lease_release(
    monkeypatch, tmp_path,
):
    """The durable completion flag is not sufficient while its lease is live."""
    session_id = "delete-complete-before-unlock"
    state, session_dir, _marker = _prepare_delete_case(
        monkeypatch, tmp_path, session_id,
    )
    lease = session_deletion._try_acquire_cleanup_lease_sync(session_id)
    assert lease is not None
    try:
        tombstone = tombstone_generation(
            session_id, "owner", state.lifecycle_generation,
        )
        session_deletion.shutil.rmtree(session_dir)
        assert complete_tombstone_cleanup(
            session_id, "owner", int(tombstone["generation"]),
        ) is True
        assert read_lifecycle_record(session_id)["cleanup_complete"] is True

        assert claim_import_generation(
            session_id, "owner", session_dir,
        ) is None
        assert not session_dir.exists()
    finally:
        lease.release()

    assert claim_import_generation(session_id, "owner", session_dir) == 2


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


def test_failed_import_producer_cleans_account_cancel_marker(monkeypatch, tmp_path):
    """Import cleanup owns a generation cancelled by account deletion."""
    session_id = "account-tombstoned-import"
    session_root = tmp_path / "sessions"
    session_dir = session_root / session_id
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_deletion, "SESSION_DIR", session_root)
    integrity_dir = tmp_path / "integrity" / session_id
    monkeypatch.setattr(session_deletion, "INTEGRITY_DIR", tmp_path / "integrity")
    monkeypatch.setattr(
        session_deletion, "get_session_dir", lambda _sid: session_dir,
    )
    session_lifecycle.reset_lifecycle_locks()
    assert claim_import_generation(session_id, "owner", session_dir) == 1
    (session_dir / "partial.txt").write_text("partial", encoding="utf-8")
    integrity_dir.mkdir(parents=True)
    (integrity_dir / "live.pkl.sig").write_text("partial", encoding="utf-8")

    assert request_import_generation_cancellation(session_id, "owner") == 1
    before = read_lifecycle_record(session_id)
    assert before is not None
    assert before["status"] == "importing"
    assert before["import_cancel_requested"] is True

    state = AnalysisState(id=session_id, user_id="owner")
    session_lifecycle.bind_import_generation(state)
    with pytest.raises(session_lifecycle.SessionGenerationError):
        session_lifecycle.commit_import_generation(state)

    assert session_deletion.cleanup_failed_import_generation(session_id) is True
    after = read_lifecycle_record(session_id)
    assert after is not None and after["cleanup_complete"] is True
    assert not session_dir.exists()
    assert not integrity_dir.exists()
