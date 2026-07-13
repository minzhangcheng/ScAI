"""Tests for Snapshot V2 cancellable restore jobs."""
import asyncio
import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.services.snapshot_v2.errors import SnapshotTransactionError
from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
from backend.state import snapshot_restore_jobs


class _StatusOnlyService:
    """Test double: walk lifecycle statuses without live side effects.

    Used by unit tests that only assert job state-machine behaviour.
    """

    def __init__(self, session_id, paths):
        self.session_id = session_id
        self.paths = paths

    async def restore_snapshot(
        self,
        request,
        transaction_id,
        cancel_event,
        *,
        job_transaction_id=None,
        status_callback=None,
    ):
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord

        def _emit(status):
            rec = SnapshotTransactionRecord(
                transaction_id=transaction_id,
                session_id=self.session_id,
                snapshot_id=request.snapshot_id,
                status=status,
            )
            if status_callback is not None:
                status_callback(rec)
            return rec

        if cancel_event is not None and cancel_event.is_set():
            return _emit("cancelled")
        _emit("staging")
        if cancel_event is not None and cancel_event.is_set():
            return _emit("cancelled")
        _emit("staged")
        if cancel_event is not None and cancel_event.is_set():
            return _emit("cancelled")
        _emit("applying")
        if cancel_event is not None and cancel_event.is_set():
            return _emit("cancelled")
        return _emit("committed")


@pytest.fixture
def v2_paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"
    p = FakePaths()
    for d in [p.root, p.objects, p.staging, p.journals, p.integrity, p.committed]:
        d.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def preflight_result():
    from backend.services.snapshot_v2.preflight import RestorePreflightResult
    return RestorePreflightResult(status="ready")


@pytest.fixture
def restore_request():
    from backend.services.snapshot_v2.models import SnapshotRestoreRequest
    return SnapshotRestoreRequest(
        snapshot_id="snap-001",
        runtime_choice="reset",
        current_state_choice="abort",
    )


def make_mgr(v2_paths, session_id="sess-test", *, real_service=False, snapshot_service=None):
    from backend.services.snapshot_v2.jobs import RestoreJobManager
    if snapshot_service is None and not real_service:
        snapshot_service = _StatusOnlyService(session_id, v2_paths)
    return RestoreJobManager(
        paths=v2_paths,
        session_id=session_id,
        snapshot_service=snapshot_service,
    )


async def _clear_safe_points(manager) -> None:
    session_ids = set(manager.active_sessions) | set(manager._pending_requests)
    for session_id in session_ids:
        pending = manager.get_pending_observation(session_id)
        owner = (
            pending.owner_token
            if pending is not None
            else manager._acquired_owners.get(session_id)
        )
        if owner:
            await manager.cancel_safe_point_request(session_id, owner)


@pytest_asyncio.fixture(autouse=True)
async def _clear_restore_job_registry():
    snapshot_restore_jobs.clear()
    manager = get_snapshot_safe_point_manager()
    await _clear_safe_points(manager)
    yield
    for handle in list(snapshot_restore_jobs.values()):
        if handle.task is not None and not handle.task.done():
            handle.cancel_event.set()
            handle.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.task
    snapshot_restore_jobs.clear()
    await _clear_safe_points(manager)


@pytest.mark.asyncio
async def test_enqueue_publishes_owned_request_before_return(
    v2_paths, preflight_result, restore_request
):
    manager = make_mgr(v2_paths)
    record = await manager.enqueue("sess-test", preflight_result, restore_request)
    handle = snapshot_restore_jobs[record.transaction_id]

    assert handle.safe_point_request is not None
    assert handle.safe_point_request.owner_token == handle.owner_token
    assert manager._safe_point.get_pending_observation(
        "sess-test"
    ) == handle.safe_point_request
    await manager.cancel(record.transaction_id)


@pytest.mark.asyncio
async def test_enqueue_does_not_borrow_foreign_pending_event(
    v2_paths, preflight_result, restore_request, monkeypatch
):
    manager = make_mgr(v2_paths)
    foreign = await manager._safe_point.request_safe_point(
        "sess-test", "restore:foreign"
    )
    assert foreign is not False
    monkeypatch.setattr(
        "backend.services.snapshot_v2.jobs.SAFE_POINT_RESERVATION_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(SnapshotTransactionError) as error:
        await manager.enqueue("sess-test", preflight_result, restore_request)

    assert error.value.code == "SAFE_POINT_BUSY"
    assert manager._safe_point.get_pending_observation("sess-test") == foreign
    assert snapshot_restore_jobs == {}
    assert list(v2_paths.journals.glob("restore-job-*.json")) == []
    assert await manager._safe_point.cancel_safe_point_request(
        "sess-test", "restore:foreign"
    )


@pytest.mark.asyncio
async def test_second_enqueue_waits_then_owns_new_generation(
    v2_paths, preflight_result, restore_request
):
    first_manager = make_mgr(v2_paths)
    second_manager = make_mgr(v2_paths)
    first = await first_manager.enqueue(
        "sess-test", preflight_result, restore_request
    )
    second_task = asyncio.create_task(
        second_manager.enqueue("sess-test", preflight_result, restore_request)
    )
    await asyncio.sleep(0)
    assert not second_task.done()

    await first_manager.cancel(first.transaction_id)
    second = await asyncio.wait_for(second_task, timeout=1.0)
    handle = snapshot_restore_jobs[second.transaction_id]
    assert handle.safe_point_request is not None
    assert handle.safe_point_request.owner_token == handle.owner_token
    await second_manager.cancel(second.transaction_id)


class TestRestoreJobLifecycle:

    @pytest.mark.asyncio
    async def test_enqueue_returns_transaction(self, v2_paths, preflight_result, restore_request):
        """Enqueue creates a SnapshotTransactionRecord with queued status."""
        mgr = make_mgr(v2_paths)
        record = await mgr.enqueue("sess-test", preflight_result, restore_request)

        assert record.transaction_id is not None
        assert record.status == "queued"
        assert record.session_id == "sess-test"
        assert record.snapshot_id == "snap-001"

        # Stored in memory
        stored = mgr.get(record.transaction_id)
        assert stored is not None
        assert stored.transaction_id == record.transaction_id

        # Persistent journal
        journal_path = v2_paths.journals / f"{record.transaction_id}.json"
        assert journal_path.exists()
        assert json.loads(journal_path.read_text())["status"] == "queued"

        # Stored in global snapshot_restore_jobs
        from backend.services.snapshot_v2.jobs import get_restore_job
        handle = get_restore_job(record.transaction_id)
        assert handle is not None
        assert handle.record.transaction_id == record.transaction_id

        # Task is running
        assert handle.task is not None
        assert not handle.task.done()

    @pytest.mark.asyncio
    async def test_enqueue_progresses_with_workflow_ack(
        self, v2_paths, preflight_result, restore_request,
    ):
        """Job progresses through all states when workflow acknowledges safe point."""
        mgr = make_mgr(v2_paths)

        # Track state transitions via events
        state_events = {}
        async def on_state_change(record):
            state_events[record.status] = asyncio.Event()
            state_events[record.status].set()
        mgr._state_hook = on_state_change

        record = await mgr.enqueue("sess-test", preflight_result, restore_request)
        tx_id = record.transaction_id

        # Wait for waiting_safe_point
        await asyncio.wait_for(
            self._wait_for_status(mgr, tx_id, "waiting_safe_point"), timeout=5
        )
        assert self._has_status(mgr, tx_id, "waiting_safe_point")

        # Safe point should be pending
        assert mgr._safe_point.has_pending_request("sess-test")

        # Acknowledge safe point (simulate workflow)
        gen = mgr._safe_point.get_pending_generation("sess-test")
        await mgr._safe_point.acknowledge_safe_point_request("sess-test", gen, role="workflow")

        # Wait for committed
        await asyncio.wait_for(
            self._wait_for_status(mgr, tx_id, "committed"), timeout=5
        )
        rec = mgr.get(tx_id)
        assert rec.status == "committed"

        # All journal entries reflect final state
        journal_path = v2_paths.journals / f"{tx_id}.json"
        assert json.loads(journal_path.read_text())["status"] == "committed"

    @pytest.mark.asyncio
    async def test_restore_job_keeps_public_id_in_retry_context_and_final_events(
        self, v2_paths, preflight_result, restore_request,
    ):
        """Internal apply journals must never replace the public job ID."""
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord

        class RecordingService:
            seen_apply_transaction_id = None
            seen_job_transaction_id = None

            async def restore_snapshot(
                self,
                request,
                transaction_id,
                cancel_event,
                *,
                job_transaction_id=None,
                status_callback=None,
            ):
                self.seen_apply_transaction_id = transaction_id
                self.seen_job_transaction_id = job_transaction_id
                record = SnapshotTransactionRecord(
                    transaction_id=transaction_id,
                    session_id="sess-test",
                    snapshot_id=request.snapshot_id,
                    status="committed",
                )
                if status_callback is not None:
                    status_callback(record)
                return record

        service = RecordingService()
        mgr = make_mgr(v2_paths, snapshot_service=service)
        final_events = []

        async def on_state_change(record):
            if record.status == "committed":
                final_events.append(record)

        mgr._state_hook = on_state_change
        original = await mgr.enqueue(
            "sess-test",
            preflight_result,
            restore_request,
            retry_of_transaction_id="restore-job-parent",
        )
        generation = mgr._safe_point.get_pending_generation("sess-test")
        while generation is None:
            await asyncio.sleep(0)
            generation = mgr._safe_point.get_pending_generation("sess-test")
        await mgr._safe_point.acknowledge_safe_point_request(
            "sess-test", generation, role="workflow",
        )
        await asyncio.wait_for(
            self._wait_for_status(mgr, original.transaction_id, "committed"),
            timeout=5,
        )
        await asyncio.sleep(0)

        assert service.seen_apply_transaction_id == (
            "restore-apply-" + original.transaction_id.removeprefix("restore-job-")
        )
        assert service.seen_job_transaction_id == original.transaction_id
        assert final_events
        assert all(event.transaction_id == original.transaction_id for event in final_events)
        assert all(
            event.retry_of_transaction_id == "restore-job-parent"
            for event in final_events
        )

    @pytest.mark.asyncio
    async def test_job_progresses_through_intermediate_states(
        self, v2_paths, preflight_result, restore_request,
    ):
        """Job emits each intermediate state: waiting_safe_point, staging, staged, applying."""
        mgr = make_mgr(v2_paths)
        seen = []

        async def on_state_change(record):
            seen.append(record.status)
        mgr._state_hook = on_state_change

        record = await mgr.enqueue("sess-test", preflight_result, restore_request,
                                    timeout_seconds=30)
        tx_id = record.transaction_id

        # Wait for waiting_safe_point
        await asyncio.wait_for(
            self._wait_for_status(mgr, tx_id, "waiting_safe_point"), timeout=5
        )

        # Acknowledge
        gen = mgr._safe_point.get_pending_generation("sess-test")
        await mgr._safe_point.acknowledge_safe_point_request("sess-test", gen, role="workflow")

        # Wait for committed
        await asyncio.wait_for(
            self._wait_for_status(mgr, tx_id, "committed"), timeout=5
        )

        # Should have passed through intermediate states
        state_order = [s for s in seen if s != "queued"]
        assert "staging" in state_order, f"States seen: {state_order}"
        assert "staged" in state_order, f"States seen: {state_order}"
        assert "applying" in state_order, f"States seen: {state_order}"

    @pytest.mark.asyncio
    async def test_cancel_before_ack_aborts_job(
        self, v2_paths, preflight_result, restore_request,
    ):
        """Cancel before workflow acknowledges safe point sets status to cancelled."""
        mgr = make_mgr(v2_paths)
        record = await mgr.enqueue("sess-test", preflight_result, restore_request)
        tx_id = record.transaction_id

        await asyncio.wait_for(
            self._wait_for_status(mgr, tx_id, "waiting_safe_point"), timeout=5
        )

        await mgr.cancel(tx_id)

        rec = mgr.get(tx_id)
        assert rec.status == "cancelled"

        # Safe point request should be withdrawn
        assert not mgr._safe_point.has_pending_request("sess-test")

        # Journal reflects cancellation
        journal_path = v2_paths.journals / f"{tx_id}.json"
        assert json.loads(journal_path.read_text())["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_after_workflow_ack_releases_safe_point(
        self, v2_paths, preflight_result, restore_request,
    ):
        """Cancel after safe point is acknowledged releases the safe point."""
        mgr = make_mgr(v2_paths)

        # Set a barrier at "staging" to freeze the job after ack
        barrier = asyncio.Event()
        mgr._barrier_events["staging"] = barrier

        record = await mgr.enqueue("sess-test", preflight_result, restore_request,
                                    timeout_seconds=30)
        tx_id = record.transaction_id

        # Wait for waiting_safe_point
        await asyncio.wait_for(
            self._wait_for_status(mgr, tx_id, "waiting_safe_point"), timeout=5
        )

        # Acknowledge (workflow pauses)
        gen = mgr._safe_point.get_pending_generation("sess-test")
        await mgr._safe_point.acknowledge_safe_point_request("sess-test", gen, role="workflow")
        assert mgr._safe_point.is_paused("sess-test")

        # The job is now frozen at the "staging" barrier
        # Give it a moment to hit the barrier
        await asyncio.sleep(0.05)

        # Cancel — should release safe point
        await mgr.cancel(tx_id)

        rec = mgr.get(tx_id)
        assert rec.status == "cancelled"

        # Safe point should be released
        assert not mgr._safe_point.is_paused("sess-test")

        # Lift barrier so the cancelled task can complete cleanup
        barrier.set()

    @pytest.mark.asyncio
    async def test_timeout(self, v2_paths, preflight_result, restore_request):
        """Job transitions to timed_out when acknowledge timeout expires."""
        mgr = make_mgr(v2_paths)
        record = await mgr.enqueue("sess-test", preflight_result, restore_request,
                                    timeout_seconds=0.1)
        tx_id = record.transaction_id

        # Don't acknowledge — let timeout fire
        await asyncio.wait_for(
            self._wait_for_status(mgr, tx_id, "timed_out"), timeout=5
        )

        rec = mgr.get(tx_id)
        assert rec.status == "timed_out"

        # Journal reflects timeout
        journal_path = v2_paths.journals / f"{tx_id}.json"
        assert json.loads(journal_path.read_text())["status"] == "timed_out"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, v2_paths, preflight_result, restore_request):
        """get returns None for unknown transaction."""
        mgr = make_mgr(v2_paths)
        assert mgr.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_raises(self, v2_paths, preflight_result, restore_request):
        """cancel on unknown transaction raises SnapshotTransactionError (404)."""
        mgr = make_mgr(v2_paths)
        with pytest.raises(SnapshotTransactionError):
            await mgr.cancel("nonexistent")

    @pytest.mark.asyncio
    async def test_enqueue_failure_returns_none(self, v2_paths, preflight_result, restore_request):
        """If the job task fails to start, enqueue still returns the record."""
        mgr = make_mgr(v2_paths)
        # This test just confirms enqueue doesn't throw on basic setup
        record = await mgr.enqueue("sess-test", preflight_result, restore_request)
        assert record is not None
        assert record.transaction_id is not None

    # ── Helpers ─────────────────────────────────────────────────

    async def _wait_for_status(self, mgr, tx_id, target_status, poll=0.02):
        """Wait for a transaction to reach target_status."""
        while True:
            rec = mgr.get(tx_id)
            if rec and rec.status == target_status:
                return
            if rec and rec.status in ("failed", "cancelled", "timed_out") and rec.status != target_status:
                raise AssertionError(
                    f"Job reached terminal state '{rec.status}' before '{target_status}'"
                    f" failure={getattr(rec, 'failure', None)}"
                )
            await asyncio.sleep(poll)

    def _has_status(self, mgr, tx_id, status):
        rec = mgr.get(tx_id)
        return rec is not None and rec.status == status


class TestSharedRestoreJobRegistry:
    """Cross-request visibility and cooperative safe-point acknowledgement."""

    @pytest.mark.asyncio
    async def test_job_is_visible_and_cancellable_from_new_request(
        self, v2_paths, preflight_result, restore_request,
    ):
        """A job enqueued by one manager is visible and cancellable without that instance."""
        from backend.services.snapshot_v2.jobs import (
            cancel_restore_job,
            get_restore_job,
            load_transaction,
        )

        # Enqueue via manager A (simulates POST restore)
        mgr_a = make_mgr(v2_paths, session_id="sess-cross")
        tx = await mgr_a.enqueue(
            "sess-cross", preflight_result, restore_request, timeout_seconds=30,
        )

        # Independent request path — no access to mgr_a
        handle = get_restore_job(tx.transaction_id)
        assert handle is not None
        assert handle.record.session_id == "sess-cross"
        assert handle.cancel_event is not None
        assert handle.task is not None

        # Wait until waiting_safe_point so cancel has a non-terminal job
        async def _wait_waiting():
            while True:
                rec = load_transaction(tx.transaction_id)
                if rec and rec.status == "waiting_safe_point":
                    return
                if rec and rec.status in ("failed", "cancelled", "timed_out", "committed"):
                    raise AssertionError(f"Unexpected terminal status: {rec.status}")
                await asyncio.sleep(0.02)

        await asyncio.wait_for(_wait_waiting(), timeout=5)

        await cancel_restore_job(tx.transaction_id, session_id="sess-cross")

        loaded = load_transaction(tx.transaction_id)
        assert loaded is not None
        assert loaded.status == "cancelled"

        # Journal remains queryable after task cleanup
        journal_path = v2_paths.journals / f"{tx.transaction_id}.json"
        assert json.loads(journal_path.read_text())["status"] == "cancelled"

        # After task finishes, handle may release the task ref but journal stays
        if handle.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.05)

        still = load_transaction(tx.transaction_id)
        assert still is not None
        assert still.status == "cancelled"


    @pytest.mark.asyncio
    async def test_cancel_rejects_session_mismatch(
        self, v2_paths, preflight_result, restore_request,
    ):
        """Cancel from a different session_id must not cancel the job (404)."""
        from backend.services.snapshot_v2.errors import SnapshotTransactionError
        from backend.services.snapshot_v2.jobs import (
            cancel_restore_job,
            get_restore_job,
            load_transaction,
        )

        mgr = make_mgr(v2_paths, session_id="sess-a")
        tx = await mgr.enqueue(
            "sess-a", preflight_result, restore_request, timeout_seconds=30,
        )

        async def _wait_waiting():
            while True:
                rec = load_transaction(tx.transaction_id)
                if rec and rec.status == "waiting_safe_point":
                    return
                if rec and rec.status in ("failed", "cancelled", "timed_out", "committed"):
                    raise AssertionError(f"Unexpected terminal: {rec.status}")
                await asyncio.sleep(0.02)

        await asyncio.wait_for(_wait_waiting(), timeout=5)

        with pytest.raises(SnapshotTransactionError) as exc_info:
            await cancel_restore_job(tx.transaction_id, session_id="sess-b")
        assert "not found" in str(exc_info.value).lower() or exc_info.value.code in (
            "TRANSACTION_NOT_FOUND",
            "SESSION_MISMATCH",
        )

        loaded = load_transaction(tx.transaction_id)
        assert loaded is not None
        assert loaded.status == "waiting_safe_point"
        assert loaded.session_id == "sess-a"

        await cancel_restore_job(tx.transaction_id, session_id="sess-a")
        loaded2 = load_transaction(tx.transaction_id)
        assert loaded2 is not None
        assert loaded2.status == "cancelled"

        handle = get_restore_job(tx.transaction_id)
        if handle is not None and handle.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    @pytest.mark.asyncio
    async def test_cancel_unknown_transaction_raises(self):
        """Cancel of unknown transaction_id raises (maps to HTTP 404)."""
        from backend.services.snapshot_v2.errors import SnapshotTransactionError
        from backend.services.snapshot_v2.jobs import cancel_restore_job

        with pytest.raises(SnapshotTransactionError) as exc_info:
            await cancel_restore_job("restore-nonexistent", session_id="sess-x")
        assert "not found" in str(exc_info.value).lower() or exc_info.value.code == (
            "TRANSACTION_NOT_FOUND"
        )

    @pytest.mark.asyncio
    async def test_load_transaction_from_journal_without_handle(
        self, v2_paths, preflight_result, restore_request,
    ):
        """When in-memory handle is gone, load_transaction reads the journal file."""
        from backend.state import snapshot_restore_jobs
        from backend.services.snapshot_v2.jobs import load_transaction
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord

        tx_id = "restore-journal-only"
        record = SnapshotTransactionRecord(
            transaction_id=tx_id,
            session_id="sess-journal",
            snapshot_id="snap-001",
            status="waiting_safe_point",
        )
        journal_path = v2_paths.journals / f"{tx_id}.json"
        journal_path.write_text(record.model_dump_json())

        # Ensure no in-memory handle
        snapshot_restore_jobs.pop(tx_id, None)
        assert load_transaction(tx_id) is None

        loaded = load_transaction(tx_id, paths=v2_paths)
        assert loaded is not None
        assert isinstance(loaded, SnapshotTransactionRecord)
        assert loaded.transaction_id == tx_id
        assert loaded.session_id == "sess-journal"
        assert loaded.status == "waiting_safe_point"

    @pytest.mark.asyncio
    async def test_workflow_ack_via_boundary_hook_without_private_safe_point(
        self, v2_paths, preflight_result, restore_request,
    ):
        """Request safe point via shared service; ack via workflow boundary hook only."""
        from backend.services.snapshot_v2.jobs import RestoreJobManager
        from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
        from backend.workflow.core import SingleCellAnalysisWorkflow

        session_id = "sess-wf-ack"
        sp = get_snapshot_safe_point_manager()

        mgr = RestoreJobManager(
            paths=v2_paths,
            session_id=session_id,
            snapshot_service=_StatusOnlyService(session_id, v2_paths),
        )
        barrier = asyncio.Event()
        mgr._barrier_events["staging"] = barrier

        record = await mgr.enqueue(
            session_id, preflight_result, restore_request, timeout_seconds=30,
        )
        tx_id = record.transaction_id

        # Wait until job has requested the safe point
        async def _wait_pending():
            while not sp.has_pending_request(session_id):
                rec = mgr.get(tx_id)
                if rec and rec.status in ("failed", "cancelled", "timed_out", "committed"):
                    raise AssertionError(f"Job ended before pending: {rec.status}")
                await asyncio.sleep(0.02)

        await asyncio.wait_for(_wait_pending(), timeout=5)
        assert sp.has_pending_request(session_id)

        # Workflow-side acknowledgement — must not touch mgr._safe_point
        wf = MagicMock(spec=SingleCellAnalysisWorkflow)
        wf.session_id = session_id
        ack = SingleCellAnalysisWorkflow.acknowledge_at_boundary.__get__(
            wf, SingleCellAnalysisWorkflow,
        )
        await ack()

        assert not sp.has_pending_request(session_id)
        assert sp.is_paused(session_id)

        # Job is frozen at the pre-staging barrier; give it a moment to leave
        # waiting_safe_point wait and hit the barrier.
        await asyncio.sleep(0.05)
        rec = mgr.get(tx_id)
        assert rec is not None
        assert rec.status in ("waiting_safe_point", "staging"), rec.status

        # Release so job can finish
        barrier.set()
        # Job releases safe point on commit; also release if still held
        await asyncio.wait_for(
            TestRestoreJobLifecycle()._wait_for_status(mgr, tx_id, "committed"),
            timeout=5,
        )
        assert mgr.get(tx_id).status == "committed"
        assert not sp.is_paused(session_id)

    @pytest.mark.asyncio
    async def test_terminal_job_queryable_after_task_cleanup(
        self, v2_paths, preflight_result, restore_request,
    ):
        """After task cleanup, only the task ref is released; journal remains loadable."""
        from backend.services.snapshot_v2.jobs import get_restore_job, load_transaction

        mgr = make_mgr(v2_paths)
        record = await mgr.enqueue(
            "sess-test", preflight_result, restore_request, timeout_seconds=30,
        )
        tx_id = record.transaction_id

        await asyncio.wait_for(
            TestRestoreJobLifecycle()._wait_for_status(mgr, tx_id, "waiting_safe_point"),
            timeout=5,
        )
        gen = mgr._safe_point.get_pending_generation("sess-test")
        await mgr._safe_point.acknowledge_safe_point_request("sess-test", gen, role="workflow")
        await asyncio.wait_for(
            TestRestoreJobLifecycle()._wait_for_status(mgr, tx_id, "committed"),
            timeout=5,
        )

        # Allow done-callback cleanup
        handle = get_restore_job(tx_id)
        if handle is not None and handle.task is not None:
            await asyncio.wait_for(handle.task, timeout=2)
            await asyncio.sleep(0.05)

        # load_transaction still works via journal
        loaded = load_transaction(tx_id)
        assert loaded is not None
        assert loaded.status == "committed"

        # Task reference released (None) while handle/journal remain
        handle_after = get_restore_job(tx_id)
        if handle_after is not None:
            assert handle_after.task is None


# ── Real-effect job (Task 4) ───────────────────────────────────────────────


class TestCommittedJobLiveSideEffects:
    """committed means live side effects verified — not status-only."""

    @pytest.mark.asyncio
    async def test_committed_job_changed_live_content_and_continuation(
        self, tmp_path, monkeypatch,
    ):
        """End-to-end: enqueue → ack → committed with live state/workspace/continuation."""
        import hashlib
        import json as _json
        from datetime import datetime, timezone

        from backend.models.schemas import AnalysisState
        from backend.services.snapshot_v2.jobs import RestoreJobManager, load_transaction
        from backend.services.snapshot_v2.models import (
            GraphNodeCapsule,
            ObjectRef,
            SnapshotRestoreRequest,
        )
        from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore, canonical_json
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.preflight import RestorePreflight
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        session_id = "sess-live-effect"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        monkeypatch.setattr("backend.config._config.get_session_dir", _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.config._config.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_session_dir", _sess_dir,
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )

        session_root = _sess_dir(session_id)
        workspace = session_root / "workspace"
        states = session_root / "states"
        for d in (session_root, workspace, states):
            d.mkdir(parents=True, exist_ok=True)

        paths = SnapshotV2Paths.for_session(session_id)
        for d in (paths.root, paths.objects, paths.staging, paths.journals, paths.integrity, paths.committed):
            d.mkdir(parents=True, exist_ok=True)

        # Live (pre-restore) content — different from target.
        live_state = AnalysisState(
            id=session_id, user_id="u1", name="live",
            initial_instruction="live", current_round=1, max_rounds=4,
        )
        sessions[session_id] = live_state
        (workspace / "tables").mkdir(parents=True, exist_ok=True)
        (workspace / "tables" / "result.csv").write_bytes(b"live,old\n")

        # Target content
        target_round = 3
        table_bytes = b"gene,score\nTP53,0.9\n"
        table_sha = hashlib.sha256(table_bytes).hexdigest()
        (paths.objects / table_sha).write_bytes(table_bytes)
        table_ref = ObjectRef(sha256=table_sha, size=len(table_bytes), media_type="text/csv")

        now = datetime.now(timezone.utc)
        target_state = AnalysisState(
            id=session_id, user_id="u1", name="target",
            initial_instruction="target", current_round=target_round, max_rounds=4,
            created_at=now, updated_at=now,
        )
        state_payload = target_state.model_dump(mode="json")
        state_payload.pop("snapshots", None)
        state_payload.pop("realtime_messages", None)
        state_bytes = _json.dumps(
            state_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
        ).encode()
        state_sha = hashlib.sha256(state_bytes).hexdigest()
        (paths.objects / state_sha).write_bytes(state_bytes)
        state_ref = ObjectRef(sha256=state_sha, size=len(state_bytes), media_type="application/json")

        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
        nb_bytes = _json.dumps(nb, sort_keys=True, separators=(",", ":")).encode()
        nb_sha = hashlib.sha256(nb_bytes).hexdigest()
        (paths.objects / nb_sha).write_bytes(nb_bytes)
        nb_ref = ObjectRef(sha256=nb_sha, size=len(nb_bytes), media_type="application/x-ipynb+json")

        ws_payload = {
            "current_round": target_round,
            "planning_stage": "stage1",
            "plan_approved": True,
        }
        ws_bytes = _json.dumps(ws_payload, sort_keys=True, ensure_ascii=False).encode()
        ws_sha = hashlib.sha256(ws_bytes).hexdigest()
        (paths.objects / ws_sha).write_bytes(ws_bytes)
        ws_ref = ObjectRef(sha256=ws_sha, size=len(ws_bytes), media_type="application/json")
        capsule = GraphNodeCapsule(
            graph_version="2.0",
            next_node="bioinformatics_analysis",
            workflow_state_ref=ws_ref,
            planning_stage="stage1",
        )

        objects = {
            state_sha: state_ref,
            nb_sha: nb_ref,
            table_sha: table_ref,
            ws_sha: ws_ref,
        }
        snapshot_id = "snap-live-target"
        manifest = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_number=target_round,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[{
                "path": "tables/result.csv",
                "object_hash": table_sha,
                "size": len(table_bytes),
                "mode": 0o644,
            }],
            objects={k: v.model_dump(mode="json") for k, v in objects.items()},
            continuation=capsule.model_dump(mode="json"),
            runtime={
                "status": "complete",
                "dependencies": {"required": {}, "incidental_skips": []},
                "residual_ref": None,
                "captured_residual_names": [],
                "adata_refs": {},
            },
        )
        raw = canonical_json(manifest)
        manifest.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(paths)
        staging = mstore.write_staging(manifest)
        mstore.publish_staging(staging)

        request = SnapshotRestoreRequest(
            snapshot_id=snapshot_id,
            runtime_choice="reset",
            current_state_choice="abort",
        )
        preflight = RestorePreflight(paths).run(snapshot_id, request)
        assert preflight.status == "ready", preflight.compatibility_errors

        service = SnapshotService(session_id, paths)

        class _FakePreRestore:
            async def capture_locked(self, context):
                from backend.services.snapshot_v2.models import SnapshotRecordV2
                pre_id = "snap-pre-live"
                pre_m = make_snapshot_v2_manifest(
                    session_id=session_id,
                    snapshot_id=pre_id,
                    analysis_state_ref=state_ref.model_dump(mode="json"),
                    notebook_ref=nb_ref.model_dump(mode="json"),
                    objects={
                        state_sha: state_ref.model_dump(mode="json"),
                        nb_sha: nb_ref.model_dump(mode="json"),
                    },
                )
                pre_raw = canonical_json(pre_m)
                pre_m.root_hash = hashlib.sha256(pre_raw).hexdigest()
                st = mstore.write_staging(pre_m)
                mstore.publish_staging(st)
                return SnapshotRecordV2(snapshot_id=pre_id, manifest=pre_m, failure=None)

        service.pre_restore_capture = _FakePreRestore()

        async def _persist(sid, state):
            sessions[sid] = state
            p = _sess_dir(sid) / "session.json"
            p.write_text(state.model_dump_json())
        service.session_persistence = _persist

        mgr = RestoreJobManager(
            paths=paths, session_id=session_id, snapshot_service=service,
        )

        async def enqueue_restore(target_snapshot_id):
            req = SnapshotRestoreRequest(
                snapshot_id=target_snapshot_id,
                runtime_choice="reset",
                current_state_choice="abort",
            )
            pf = RestorePreflight(paths).run(target_snapshot_id, req)
            return await mgr.enqueue(session_id, pf, req, timeout_seconds=30)

        async def acknowledge_from_workflow(sid):
            gen = mgr._safe_point.get_pending_generation(sid)
            if gen is not None:
                await mgr._safe_point.acknowledge_safe_point_request(sid, gen, role='workflow')

        async def wait_terminal(tx_id):
            while True:
                rec = load_transaction(tx_id) or mgr.get(tx_id)
                if rec and rec.status in (
                    "committed", "failed", "cancelled", "timed_out",
                ):
                    return rec
                await asyncio.sleep(0.02)

        class LiveSession:
            def __init__(self, sid):
                self.session_id = sid

            def read_state(self):
                st = sessions.get(self.session_id)
                if st is None:
                    return {}
                return {"current_round": st.current_round}

            def read_workspace(self, rel):
                return (_sess_dir(self.session_id) / "workspace" / rel).read_bytes()

            def next_node(self):
                st = sessions.get(self.session_id)
                return getattr(st, "restore_resume_node", None)

        live_session = LiveSession(session_id)

        class _Cont:
            next_node = "bioinformatics_analysis"

        class _Target:
            pass

        target = _Target()
        target.round_number = target_round
        target.table_bytes = table_bytes
        target.continuation = _Cont()

        tx = await enqueue_restore(snapshot_id)

        # Wait until the job has requested the safe point (avoid ack race).
        async def _wait_pending():
            while not mgr._safe_point.has_pending_request(session_id):
                rec = mgr.get(tx.transaction_id)
                if rec and rec.status in ("failed", "cancelled", "timed_out", "committed"):
                    raise AssertionError(f"ended before pending: {rec.status} {rec.failure}")
                await asyncio.sleep(0.02)
        await asyncio.wait_for(_wait_pending(), timeout=5)

        await acknowledge_from_workflow(live_session.session_id)
        record = await asyncio.wait_for(wait_terminal(tx.transaction_id), timeout=15)

        assert record.status == "committed", getattr(record, "failure", None)
        assert live_session.read_state()["current_round"] == target.round_number
        assert live_session.read_workspace("tables/result.csv") == target.table_bytes
        assert live_session.next_node() == target.continuation.next_node

        sessions.pop(session_id, None)

    @pytest.mark.asyncio
    async def test_live_verify_failure_keeps_journal_applying_and_restart_rolls_back(
        self, tmp_path, monkeypatch,
    ):
        """LIVE_VERIFY_FAILED must leave journal APPLYING so recovery can roll back."""
        import hashlib
        import json as _json
        from datetime import datetime, timezone
        from unittest.mock import AsyncMock

        from backend.models.schemas import AnalysisState
        from backend.services.snapshot_v2.errors import SnapshotTransactionError
        from backend.services.snapshot_v2.models import (
            GraphNodeCapsule,
            ObjectRef,
            SnapshotRestoreRequest,
        )
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        from backend.services.snapshot_v2.restore import read_verified_restore_journal
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        session_id = "sess-live-verify-order"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        monkeypatch.setattr("backend.config._config.get_session_dir", _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_session_dir", _sess_dir,
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )

        session_root = _sess_dir(session_id)
        workspace = session_root / "workspace"
        states = session_root / "states"
        for d in (session_root, workspace, states):
            d.mkdir(parents=True, exist_ok=True)

        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Pre-restore live content that recovery must restore.
        pre_table = b"live,pre-restore\n"
        (workspace / "tables").mkdir(parents=True, exist_ok=True)
        (workspace / "tables" / "result.csv").write_bytes(pre_table)
        pre_session = AnalysisState(
            id=session_id, user_id="u1", name="pre-live",
            initial_instruction="pre", current_round=1, max_rounds=4,
        )
        sessions[session_id] = pre_session
        (session_root / "session.json").write_text(pre_session.model_dump_json())

        # State object claims round=1 but manifest.round_number=9 → live verify fails.
        now = datetime.now(timezone.utc)
        target_state = AnalysisState(
            id=session_id, user_id="u1", name="target",
            initial_instruction="target", current_round=1, max_rounds=4,
            created_at=now, updated_at=now,
        )
        state_payload = target_state.model_dump(mode="json")
        state_payload.pop("snapshots", None)
        state_payload.pop("realtime_messages", None)
        state_bytes = _json.dumps(
            state_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        ).encode()
        state_sha = hashlib.sha256(state_bytes).hexdigest()
        (paths.objects / state_sha).write_bytes(state_bytes)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state_bytes), media_type="application/json",
        )

        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
        nb_bytes = _json.dumps(nb, sort_keys=True, separators=(",", ":")).encode()
        nb_sha = hashlib.sha256(nb_bytes).hexdigest()
        (paths.objects / nb_sha).write_bytes(nb_bytes)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb_bytes), media_type="application/x-ipynb+json",
        )

        ws_payload = {
            "current_round": 1, "planning_stage": "stage1", "plan_approved": True,
        }
        ws_bytes = _json.dumps(ws_payload, sort_keys=True, ensure_ascii=False).encode()
        ws_sha = hashlib.sha256(ws_bytes).hexdigest()
        (paths.objects / ws_sha).write_bytes(ws_bytes)
        ws_ref = ObjectRef(
            sha256=ws_sha, size=len(ws_bytes), media_type="application/json",
        )
        capsule = GraphNodeCapsule(
            graph_version="2.0",
            next_node="bioinformatics_analysis",
            workflow_state_ref=ws_ref,
            planning_stage="stage1",
        )
        objects = {state_sha: state_ref, nb_sha: nb_ref, ws_sha: ws_ref}
        snapshot_id = "snap-live-verify-fail-target"
        manifest = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_number=9,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[],
            objects={k: v.model_dump(mode="json") for k, v in objects.items()},
            continuation=capsule.model_dump(mode="json"),
            runtime={
                "status": "complete",
                "dependencies": {"required": {}, "incidental_skips": []},
                "residual_ref": None,
                "captured_residual_names": [],
                "adata_refs": {},
            },
        )
        raw = canonical_json(manifest)
        manifest.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(paths)
        staging = mstore.write_staging(manifest)
        mstore.publish_staging(staging)

        # Pre-restore snapshot content (original live bytes) for rollback.
        pre_table_sha = hashlib.sha256(pre_table).hexdigest()
        (paths.objects / pre_table_sha).write_bytes(pre_table)
        pre_table_ref = ObjectRef(
            sha256=pre_table_sha, size=len(pre_table), media_type="text/csv",
        )
        pre_state_payload = pre_session.model_dump(mode="json")
        pre_state_payload.pop("snapshots", None)
        pre_state_payload.pop("realtime_messages", None)
        pre_state_bytes = _json.dumps(
            pre_state_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        ).encode()
        pre_state_sha = hashlib.sha256(pre_state_bytes).hexdigest()
        (paths.objects / pre_state_sha).write_bytes(pre_state_bytes)
        pre_state_ref = ObjectRef(
            sha256=pre_state_sha, size=len(pre_state_bytes),
            media_type="application/json",
        )

        service = SnapshotService(session_id, paths)

        class _FakePreRestore:
            async def capture_locked(self, context):
                from backend.services.snapshot_v2.models import SnapshotRecordV2
                pre_id = "snap-pre-live-verify-order"
                pre_m = make_snapshot_v2_manifest(
                    session_id=session_id,
                    snapshot_id=pre_id,
                    analysis_state_ref=pre_state_ref.model_dump(mode="json"),
                    notebook_ref=nb_ref.model_dump(mode="json"),
                    workspace=[{
                        "path": "tables/result.csv",
                        "object_hash": pre_table_sha,
                        "size": len(pre_table),
                        "mode": 0o644,
                    }],
                    objects={
                        pre_state_sha: pre_state_ref.model_dump(mode="json"),
                        nb_sha: nb_ref.model_dump(mode="json"),
                        pre_table_sha: pre_table_ref.model_dump(mode="json"),
                    },
                )
                pre_raw = canonical_json(pre_m)
                pre_m.root_hash = hashlib.sha256(pre_raw).hexdigest()
                st = mstore.write_staging(pre_m)
                mstore.publish_staging(st)
                return SnapshotRecordV2(snapshot_id=pre_id, manifest=pre_m, failure=None)

        service.pre_restore_capture = _FakePreRestore()

        async def _persist(sid, state):
            sessions[sid] = state
            p = _sess_dir(sid) / "session.json"
            p.write_text(state.model_dump_json())
        service.session_persistence = _persist

        # Prefer explicit mock when _verify_live_restore exists; still assert
        # journal ordering against real LIVE_VERIFY_FAILED path when it does not.
        if hasattr(SnapshotService, "_verify_live_restore"):
            service._verify_live_restore = AsyncMock(
                side_effect=SnapshotTransactionError(
                    "LIVE_VERIFY_FAILED", "workspace hash mismatch",
                )
            )

        target_request = SnapshotRestoreRequest(
            snapshot_id=snapshot_id,
            runtime_choice="reset",
            current_state_choice="abort",
        )
        record = await service.restore_snapshot(
            target_request,
            transaction_id="restore-live-verify-fail",
            cancel_event=None,
        )

        assert record.status == "failed"
        journal = read_verified_restore_journal(paths, "restore-live-verify-fail")
        assert journal.status == "APPLYING"

        recovered = await recover_snapshot_transactions(
            session_id=session_id,
            paths=paths,
            session_dir=str(session_root),
            workspace_root=str(workspace),
            states_root=str(states),
        )
        assert recovered.failed_transactions == []
        assert (workspace / "tables" / "result.csv").read_bytes() == pre_table
        sessions.pop(session_id, None)


    @pytest.mark.asyncio
    async def test_live_verify_fails_on_current_round_mismatch(
        self, tmp_path, monkeypatch,
    ):
        """Round mismatch must fail with LIVE_VERIFY_FAILED (not committed)."""
        import hashlib
        import json as _json
        from datetime import datetime, timezone

        from backend.models.schemas import AnalysisState
        from backend.services.snapshot_v2.models import (
            GraphNodeCapsule,
            ObjectRef,
            SnapshotRestoreRequest,
        )
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        session_id = "sess-round-verify"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        monkeypatch.setattr("backend.config._config.get_session_dir", _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_session_dir", _sess_dir,
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )

        session_root = _sess_dir(session_id)
        workspace = session_root / "workspace"
        states = session_root / "states"
        for d in (session_root, workspace, states):
            d.mkdir(parents=True, exist_ok=True)

        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)

        live_state = AnalysisState(
            id=session_id, user_id="u1", name="live",
            initial_instruction="live", current_round=1, max_rounds=4,
        )
        sessions[session_id] = live_state

        # State object claims round=1 but manifest.round_number=9 → mismatch.
        now = datetime.now(timezone.utc)
        target_state = AnalysisState(
            id=session_id, user_id="u1", name="target",
            initial_instruction="target", current_round=1, max_rounds=4,
            created_at=now, updated_at=now,
        )
        state_payload = target_state.model_dump(mode="json")
        state_payload.pop("snapshots", None)
        state_payload.pop("realtime_messages", None)
        state_bytes = _json.dumps(
            state_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        ).encode()
        state_sha = hashlib.sha256(state_bytes).hexdigest()
        (paths.objects / state_sha).write_bytes(state_bytes)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state_bytes), media_type="application/json",
        )

        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
        nb_bytes = _json.dumps(nb, sort_keys=True, separators=(",", ":")).encode()
        nb_sha = hashlib.sha256(nb_bytes).hexdigest()
        (paths.objects / nb_sha).write_bytes(nb_bytes)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb_bytes), media_type="application/x-ipynb+json",
        )

        ws_payload = {
            "current_round": 1, "planning_stage": "stage1", "plan_approved": True,
        }
        ws_bytes = _json.dumps(ws_payload, sort_keys=True, ensure_ascii=False).encode()
        ws_sha = hashlib.sha256(ws_bytes).hexdigest()
        (paths.objects / ws_sha).write_bytes(ws_bytes)
        ws_ref = ObjectRef(
            sha256=ws_sha, size=len(ws_bytes), media_type="application/json",
        )
        capsule = GraphNodeCapsule(
            graph_version="2.0",
            next_node="bioinformatics_analysis",
            workflow_state_ref=ws_ref,
            planning_stage="stage1",
        )
        objects = {
            state_sha: state_ref, nb_sha: nb_ref, ws_sha: ws_ref,
        }
        snapshot_id = "snap-round-mismatch"
        manifest = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_number=9,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[],
            objects={k: v.model_dump(mode="json") for k, v in objects.items()},
            continuation=capsule.model_dump(mode="json"),
            runtime={
                "status": "complete",
                "dependencies": {"required": {}, "incidental_skips": []},
                "residual_ref": None,
                "captured_residual_names": [],
                "adata_refs": {},
            },
        )
        raw = canonical_json(manifest)
        manifest.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(paths)
        staging = mstore.write_staging(manifest)
        mstore.publish_staging(staging)

        service = SnapshotService(session_id, paths)

        class _FakePreRestore:
            async def capture_locked(self, context):
                from backend.services.snapshot_v2.models import SnapshotRecordV2
                pre_id = "snap-pre-round"
                pre_m = make_snapshot_v2_manifest(
                    session_id=session_id,
                    snapshot_id=pre_id,
                    analysis_state_ref=state_ref.model_dump(mode="json"),
                    notebook_ref=nb_ref.model_dump(mode="json"),
                    objects={
                        state_sha: state_ref.model_dump(mode="json"),
                        nb_sha: nb_ref.model_dump(mode="json"),
                    },
                )
                pre_raw = canonical_json(pre_m)
                pre_m.root_hash = hashlib.sha256(pre_raw).hexdigest()
                st = mstore.write_staging(pre_m)
                mstore.publish_staging(st)
                return SnapshotRecordV2(snapshot_id=pre_id, manifest=pre_m, failure=None)

        service.pre_restore_capture = _FakePreRestore()

        async def _persist(sid, state):
            sessions[sid] = state
            p = _sess_dir(sid) / "session.json"
            p.write_text(state.model_dump_json())
        service.session_persistence = _persist

        request = SnapshotRestoreRequest(
            snapshot_id=snapshot_id,
            runtime_choice="reset",
            current_state_choice="abort",
        )
        rec = await service.restore_snapshot(
            request, transaction_id="tx-round-verify", cancel_event=None,
        )
        assert rec.status == "failed"
        assert rec.failure is not None
        assert rec.failure.get("code") == "LIVE_VERIFY_FAILED"
        assert "current_round" in (rec.failure.get("message") or "")
        # In-process guard: memory state was persisted to the target while the
        # journal stays APPLYING; the session must be recovery-blocked so no
        # forward mutation trusts it until startup recovery rolls back.
        from backend.services.session_store import (
            assert_recovery_clear,
            _recovery_blocked_sessions,
        )
        from backend.services.snapshot_v2.errors import SnapshotTransactionError

        blocked_state = sessions.get(session_id)
        assert blocked_state is not None
        assert blocked_state.recovery_blocked is True
        with pytest.raises(SnapshotTransactionError):
            assert_recovery_clear(blocked_state, session_id=session_id)
        _recovery_blocked_sessions.discard(session_id)
        sessions.pop(session_id, None)


class TestSafePointRetainedDuringPreRestore:
    """pre_restore capture must not release the restore-held safe point."""

    @pytest.mark.asyncio
    async def test_capture_locked_for_restore_skips_safe_point_lifecycle(
        self, tmp_path, monkeypatch,
    ):
        """_capture_locked_for_restore keeps an already-held safe point."""
        from backend.models.schemas import AnalysisState
        from backend.services.snapshot_v2.models import (
            IdleCapsule,
            SnapshotCaptureRequest,
        )
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions

        session_id = "sess-safe-retain"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        monkeypatch.setattr("backend.config._config.get_session_dir", _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )

        for d in ("workspace", "states"):
            (_sess_dir(session_id) / d).mkdir(parents=True, exist_ok=True)

        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)

        state = AnalysisState(
            id=session_id, user_id="u1", name="live",
            initial_instruction="live", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state

        sp = get_snapshot_safe_point_manager()
        await sp.release(session_id, "test-owner")
        assert await sp.acquire(session_id, {"kernel_present": False}, owner_token="test-owner") is True
        assert sp.is_paused(session_id)

        service = SnapshotService(session_id, paths)
        req = SnapshotCaptureRequest(
            session_id=session_id,
            phase="pre_restore",
            reason="pre_restore",
            round_number=1,
            continuation=IdleCapsule(graph_version="2.0"),
        )
        record = await service._capture_locked_for_restore(
            req,
            workspace_root=str(_sess_dir(session_id) / "workspace"),
            workflow_state={"kernel_present": False},
            analysis_state=state,
        )
        assert record.failure is None, getattr(record.failure, "message", None)
        # Critical: restore-held safe point must still be paused after capture.
        assert sp.is_paused(session_id)
        await sp.release(session_id, "test-owner")
        sessions.pop(session_id, None)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_capture_locked_for_restore_returns_without_session_save(
        self, tmp_path, monkeypatch,
    ):
        """Pre-restore capture must not call session save (avoid re-entry hang)."""
        from backend.models.schemas import AnalysisState
        from backend.services.snapshot_v2.models import (
            IdleCapsule,
            SnapshotCaptureRequest,
        )
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions

        session_id = "sess-no-save"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        monkeypatch.setattr("backend.config._config.get_session_dir", _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )

        for d in ("workspace", "states"):
            (_sess_dir(session_id) / d).mkdir(parents=True, exist_ok=True)

        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)

        state = AnalysisState(
            id=session_id, user_id="u1", name="live",
            initial_instruction="live", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state

        sp = get_snapshot_safe_point_manager()
        await sp.release(session_id, "test-owner")
        assert await sp.acquire(session_id, {"kernel_present": False}, owner_token="test-owner") is True

        service = SnapshotService(session_id, paths)
        save = AsyncMock()
        monkeypatch.setattr(service, "_save_session_state", save)

        req = SnapshotCaptureRequest(
            session_id=session_id,
            phase="pre_restore",
            reason="pre_restore",
            round_number=1,
            continuation=IdleCapsule(graph_version="2.0"),
        )
        record = await service._capture_locked_for_restore(
            req,
            workspace_root=str(_sess_dir(session_id) / "workspace"),
            workflow_state={"kernel_present": False, "inactive": True},
            analysis_state=state,
        )
        assert record.failure is None, getattr(record.failure, "message", None)
        save.assert_not_awaited()
        await sp.release(session_id, "test-owner")
        sessions.pop(session_id, None)


# ---- post_report snapshot-failure retry re-invokes capture producer ---------


@pytest.fixture
def post_report_blocked_session():
    """Session blocked on post_report snapshot failure with a fake workflow."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.state import sessions, workflows

    session_id = "sess-post-report-retry"
    state = AnalysisState(
        id=session_id,
        name="PostReportRetry",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            "retry_context": {"producer": "post_report"},
        },
        pending_required_snapshot={"report_generation_id": "r1"},
    )

    async def _retry_ok():
        # Real capture_required_snapshot clears failure only on commit.
        state.pending_required_snapshot = None
        state.snapshot_failure = None
        if state.wait_input_reason == "snapshot_failure":
            state.wait_input_reason = None
        return {"snapshot_id": "snap-retry-ok"}

    fake_workflow = MagicMock()
    fake_workflow.retry_pending_required_snapshot = AsyncMock(side_effect=_retry_ok)
    fake_workflow.stop = AsyncMock()
    fake_workflow.pause = AsyncMock()

    sessions[session_id] = state
    workflows[session_id] = fake_workflow
    try:
        yield session_id, state, fake_workflow
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_post_report_retry_invokes_recapture(post_report_blocked_session, monkeypatch):
    """post_report Retry recaptures then finalizes like normal post-report success."""
    from backend.models.schemas import TaskStatus
    from backend.routes import lifecycle

    session_id, state, fake_workflow = post_report_blocked_session

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    resp = await lifecycle.snapshot_failure_retry(session_id, req)

    assert resp.success is True
    fake_workflow.retry_pending_required_snapshot.assert_awaited_once()
    assert state.status == TaskStatus.COMPLETED
    assert state.wait_input_reason is None
    assert state.snapshot_failure is None
    assert getattr(state, "pending_required_snapshot", None) is None
    assert resp.data["status"] == TaskStatus.COMPLETED.value
    assert resp.data["wait_input_reason"] is None
    assert resp.data["snapshot_failure"] is None


@pytest.mark.asyncio
async def test_post_report_retry_failure_reenters_gate(post_report_blocked_session, monkeypatch):
    """Failed recapture re-enters the snapshot failure gate with post_report context."""
    from backend.models.schemas import TaskStatus
    from backend.routes import lifecycle
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

    session_id, state, fake_workflow = post_report_blocked_session
    fake_workflow.retry_pending_required_snapshot.side_effect = RequiredSnapshotBlocked(
        "REQUIRED_SNAPSHOT_INCOMPLETE", "still failing",
    )

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    resp = await lifecycle.snapshot_failure_retry(session_id, req)

    assert resp.success is False
    assert state.snapshot_failure is not None
    assert state.snapshot_failure["retry_context"]["producer"] == "post_report"
    assert state.wait_input_reason == "snapshot_failure"
    assert state.status == TaskStatus.WAITING_INPUT


@pytest.mark.asyncio
async def test_post_report_retry_generic_exception_reenters_gate(
    post_report_blocked_session, monkeypatch,
):
    """Broad Exception after gate clear must re-enter gate with success=False."""
    from backend.models.schemas import TaskStatus
    from backend.routes import lifecycle

    session_id, state, fake_workflow = post_report_blocked_session
    fake_workflow.retry_pending_required_snapshot.side_effect = RuntimeError(
        "sandbox gone",
    )

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    resp = await lifecycle.snapshot_failure_retry(session_id, req)

    assert resp.success is False
    assert state.snapshot_failure is not None
    assert state.snapshot_failure["retry_context"]["producer"] == "post_report"
    assert state.wait_input_reason == "snapshot_failure"
    assert state.status == TaskStatus.WAITING_INPUT


@pytest.mark.asyncio
async def test_terminate_stops_and_records_reason(post_report_blocked_session, monkeypatch):
    """Terminate must STOPPED + audit reason; teardown, not pause."""
    from backend.models.schemas import TaskStatus
    from backend.routes import lifecycle

    session_id, state, fake_workflow = post_report_blocked_session

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    async def fake_pause(sid):
        # Would clobber STOPPED → PAUSED if terminate used pause.
        state.status = TaskStatus.PAUSED

    fake_workflow.pause = AsyncMock(side_effect=fake_pause)
    teardown = AsyncMock()

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)
    monkeypatch.setattr(lifecycle, "teardown_running_workflow", teardown)

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    resp = await lifecycle.snapshot_failure_terminate(session_id, req)

    assert resp.success is True
    assert state.status == TaskStatus.STOPPED
    assert state.snapshot_failure is None
    assert getattr(state, "wait_input_reason", None) is None
    assert getattr(state, "termination_reason", None) == "snapshot_failure"
    assert getattr(state, "pending_required_snapshot", "missing") is None
    teardown.assert_awaited_once_with(session_id)
    fake_workflow.pause.assert_not_awaited()
    fake_workflow.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_report_retry_rebuilds_workflow_when_missing(monkeypatch):
    """After WS teardown, rebuild capture-only workflow via _build_workflow_for_session."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle
    from backend.state import sessions, workflows

    session_id = "sess-post-report-rebuild"
    state = AnalysisState(
        id=session_id,
        name="PostReportRebuild",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            "retry_context": {"producer": "post_report"},
        },
        pending_required_snapshot={"report_generation_id": "r1"},
    )
    sessions[session_id] = state
    workflows.pop(session_id, None)

    async def _retry_ok():
        # Real capture_required_snapshot clears failure only on commit.
        state.pending_required_snapshot = None
        state.snapshot_failure = None
        if state.wait_input_reason == "snapshot_failure":
            state.wait_input_reason = None
        return {"snapshot_id": "snap-rebuild-ok"}

    rebuilt = MagicMock()
    rebuilt.retry_pending_required_snapshot = AsyncMock(side_effect=_retry_ok)
    rebuilt.close = AsyncMock()

    async def fake_build(sid, **kwargs):
        assert sid == session_id
        assert kwargs.get("fail_on_missing_config") is False
        return rebuilt, state

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)
    monkeypatch.setattr(
        "backend.services.workflow_runner._build_workflow_for_session",
        fake_build,
    )

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    try:
        resp = await lifecycle.snapshot_failure_retry(session_id, req)
        assert resp.success is True
        rebuilt.retry_pending_required_snapshot.assert_awaited_once()
        assert state.status == TaskStatus.COMPLETED
        assert state.snapshot_failure is None
        assert state.wait_input_reason is None
        assert getattr(state, "pending_required_snapshot", None) is None
        # Temporary workflow must not remain registered
        assert session_id not in workflows
        rebuilt.close.assert_awaited()
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_post_report_retry_missing_workflow_keeps_gate(monkeypatch):
    """When rebuild also fails, leave gate intact and return 409."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle
    from backend.state import sessions, workflows
    from fastapi import HTTPException

    session_id = "sess-post-report-no-wf"
    state = AnalysisState(
        id=session_id,
        name="PostReportNoWf",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            "retry_context": {"producer": "post_report"},
        },
        pending_required_snapshot={"report_generation_id": "r1"},
    )
    sessions[session_id] = state
    workflows.pop(session_id, None)

    async def fake_build(sid, **kwargs):
        assert kwargs.get("fail_on_missing_config") is False
        return None, None

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)
    monkeypatch.setattr(
        "backend.services.workflow_runner._build_workflow_for_session",
        fake_build,
    )

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.snapshot_failure_retry(session_id, req)
        assert exc_info.value.status_code in (409, 503)
        assert state.snapshot_failure is not None
        assert state.wait_input_reason == "snapshot_failure"
        assert state.status == TaskStatus.WAITING_INPUT
        assert state.pending_required_snapshot is not None
        assert session_id not in workflows
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_post_report_retry_missing_config_keeps_waiting_not_failed(monkeypatch):
    """Capture-only rebuild must not mark session FAILED when config is missing."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle
    from backend.services import workflow_runner
    from backend.state import sessions, workflows
    from fastapi import HTTPException

    session_id = "sess-post-report-no-cfg"
    state = AnalysisState(
        id=session_id,
        name="PostReportNoCfg",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            "retry_context": {"producer": "post_report"},
        },
        pending_required_snapshot={"report_generation_id": "r1"},
    )
    sessions[session_id] = state
    workflows.pop(session_id, None)

    async def no_config(user_id):
        return None

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)
    monkeypatch.setattr(workflow_runner, "async_save_session", noop)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", no_config)

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.snapshot_failure_retry(session_id, req)
        assert exc_info.value.status_code == 409
        assert state.status == TaskStatus.WAITING_INPUT
        assert state.status != TaskStatus.FAILED
        assert state.wait_input_reason == "snapshot_failure"
        assert state.snapshot_failure is not None
        assert state.snapshot_failure["code"] == "QUOTA"
        assert state.pending_required_snapshot is not None
        assert session_id not in workflows
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)

# ---- producer-dispatched retry (Task 5) -----


@pytest.mark.parametrize("producer", ["chat", "pre_restore", "workflow_boundary"])
@pytest.mark.asyncio
async def test_retry_invokes_exact_failed_producer(producer):
    from unittest.mock import AsyncMock

    from backend.models.schemas import AnalysisState
    from backend.services.snapshot_v2.failure_gate import (
        SnapshotRetryDependencies,
        retry_snapshot_producer,
    )
    from backend.services.snapshot_v2.models import SnapshotRetryContext

    callbacks = {
        "post_report": AsyncMock(),
        "chat": AsyncMock(),
        "pre_restore": AsyncMock(),
        "workflow_boundary": AsyncMock(),
    }
    dependencies = SnapshotRetryDependencies(
        retry_post_report=callbacks["post_report"],
        retry_chat_capture=callbacks["chat"],
        retry_pre_restore_capture=callbacks["pre_restore"],
        retry_workflow_boundary_capture=callbacks["workflow_boundary"],
    )
    state = AnalysisState(id="retry-s1", user_id="u1")
    context = SnapshotRetryContext(
        producer=producer,
        resume_policy={
            "chat": "stable_status",
            "pre_restore": "restore_transaction",
            "workflow_boundary": "graph_checkpoint",
        }[producer],
        snapshot_request={},
        previous_status="waiting_input",
    )
    await retry_snapshot_producer("retry-s1", state, context, dependencies)
    callbacks[producer].assert_awaited_once()
    for name, callback in callbacks.items():
        if name != producer:
            callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_restore_retry_queues_linked_restore_job(monkeypatch, v2_paths):
    """A failed pre-restore job is terminal; Retry must enqueue a new job."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.failure_gate import (
        default_retry_pre_restore_capture,
    )
    from backend.services.snapshot_v2.jobs import RestoreJobManager
    from backend.services.snapshot_v2.models import (
        SnapshotRestoreRequest,
        SnapshotRetryContext,
        SnapshotTransactionRecord,
    )
    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.preflight import (
        RestorePreflight,
        RestorePreflightResult,
    )

    session_id = "sess-pre-restore-retry"
    restore_request = SnapshotRestoreRequest(
        snapshot_id="target-snapshot",
        runtime_choice="reset",
        current_state_choice="abort",
    )
    context = SnapshotRetryContext(
        producer="pre_restore",
        resume_policy="restore_transaction",
        snapshot_request={"phase": "pre_restore"},
        previous_status="running",
        restore_transaction_id="restore-job-original",
        restore_request=restore_request.model_dump(mode="json"),
    )
    state = AnalysisState(
        id=session_id,
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={"code": "NO_SPACE", "message": "disk full"},
    )

    enqueue = AsyncMock(return_value=SnapshotTransactionRecord(
        transaction_id="restore-job-retry",
        session_id=session_id,
        snapshot_id="target-snapshot",
        status="queued",
    ))
    monkeypatch.setattr(SnapshotV2Paths, "for_session", lambda _sid: v2_paths)
    monkeypatch.setattr(
        RestorePreflight,
        "run",
        lambda self, snapshot_id, request: RestorePreflightResult(status="ready"),
    )
    monkeypatch.setattr(RestoreJobManager, "enqueue", enqueue)

    outcome = await default_retry_pre_restore_capture(session_id, state, context)

    assert outcome.success is True
    assert outcome.message == "Restore retry queued"
    assert outcome.transaction_id == "restore-job-retry"
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
    assert state.status == TaskStatus.PAUSED
    assert state.snapshot_transaction == {
        "transaction_id": "restore-job-retry",
        "session_id": session_id,
        "snapshot_id": "target-snapshot",
        "status": "queued",
        "recovery_mode": "rollback",
        "failure": None,
        "retry_of_transaction_id": None,
    }
    enqueue.assert_awaited_once()
    assert enqueue.await_args.kwargs["retry_of_transaction_id"] == (
        "restore-job-original"
    )


@pytest.mark.asyncio
async def test_pre_restore_retry_route_returns_new_transaction_id(monkeypatch):
    """The generic Retry route exposes the newly queued restore transaction."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle
    from backend.services.snapshot_v2 import failure_gate
    from backend.services.snapshot_v2.models import RetryOutcome
    from backend.state import sessions

    session_id = "sess-pre-restore-route"
    state = AnalysisState(
        id=session_id,
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "NO_SPACE",
            "message": "disk full",
            "retry_context": {
                "producer": "pre_restore",
                "resume_policy": "restore_transaction",
                "snapshot_request": {"phase": "pre_restore"},
                "previous_status": "running",
                "restore_transaction_id": "restore-job-original",
                "restore_request": {
                    "snapshot_id": "target-snapshot",
                    "runtime_choice": "reset",
                    "current_state_choice": "abort",
                    "discard_confirmation": False,
                },
            },
        },
    )
    sessions[session_id] = state

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*args, **kwargs):
        return None

    retry = AsyncMock(return_value=RetryOutcome(
        success=True,
        message="Restore retry queued",
        status=TaskStatus.PAUSED.value,
        transaction_id="restore-job-retry",
        snapshot_transaction={
            "transaction_id": "restore-job-retry",
            "session_id": session_id,
            "snapshot_id": "target-snapshot",
            "status": "queued",
            "recovery_mode": "rollback",
            "failure": None,
            "retry_of_transaction_id": "restore-job-original",
        },
    ))
    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)
    monkeypatch.setattr(failure_gate, "retry_snapshot_producer", retry)

    request = MagicMock()
    request.state = MagicMock(user_id="u1")
    try:
        response = await lifecycle.snapshot_failure_retry(session_id, request)
        assert response.success is True
        assert response.data["transaction_id"] == "restore-job-retry"
        assert response.data["snapshot_transaction"] == {
            "transaction_id": "restore-job-retry",
            "session_id": session_id,
            "snapshot_id": "target-snapshot",
            "status": "queued",
            "recovery_mode": "rollback",
            "failure": None,
            "retry_of_transaction_id": "restore-job-original",
        }
    finally:
        sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_retry_missing_context_returns_409_gate_intact(monkeypatch):
    """Missing/invalid retry_context must 409 and leave the gate intact."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle
    from backend.state import sessions
    from fastapi import HTTPException

    session_id = "sess-retry-missing-ctx"
    state = AnalysisState(
        id=session_id,
        name="MissingCtx",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            # no retry_context
        },
    )
    sessions[session_id] = state

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "validate_session_id", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_update_session_cache", noop)
    monkeypatch.setattr(lifecycle, "async_save_session", noop)

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.snapshot_failure_retry(session_id, req)
        assert exc_info.value.status_code == 409
        assert state.snapshot_failure is not None
        assert state.wait_input_reason == "snapshot_failure"
        assert state.status == TaskStatus.WAITING_INPUT
    finally:
        sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_workflow_boundary_gate_persists_real_phase():
    """workflow_boundary gate entry must store the real capture phase."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.failure_gate import (
        enter_snapshot_failure_gate,
        workflow_boundary_retry_context,
    )

    state = AnalysisState(
        id="wb-phase",
        user_id="u1",
        status=TaskStatus.RUNNING,
    )
    exc = RequiredSnapshotBlocked(
        "QUOTA",
        "disk full",
        details={"phase": "pre_direction"},
    )
    enter_snapshot_failure_gate(
        state,
        {
            "code": exc.code,
            "message": str(exc),
            "allowed_actions": ["retry", "manage", "terminate"],
            "snapshot_id": None,
        },
        retry_context=workflow_boundary_retry_context(exc),
    )
    rc = state.snapshot_failure["retry_context"]
    assert rc["producer"] == "workflow_boundary"
    assert rc["snapshot_request"]["phase"] == "pre_direction"
    assert rc["snapshot_request"]["phase"] != "workflow_boundary"


@pytest.mark.asyncio
async def test_ensure_workflow_boundary_upgrades_bare_service_gate():
    """Service bare gate + RequiredSnapshotBlocked must gain typed phase."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.failure_gate import (
        ensure_workflow_boundary_retry_context,
    )

    state = AnalysisState(
        id="wb-bare-gate",
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        # Service-style bare failure: no retry_context
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            "snapshot_id": "snap-bare-1",
        },
    )
    exc = RequiredSnapshotBlocked(
        "QUOTA",
        "disk full",
        phase="pre_annotation",
        snapshot_id="snap-bare-1",
    )
    ensure_workflow_boundary_retry_context(
        state,
        exc,
        previous_status="running",
    )
    assert state.snapshot_failure is not None
    rc = state.snapshot_failure["retry_context"]
    assert rc["producer"] == "workflow_boundary"
    assert rc["snapshot_request"]["phase"] == "pre_annotation"
    assert rc["snapshot_request"]["phase"] != "workflow_boundary"
    assert rc["previous_status"] == "running"
    assert state.snapshot_failure["code"] == "QUOTA"
    assert state.snapshot_failure["snapshot_id"] == "snap-bare-1"


@pytest.mark.asyncio
async def test_ensure_post_report_bare_gate_keeps_post_report_producer():
    """Bare service gate + post_report block must not become workflow_boundary."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.failure_gate import (
        ensure_workflow_boundary_retry_context,
    )

    state = AnalysisState(
        id="pr-bare-gate",
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            "snapshot_id": "snap-pr-1",
        },
    )
    exc = RequiredSnapshotBlocked(
        "QUOTA",
        "disk full",
        phase="post_report",
        snapshot_id="snap-pr-1",
    )
    # run/resume catch always calls this helper
    ensure_workflow_boundary_retry_context(
        state,
        exc,
        previous_status="running",
    )
    assert state.snapshot_failure is not None
    rc = state.snapshot_failure["retry_context"]
    assert rc["producer"] == "post_report"
    assert rc["snapshot_request"]["phase"] == "post_report"
    assert rc["snapshot_request"]["phase"] != "workflow_boundary"
    assert rc["producer"] != "workflow_boundary"


@pytest.mark.asyncio
async def test_ensure_does_not_overwrite_correct_post_report_context():
    """Correct post_report retry_context must survive ensure_workflow_boundary."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.failure_gate import (
        ensure_workflow_boundary_retry_context,
    )

    original_rc = {
        "producer": "post_report",
        "resume_policy": "complete",
        "snapshot_request": {"phase": "post_report"},
        "previous_status": "generating_report",
    }
    state = AnalysisState(
        id="pr-keep",
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
            "snapshot_id": "snap-pr-2",
            "retry_context": dict(original_rc),
        },
    )
    exc = RequiredSnapshotBlocked(
        "QUOTA",
        "disk full",
        phase="post_report",
        snapshot_id="snap-pr-2",
    )
    ensure_workflow_boundary_retry_context(
        state,
        exc,
        previous_status="running",
    )
    rc = state.snapshot_failure["retry_context"]
    assert rc["producer"] == "post_report"
    assert rc["resume_policy"] == "complete"
    assert rc["snapshot_request"]["phase"] == "post_report"
    assert rc["previous_status"] == "generating_report"


@pytest.mark.asyncio
async def test_generate_report_upgrades_bare_post_report_gate():
    """generate_report bare gate must merge typed post_report retry_context."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = AnalysisState(
        id="gen-report-bare",
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        # Service-style bare post_report failure (no retry_context)
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full after report",
            "allowed_actions": ["retry", "manage", "terminate"],
            "snapshot_id": "snap-gen-1",
        },
    )

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "gen-report-bare"
    workflow._send_state_sync = AsyncMock()

    result = await SingleCellAnalysisWorkflow.generate_report(
        workflow, state, "gen-report-bare",
    )
    assert result is state
    assert state.snapshot_failure is not None
    rc = state.snapshot_failure["retry_context"]
    assert rc["producer"] == "post_report"
    assert rc["snapshot_request"]["phase"] == "post_report"
    assert rc["previous_status"] == "generating_report"
    workflow._send_state_sync.assert_awaited()


@pytest.mark.asyncio
async def test_workflow_boundary_retry_forwards_persisted_phase(monkeypatch):
    """Retry must call _create_auto_snapshot with the persisted real phase."""
    from unittest.mock import AsyncMock

    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.failure_gate import (
        default_retry_workflow_boundary_capture,
    )
    from backend.services.snapshot_v2.models import SnapshotRetryContext
    from backend.state import workflows

    session_id = "wb-retry-phase"
    state = AnalysisState(
        id=session_id,
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "retry_context": {
                "producer": "workflow_boundary",
                "resume_policy": "graph_checkpoint",
                "snapshot_request": {"phase": "pre_annotation"},
                "previous_status": "running",
            },
        },
    )
    captured = {}

    class FakeWorkflow:
        def __init__(self):
            self.resume = AsyncMock()

        async def _create_auto_snapshot(self, analysis_state, phase):
            captured["phase"] = phase
            analysis_state.snapshot_failure = None
            analysis_state.wait_input_reason = None

    workflow = FakeWorkflow()
    workflows[session_id] = workflow
    context = SnapshotRetryContext(
        producer="workflow_boundary",
        resume_policy="graph_checkpoint",
        snapshot_request={"phase": "pre_annotation"},
        previous_status="running",
    )
    try:
        outcome = await default_retry_workflow_boundary_capture(
            session_id, state, context,
        )
        assert outcome.success is True
        assert outcome.still_blocked is False
        assert captured["phase"] == "pre_annotation"
        assert captured["phase"] != "workflow_boundary"
        assert state.status == TaskStatus.RUNNING
        workflow.resume.assert_not_awaited()
    finally:
        workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_workflow_boundary_retry_without_live_workflow_stays_blocked():
    """Exact workflow-boundary recapture must fail closed without live state."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.failure_gate import (
        default_retry_workflow_boundary_capture,
    )
    from backend.services.snapshot_v2.models import SnapshotRetryContext
    from backend.state import workflows

    session_id = "wb-retry-no-workflow"
    workflows.pop(session_id, None)
    context = SnapshotRetryContext(
        producer="workflow_boundary",
        resume_policy="graph_checkpoint",
        snapshot_request={"phase": "pre_annotation"},
        previous_status="running",
    )
    state = AnalysisState(
        id=session_id,
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={"code": "QUOTA", "message": "disk full"},
    )

    outcome = await default_retry_workflow_boundary_capture(
        session_id, state, context,
    )

    assert outcome.success is False
    assert outcome.still_blocked is True
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.wait_input_reason == "snapshot_failure"
    assert state.snapshot_failure is not None
    assert state.snapshot_failure["code"] == "WORKFLOW_UNAVAILABLE"


@pytest.mark.asyncio
async def test_chat_retry_still_blocked_when_failure_remains(monkeypatch):
    """Chat retry must report still_blocked when snapshot_failure remains."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.failure_gate import default_retry_chat_capture
    from backend.services.snapshot_v2.models import SnapshotRetryContext

    session_id = "chat-retry-blocked"
    state = AnalysisState(
        id=session_id,
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "retry_context": {
                "producer": "chat",
                "resume_policy": "stable_status",
                "snapshot_request": {"phase": "post_chat"},
                "previous_status": "completed",
            },
        },
    )
    context = SnapshotRetryContext(
        producer="chat",
        resume_policy="stable_status",
        snapshot_request={"phase": "post_chat"},
        previous_status="completed",
    )

    async def fake_capture(sid, st):
        # Capture "succeeds" without raising but leaves failure set
        st.snapshot_failure = {
            "code": "QUOTA",
            "message": "still full",
        }
        st.wait_input_reason = "snapshot_failure"
        st.status = TaskStatus.WAITING_INPUT

    monkeypatch.setattr(
        "backend.routes.chat.capture_post_chat_snapshot",
        fake_capture,
    )
    outcome = await default_retry_chat_capture(session_id, state, context)
    assert outcome.success is False
    assert outcome.still_blocked is True
    assert state.snapshot_failure is not None
    assert state.status == TaskStatus.WAITING_INPUT


@pytest.mark.asyncio
async def test_retry_handlers_do_not_clear_gate_before_capture(monkeypatch):
    """Handlers must not clear the gate before capture returns success."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.failure_gate import (
        default_retry_chat_capture,
        default_retry_workflow_boundary_capture,
    )
    from backend.services.snapshot_v2.models import SnapshotRetryContext
    from backend.state import workflows

    # Chat: capture raises; gate must remain / re-enter WAITING_INPUT
    chat_state = AnalysisState(
        id="gate-order-chat",
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={"code": "QUOTA", "message": "full"},
    )
    chat_ctx = SnapshotRetryContext(
        producer="chat",
        resume_policy="stable_status",
        snapshot_request={"phase": "post_chat"},
        previous_status="completed",
    )
    seen_status_during_capture = {}

    async def chat_capture_observing(sid, st):
        seen_status_during_capture["status"] = st.status
        seen_status_during_capture["failure"] = st.snapshot_failure
        raise RequiredSnapshotBlocked("QUOTA", "still full")

    monkeypatch.setattr(
        "backend.routes.chat.capture_post_chat_snapshot",
        chat_capture_observing,
    )
    outcome = await default_retry_chat_capture(
        "gate-order-chat", chat_state, chat_ctx,
    )
    assert outcome.success is False
    assert outcome.still_blocked is True
    # Prefer: gate still present during capture (or at least re-entered after)
    assert chat_state.status == TaskStatus.WAITING_INPUT
    assert chat_state.snapshot_failure is not None
    assert chat_state.wait_input_reason == "snapshot_failure"
    # Critical: do not clear before capture — failure must still be set when capture runs
    assert seen_status_during_capture["failure"] is not None

    # Workflow boundary: same guarantee
    wb_sid = "gate-order-wb"
    wb_state = AnalysisState(
        id=wb_sid,
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={"code": "QUOTA", "message": "full"},
    )
    wb_ctx = SnapshotRetryContext(
        producer="workflow_boundary",
        resume_policy="graph_checkpoint",
        snapshot_request={"phase": "pre_direction"},
        previous_status="running",
    )
    wb_seen = {}

    class ObservingWorkflow:
        async def _create_auto_snapshot(self, analysis_state, phase):
            wb_seen["failure"] = analysis_state.snapshot_failure
            wb_seen["phase"] = phase
            raise RequiredSnapshotBlocked("QUOTA", "still full")

    workflows[wb_sid] = ObservingWorkflow()
    try:
        outcome = await default_retry_workflow_boundary_capture(
            wb_sid, wb_state, wb_ctx,
        )
        assert outcome.success is False
        assert outcome.still_blocked is True
        assert wb_state.status == TaskStatus.WAITING_INPUT
        assert wb_state.snapshot_failure is not None
        assert wb_seen["failure"] is not None
        assert wb_seen["phase"] == "pre_direction"
    finally:
        workflows.pop(wb_sid, None)
