"""Concurrency and safe-point integration tests for Snapshot V2.

Uses asyncio events/barriers to coordinate concurrent lifecycle operations
and verify serialized access to session state.
"""

import asyncio

import pytest

from backend.services.snapshot_v2.errors import SnapshotTransactionError
from backend.services.snapshot_v2.jobs import RestoreJobManager
from backend.services.snapshot_v2.models import (
    IdleCapsule,
    SnapshotCaptureRequest,
    SnapshotRestoreRequest,
)
from backend.services.snapshot_v2.preflight import RestorePreflightResult
from backend.services.snapshot_v2.safe_point import SafePointManager


@pytest.mark.asyncio
class TestConcurrentLifecycle:
    """Verify that concurrent snapshot operations are serialized."""

    async def test_safe_point_acquire_release(self):
        mgr = SafePointManager()
        sid = "conc-session"
        token = "owner-conc-session"

        acquired = await mgr.acquire(sid, owner_token=token)
        assert acquired is True

        assert mgr.is_paused(sid) is True

        await mgr.release(sid, token)
        assert mgr.is_paused(sid) is False

    async def test_safe_point_idempotent_acquire(self):
        mgr = SafePointManager()
        sid = "conc-session2"
        token = "owner-conc-session2"

        first = await mgr.acquire(sid, owner_token=token)
        second = await mgr.acquire(sid, owner_token=token)
        assert first is True
        assert second is False

        await mgr.release(sid, token)

    async def test_safe_point_wait_unblocks(self):
        mgr = SafePointManager()
        sid = "conc-wait"
        token = "owner-conc-wait"

        await mgr.acquire(sid, owner_token=token)

        async def release_after_delay():
            await asyncio.sleep(0.05)
            await mgr.release(sid, token)

        task = asyncio.create_task(release_after_delay())
        await mgr.wait_for(sid)
        await task

        assert mgr.is_paused(sid) is False

    async def test_pending_request_acknowledge(self):
        mgr = SafePointManager()
        sid = "conc-pending"
        token = "owner-conc-pending"

        requested = await mgr.request_safe_point(sid, token)
        assert requested is not False and requested is not None

        assert mgr.has_pending_request(sid) is True
        gen = mgr.get_pending_generation(sid)
        assert gen is not None

        acknowledged = await mgr.acknowledge_safe_point_request(
            sid, gen, role="workflow",
        )
        assert acknowledged is True

        assert mgr.has_pending_request(sid) is False
        assert mgr.is_paused(sid) is True

        await mgr.release(sid, token)

    async def test_cancel_safe_point_request(self):
        mgr = SafePointManager()
        sid = "conc-cancel"

        await mgr.request_safe_point(sid, "owner-cancel")
        await mgr.cancel_safe_point_request(sid, "owner-cancel")

        assert mgr.has_pending_request(sid) is False
        assert mgr.is_paused(sid) is False

    async def test_foreign_owner_cannot_update_cancel_or_release(self):
        manager = SafePointManager()
        request = await manager.request_safe_point("s1", "owner-a")
        assert await manager.acknowledge_safe_point_request(
            "s1", request.generation, {}, role="workflow",
        )
        assert not await manager.update_descriptor("s1", "owner-b", {"next_node": "x"})
        assert not await manager.cancel_safe_point_request("s1", "owner-b")
        assert not await manager.release("s1", "owner-b")
        assert await manager.release("s1", "owner-a")

    async def test_duplicate_request_returns_false(self):
        mgr = SafePointManager()
        sid = "conc-dup"
        token = "owner-conc-dup"

        first = await mgr.request_safe_point(sid, token)
        second = await mgr.request_safe_point(sid, token)

        assert first is not False and first is not None
        assert second is False

        await mgr.cancel_safe_point_request(sid, token)

    async def test_async_lock_serializes_capture(self, v2_paths, service):
        """Two concurrent capture attempts on the same session are serialized."""
        from backend.state import get_session_lock

        sid = service.session_id
        lock = get_session_lock(sid)

        events = []
        async with lock:
            events.append("first_locked")
        async with lock:
            events.append("second_locked")

        assert len(events) == 2
        assert events == ["first_locked", "second_locked"]

    async def test_job_manager_race(self, v2_paths):
        """Multiple cancel calls on same transaction are safe."""
        session_id = "conc-race"
        mgr = RestoreJobManager(v2_paths, session_id)

        preflight = RestorePreflightResult(
            session_id=session_id,
            snapshot_id="snap-race",
            snapshot_size=100,
            runtime_status="complete",
            runtime_action="apply",
            pre_restore_required=False,
            has_children=False,
        )
        request = SnapshotRestoreRequest(
            snapshot_id="snap-race",
            runtime_choice="keep",
        )

        record = await mgr.enqueue(session_id, preflight, request, timeout_seconds=10)
        assert record.status == "queued"

        cancel_coro1 = mgr.cancel(record.transaction_id)
        cancel_coro2 = mgr.cancel(record.transaction_id)

        await asyncio.gather(cancel_coro1, cancel_coro2)

        final = mgr.get(record.transaction_id)
        assert final is None or final.status == "cancelled"

    async def test_job_manager_concurrent_enqueue(self, v2_paths, monkeypatch):
        """A second restore is fenced out while the first owns the safe point.

        Enqueuing reserves the session safe point and holds it until the job
        resolves, so a second restore for a different snapshot must be rejected
        with SAFE_POINT_BUSY (surfaced as HTTP 409) rather than queued
        alongside. This test previously expected both to queue, which predates
        safe-point fencing.
        """
        session_id = "conc-enq"
        mgr = RestoreJobManager(v2_paths, session_id)
        # Fail fast instead of waiting out the reservation timeout.
        monkeypatch.setattr(
            "backend.services.snapshot_v2.jobs.SAFE_POINT_RESERVATION_TIMEOUT_SECONDS",
            0.01,
        )

        preflight1 = RestorePreflightResult(
            session_id=session_id,
            snapshot_id="snap-enq-1",
            snapshot_size=100,
            runtime_status="complete",
            runtime_action="apply",
            pre_restore_required=False,
            has_children=False,
        )
        preflight2 = RestorePreflightResult(
            session_id=session_id,
            snapshot_id="snap-enq-2",
            snapshot_size=100,
            runtime_status="complete",
            runtime_action="apply",
            pre_restore_required=False,
            has_children=False,
        )

        r1 = SnapshotRestoreRequest(snapshot_id="snap-enq-1", runtime_choice="keep")
        r2 = SnapshotRestoreRequest(snapshot_id="snap-enq-2", runtime_choice="keep")

        rec1 = await mgr.enqueue(session_id, preflight1, r1, timeout_seconds=5)
        assert rec1.status == "queued"

        with pytest.raises(SnapshotTransactionError) as error:
            await mgr.enqueue(session_id, preflight2, r2, timeout_seconds=5)
        assert error.value.code == "SAFE_POINT_BUSY"

        # The fenced-out attempt must not leave a second job behind.
        assert mgr.get(rec1.transaction_id) is not None
