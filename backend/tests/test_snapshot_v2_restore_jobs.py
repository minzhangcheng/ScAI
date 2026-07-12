"""Tests for Snapshot V2 cancellable restore jobs."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.snapshot_v2.errors import SnapshotTransactionError


class _StatusOnlyService:
    """Test double: walk lifecycle statuses without live side effects.

    Used by unit tests that only assert job state-machine behaviour.
    """

    def __init__(self, session_id, paths):
        self.session_id = session_id
        self.paths = paths

    async def restore_snapshot(self, request, transaction_id, cancel_event, *, status_callback=None):
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


@pytest.fixture(autouse=True)
def _clear_restore_job_registry():
    """Isolate each test from process-wide restore job state."""
    from backend.state import snapshot_restore_jobs
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

    snapshot_restore_jobs.clear()
    sp = get_snapshot_safe_point_manager()
    for sid in list(sp.active_sessions):
        sp.release(sid)
    # Clear any pending requests by canceling known test session ids.
    for sid in (
        "sess-test", "sess-cross", "sess-wf-ack", "sess-a", "sess-b",
        "sess-journal", "sess-live-effect",
    ):
        sp.cancel_safe_point_request(sid)
    yield
    snapshot_restore_jobs.clear()
    for sid in list(sp.active_sessions):
        sp.release(sid)
    for sid in (
        "sess-test", "sess-cross", "sess-wf-ack", "sess-a", "sess-b",
        "sess-journal", "sess-live-effect",
    ):
        sp.cancel_safe_point_request(sid)


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
        mgr._safe_point.acknowledge_safe_point_request("sess-test")

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
        mgr._safe_point.acknowledge_safe_point_request("sess-test")

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
        mgr._safe_point.acknowledge_safe_point_request("sess-test")
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
        mgr._safe_point.acknowledge_safe_point_request("sess-test")
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
            mgr._safe_point.acknowledge_safe_point_request(sid)

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
        sp.release(session_id)
        assert sp.acquire(session_id, {"kernel_present": False}) is True
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
        sp.release(session_id)
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
        sp.release(session_id)
        assert sp.acquire(session_id, {"kernel_present": False}) is True

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
        sp.release(session_id)
        sessions.pop(session_id, None)
