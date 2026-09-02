"""Regression tests for the safe-point concurrency fabric (0.36.12 follow-up).

Every gate, producer and coordinator in this group has the same invariant: a
restore reservation is a *handover* that some live participant must complete.
Whoever is holding the session when the reservation is filed owns that duty, and
if it neither acknowledges nor notices that it has become idle, the reservation
is orphaned — the restore burns its full timeout and every other snapshot
operation on the session answers SAFE_POINT_BUSY meanwhile.

One test per defect, named for the failure it prevents. Each was verified to
fail against the pre-fix source.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
from backend.state import snapshot_restore_jobs


# ── shared minimal fixtures ──────────────────────────────────────────────────


class _Paths:
    def __init__(self, root: Path):
        self.root = root / "v2"
        self.objects = root / "objects"
        self.staging = root / "staging"
        self.journals = root / "journals"
        self.integrity = root / "integrity"
        self.committed = root / "committed"

    def ensure_dirs(self):
        for d in (
            self.root, self.objects, self.staging,
            self.journals, self.integrity, self.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def v2_paths(tmp_path):
    paths = _Paths(tmp_path)
    paths.ensure_dirs()
    return paths


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


class _StatusOnlyService:
    """Test double: walk the restore lifecycle without live side effects."""

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
        expected_lifecycle_generation=None,
    ):
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord

        return SnapshotTransactionRecord(
            transaction_id=transaction_id,
            session_id=self.session_id,
            snapshot_id=request.snapshot_id,
            status="committed",
        )


def _make_mgr(v2_paths, session_id):
    from backend.services.snapshot_v2.jobs import RestoreJobManager

    return RestoreJobManager(
        paths=v2_paths,
        session_id=session_id,
        snapshot_service=_StatusOnlyService(session_id, v2_paths),
    )


async def _wait_for_status(manager, tx_id, status, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        record = manager.get(tx_id)
        if record is not None and record.status == status:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"restore job {tx_id} never reached {status!r} "
        f"(last status: {getattr(manager.get(tx_id), 'status', None)!r})"
    )


@pytest_asyncio.fixture(autouse=True)
async def _clean_safe_point_registry():
    """Restore jobs and safe points are process-wide; leave nothing behind."""
    manager = get_snapshot_safe_point_manager()
    yield
    for handle in list(snapshot_restore_jobs.values()):
        if handle.task is not None and not handle.task.done():
            handle.cancel_event.set()
            handle.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.task
    snapshot_restore_jobs.clear()
    for session_id in set(manager.active_sessions) | set(manager._pending_requests):
        pending = manager.get_pending_observation(session_id)
        owner = (
            pending.owner_token
            if pending is not None
            else manager._acquired_owners.get(session_id)
        )
        if owner:
            await manager.cancel_safe_point_request(session_id, owner)
        manager.clear_superseded(session_id)


# ── F29. The shared HITL scaffold hands the safe point over ──────────────────
#
# `_drive_hitl_gate` is the seventh HITL gate: it backs the LLM-failure,
# disabled-agent and marker-DB dialogs. Like every other gate it blocks *inside*
# a graph node, so the run task stays live and the graph never reaches another
# node entry — the only other place the cooperative safe point is acknowledged.


@pytest.mark.asyncio
async def test_shared_hitl_scaffold_hands_the_safe_point_to_a_pending_restore(
    monkeypatch,
):
    from backend.constants import LlmFailureAction
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.workflow import helpers
    from backend.workflow.core import SingleCellAnalysisWorkflow

    monkeypatch.setattr(helpers, "HITL_SAFE_POINT_POLL_SECONDS", 0.02)

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "sess-hitl-scaffold"
    workflow._llm_failure_lock = asyncio.Lock()
    workflow._llm_failure_events = {}
    workflow._llm_failure_generations = {}
    workflow._unattended_gate_retries = {}
    workflow._llm_failure_actions = {}
    workflow._settings_provider = None
    workflow._set_status = lambda state, status: setattr(state, "status", status)

    async def _send_state_sync(*_a, **_kw):
        return None

    workflow._send_state_sync = _send_state_sync

    acknowledged = asyncio.Event()

    async def _await_restore():
        acknowledged.set()

    workflow.await_restore_safe_point = _await_restore

    analysis_state = AnalysisState(
        session_id=workflow.session_id, user_id="u1", status=TaskStatus.RUNNING,
    )

    gate = asyncio.ensure_future(
        SingleCellAnalysisWorkflow._drive_hitl_gate(
            workflow,
            "pi_synthesis",
            analysis_state,
            reason="llm_failure",
            info_attr="llm_failure_info",
            info_payload={"agent": "pi"},
            enter_change="llm_failure",
            exit_change="llm_retry",
            clear_cache_on_retry=False,
        )
    )
    try:
        await asyncio.wait_for(acknowledged.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        gate.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate
        pytest.fail(
            "the shared HITL scaffold never invoked the safe-point hook: a "
            "restore requested at the LLM-failure / disabled-agent / marker-DB "
            "dialog has no acknowledger and can only time out"
        )

    # The gate must still be a gate: the hook does not resolve it.
    assert not gate.done()
    workflow._llm_failure_actions["pi_synthesis"] = LlmFailureAction.SKIP.value
    workflow._llm_failure_events["pi_synthesis"].set()
    assert await asyncio.wait_for(gate, timeout=2.0) == LlmFailureAction.SKIP
    assert analysis_state.wait_input_reason is None


def test_shared_hitl_scaffold_has_no_bare_event_wait():
    """A gate that blocks on a bare event reintroduces the deadlock silently.

    The sibling guard in test_snapshot_v2_edge_hardening.py counts
    ``wait_with_skip_time`` calls against hooks, which cannot see a scaffold
    that calls neither — 0 == 0 passes. Assert on the bare wait instead.
    """
    from backend.workflow.core_parts import interventions

    source = inspect.getsource(interventions._drive_hitl_gate)
    assert "await event.wait()" not in source, (
        "_drive_hitl_gate blocks on a bare event.wait(); a restore requested "
        "while the user sits at this gate cannot be acknowledged"
    )
    assert 'safe_point_hook=getattr(self, "await_restore_safe_point"' in source


# ── F08. Quiescence is re-probed while the restore job waits ─────────────────


@pytest.mark.asyncio
async def test_restore_acknowledges_a_session_that_goes_idle_while_waiting(
    monkeypatch, v2_paths, preflight_result, restore_request,
):
    """Quiescence proven once at enqueue is not proof for the whole wait.

    A chat delegation is live when the restore is filed, so
    ``_acknowledge_if_quiescent`` declines — but the chat path contains no
    acknowledgement site at all. When it ends the session is completely idle and
    nobody is left to hand the reservation over.
    """
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2 import jobs as jobs_mod
    from backend.state import _run_tasks, chat_tasks, sessions, workflows

    monkeypatch.setattr(jobs_mod, "QUIESCENCE_RECHECK_SECONDS", 0.05, raising=False)

    session_id = "sess-goes-idle-while-waiting"
    sessions[session_id] = AnalysisState(
        session_id=session_id,
        user_id="u1",
        status=TaskStatus.RUNNING,
        current_round=2,
    )
    workflows.pop(session_id, None)
    _run_tasks.pop(session_id, None)

    release = asyncio.Event()
    chat_tasks[session_id] = asyncio.ensure_future(release.wait())
    try:
        manager = _make_mgr(v2_paths, session_id)
        record = await manager.enqueue(
            session_id, preflight_result, restore_request, timeout_seconds=30,
        )
        await asyncio.sleep(0)
        assert manager._safe_point.has_pending_request(session_id), (
            "precondition: the live chat task must make the enqueue-time "
            "quiescence probe decline"
        )

        release.set()
        await chat_tasks.pop(session_id)

        await _wait_for_status(manager, record.transaction_id, "committed")
        assert not manager._safe_point.has_pending_request(session_id)
    finally:
        task = chat_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_restore_never_acknowledges_under_a_live_run_task(
    monkeypatch, v2_paths, preflight_result, restore_request,
):
    """The re-probe must not degrade into acknowledging a running graph."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2 import jobs as jobs_mod
    from backend.state import _run_tasks, sessions, workflows

    monkeypatch.setattr(jobs_mod, "QUIESCENCE_RECHECK_SECONDS", 0.05, raising=False)

    session_id = "sess-live-run-never-acked"
    sessions[session_id] = AnalysisState(
        session_id=session_id, user_id="u1", status=TaskStatus.RUNNING,
    )
    workflows.pop(session_id, None)
    running = asyncio.Event()
    _run_tasks[session_id] = asyncio.ensure_future(running.wait())
    try:
        manager = _make_mgr(v2_paths, session_id)
        record = await manager.enqueue(
            session_id, preflight_result, restore_request, timeout_seconds=30,
        )
        await asyncio.sleep(0.4)  # several recheck intervals

        assert manager._safe_point.has_pending_request(session_id)
        assert manager.get(record.transaction_id).status == "waiting_safe_point"
        await manager.cancel(record.transaction_id)
    finally:
        running.set()
        task = _run_tasks.pop(session_id, None)
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        sessions.pop(session_id, None)


# ── F09. The post-report capture hands a pending reservation over ────────────


class _Handover(Exception):
    """Sentinel: raised from the hook to prove control reached the handover."""


def _post_report_workflow(session_id, *, pending, kernel):
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.workflow.core import SingleCellAnalysisWorkflow

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = session_id
    workflow.sandbox = MagicMock(km=object()) if kernel else None

    safe_point = MagicMock()
    safe_point.is_paused.return_value = False
    safe_point.has_pending_request.return_value = pending
    safe_point.is_eligibility_trusted.return_value = True
    service = MagicMock()
    service.safe_point = safe_point

    async def _capture_required_snapshot(*_a, **_kw):
        # What the real acquire does: `_acquire_unlocked` treats a pending
        # request exactly like an acquired hold and refuses.
        if safe_point.has_pending_request(session_id):
            raise RequiredSnapshotBlocked(
                "SAFE_POINT_BUSY", "safe point reserved by a restore job",
            )
        return MagicMock(snapshot_id="snap-post-report")

    service.capture_required_snapshot = _capture_required_snapshot
    workflow.snapshot_service = service
    return workflow


@pytest.mark.parametrize("kernel", [False, True])
@pytest.mark.asyncio
async def test_post_report_capture_hands_over_a_pending_safe_point(kernel):
    """Report nodes contain no other acknowledgement site.

    With no live kernel the capture refuses outright ("cannot steal an existing
    safe-point hold"); with a live kernel it skips the guard entirely and loses
    the acquire two frames down. Either way the reservation it refuses is one
    only this producer could have completed.
    """
    from backend.models.schemas import AnalysisState
    from backend.workflow.core import SingleCellAnalysisWorkflow

    workflow = _post_report_workflow(
        "sess-post-report-handover", pending=True, kernel=kernel,
    )

    async def _await_restore():
        raise _Handover()

    workflow.await_restore_safe_point = _await_restore

    analysis_state = AnalysisState(
        session_id=workflow.session_id, user_id="u1", current_round=1,
    )
    with pytest.raises(_Handover):
        await SingleCellAnalysisWorkflow._capture_post_report_snapshot(
            workflow, analysis_state, "gen-1",
        )


@pytest.mark.asyncio
async def test_post_report_capture_does_not_wait_when_nothing_is_pending():
    """An already-held foreign hold must keep failing closed."""
    from backend.models.schemas import AnalysisState
    from backend.workflow.core import SingleCellAnalysisWorkflow

    workflow = _post_report_workflow(
        "sess-post-report-no-pending", pending=False, kernel=False,
    )
    called = []

    async def _await_restore():
        called.append(True)

    workflow.await_restore_safe_point = _await_restore

    analysis_state = AnalysisState(
        session_id=workflow.session_id, user_id="u1", current_round=1,
    )
    with contextlib.suppress(Exception):
        await SingleCellAnalysisWorkflow._capture_post_report_snapshot(
            workflow, analysis_state, "gen-1",
        )
    assert called == [], (
        "the handover must be conditional on a pending request; waiting on an "
        "acquired foreign hold turns a visible gate into an unbounded wait"
    )


# ── F10. A superseded report-only run unwinds instead of overwriting ─────────


@pytest.mark.asyncio
async def test_generate_report_unwinds_when_a_restore_supersedes_it():
    """`_pi_synthesis` opens with the safe-point boundary hook, so a committed
    restore unwinds through the checkpointer-load block. The generic handler
    there mislabels it as "could not load state from checkpointer" and keeps
    generating — reports, and a post-report snapshot, built from the state the
    restore just replaced.
    """
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.workflow.core import SingleCellAnalysisWorkflow
    from backend.workflow.helpers import WorkflowSupersededError
    from backend.state import sessions

    session_id = "sess-report-superseded"
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = session_id

    generated = []

    async def _noop(*_a, **_kw):
        return None

    workflow.initialize = _noop
    workflow._send_message = _noop
    workflow._send_state_sync = _noop
    workflow._set_status = lambda state, status: setattr(state, "status", status)

    async def _pi_synthesis(_state):
        raise WorkflowSupersededError("snapshot restore replaced this run")

    workflow._pi_synthesis = _pi_synthesis

    async def _generate_final_report(state):
        generated.append("final")
        return state

    workflow._generate_final_report = _generate_final_report
    workflow._generate_graphic_abstract = _generate_final_report
    workflow._generate_detailed_report = _generate_final_report

    class _Graph:
        async def aget_state(self, _config):
            return MagicMock(values={"analysis_state": None})

    workflow.compiled_graph = _Graph()

    stale = AnalysisState(
        session_id=session_id,
        user_id="u1",
        status=TaskStatus.GENERATING_REPORT,
        current_round=2,
    )
    restored = AnalysisState(
        session_id=session_id,
        user_id="u1",
        status=TaskStatus.WAITING_INPUT,
        current_round=1,
    )
    sessions[session_id] = restored
    sp = get_snapshot_safe_point_manager()
    sp.mark_superseded(session_id)
    try:
        result = await SingleCellAnalysisWorkflow.generate_report(
            workflow, stale, session_id,
        )
    finally:
        sessions.pop(session_id, None)
        sp.clear_superseded(session_id)

    assert generated == [], (
        "report generation continued after the restore superseded this run; "
        "the reports and the post-report snapshot would be built from the "
        "pre-restore state"
    )
    assert result is restored, (
        "generate_report must return the state the restore installed, never "
        "the pre-restore object it was invoked with"
    )
    assert stale.status == TaskStatus.GENERATING_REPORT, (
        "WorkflowSupersededError must not touch session state"
    )


@pytest.mark.asyncio
async def test_report_state_callback_refuses_to_write_over_a_restored_session(
    monkeypatch,
):
    """The headless report runner's callback has no is-current guard.

    The WebSocket runner keys its writes on workflow identity
    (`_write_state_if_current`); the report-only workflow is never registered in
    `workflows`, so it has no such anchor and used to persist its pre-restore
    state straight over the object the restore installed.
    """
    from types import SimpleNamespace

    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services import workflow_runner
    import backend.workflow as workflow_module
    from backend.state import sessions

    session_id = "sess-report-callback-guard"
    captured = {}

    class _CapturingWorkflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            captured["state_callback"] = state_callback

    saved = []
    cached = []

    async def _fake_save_session(sid, state):
        saved.append((sid, state))

    async def _fake_update_session_cache(sid, state):
        cached.append((sid, state))
        sessions[sid] = state

    async def _fake_resolve_config_for_user(_user_id):
        return SimpleNamespace(max_rounds=3)

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _CapturingWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _fake_resolve_config_for_user)
    monkeypatch.setattr(workflow_runner, "async_save_session", _fake_save_session)
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _fake_update_session_cache)

    stale = AnalysisState(
        session_id=session_id, user_id="u1", status=TaskStatus.GENERATING_REPORT,
    )
    sessions[session_id] = stale
    try:
        workflow, state = await workflow_runner._build_workflow_for_session(session_id)
        assert state is stale

        restored = AnalysisState(
            session_id=session_id, user_id="u1", status=TaskStatus.WAITING_INPUT,
        )
        sessions[session_id] = restored

        await captured["state_callback"](stale)

        assert sessions[session_id] is restored, (
            "the report run wrote its pre-restore state back over the session "
            "the restore just installed"
        )
        assert saved == [], "the stale state was persisted to disk"
    finally:
        sessions.pop(session_id, None)


# ── F11. The report-only runner closes the workflow it builds ────────────────


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.asyncio
async def test_report_generation_closes_its_workflow(monkeypatch, raises):
    """`close()` is what compacts and closes the checkpointer connection.

    Leaving it open keeps a WAL connection on a database file that a concurrent
    restore replaces by rename — and the workflow is never registered in
    `workflows`, so `_retire_superseded_workflow` cannot find it either.
    """
    from backend.services import workflow_runner

    closed = []

    class _StubWorkflow:
        def __init__(self):
            self._db_conn = object()

        async def generate_report(self, _state, _thread_id):
            if raises:
                raise RuntimeError("report generation blew up")

        async def close(self, *, save_runtime=True):
            closed.append(True)
            self._db_conn = None

    workflow = _StubWorkflow()

    session_id = "sess-report-close"
    state = MagicMock()

    async def _fake_build(session_id, **_kw):
        return workflow, state

    monkeypatch.setattr(workflow_runner, "_build_workflow_for_session", _fake_build)
    workflow_runner.sessions[session_id] = state
    try:
        await workflow_runner._run_report_generation(session_id)
    finally:
        workflow_runner.sessions.pop(session_id, None)
        workflow_runner.report_workflows.pop(session_id, None)

    assert closed == [True], (
        "the report-only runner leaked its workflow: the aiosqlite WAL "
        "connection, the LLM clients and the settings provider stay open"
    )
    assert workflow._db_conn is None


@pytest.mark.asyncio
async def test_report_generation_retains_incomplete_workflow_for_deletion(
    monkeypatch,
):
    from backend.services import workflow_runner

    session_id = "sess-report-close-incomplete"
    state = MagicMock()

    class _StubWorkflow:
        async def generate_report(self, _state, _thread_id):
            return None

        async def close(self, *, save_runtime=True):
            return False

    workflow = _StubWorkflow()

    async def _fake_build(_session_id, **_kw):
        return workflow, state

    monkeypatch.setattr(workflow_runner, "_build_workflow_for_session", _fake_build)
    workflow_runner.sessions[session_id] = state
    try:
        await workflow_runner._run_report_generation(session_id)
        assert workflow_runner.report_workflows[session_id] is workflow
    finally:
        workflow_runner.sessions.pop(session_id, None)
        workflow_runner.report_workflows.pop(session_id, None)
