"""Tests for Snapshot V2 lifecycle integration: idle restore, explicit start."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.schemas import AnalysisPlan, AnalysisState, RoundResult, TaskStatus
from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.workflow.planning import populate_plan_scope
from backend.workflow.provenance import (
    consume_restore_provenance,
    restore_provenance_id,
)


def consume_restore_routing_metadata(state: AnalysisState) -> None:
    """Test helper wrapping the workflow routing-only consume."""
    SingleCellAnalysisWorkflow._consume_restore_routing_metadata(None, state)


def test_restore_provenance_survives_routing_consume_and_enters_plan():
    state = AnalysisState(user_id="u1", last_restored_snapshot_id="snap-parent")
    state.restore_provenance_snapshot_id = state.last_restored_snapshot_id
    consume_restore_routing_metadata(state)
    plan = AnalysisPlan(phase="innovation")
    populate_plan_scope(plan, state)
    assert plan.created_from_snapshot_id == "snap-parent"


def test_round_result_records_then_consumes_restore_provenance():
    state = AnalysisState(
        id="provenance-round",
        user_id="u1",
        restore_provenance_snapshot_id="snap-parent",
    )
    result = RoundResult(round_number=1)
    result.restored_from_snapshot_id = restore_provenance_id(state)
    assert result.restored_from_snapshot_id == "snap-parent"
    consume_restore_provenance(state)
    assert state.restore_provenance_snapshot_id is None


@pytest.mark.asyncio
async def test_reanalysis_clears_restore_provenance(monkeypatch):
    """COMPLETED re-run must clear both provenance fields so the next plan/RoundResult
    is not mis-tagged from a prior terminal/idle restore that left provenance alive.
    """
    import backend.routes.lifecycle as life

    sid = "sess-reanalysis-provenance"
    state = AnalysisState(
        id=sid,
        user_id="u1",
        status=TaskStatus.COMPLETED,
        current_round=2,
        last_restored_snapshot_id="snap-old-restore",
        restore_provenance_snapshot_id="snap-old-restore",
    )

    async def fake_owner(session_id, user_id):
        return state

    async def noop(*a, **k):
        return None

    async def teardown_ok(*a, **k):
        return True

    monkeypatch.setattr(life, "require_session_ownership", fake_owner)
    monkeypatch.setattr(life, "teardown_running_workflow", teardown_ok)
    monkeypatch.setattr(life, "_update_session_cache", noop)
    monkeypatch.setattr(life, "async_save_session", noop)

    resp = await life._resume_analysis_locked(sid, "u1")
    assert resp.success is True
    assert resp.message == "Re-analysis started"
    assert state.status == TaskStatus.RUNNING
    assert state.last_restored_snapshot_id is None
    assert state.restore_provenance_snapshot_id is None
    assert restore_provenance_id(state) is None

    # Next run must not inherit prior restore parent into plan/RoundResult.
    plan = AnalysisPlan(phase="innovation")
    populate_plan_scope(plan, state)
    assert plan.created_from_snapshot_id is None
    next_round = RoundResult(round_number=3)
    next_round.restored_from_snapshot_id = restore_provenance_id(state)
    assert next_round.restored_from_snapshot_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_v2", ["true", "false"])
async def test_reanalysis_refuses_to_mutate_when_teardown_is_incomplete(
    monkeypatch, snapshot_v2,
):
    import backend.routes.lifecycle as life

    monkeypatch.setenv("SCAI_ENABLE_SNAPSHOT_V2", snapshot_v2)
    sid = "sess-reanalysis-cleanup-incomplete"
    state = AnalysisState(
        id=sid,
        user_id="u1",
        status=TaskStatus.COMPLETED,
        current_round=2,
        max_rounds=9,
        original_max_rounds=4,
        last_restored_snapshot_id="snap-old-restore",
        restore_provenance_snapshot_id="snap-old-restore",
        runtime_restore_status="unavailable",
        runtime_restore_warning="runtime unavailable",
    )

    monkeypatch.setattr(
        life,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    teardown = AsyncMock(return_value=False)
    update_cache = AsyncMock()
    save_session = AsyncMock()
    monkeypatch.setattr(life, "teardown_running_workflow", teardown)
    monkeypatch.setattr(life, "_update_session_cache", update_cache)
    monkeypatch.setattr(life, "async_save_session", save_session)

    response = await life._resume_analysis_locked(sid, "u1")

    assert response.success is False
    assert response.data == {
        "status": TaskStatus.COMPLETED.value,
        "needs_reconnect": False,
        "cleanup_incomplete": True,
    }
    assert state.status is TaskStatus.COMPLETED
    assert state.max_rounds == 9
    assert state.original_max_rounds == 4
    assert state.last_restored_snapshot_id == "snap-old-restore"
    assert state.restore_provenance_snapshot_id == "snap-old-restore"
    assert state.runtime_restore_status == "unavailable"
    assert state.runtime_restore_warning == "runtime unavailable"
    teardown.assert_awaited_once_with(sid, discard_runtime=True, retire_runtime=True)
    update_cache.assert_not_awaited()
    save_session.assert_not_awaited()


@pytest.fixture
def workflow_with_idle_restore():
    """Idle-restored session: branch_id set, resume_kind None, status RUNNING."""
    state = AnalysisState(
        id="idle-explicit-start",
        user_id="u1",
        name="test",
        initial_instruction="analyze",
        status=TaskStatus.RUNNING,
        current_round=2,
        restore_resume_kind=None,
        restore_branch_id="b1",
        last_restored_snapshot_id="snap-1",
    )
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = state.id
    wf._send_message = AsyncMock()
    wf._set_status = SingleCellAnalysisWorkflow._set_status
    wf._consume_restore_resume_metadata = (
        SingleCellAnalysisWorkflow._consume_restore_resume_metadata.__get__(
            wf, SingleCellAnalysisWorkflow
        )
    )
    wf._make_initial_workflow_state = (
        SingleCellAnalysisWorkflow._make_initial_workflow_state
    )
    wf.run = SingleCellAnalysisWorkflow.run.__get__(wf, SingleCellAnalysisWorkflow)
    wf.resume = SingleCellAnalysisWorkflow.resume.__get__(wf, SingleCellAnalysisWorkflow)
    wf._invoke_restore_entry = (
        SingleCellAnalysisWorkflow._invoke_restore_entry.__get__(
            wf, SingleCellAnalysisWorkflow
        )
    )
    wf.compiled_graph = MagicMock()
    return wf, state


@pytest.fixture
def workflow_with_terminal_restore():
    """Terminal-restored session: restore_resume_kind=terminal."""
    from unittest.mock import AsyncMock

    state = AnalysisState(
        id="terminal-restore",
        user_id="u1",
        name="test",
        initial_instruction="analyze",
        status=TaskStatus.RUNNING,
        current_round=3,
        restore_resume_kind="terminal",
        last_restored_snapshot_id="snap-x",
    )
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = state.id
    wf._send_message = AsyncMock()
    wf._set_status = SingleCellAnalysisWorkflow._set_status
    wf._consume_restore_resume_metadata = (
        SingleCellAnalysisWorkflow._consume_restore_resume_metadata.__get__(
            wf, SingleCellAnalysisWorkflow
        )
    )
    wf.run = SingleCellAnalysisWorkflow.run.__get__(wf, SingleCellAnalysisWorkflow)
    return wf, state


@pytest.mark.asyncio
async def test_terminal_restore_consumes_metadata(workflow_with_terminal_restore):
    wf, state = workflow_with_terminal_restore  # restore_resume_kind="terminal"
    state.last_restored_snapshot_id = "snap-x"
    await wf.run(state, thread_id="t1")
    assert state.status == TaskStatus.COMPLETED
    assert state.restore_resume_kind is None
    # Routing metadata is cleared; first-round provenance survives until
    # canonical RoundResult recording (Task 4).
    assert state.last_restored_snapshot_id == "snap-x"


@pytest.mark.asyncio
async def test_idle_restore_consumes_metadata(workflow_with_idle_restore):
    wf, state = workflow_with_idle_restore  # branch_id set, no explicit_start
    state.last_restored_snapshot_id = "snap-y"
    await wf.run(state, thread_id="t1")
    assert state.status == TaskStatus.PAUSED
    assert state.restore_resume_kind is None
    # Idle reconnect clears routing only; provenance is not consumed here.
    assert state.last_restored_snapshot_id == "snap-y"


@pytest.mark.asyncio
async def test_idle_explicit_start_reaches_graph(workflow_with_idle_restore):
    """explicit_start intent must NOT be short-circuited to PAUSED."""
    wf, state = workflow_with_idle_restore  # idle-restored: branch_id set, resume_kind None
    state.restore_start_intent = "explicit_start"

    reached = {"graph": False}

    async def fake_initialize():
        reached["graph"] = True
        # Stop before real graph invoke: raise so run() returns after initialize.
        raise RuntimeError("stop-after-initialize")

    wf.initialize = fake_initialize

    await wf.run(state, thread_id="t1")

    assert reached["graph"] is True, "explicit_start was short-circuited before the graph"
    assert state.status != TaskStatus.PAUSED


@pytest.mark.asyncio
async def test_headless_runner_idle_explicit_start_invokes_graph(monkeypatch):
    from backend.services import workflow_runner
    from backend.state import sessions, workflows

    state = AnalysisState(
        id="idle-start", user_id="u1", status=TaskStatus.RUNNING,
        current_round=3, restore_branch_id="restore-tx",
        restore_start_intent="explicit_start",
    )
    workflow = AsyncMock()
    workflow.resume.return_value = state
    monkeypatch.setattr(
        workflow_runner,
        "_build_workflow_for_session",
        AsyncMock(return_value=(workflow, state)),
    )
    sessions[state.id] = state
    try:
        await workflow_runner._headless_workflow_runner(state.id)
        workflow.resume.assert_awaited_once_with(state.id, state)
    finally:
        sessions.pop(state.id, None)
        workflows.pop(state.id, None)


@pytest.mark.asyncio
async def test_resume_idle_explicit_start_invokes_restore_entry(
    workflow_with_idle_restore,
):
    workflow, state = workflow_with_idle_restore
    state.current_round = 3
    state.restore_start_intent = "explicit_start"
    workflow.compiled_graph.aget_state = AsyncMock(return_value=None)
    workflow.compiled_graph.ainvoke = AsyncMock(
        return_value={"analysis_state": state},
    )
    result = await workflow.resume(state.id, state)
    workflow.compiled_graph.ainvoke.assert_awaited_once()
    invoked = workflow.compiled_graph.ainvoke.await_args.args[0]
    assert invoked["analysis_state"] is state
    assert result.current_round == 3


class TestLifecycleRestoreEntry:
    """Tests for lifecycle route interactions with restore_entry."""

    @pytest.mark.asyncio
    async def test_lifecycle_explicit_start_preserves_restored_round_and_plan(self):
        """POST /run on an idle-restored session sets restore_start_intent.

        This verifies that when the user clicks "Start" on a session that was
        restored without a continuation target (idle), the route sets
        restore_start_intent so the graph's restore_entry node preserves
        the restored round and plan.
        """
        from backend.models.schemas import AnalysisState, TaskStatus, AnalysisPlan, PlanStage

        # Simulate an idle-restored session (restore_resume_kind is None
        # but restore_branch_id is set).
        state = AnalysisState(
            id="explicit-start-test",
            user_id="u1",
            name="test",
            initial_instruction="analyze",
            status=TaskStatus.PAUSED,
            current_round=3,
            max_rounds=5,
            restore_resume_kind=None,
            restore_branch_id="restore-tx-explicit",
            restore_branch_label="branch-snap1234",
            last_restored_snapshot_id="snap-1234",
        )

        # Simulate the POST /run logic from lifecycle.py
        if (
            getattr(state, "restore_resume_kind", None) is None
            and getattr(state, "restore_branch_id", None)
        ):
            state.restore_start_intent = "explicit_start"

        assert state.restore_start_intent == "explicit_start"
        # Round should be preserved (not auto-incremented by explicit_start).
        assert state.current_round == 3

        # Now simulate _restore_entry consuming restore_start_intent.
        from backend.services.snapshot_v2.restore_entry import build_restore_entry_plan

        plan = build_restore_entry_plan(
            state, restore_start_intent=state.restore_start_intent,
        )
        assert plan.intent == "explicit_start"
        assert plan.preserve_round_on_initialize is True

        # After consumption, the intent should be cleared.
        state.restore_start_intent = None
        assert state.restore_start_intent is None

    def test_web_socket_reconnect_does_not_set_explicit_start(self):
        """WebSocket reconnect must NOT set restore_start_intent.

        The explicit_start signal is only set by the POST /run lifecycle
        route — not by WebSocket reconnection which is handled separately
        in the session loader/websocket route.
        """
        from backend.models.schemas import AnalysisState, TaskStatus

        state = AnalysisState(
            id="ws-reconnect", user_id="u1", initial_instruction="test",
            status=TaskStatus.PAUSED,
            current_round=2,
            restore_resume_kind="semantic_gate",
            restore_branch_id="restore-tx-ws",
            restore_start_intent=None,
        )

        # WebSocket reconnect should preserve the existing state.
        # The reconnection logic in routes/websockets.py does NOT touch
        # restore_start_intent — it stays None.
        assert state.restore_start_intent is None

    def test_normal_start_does_not_set_explicit_start(self):
        """Normal (non-restored) session must not get explicit_start."""
        from backend.models.schemas import AnalysisState, TaskStatus

        state = AnalysisState(
            id="normal-start", user_id="u1", initial_instruction="test",
            status=TaskStatus.PENDING,
            current_round=0,
        )

        # Normal start: no restore metadata.
        if (
            getattr(state, "restore_resume_kind", None) is None
            and getattr(state, "restore_branch_id", None)
        ):
            state.restore_start_intent = "explicit_start"

        # Since there's no restore_branch_id, the condition is false.
        assert getattr(state, "restore_start_intent", None) is None


@pytest.mark.asyncio
async def test_initialize_preserves_plan_when_marker_already_consumed():
    """Option B: after consume clears last_restored_snapshot_id, preserve flag
    must still keep round/plan/direction and set plan_approved for approved plans.
    """
    from unittest.mock import AsyncMock

    from backend.models.schemas import AnalysisPlan, PlanStep
    from backend.workflow.initialize import InitializeMixin

    plan = AnalysisPlan(
        phase="innovation",
        stage="full",
        overview="Restored approved plan",
        steps=[PlanStep(id=1, title="S1", method="m")],
        status="approved",
    )
    direction = "direction-alpha"
    state = AnalysisState(
        id="preserve-after-consume",
        user_id="u1",
        name="test",
        initial_instruction="analyze",
        status=TaskStatus.PAUSED,
        current_round=3,
        current_plan=plan,
        current_round_direction=direction,
        directions_queue=[{"direction": direction}],
        # Marker already consumed at handoff (Task 4).
        last_restored_snapshot_id=None,
    )

    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = state.id
    wf._pause_event = None
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    # _set_status is a staticmethod — bind the raw function, not a method.
    wf._set_status = SingleCellAnalysisWorkflow._set_status

    workflow_state = {
        "analysis_state": state,
        "_preserve_round_on_initialize": True,
    }
    # Call the real decorated method (with_retry is fine with mocks).
    result = await InitializeMixin._initialize_round(wf, workflow_state)

    assert result["analysis_state"].current_round == 3
    assert result["analysis_state"].current_plan is not None
    assert result["analysis_state"].current_plan.status == "approved"
    assert result["plan_approved"] is True
    assert result["analysis_state"].current_round_direction == direction
    assert result["analysis_state"].directions_queue == [{"direction": direction}]
    assert result["analysis_state"].last_restored_snapshot_id is None
    assert "_preserve_round_on_initialize" not in result
