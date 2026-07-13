from backend.workflow.restore_targets import SNAPSHOT_RESUME_TARGET, stable_restore_branch_id, stable_restore_branch_label


def test_pre_round_wait_maps_to_round_wait_semantic_gate():
    target = SNAPSHOT_RESUME_TARGET["pre_round_wait"]
    assert target == {"kind": "semantic_gate", "gate": "round_wait", "node": "wait_for_user_input"}


def test_pre_annotation_gate_is_not_graph_node():
    target = SNAPSHOT_RESUME_TARGET["pre_annotation"]
    assert target["kind"] == "semantic_gate"
    assert target["gate"] == "annotation_confirmation"
    assert target["node"] is None


def test_post_report_is_terminal():
    target = SNAPSHOT_RESUME_TARGET["post_report"]
    assert target["kind"] == "terminal"
    assert target["gate"] is None
    assert target["node"] is None


def test_stable_restore_branch_id_uses_snapshot_id_not_label():
    assert stable_restore_branch_id("snapshot-123", "branch-A") == "restore-snapshot-123"
    assert stable_restore_branch_id("snapshot-123", "branch-A") == stable_restore_branch_id("snapshot-123", "branch-B")


def test_stable_restore_branch_label_uses_snapshot_id_when_no_label_exists():
    assert stable_restore_branch_label("snapshot-1234567890", None) == "branch-snapshot"


import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.models.schemas import AnalysisPhase, AnalysisState, TaskStatus
from backend.workflow.core import SingleCellAnalysisWorkflow


class _FakeCompiledGraph:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, workflow_state, config):
        self.calls.append((workflow_state, config))
        state = workflow_state["analysis_state"]
        # Routing metadata is consumed before graph continue; provenance may
        # still be present until the first restored RoundResult is recorded.
        assert state.restore_resume_kind is None
        state.current_round += 1
        return {"analysis_state": state}


def _workflow(max_rounds=4):
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "s"
    workflow.config = SimpleNamespace(max_rounds=max_rounds)
    workflow.compiled_graph = _FakeCompiledGraph()
    workflow._current_analysis_state = None
    workflow._send_message = AsyncMock()
    workflow._send_state_sync = AsyncMock()
    return workflow


@pytest.mark.asyncio
async def test_resume_restored_wait_state_clears_restore_flags_before_next_round(monkeypatch):
    workflow = _workflow(max_rounds=4)
    state = AnalysisState(
        id="s",
        user_id="u",
        status=TaskStatus.WAITING_INPUT,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_round=3,
        max_rounds=4,
        last_restored_snapshot_id="snap-1",
        restore_resume_kind="semantic_gate",
        restore_resume_gate="round_wait",
        restore_resume_node="wait_for_user_input",
        restore_branch_id="restore-snap-1",
    )

    async def fake_check_convergence(graph_state):
        graph_state["should_continue"] = True
        return graph_state

    monkeypatch.setattr(workflow, "_check_convergence", fake_check_convergence)
    monkeypatch.setattr(workflow, "_check_convergence_condition", lambda graph_state: "continue")

    result = await workflow.resume_restored_wait_state(state)

    assert len(workflow.compiled_graph.calls) == 1
    assert result.current_round == 4
    assert result.restore_resume_kind is None
    # Provenance survives wait-state resume until canonical round recording.
    assert result.last_restored_snapshot_id == "snap-1"


@pytest.mark.asyncio
async def test_resume_restored_wait_state_routes_to_report_when_max_round_reached(monkeypatch):
    workflow = _workflow(max_rounds=3)
    state = AnalysisState(
        id="s",
        user_id="u",
        status=TaskStatus.WAITING_INPUT,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_round=3,
        max_rounds=3,
        last_restored_snapshot_id="snap-1",
        restore_resume_kind="semantic_gate",
        restore_resume_gate="round_wait",
        restore_resume_node="wait_for_user_input",
    )
    called = {"report": False}

    async def fake_report(graph_state):
        called["report"] = True
        graph_state["analysis_state"].status = TaskStatus.COMPLETED
        return graph_state

    monkeypatch.setattr(workflow, "_generate_final_report", fake_report)

    result = await workflow.resume_restored_wait_state(state)

    assert called == {"report": True}
    assert workflow.compiled_graph.calls == []
    assert result.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_uses_restored_wait_continuation_for_headless_entry(monkeypatch):
    workflow = _workflow(max_rounds=4)
    state = AnalysisState(
        id="s",
        user_id="u",
        status=TaskStatus.RUNNING,
        current_round=3,
        max_rounds=4,
        restore_resume_kind="semantic_gate",
        restore_resume_gate="round_wait",
    )
    workflow.initialize = AsyncMock()
    workflow.resume_restored_wait_state = AsyncMock(return_value=state)

    result = await workflow.run(state, "s")

    assert result is state
    workflow.resume_restored_wait_state.assert_awaited_once_with(state)


@pytest.mark.asyncio
async def test_lifecycle_resume_restored_round_wait_without_workflow_sets_running(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(
        id="s1",
        user_id="u",
        status=TaskStatus.WAITING_INPUT,
        current_round=3,
        max_rounds=4,
        last_restored_snapshot_id="snap-1",
        restore_resume_kind="semantic_gate",
        restore_resume_gate="round_wait",
        restore_resume_node="wait_for_user_input",
    )

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))

    async def _fake_require_session_ownership(session_id: str, user_id: str):
        assert session_id == "s1"
        assert user_id == "u"
        return state

    pause_and_restart = AsyncMock()
    save = AsyncMock()
    update_cache = AsyncMock()

    monkeypatch.setattr(lifecycle, "require_session_ownership", _fake_require_session_ownership)
    monkeypatch.setattr(lifecycle, "_pause_and_restart_ws", pause_and_restart)
    monkeypatch.setattr(lifecycle, "async_save_session", save)
    monkeypatch.setattr(lifecycle, "_update_session_cache", update_cache)
    monkeypatch.setattr(lifecycle, "workflows", {})

    resp = await lifecycle.resume_analysis("s1", request)

    assert resp.success is True
    assert resp.data == {"needs_reconnect": True, "restored_wait_continuation": True}
    assert state.status == TaskStatus.RUNNING
    pause_and_restart.assert_not_awaited()
    update_cache.assert_awaited_once_with("s1", state)
    save.assert_awaited_once_with("s1", state)


@pytest.mark.asyncio
async def test_lifecycle_resume_regular_waiting_input_without_workflow_still_restarts_ws(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(
        id="s1",
        user_id="u",
        status=TaskStatus.WAITING_INPUT,
        current_round=1,
        max_rounds=3,
    )

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))

    async def _fake_require_session_ownership(session_id: str, user_id: str):
        return state

    pause_and_restart = AsyncMock()

    monkeypatch.setattr(lifecycle, "require_session_ownership", _fake_require_session_ownership)
    monkeypatch.setattr(lifecycle, "_pause_and_restart_ws", pause_and_restart)
    monkeypatch.setattr(lifecycle, "workflows", {})

    resp = await lifecycle.resume_analysis("s1", request)

    assert resp.success is True
    assert resp.data == {"needs_reconnect": True}
    pause_and_restart.assert_awaited_once_with("s1", state)


def test_snapshot_restore_branch_identity_is_default_safety_policy():
    snapshot_id = "snapshot-main-head"
    label = stable_restore_branch_label(snapshot_id, None)

    assert label == "branch-snapshot"
    assert stable_restore_branch_id(snapshot_id, label) == "restore-snapshot-main-head"
