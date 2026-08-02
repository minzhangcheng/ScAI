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
    def __init__(self, workflow):
        self.workflow = workflow
        self.calls = []
        self.routes = []

    async def ainvoke(self, workflow_state, config):
        restored = self.workflow._restore_entry(workflow_state)
        route = self.workflow._route_from_restore_entry(restored)
        self.calls.append((restored, config))
        self.routes.append(route)
        state = restored["analysis_state"]
        return {"analysis_state": state}


def _workflow(max_rounds=4):
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "s"
    workflow.config = SimpleNamespace(max_rounds=max_rounds)
    workflow.restore_entry_loader = None
    workflow.compiled_graph = _FakeCompiledGraph(workflow)
    workflow._current_analysis_state = None
    workflow._send_message = AsyncMock()
    workflow._send_state_sync = AsyncMock()
    return workflow


@pytest.mark.asyncio
async def test_resume_restored_wait_state_routes_round_action_to_exact_node():
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

    result = await workflow.resume_restored_wait_state(state)

    assert len(workflow.compiled_graph.calls) == 1
    assert workflow.compiled_graph.routes == ["route_check_convergence"]
    assert result.current_round == 3
    assert result.restore_resume_kind is None
    # Provenance survives wait-state resume until canonical round recording.
    assert result.last_restored_snapshot_id == "snap-1"


@pytest.mark.asyncio
async def test_resume_restored_wait_state_does_not_prejudge_max_round_action():
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
    result = await workflow.resume_restored_wait_state(state)

    assert workflow.compiled_graph.routes == ["route_check_convergence"]
    assert len(workflow.compiled_graph.calls) == 1
    assert result.status == TaskStatus.RUNNING


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
    assert resp.data == {
        "needs_reconnect": True,
        "restored_wait_continuation": True,
        "next_node": "check_convergence",
    }
    assert state.status == TaskStatus.RUNNING
    assert state.restore_resume_kind == "graph_node"
    assert state.restore_resume_node == "check_convergence"
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gate", "target", "expected_status", "payload"),
    [
        (
            "direction_selection",
            "check_convergence",
            TaskStatus.WAITING_INPUT,
            {"pi_instructions": {"direction": "axis-a"}},
        ),
        (
            "annotation_confirmation",
            "bioinformatics_analysis",
            TaskStatus.WAITING_ANNOTATION,
            {"planning_stage": "stage1", "plan_approved": True},
        ),
        (
            "plan_revision",
            "planning_dialogue",
            TaskStatus.WAITING_INPUT,
            {
                "planning_stage": "stage2",
                "plan_revision_needed": True,
                "plan_revision_context": {"reason": "missing method"},
                "plan_revision_resume": True,
            },
        ),
    ],
)
async def test_restored_semantic_action_ws_and_graph_target_are_exact(
    gate, target, expected_status, payload,
):
    """Restore → action → reconnect state_sync → restore_entry exact node."""
    from backend.routes.websockets import _send_restore_state_sync
    from backend.services.snapshot_v2.dispatcher import (
        ContinuationDispatcher,
        StagedCheckpoint,
    )
    from backend.workflow.restore_targets import prepare_restored_semantic_action

    state = AnalysisState(
        id=f"semantic-{gate}",
        user_id="owner",
        current_round=4,
    )
    staged = StagedCheckpoint(
        kind="semantic_gate",
        gate=gate,
        payload={
            "gate": gate,
            "current_round": 4,
            "next_node": target,
            **payload,
        },
        branch_id=f"restore-{gate}",
        branch_label=f"branch-{gate}",
        parent_snapshot_id=f"snapshot-{gate}",
        transaction_id=f"tx-{gate}",
    )
    ContinuationDispatcher(None).activate(state, staged)
    assert state.restore_resume_kind == "semantic_gate"
    assert state.restore_resume_gate == gate
    assert state.status == expected_status

    assert prepare_restored_semantic_action(
        state, expected_gate=gate,
    ) == target
    assert state.restore_resume_kind == "graph_node"
    assert state.restore_resume_node == target

    sent = []

    class _WebSocket:
        async def send_json(self, message):
            sent.append(message)

    await _send_restore_state_sync(_WebSocket(), state, state.id)
    sync = sent[0]["content"]
    assert sync["restore_resume_kind"] == "graph_node"
    assert sync["restore_resume_node"] == target
    assert sync["restore_resume_gate"] is None

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = state.id
    workflow.restore_entry_loader = None
    graph_state = workflow._restore_entry({
        "analysis_state": state,
        "current_round": state.current_round,
    })
    assert workflow._route_from_restore_entry(graph_state) == f"route_{target}"
    assert graph_state["_restore_resume_node"] == target
    for key, value in payload.items():
        assert graph_state[key] == value
