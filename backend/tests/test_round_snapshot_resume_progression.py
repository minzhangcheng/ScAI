"""Graph-level restore progression regression tests.

The historical defect (SR-1): a restored Round 1 direction action converted the
semantic gate to ``check_convergence``, but ``initialize_round`` then treated
the durable ``last_restored_snapshot_id`` as a same-round control. It skipped
the round increment and the plan/cache reset, so the child initialized another
Round 1 instead of Round 2.

These tests build a real on-disk bundle through the capture path, restore it,
answer the gate through the same route the browser uses, and then walk the real
routing functions. They must fail on the parent commit.
"""
from __future__ import annotations

import sqlite3
import types
import uuid

import pytest

from backend.constants import (
    GATE_DIRECTION_SELECTION,
)
from backend.models.schemas import (
    AgentRoundCache,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    PlanStep,
    RoundResult,
    TaskStatus,
)

from backend.tests.test_round_snapshot_capture import (  # noqa: E402
    _Sandbox,
    _Workflow,
    _seed_session,
    _state,
)

from backend.config import _config, get_session_states
from backend.services.round_snapshots.restore import restore_snapshot_into_new_session
from backend.services.round_snapshots.writer import capture_decision_snapshot


OWNER = "owner-1"


def _write_stateful_checkpoint(
    session_id: str,
    analysis_state: AnalysisState,
    round_number: int,
    *extra_channels: dict,
) -> None:
    """Replace the harness checkpoint pair with the channels this test needs."""
    from backend.utils.checkpoint import DurableJsonPlusSerializer
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    checkpoint = (
        get_session_states(session_id)
        / GENERATION_PAIR_DIRNAME
        / "57"
        / "checkpoints.sqlite"
    )
    serde = DurableJsonPlusSerializer(allowed_json_modules=[
        ("backend.models.schemas", name)
        for name in (
            "AgentType", "AnalysisState", "AnalysisPhase", "RoundResult",
            "TaskStatus", "MessageRole",
        )
    ])
    values = {
        "analysis_state": analysis_state,
        "current_round": round_number,
        "planning_stage": "full",
        "plan_approved": True,
        "should_continue": True,
    }
    for channel in extra_channels:
        values.update(channel)
    kind, blob = serde.dumps_typed({
        "v": 1,
        "id": "checkpoint-1",
        "channel_values": values,
        "channel_versions": {},
        "versions_seen": {},
    })
    with sqlite3.connect(checkpoint) as connection:
        connection.execute("DELETE FROM checkpoints")
        connection.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
            (session_id, "", "checkpoint-1", kind, blob),
        )


def _state_for_bundle(session_id: str, round_number: int, *, max_rounds: int = 6):
    state = _state(session_id, round_number=round_number, name="Progression")
    state.state_version = 1189
    state.max_rounds = max_rounds
    state.status = TaskStatus.PAUSED
    state.analysis_phase = AnalysisPhase.INNOVATION
    state.current_round_cache = AgentRoundCache(
        round_number=round_number,
        results={"bioinformatics": {"stale": True}},
    )
    state.current_plan = AnalysisPlan(
        phase="innovation",
        stage="full",
        overview="approved before restore",
        status="approved",
        steps=[
            PlanStep(id=1, title="locked", method="m", status="completed"),
            PlanStep(id=2, title="pending", method="m", status="pending"),
        ],
    )
    state.round_results = [
        RoundResult(
            round_number=round_number,
            phase="innovation",
            direction="Original direction",
            pi_summary="one result before restore",
            bioinformatics_findings=["original finding"],
        )
    ]
    return state


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    """Create a source session plus a real captured direction-round bundle."""
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")

    async def _build(
        *,
        gate: str = GATE_DIRECTION_SELECTION,
        round_number: int = 1,
        phase: AnalysisPhase = AnalysisPhase.INNOVATION,
        prepare_state=None,
        extra_channels=None,
        extra_capsule_state=None,
    ):
        session_id = "src-" + uuid.uuid4().hex[:12]
        digest = _seed_session(session_id, revision=57, round_number=round_number)
        state = _state_for_bundle(session_id, round_number)
        state.analysis_phase = phase
        if prepare_state is not None:
            prepare_state(state)
        channels = {
            "current_round": round_number,
            "planning_stage": "full",
            "plan_approved": True,
            "should_continue": True,
        }
        if extra_channels:
            channels.update(extra_channels)
        _write_stateful_checkpoint(session_id, state, round_number, channels)
        workflow = _Workflow(
            _Sandbox(session_id, 57, digest, round_number=round_number)
        )
        capsule_state = dict(channels)
        if extra_capsule_state:
            capsule_state.update(extra_capsule_state)
        snapshot = await capture_decision_snapshot(
            workflow, state, gate=gate, workflow_state=capsule_state,
        )
        assert snapshot is not None
        new_id = await restore_snapshot_into_new_session(
            source_session_id=session_id, snapshot=snapshot, user_id=OWNER,
        )
        return new_id, session_id, snapshot

    return _build


class _Harness:
    """The real routing/node functions, with transport collaborators stubbed."""

    def __init__(self, session_id: str):
        from backend.workflow.core import SingleCellAnalysisWorkflow

        self.wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        self.wf.session_id = session_id
        self.wf.config = types.SimpleNamespace(convergence_rounds=2, max_rounds=6)
        self.wf.restore_entry_loader = None
        self.wf._send_message = self._send_message
        self.wf._send_state_sync = self._send_state_sync
        self.wf._set_status = self._set_status
        self.wf.pi_agent = types.SimpleNamespace(
            process_user_input=self._process_user_input,
        )

    async def _send_message(self, *args, **kwargs):
        return None

    async def _send_state_sync(self, *args, **kwargs):
        return None

    @staticmethod
    def _set_status(state, status):
        state.status = status

    @staticmethod
    async def _process_user_input(*args, **kwargs):
        return {"action": "continue", "response": "ok", "adjusted_instructions": None}

    def entry_state(self, state):
        return self.wf._make_initial_workflow_state(state)

    async def advance_semantic_continuation(self, restored):
        """restore_entry -> check_convergence -> initialize_round."""
        assert restored.status == TaskStatus.RUNNING
        ws = self.wf._restore_entry(self.entry_state(restored))
        assert self.wf._route_from_restore_entry(ws) == "route_check_convergence"
        ws = await self.wf._check_convergence(ws)
        assert self.wf._check_convergence_condition(ws) == "continue"
        ws = await self.wf._initialize_round(ws)
        return ws


async def _current(session_id: str) -> AnalysisState:
    from backend.services.session_store import ensure_session_loaded

    return await ensure_session_loaded(session_id)


def _compile_restore_progression_graph(harness):
    """Compile the real restore/init/convergence nodes in a small graph.

    The production graph has many expensive agent nodes between initialization
    and the next gate. This graph keeps the same StateGraph channels and the
    real restore entry and convergence routers, ending at a sink immediately
    after initialization so the routing contract can be exercised without
    making an LLM call.
    """
    from langgraph.graph import END, StateGraph

    graph = StateGraph(__import__(
        "backend.workflow.state", fromlist=["WorkflowState"],
    ).WorkflowState)
    graph.add_node("restore_entry", harness.wf._restore_entry)
    graph.add_node("check_convergence", harness.wf._check_convergence)
    graph.add_node("initialize_round", harness.wf._initialize_round)
    graph.add_node("sink", lambda _state: {})
    graph.set_entry_point("restore_entry")
    graph.add_conditional_edges(
        "restore_entry",
        harness.wf._route_from_restore_entry,
        {
            "route_check_convergence": "check_convergence",
            "route_initialize_round": "initialize_round",
            "hold": "sink",
        },
    )
    graph.add_conditional_edges(
        "check_convergence",
        harness.wf._check_convergence_condition,
        {
            "continue": "initialize_round",
            "integrate": "sink",
            "stop": "sink",
        },
    )
    graph.add_edge("initialize_round", "sink")
    graph.add_edge("sink", END)
    return graph.compile()


@pytest.mark.asyncio
async def test_real_state_graph_routes_restored_direction_through_next_round(bundle):
    """The restore path must advance through the real graph routers once."""
    from backend.models.api_requests import DirectionChoice
    from backend.routes.interaction import choose_direction

    new_id, _source, _snapshot = await bundle(
        gate=GATE_DIRECTION_SELECTION, round_number=1,
    )
    request = types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER))
    response = await choose_direction(
        new_id,
        request,
        DirectionChoice(directions=[{"direction": "Kupffer heterogeneity", "notes": ""}]),
    )
    assert response.success

    restored = await _current(new_id)
    restored.status = TaskStatus.RUNNING
    harness = _Harness(new_id)
    graph = _compile_restore_progression_graph(harness)
    result = await graph.ainvoke(harness.entry_state(restored))

    assert result["analysis_state"].current_round == 2
    assert result["current_round"] == 2
    assert result["analysis_state"].current_plan is None
    assert result["analysis_state"].current_round_cache is None
    assert result["_preserve_round_on_initialize"] is False
