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
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_PLAN_REVISION,
    GATE_ROUND_WAIT,
)
from backend.models.schemas import (
    AgentRoundCache,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    DataDrivenMechanismProgram,
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

try:
    from backend.config import _config, get_session_states
    from backend.services.round_snapshots.layout import session_root
    from backend.services.round_snapshots.restore import restore_snapshot_into_new_session
    from backend.services.round_snapshots.writer import capture_decision_snapshot
except ImportError as exc:  # pragma: no cover - module under test
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


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


def _origin(source: str, snapshot: str) -> str:
    return f"{source}/{snapshot}"


@pytest.mark.asyncio
async def test_restored_direction_action_advances_to_round_two(bundle):
    """The graph-progress half of spec §6.4's direction row and §6.5."""
    from backend.models.api_requests import DirectionChoice
    from backend.routes.interaction import choose_direction

    new_id, source, snapshot = await bundle(gate=GATE_DIRECTION_SELECTION, round_number=1)
    restored = await _current(new_id)
    assert restored.last_restored_snapshot_id == _origin(source, snapshot)
    assert restored.restore_provenance_snapshot_id == _origin(source, snapshot)
    original_result_count = len(restored.round_results)
    assert original_result_count == 2
    assert [result.lineage_id for result in restored.round_results] == [
        "main", restored.restore_branch_id,
    ]
    assert [result.round_number for result in restored.round_results] == [1, 1]

    request = types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER))
    response = await choose_direction(
        new_id,
        request,
        DirectionChoice(directions=[{"direction": "Kupffer heterogeneity", "notes": ""}]),
    )
    assert response.success
    restored = await _current(new_id)
    assert restored.restore_resume_kind == "graph_node"
    assert restored.restore_resume_node == "check_convergence"
    assert restored.status == TaskStatus.PAUSED
    restored.status = TaskStatus.RUNNING

    harness = _Harness(new_id)
    ws = await harness.advance_semantic_continuation(restored)

    # 4: planning/initialization sees round 2, not another round 1.
    assert ws["analysis_state"].current_round == 2
    assert ws["current_round"] == 2
    # 5: the source record and its branch-local inherited view remain singular.
    assert len(ws["analysis_state"].round_results) == original_result_count
    assert [r.round_number for r in ws["analysis_state"].round_results] == [1, 1]
    # 6: the selected direction is available to round 2 planning.
    assert any(
        item.get("direction") == "Kupffer heterogeneity"
        for item in ws["analysis_state"].directions_queue
    )
    # 7: prior approved plan and per-round cache are not reused.
    assert ws["analysis_state"].current_plan is None
    assert ws["analysis_state"].current_round_cache is None
    assert ws["plan_approved"] is False
    # One-shot explicit-start flag is consumed in the graph channel.
    assert ws["_preserve_round_on_initialize"] is False

    # 8/9: the inherited branch record already carries the origin; new results
    # consume the pending marker without duplicating that attribution.
    from backend.workflow.provenance import (
        consume_restore_provenance,
        restore_provenance_id,
    )
    from backend.workflow.synthesis import _upsert_round_result

    analysis = ws["analysis_state"]
    origin = _origin(source, snapshot)
    assert restore_provenance_id(analysis) == origin
    first = RoundResult(
        round_number=2,
        phase="innovation",
        direction="Kupffer heterogeneity",
        pi_summary="first new result",
        bioinformatics_findings=["new"],
    )
    _upsert_round_result(
        analysis.round_results, first, lineage_id=analysis.restore_branch_id,
        branch_id=analysis.restore_branch_id,
        branch_label=analysis.restore_branch_label,
        restored_from_snapshot_id=restore_provenance_id(analysis),
    )
    consume_restore_provenance(analysis)
    assert first.round_number == 2
    assert first.restored_from_snapshot_id is None
    assert analysis.last_restored_snapshot_id == origin
    assert restore_provenance_id(analysis) is None

    second = RoundResult(
        round_number=3,
        phase="innovation",
        direction="Second",
        pi_summary="later",
        bioinformatics_findings=["later"],
    )
    _upsert_round_result(
        analysis.round_results, second, lineage_id=analysis.restore_branch_id,
        branch_id=analysis.restore_branch_id,
        branch_label=analysis.restore_branch_label,
        restored_from_snapshot_id=restore_provenance_id(analysis),
    )
    assert second.restored_from_snapshot_id is None
    assert sum(
        result.restored_from_snapshot_id == origin
        for result in analysis.round_results
    ) == 1

    # 10: parent listing survives the result once the session is persisted.
    from backend.routes.round_snapshots import _restored_from
    from backend.services.session_store import async_save_session

    await async_save_session(new_id, analysis, force=True)
    assert _restored_from(source) == {snapshot: [new_id]}

    # 11: no duplicate r001-direction snapshot was produced on the child.
    child_snapshots = session_root(new_id)
    if child_snapshots.is_dir():
        assert not any(
            entry.name.startswith("r001-direction-")
            for entry in child_snapshots.iterdir()
        )


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


@pytest.mark.asyncio
async def test_restored_round_wait_action_advances_to_next_round(bundle):
    """The ordinary round-completion row of spec §6.4."""
    from backend.routes.lifecycle import resume_analysis

    new_id, source, snapshot = await bundle(gate=GATE_ROUND_WAIT, round_number=3)
    restored = await _current(new_id)
    assert restored.current_round == 3
    original = [r.round_number for r in restored.round_results]

    request = types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER))
    response = await resume_analysis(new_id, request)
    assert response.success
    restored = await _current(new_id)
    assert restored.restore_resume_node == "check_convergence"
    restored.status = TaskStatus.RUNNING

    harness = _Harness(new_id)
    ws = await harness.advance_semantic_continuation(restored)
    assert ws["analysis_state"].current_round == 4
    assert [r.round_number for r in ws["analysis_state"].round_results] == original
    assert ws["analysis_state"].current_plan is None
    assert ws["analysis_state"].current_round_cache is None


@pytest.mark.asyncio
async def test_restored_integration_direction_advances_to_next_round(bundle):
    """The integration-direction row of spec §6.4."""
    from backend.models.api_requests import DirectionChoice
    from backend.routes.interaction import choose_direction

    new_id, source, snapshot = await bundle(
        gate=GATE_INTEGRATION_SELECTION,
        round_number=4,
        phase=AnalysisPhase.INTEGRATION,
        prepare_state=lambda state: setattr(
            state, "integration_gate_candidates",
            [{"direction": "cross-direction correlation"}],
        ),
    )
    request = types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER))
    response = await choose_direction(
        new_id,
        request,
        DirectionChoice(directions=[{"direction": "new integrated direction", "notes": ""}]),
    )
    assert response.success
    restored = await _current(new_id)
    assert restored.restore_resume_node == "check_convergence"
    restored.status = TaskStatus.RUNNING

    harness = _Harness(new_id)
    ws = await harness.advance_semantic_continuation(restored)
    assert ws["analysis_state"].current_round == 5
    assert ws["analysis_state"].current_plan is None
    assert ws["analysis_state"].current_round_cache is None


@pytest.mark.asyncio
async def test_explicit_start_preserves_round_and_plan_exactly_once(bundle):
    """The explicit-start row: captured round survives once, then normal."""
    new_id, source, snapshot = await bundle(gate=GATE_ROUND_WAIT, round_number=3)
    restored = await _current(new_id)
    restored.status = TaskStatus.RUNNING
    restored.analysis_phase = AnalysisPhase.EXPLORATION
    restored.restore_resume_kind = None
    restored.restore_resume_gate = None
    restored.restore_resume_node = None
    restored.restore_resume_workflow_state = None
    restored.restore_start_intent = "explicit_start"
    original_plan_id = restored.current_plan.id
    original_cache_round = restored.current_round_cache.round_number

    harness = _Harness(new_id)
    ws = harness.wf._restore_entry(harness.entry_state(restored))
    assert ws["_preserve_round_on_initialize"] is True
    assert harness.wf._route_from_restore_entry(ws) == "route_initialize_round"
    ws = await harness.wf._initialize_round(ws)
    assert ws["analysis_state"].current_round == 3
    assert ws["analysis_state"].current_plan is not None
    assert ws["analysis_state"].current_plan.id == original_plan_id
    assert ws["analysis_state"].current_round_cache is not None
    assert ws["analysis_state"].current_round_cache.round_number == original_cache_round
    assert ws["_preserve_round_on_initialize"] is False

    # The next ordinary initialization increments and resets normally.
    ws_next = await harness.wf._initialize_round(
        {**ws, "analysis_state": ws["analysis_state"]}
    )
    assert ws_next["analysis_state"].current_round == 4
    assert ws_next["analysis_state"].current_plan is None
    assert ws_next["analysis_state"].current_round_cache is None


@pytest.mark.asyncio
async def test_completed_reanalysis_starts_fresh_next_round_through_graph(
    bundle, monkeypatch,
):
    """Completed re-analysis starts a new round, unlike snapshot explicit start."""
    from backend.routes import lifecycle
    from backend.routes.websockets import should_resume_workflow

    new_id, _source, _snapshot = await bundle(
        gate=GATE_ROUND_WAIT, round_number=3,
    )
    state = await _current(new_id)
    prior_plan_id = state.current_plan.id
    assert state.current_round_cache is not None
    state.status = TaskStatus.COMPLETED

    monkeypatch.setattr(
        lifecycle, "require_session_ownership", _async_return(state),
    )
    monkeypatch.setattr(lifecycle, "_update_session_cache", _async_none())
    monkeypatch.setattr(lifecycle, "async_save_session", _async_none())
    monkeypatch.setattr(
        lifecycle, "teardown_running_workflow", _async_return(True),
    )

    response = await lifecycle.resume_analysis(
        new_id, types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER)),
    )
    assert response.success
    assert state.restore_start_intent == "completed_reanalysis"
    assert state.max_rounds == 4
    assert should_resume_workflow(state) is False

    # Use a compiled StateGraph so the intent and preserve flag cross the same
    # schema merge/router boundary as production. The production restore and
    # initialize nodes must advance and reset the completed round exactly once.
    harness = _Harness(new_id)
    graph = _compile_restore_progression_graph(harness)
    result = await graph.ainvoke(harness.entry_state(state))
    advanced = result["analysis_state"]

    assert advanced.current_round == 4
    assert result["current_round"] == 4
    assert advanced.current_plan is None
    assert advanced.current_round_cache is None
    assert advanced.restore_start_intent is None
    assert result["_restore_entry_plan"]["intent"] == "normal"
    assert result["_preserve_round_on_initialize"] is False
    assert result["plan_approved"] is False
    assert prior_plan_id is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate,target_node,plan_step_status,cache_present",
    [
        (GATE_ANNOTATION_CONFIRMATION, "bioinformatics_analysis", "completed", True),
        (GATE_PLAN_REVISION, "planning_dialogue", "completed", True),
    ],
)
async def test_non_convergence_continuations_keep_execution_context(
    gate, target_node, plan_step_status, cache_present,
):
    """Spec §6.4 rows that must not re-initialize or reset the plan."""
    from backend.workflow.provenance import consume_restore_provenance  # noqa: F401
    from backend.workflow.restore_entry import build_restore_entry_plan
    from backend.workflow.restore_targets import SEMANTIC_GATE_ACTION_NEXT_NODES, prepare_restored_semantic_action

    state = AnalysisState(
        user_id=OWNER,
        current_round=2,
        status=TaskStatus.WAITING_INPUT,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_plan=AnalysisPlan(
            phase="innovation",
            overview="context",
            status="approved",
            steps=[PlanStep(id=1, title="kept", method="m", status=plan_step_status)],
        ),
        current_round_cache=AgentRoundCache(round_number=2, results={}),
        restore_resume_kind="semantic_gate",
        restore_resume_gate=gate,
        restore_resume_workflow_state={"current_round": 2},
    )
    target = prepare_restored_semantic_action(state)
    assert target == target_node
    assert SEMANTIC_GATE_ACTION_NEXT_NODES[gate] == target_node
    assert state.restore_resume_kind == "graph_node"

    plan = build_restore_entry_plan(state)
    assert plan.intent == "graph_node"
    assert plan.target_node == target_node
    assert plan.workflow_state["current_round"] == 2
    assert plan.preserve_round_on_initialize is False

    harness = _Harness("synthetic-session")
    ws = harness.wf._restore_entry({"analysis_state": state, "messages": []})
    assert harness.wf._route_from_restore_entry(ws) == f"route_{target_node}"
    assert state.current_round == 2
    assert state.current_plan is not None
    assert state.current_plan.steps[0].status == plan_step_status
    assert state.current_round_cache is not None


@pytest.mark.asyncio
async def test_pre_execution_graph_node_keeps_captured_plan():
    """A recorded execution-node restore retains the captured plan."""
    from backend.workflow.restore_entry import build_restore_entry_plan

    state = AnalysisState(
        user_id=OWNER,
        current_round=2,
        status=TaskStatus.WAITING_INPUT,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_plan=AnalysisPlan(
            phase="innovation",
            overview="captured",
            status="approved",
            steps=[PlanStep(id=1, title="captured", method="m", status="completed")],
        ),
        restore_resume_kind="graph_node",
        restore_resume_node="bioinformatics_analysis",
        restore_branch_id="branch-x",
    )
    plan = build_restore_entry_plan(state)
    assert plan.intent == "graph_node"
    assert plan.target_node == "bioinformatics_analysis"

    harness = _Harness("synthetic-session")
    ws = harness.wf._restore_entry({"analysis_state": state, "messages": []})
    assert harness.wf._route_from_restore_entry(ws) == "route_bioinformatics_analysis"
    assert state.current_round == 2
    assert state.current_plan is not None
    assert state.current_plan.steps[0].status == "completed"


@pytest.mark.asyncio
async def test_round_result_is_a_reviewed_redundant_channel(bundle):
    """A non-null transient ``round_result`` is omitted; canonical survives."""
    from backend.workflow.provenance import restore_provenance_id  # noqa: F401

    transient = RoundResult(
        round_number=1, phase="innovation", direction="Original direction",
        pi_summary="transient duplicate", bioinformatics_findings=["transient"],
    )
    new_id, source, snapshot = await bundle(
        gate=GATE_DIRECTION_SELECTION,
        round_number=1,
        extra_channels={"round_result": transient},
        extra_capsule_state={"round_result": True},
    )
    restored = await _current(new_id)
    assert len(restored.round_results) == 2, "lineage views were not preserved"
    assert [result.lineage_id for result in restored.round_results] == [
        "main", restored.restore_branch_id,
    ]
    assert all(
        result.pi_summary == "one result before restore"
        for result in restored.round_results
    )
    assert all(
        result.bioinformatics_findings == ["original finding"]
        for result in restored.round_results
    )
    workflow_state = restored.restore_resume_workflow_state or {}
    assert "round_result" not in workflow_state


@pytest.mark.asyncio
async def test_supported_model_nested_in_bioinformatics_result_is_json_normalized(
    bundle,
):
    """A supported model nested in a graph dict remains restorable."""
    program = DataDrivenMechanismProgram(
        label="Observed program",
        evidence_execution_ids=["exec-1"],
    )
    new_id, _source, _snapshot = await bundle(
        gate=GATE_DIRECTION_SELECTION,
        round_number=1,
        extra_channels={
            "bioinformatics_result": {
                "status": "completed",
                "mechanism_programs": [program],
            },
        },
        extra_capsule_state={"bioinformatics_result": True},
    )

    restored = await _current(new_id)
    bioinformatics_result = (
        restored.restore_resume_workflow_state or {}
    )["bioinformatics_result"]
    restored_program = bioinformatics_result["mechanism_programs"][0]
    assert isinstance(restored_program, dict)
    assert restored_program["id"] == program.id
    assert restored_program["label"] == "Observed program"
    assert restored_program["evidence_execution_ids"] == ["exec-1"]
    assert isinstance(restored_program["timestamp"], str)


@pytest.mark.asyncio
async def test_unknown_non_json_channel_is_refused_before_publication(
    monkeypatch, tmp_path,
):
    """A checkpoint channel that json.dumps cannot carry refuses atomically."""
    from backend.config import _config, ensure_session_dirs, get_session_dir
    from backend.services.round_snapshots.restore import (
        SnapshotRestoreError,
        restore_snapshot_into_new_session,
    )
    from backend.services.session_store import list_saved_sessions

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    source = "src-" + uuid.uuid4().hex[:12]
    digest = _seed_session(source, revision=57, round_number=1)
    state = _state_for_bundle(source, 1)
    channels = {
        "current_round": 1,
        "planning_stage": "full",
        "plan_approved": True,
        "should_continue": True,
        "mystery_live_object": {1, 2, 3},
    }
    _write_stateful_checkpoint(source, state, 1, channels)
    workflow = _Workflow(_Sandbox(source, 57, digest, round_number=1))
    snapshot = await capture_decision_snapshot(
        workflow, state, gate=GATE_DIRECTION_SELECTION, workflow_state=channels,
    )
    assert snapshot is not None

    before = {summary["id"] for summary in list_saved_sessions()}
    with pytest.raises(SnapshotRestoreError) as raised:
        await restore_snapshot_into_new_session(
            source_session_id=source, snapshot=snapshot, user_id=OWNER,
        )
    assert raised.value.reason == "continuation_unreadable"
    assert {summary["id"] for summary in list_saved_sessions()} == before


@pytest.mark.asyncio
async def test_completed_reanalysis_keeps_pending_provenance_single_use(
    bundle, monkeypatch,
):
    """Re-analysis consumes the pending marker without duplicating branch origin."""
    from backend.routes import lifecycle
    from backend.workflow.provenance import (
        consume_restore_provenance,
        restore_provenance_id,
    )
    from backend.workflow.synthesis import _upsert_round_result

    new_id, source, snapshot = await bundle(
        gate=GATE_ROUND_WAIT, round_number=3,
    )
    origin = _origin(source, snapshot)
    state = await _current(new_id)
    assert state.restore_provenance_snapshot_id == origin
    state.status = TaskStatus.COMPLETED

    monkeypatch.setattr(
        lifecycle, "require_session_ownership", _async_return(state),
    )
    monkeypatch.setattr(lifecycle, "_update_session_cache", _async_none())
    monkeypatch.setattr(lifecycle, "async_save_session", _async_none())
    monkeypatch.setattr(
        lifecycle, "teardown_running_workflow", _async_return(True),
    )
    response = await lifecycle.resume_analysis(
        new_id, types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER)),
    )
    assert response.success
    assert state.status == TaskStatus.RUNNING
    assert state.last_restored_snapshot_id == origin
    assert state.restore_provenance_snapshot_id == origin, (
        "a completed re-analysis must not consume a still-pending one-shot origin"
    )
    assert sum(
        result.restored_from_snapshot_id == origin
        for result in state.round_results
    ) == 1

    first = RoundResult(
        round_number=4, phase="innovation", direction="Reanalysis",
        pi_summary="reanalysis first",
    )
    _upsert_round_result(
        state.round_results, first, lineage_id=state.restore_branch_id,
        branch_id=state.restore_branch_id,
        branch_label=state.restore_branch_label,
        restored_from_snapshot_id=restore_provenance_id(state),
    )
    consume_restore_provenance(state)
    assert first.restored_from_snapshot_id is None
    assert sum(
        result.restored_from_snapshot_id == origin
        for result in state.round_results
    ) == 1
    assert state.last_restored_snapshot_id == origin
    assert state.restore_provenance_snapshot_id is None


@pytest.mark.asyncio
async def test_completed_reanalysis_keeps_already_consumed_provenance_none(
    bundle, monkeypatch,
):
    from backend.routes import lifecycle

    new_id, source, snapshot = await bundle(
        gate=GATE_ROUND_WAIT, round_number=3,
    )
    state = await _current(new_id)
    state.restore_provenance_snapshot_id = None
    state.status = TaskStatus.COMPLETED

    monkeypatch.setattr(
        lifecycle, "require_session_ownership", _async_return(state),
    )
    monkeypatch.setattr(lifecycle, "_update_session_cache", _async_none())
    monkeypatch.setattr(lifecycle, "async_save_session", _async_none())
    monkeypatch.setattr(
        lifecycle, "teardown_running_workflow", _async_return(True),
    )
    response = await lifecycle.resume_analysis(
        new_id, types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER)),
    )
    assert response.success
    assert state.last_restored_snapshot_id == _origin(source, snapshot)
    assert state.restore_provenance_snapshot_id is None


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


def _async_none():
    async def _inner(*args, **kwargs):
        return None

    return _inner


def test_idle_restore_identity_accepts_branch_and_legacy_origin():
    """Branch id is primary; durable origin is a documented legacy fallback."""
    from types import SimpleNamespace

    from backend.workflow.core_parts.execution import _has_idle_restore_identity

    assert _has_idle_restore_identity(SimpleNamespace(
        restore_branch_id="branch-x", last_restored_snapshot_id=None,
    )) is True
    assert _has_idle_restore_identity(SimpleNamespace(
        restore_branch_id=None,
        last_restored_snapshot_id="source/r001-direction-g1",
    )) is True
    assert _has_idle_restore_identity(SimpleNamespace(
        restore_branch_id=None, last_restored_snapshot_id=None,
    )) is False
