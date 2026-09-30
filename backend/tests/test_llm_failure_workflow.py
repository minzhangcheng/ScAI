"""Bounded automatic retry and generation-scoped user intervention."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.utils.llm_client import UpstreamExhaustedError
from backend.constants import LlmFailureAction, StateSyncChangeType
from backend.models.schemas import AnalysisPlan, AnalysisState, TaskStatus


# ── Exceptional gates: bounded retries, then explicit user resolution ───


def _gate_self():
    """Workflow stub with the real HITL gate bound, as the fixtures below do."""
    from backend.workflow import SingleCellAnalysisWorkflow

    mock = MagicMock()
    mock.session_id = "test-session"
    mock._llm_failure_lock = asyncio.Lock()
    mock._llm_failure_events = {}
    mock._llm_failure_actions = {}
    mock._llm_failure_generations = {}
    mock._unattended_gate_retries = {}
    mock._send_message = AsyncMock()
    mock._send_state_sync = AsyncMock()
    mock._set_status = MagicMock()
    mock._settings_provider = MagicMock()
    for name in (
        "_wait_for_llm_intervention",
        "_wait_for_disabled_agent_intervention",
        "_drive_hitl_gate",
        "_next_generation",
        "_generation_for",
        "signal_llm_failure_action",
    ):
        setattr(
            mock, name,
            getattr(SingleCellAnalysisWorkflow, name).__get__(mock, type(mock)),
        )
    return mock


def _no_backoff(monkeypatch):
    """Keep the unattended backoff out of the suite's wall clock."""
    from backend.tests.split_facade import setattr_facade

    slept = []

    async def _record(event, attempt):
        slept.append(attempt)

    setattr_facade(
        monkeypatch, "backend.workflow.core", "_unattended_gate_backoff", _record
    )
    return slept


def _park_probe(monkeypatch):
    """Signal the moment the gate reaches the terminal indefinite wait.

    Nothing suspends between the probe and ``event.wait()``, so a test that
    observes the flag is looking at an already-parked gate: the event has been
    cleared and the signal it sends next cannot be lost.
    """
    from backend.tests.split_facade import setattr_facade
    from backend.workflow.helpers import wait_with_skip_time as real_wait

    parked = asyncio.Event()

    async def _probe(event, wait_seconds, session_id, gate_name):
        if wait_seconds == -1:
            parked.set()
        return await real_wait(event, wait_seconds, session_id, gate_name)

    setattr_facade(
        monkeypatch, "backend.workflow.core", "wait_with_skip_time", _probe
    )
    return parked


def _exhausted(cause=None):
    error = UpstreamExhaustedError(
        "upstream exhausted",
        agent_name="bioinformatics",
        model="m",
        llm_server="https://x/v1",
    )
    if cause is not None:
        error.__cause__ = cause
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("round_wait", [0, 30, -1])
@pytest.mark.parametrize("gate_kind", ["llm", "marker_db"])
async def test_exception_gate_retries_twice_then_waits(monkeypatch, round_wait, gate_kind):
    slept = _no_backoff(monkeypatch)
    parked = _park_probe(monkeypatch)
    wf = _gate_self()
    state = AnalysisState(user_id="u", round_wait_seconds=round_wait)
    wf._set_status = lambda state, status: setattr(state, "status", status)
    dialog_waits = []

    async def record_sync(state, change_type):
        if change_type == StateSyncChangeType.LLM_FAILURE.value:
            dialog_waits.append(state.llm_failure_info["wait_seconds"])

    wf._send_state_sync = AsyncMock(side_effect=record_sync)

    async def enter():
        if gate_kind == "marker_db":
            from backend.workflow.core_parts.interventions import _wait_for_marker_db_intervention
            return await _wait_for_marker_db_intervention(wf, "planning_dialogue", state)
        return await wf._wait_for_llm_intervention(_exhausted(), "planning_dialogue", state)

    assert await enter() == LlmFailureAction.RETRY
    assert await enter() == LlmFailureAction.RETRY
    task = asyncio.create_task(enter())
    try:
        await asyncio.wait_for(parked.wait(), 2)
        assert not task.done()
        assert slept == [0, 1]
        assert state.status == TaskStatus.WAITING_INPUT
        assert state.llm_failure_info["wait_seconds"] == -1
        assert state.llm_failure_info["generation"] == 3
        assert dialog_waits == [0, 0, -1]
        assert wf._send_state_sync.await_args.args[1] == StateSyncChangeType.LLM_FAILURE.value
        assert wf.signal_llm_failure_action("planning_dialogue", "retry", 3)
        assert await asyncio.wait_for(task, 2) == LlmFailureAction.RETRY
        assert state.llm_failure_info is None
        assert wf._unattended_gate_retries == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", list(LlmFailureAction))
async def test_annotation_failure_requires_action_without_upstream_recovery(monkeypatch, action):
    slept = _no_backoff(monkeypatch)
    parked = _park_probe(monkeypatch)
    wf = _gate_self()
    wf._set_status = lambda state, status: setattr(state, "status", status)
    plan = AnalysisPlan(id="annotation-plan", phase="exploration", stage="stage2", status="approved")
    state = AnalysisState(user_id="u", current_plan=plan, round_wait_seconds=0)
    error = UpstreamExhaustedError(
        "Analysis source preparation is missing or no longer exact",
        agent_name="bioinformatics", failure_kind="annotation_tool_failure",
    )
    task = asyncio.create_task(wf._wait_for_llm_intervention(
        error, "bioinformatics_analysis", state,
    ))
    try:
        await asyncio.wait_for(parked.wait(), 2)
        assert not task.done()
        assert state.status == TaskStatus.WAITING_INPUT
        assert state.llm_failure_info["failure_kind"] == "annotation_tool_failure"
        assert state.llm_failure_info["error"] == str(error)
        assert state.llm_failure_info["wait_seconds"] == -1
        assert state.current_plan is plan
        assert slept == []
        assert wf._unattended_gate_retries == {}
        assert "annotation tool failed" in wf._send_message.await_args.args[1]
        assert "LLM call failed" not in wf._send_message.await_args.args[1]
        assert wf.signal_llm_failure_action("bioinformatics_analysis", action.value, 1)
        assert await asyncio.wait_for(task, 2) == action
        wf._settings_provider.clear_client_cache.assert_not_called()
        assert state.llm_failure_info is None
        assert state.current_plan is plan
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_non_recoverable_failure_waits_without_retrying(monkeypatch):
    from backend.utils.llm_client import AuthenticationError

    slept = _no_backoff(monkeypatch)
    parked = _park_probe(monkeypatch)
    wf = _gate_self()
    state = AnalysisState(user_id="u", round_wait_seconds=0)
    auth = AuthenticationError.__new__(AuthenticationError)
    Exception.__init__(auth, "Error code: 401 - Missing Authentication header")
    task = asyncio.create_task(wf._wait_for_llm_intervention(
        _exhausted(cause=auth), "pi_planning", state
    ))
    try:
        await asyncio.wait_for(parked.wait(), 2)
        assert not task.done()
        assert slept == []
        assert state.llm_failure_info["wait_seconds"] == -1
        assert wf.signal_llm_failure_action("pi_planning", "retry", 1)
        assert await asyncio.wait_for(task, 2) == LlmFailureAction.RETRY
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("node_name", ["bioinformatics_analysis", "planning_dialogue"])
async def test_unattended_exhaustion_retains_gate_and_retry_context(
    monkeypatch, node_name,
):
    """Planning and execution failures retain the same plan until user Retry."""
    from backend.workflow import with_retry

    _no_backoff(monkeypatch)
    parked = _park_probe(monkeypatch)
    wf = _gate_self()
    wf._pause_event = None
    wf._set_status = lambda state, status: setattr(state, "status", status)

    plan = AnalysisPlan(
        id="approved-stage2-plan",
        phase="exploration",
        stage="stage2",
        status="approved",
    )
    analysis_state = AnalysisState(
        user_id="u",
        current_round=1,
        round_wait_seconds=0,
        current_plan=plan,
    )
    workflow_state = {
        "analysis_state": analysis_state,
        "current_round": 1,
        "planning_stage": "stage2",
        "plan_approved": True,
        "skip_nodes": [],
    }
    calls = 0

    @with_retry(node_name)
    async def node(self, state):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise _exhausted()
        return {**state, "bioinformatics_result": {"status": "completed"}}

    task = asyncio.create_task(node(wf, workflow_state))
    await asyncio.wait_for(parked.wait(), timeout=2.5)

    assert not task.done()
    assert analysis_state.status == TaskStatus.WAITING_INPUT
    assert analysis_state.wait_input_reason == "llm_failure"
    assert analysis_state.llm_failure_info["node_name"] == node_name
    assert analysis_state.llm_failure_info["generation"] == 3
    assert analysis_state.current_round == 1
    assert analysis_state.current_plan is plan
    assert wf._llm_failure_events[node_name]

    # The terminal branch parks in the indefinite wait rather than skipping.
    await asyncio.wait_for(parked.wait(), timeout=2.5)
    assert wf.signal_llm_failure_action(
        node_name, LlmFailureAction.RETRY.value, 3
    ) is True
    result = await asyncio.wait_for(task, timeout=2.0)

    assert calls == 4
    assert result["bioinformatics_result"]["status"] == "completed"
    assert analysis_state.current_round == 1
    assert analysis_state.current_plan is plan
    assert analysis_state.current_plan.stage == "stage2"
    assert analysis_state.current_plan.status == "approved"


# ── Test 1: @with_retry decorator LLM failure handling ────────────────


# ── Test 2: signal_llm_failure_action with generation idempotency ────


class TestSignalLlmFailureAction:
    """Test signal_llm_failure_action with generation-based idempotency."""

    @pytest.fixture
    def workflow(self):
        """Create a mock workflow with the signal method bound."""
        from backend.workflow import SingleCellAnalysisWorkflow
        mock = MagicMock(spec=SingleCellAnalysisWorkflow)
        mock.session_id = "test-session"
        mock._llm_failure_generations = {}
        mock._unattended_gate_retries = {}
        mock._llm_failure_events = {}
        mock._llm_failure_actions = {}

        # Bind the actual method
        mock.signal_llm_failure_action = SingleCellAnalysisWorkflow.signal_llm_failure_action.__get__(mock, type(mock))
        mock._generation_for = SingleCellAnalysisWorkflow._generation_for.__get__(mock, type(mock))
        return mock

    def test_rejects_wrong_generation(self, workflow):
        """signal returns False when generation doesn't match."""
        workflow._llm_failure_generations = {"pathology_analysis": 5}
        result = workflow.signal_llm_failure_action("pathology_analysis", "skip", 3)
        assert result is False

    def test_rejects_second_action_for_same_generation(self, workflow):
        workflow._llm_failure_generations = {"pathology_analysis": 5}
        event = asyncio.Event()
        workflow._llm_failure_events["pathology_analysis"] = event

        assert workflow.signal_llm_failure_action(
            "pathology_analysis", "retry", 5,
        ) is True
        assert workflow.signal_llm_failure_action(
            "pathology_analysis", "terminate", 5,
        ) is False
        assert workflow._llm_failure_actions["pathology_analysis"] == "retry"

# ── Test 3: UpstreamExhaustedError context fields ─────────────────────


# ── Test 4: Bioinformatics _bioinformatics_analysis propagation ────
