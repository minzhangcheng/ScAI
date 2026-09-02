"""Unit tests for LLM failure workflow: while-loop retry, per-node events,
generation-based idempotency, and UpstreamExhaustedError context fields."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.utils.llm_client import UpstreamExhaustedError
from backend.constants import LlmFailureAction, StateSyncChangeType
from backend.models.schemas import AnalysisState


async def _signal_action_once(mock_self, node_name, action, timeout=5.0):
    """Helper: poll for the per-node Event to appear, then signal once.

    Uses polling instead of a fixed sleep so the signal is delivered
    as soon as the decorator creates the Event and calls event.wait().

    Bounded by ``timeout``: if the Event never appears (e.g. the code under
    test diverts away from the LLM-failure gate before creating it), this
    raises instead of polling forever — turning a silent suite-wide hang
    into a fast, localized failure.
    """
    waited = 0.0
    while node_name not in mock_self._llm_failure_events:
        if waited >= timeout:
            raise AssertionError(
                f"LLM-failure Event for node '{node_name}' never appeared within "
                f"{timeout}s — the code under test did not reach the intervention "
                f"gate (it likely raised earlier or returned before getting there)."
            )
        await asyncio.sleep(0.01)
        waited += 0.01
    # Give the decorator a moment to enter event.wait() after releasing the lock
    await asyncio.sleep(0.02)
    mock_self._llm_failure_actions[node_name] = action
    mock_self._llm_failure_events[node_name].set()


# ── Unattended gates: honour round_wait_seconds instead of blocking ───


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
    mock.await_restore_safe_point = None
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
async def test_unattended_llm_gate_auto_retries_then_skips(monkeypatch):
    """Nobody answers the dialog. The gate must hand back a bounded number of
    retries and then skip — never park the run on an unanswerable dialog."""
    slept = _no_backoff(monkeypatch)
    wf = _gate_self()
    state = AnalysisState(user_id="u")
    state.round_wait_seconds = 0  # unattended

    actions = [
        await wf._wait_for_llm_intervention(_exhausted(), "bioinformatics", state)
        for _ in range(5)
    ]

    assert actions == [
        LlmFailureAction.RETRY,
        LlmFailureAction.RETRY,
        LlmFailureAction.RETRY,
        LlmFailureAction.SKIP,
        # Budget cleared on skip, so a later independent failure is not starved.
        LlmFailureAction.RETRY,
    ]
    assert slept == [0, 1, 2, 0]


@pytest.mark.asyncio
async def test_unattended_llm_gate_does_not_retry_a_non_recoverable_failure(
    monkeypatch,
):
    """The client breaks out of its own retry loop on auth errors. Re-running
    the node cannot fix a bad key, and unattended nobody will replace it."""
    from backend.utils.llm_client import AuthenticationError

    slept = _no_backoff(monkeypatch)
    wf = _gate_self()
    state = AnalysisState(user_id="u")
    state.round_wait_seconds = 0

    auth = AuthenticationError.__new__(AuthenticationError)
    Exception.__init__(auth, "Error code: 401 - Missing Authentication header")

    action = await wf._wait_for_llm_intervention(
        _exhausted(cause=auth), "pi_planning", state
    )

    assert action == LlmFailureAction.SKIP
    assert slept == []
    assert wf._unattended_gate_retries == {}


@pytest.mark.asyncio
async def test_answer_during_the_backoff_outranks_the_auto_retry(monkeypatch):
    """The gate stays answerable while it backs off. signal_llm_failure_action
    accepts an answer in that window and reports success to the caller, so the
    gate must honour it instead of returning the retry it had already chosen."""
    from backend.tests.split_facade import setattr_facade

    wf = _gate_self()
    state = AnalysisState(user_id="u")
    state.round_wait_seconds = 0

    async def _answer_late(event, attempt):
        # Exactly what the lifecycle route does when the user clicks Terminate.
        accepted = wf.signal_llm_failure_action(
            "bioinformatics",
            LlmFailureAction.TERMINATE.value,
            wf._generation_for("bioinformatics"),
        )
        assert accepted is True, "the gate must still be answerable while backing off"



    setattr_facade(
        monkeypatch, "backend.workflow.core", "_unattended_gate_backoff", _answer_late
    )

    action = await wf._wait_for_llm_intervention(
        _exhausted(), "bioinformatics", state
    )

    assert action == LlmFailureAction.TERMINATE
    # The answer must be consumed, not left behind to leak into a later gate.
    assert wf._llm_failure_actions == {}


@pytest.mark.asyncio
async def test_manual_wait_still_blocks_for_the_user(monkeypatch):
    """round_wait_seconds=-1 is the attended mode: the gate must wait for a
    real answer and must not spend the unattended budget."""
    _no_backoff(monkeypatch)
    wf = _gate_self()
    state = AnalysisState(user_id="u")
    state.round_wait_seconds = -1

    gate = asyncio.create_task(
        wf._wait_for_llm_intervention(_exhausted(), "bioinformatics", state)
    )
    await _signal_action_once(wf, "bioinformatics", LlmFailureAction.TERMINATE.value)

    assert await gate == LlmFailureAction.TERMINATE
    assert wf._unattended_gate_retries == {}


@pytest.mark.asyncio
async def test_explicit_wait_seconds_overrides_round_wait(monkeypatch):
    """A caller that supplies its own budget owns the unattended path; the
    session-wide round wait must not override it."""
    _no_backoff(monkeypatch)
    wf = _gate_self()
    state = AnalysisState(user_id="u")
    state.round_wait_seconds = -1  # would otherwise block forever

    action = await wf._wait_for_llm_intervention(
        _exhausted(), "generate_detailed_report", state,
        wait_seconds=0, auto_retry=False,
    )

    assert action == LlmFailureAction.SKIP


@pytest.mark.asyncio
async def test_unattended_disabled_agent_gate_skips_without_retrying(monkeypatch):
    """A disabled agent has an empty model. Retrying re-runs the node against
    the same empty configuration, so unattended it must skip immediately."""
    slept = _no_backoff(monkeypatch)
    wf = _gate_self()
    state = AnalysisState(user_id="u")
    state.round_wait_seconds = 0

    action = await wf._wait_for_disabled_agent_intervention(
        "pathology", "pathology_analysis", state
    )

    assert action == LlmFailureAction.SKIP
    assert slept == []


@pytest.mark.asyncio
async def test_node_entry_resets_the_unattended_retry_budget(monkeypatch):
    """The budget bounds one failure episode. A node that fails, recovers and
    later fails again must not inherit the spent budget."""
    from backend.workflow import with_retry

    _no_backoff(monkeypatch)
    wf = _gate_self()
    wf._pause_event = None
    wf._unattended_gate_retries["data_profiling"] = 3

    @with_retry("data_profiling")
    async def node(self, state):
        return {**state, "ok": True}

    analysis_state = AnalysisState(user_id="u")
    analysis_state.round_wait_seconds = 0
    result = await node(wf, {"analysis_state": analysis_state, "skip_nodes": []})

    assert result["ok"] is True
    assert "data_profiling" not in wf._unattended_gate_retries


# ── Test 1: @with_retry decorator LLM failure handling ────────────────


class TestWithRetryLlmFailure:
    """Test @with_retry while-loop retry reset and per-node Event isolation."""

    @pytest.fixture
    def mock_self(self):
        """Create a mock workflow instance with required attributes."""
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
        # Mock settings_provider for client cache clearing
        mock._settings_provider = MagicMock()
        # Bind real async methods — MagicMock auto-creates a non-awaitable stub
        mock._wait_for_llm_intervention = SingleCellAnalysisWorkflow._wait_for_llm_intervention.__get__(mock, type(mock))
        mock._drive_hitl_gate = SingleCellAnalysisWorkflow._drive_hitl_gate.__get__(mock, type(mock))
        mock._next_generation = SingleCellAnalysisWorkflow._next_generation.__get__(mock, type(mock))
        mock._generation_for = SingleCellAnalysisWorkflow._generation_for.__get__(mock, type(mock))
        return mock

    @pytest.fixture
    def base_state(self):
        """Create a base workflow state."""
        analysis_state = AnalysisState(user_id="test-user")
        return {
            "analysis_state": analysis_state,
            "skip_nodes": [],
        }

    @pytest.mark.asyncio
    async def test_while_loop_resets_attempt_on_retry(self, mock_self, base_state):
        """When user chooses retry, attempt counter resets to 0 and func re-runs."""
        from backend.workflow import with_retry

        call_count = 0

        async def failing_once_func(self, state):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise UpstreamExhaustedError(
                    "test error",
                    agent_name="pathology",
                    model="gpt-4",
                    llm_server="https://api.openai.com/v1",
                )
            return {"result": "success"}

        decorated = with_retry("pathology_analysis", max_retries=3)(failing_once_func)

        # Background task: wait for the Event, then signal RETRY
        task = asyncio.create_task(
            _signal_action_once(mock_self, "pathology_analysis", LlmFailureAction.RETRY.value)
        )
        result = await decorated(mock_self, base_state)
        await task

        # Func called twice: 1st raised UpstreamExhaustedError, 2nd succeeded
        # This proves attempt counter was reset (otherwise the while-loop
        # would have exited and returned a failure dict).
        assert call_count == 2
        assert result.get("result") == "success"

    @pytest.mark.asyncio
    async def test_per_node_event_isolation(self, mock_self, base_state):
        """Two nodes failing concurrently each get their own Event."""
        event1 = asyncio.Event()
        event2 = asyncio.Event()
        mock_self._llm_failure_events = {
            "pathology_analysis": event1,
            "bioinformatics_analysis": event2,
        }

        # Events should be separate objects
        assert (
            mock_self._llm_failure_events["pathology_analysis"]
            is not mock_self._llm_failure_events["bioinformatics_analysis"]
        )

    @pytest.mark.asyncio
    async def test_generation_increments_on_each_entry(self, mock_self, base_state):
        """Each entry into WAITING_INPUT increments generation."""
        from backend.workflow import with_retry

        async def always_failing_func(self, state):
            raise UpstreamExhaustedError(
                "test",
                agent_name="pi",
                model="gpt-4",
                llm_server="https://api.example.com",
            )

        # max_retries=0 => while attempt <= 0, only attempt=0 runs.
        # The UpstreamExhaustedError handler does NOT increment attempt;
        # it enters the user-intervention path instead.
        decorated = with_retry("pi_planning", max_retries=0)(always_failing_func)

        task = asyncio.create_task(
            _signal_action_once(mock_self, "pi_planning", LlmFailureAction.SKIP.value)
        )
        result = await decorated(mock_self, base_state)
        await task

        # Generation should have been incremented at least once
        assert mock_self._generation_for("pi_planning") >= 1

    @pytest.mark.asyncio
    async def test_non_llm_error_unchanged(self, mock_self, base_state):
        """ValueError (non-UpstreamExhaustedError) does NOT enter WAITING_INPUT."""
        from backend.workflow import with_retry

        async def failing_func(self, state):
            raise ValueError("some non-LLM error")

        # max_retries=0 => only one attempt, no retry back-off delays
        decorated = with_retry("pathology_analysis", max_retries=0)(failing_func)

        result = await decorated(mock_self, base_state)

        # Should not have set any LLM failure state
        assert base_state["analysis_state"].wait_input_reason is None
        assert base_state["analysis_state"].llm_failure_info is None
        # Should have failed_node set
        assert result.get("failed_node") == "pathology_analysis"

    @pytest.mark.asyncio
    async def test_skip_action_returns_failed_node(self, mock_self, base_state):
        """When user chooses skip, the node is recorded as failed and returned."""
        from backend.workflow import with_retry

        async def always_failing_func(self, state):
            raise UpstreamExhaustedError(
                "timeout",
                agent_name="pathology",
                model="gpt-4",
                llm_server="https://api.openai.com/v1",
            )

        decorated = with_retry("pathology_analysis", max_retries=0)(always_failing_func)

        task = asyncio.create_task(
            _signal_action_once(mock_self, "pathology_analysis", LlmFailureAction.SKIP.value)
        )
        result = await decorated(mock_self, base_state)
        await task

        assert result.get("failed_node") == "pathology_analysis"
        assert result.get("pathology_result", {}).get("status") == "failed"
        # Failure state should be cleared after skip
        assert base_state["analysis_state"].wait_input_reason is None
        assert base_state["analysis_state"].llm_failure_info is None

    @pytest.mark.asyncio
    async def test_retry_clears_client_cache(self, mock_self, base_state):
        """On retry, the provider's ownership-preserving clear API is called."""
        from backend.workflow import with_retry

        call_count = 0

        async def failing_once_func(self, state):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise UpstreamExhaustedError(
                    "test error",
                    agent_name="pathology",
                    model="gpt-4",
                    llm_server="https://api.openai.com/v1",
                )
            return {"result": "success"}

        decorated = with_retry("pathology_analysis", max_retries=3)(failing_once_func)

        task = asyncio.create_task(
            _signal_action_once(mock_self, "pathology_analysis", LlmFailureAction.RETRY.value)
        )
        result = await decorated(mock_self, base_state)
        await task

        mock_self._settings_provider.clear_client_cache.assert_called_once_with()


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

    def test_rejects_zero_generation(self, workflow):
        """signal returns False when current generation is 0 (no failure in progress)."""
        workflow._llm_failure_generations = {}
        workflow._unattended_gate_retries = {}
        result = workflow.signal_llm_failure_action("pathology_analysis", "skip", 0)
        assert result is False

    def test_accepts_matching_generation(self, workflow):
        """signal returns True and sets event when generation matches."""
        workflow._llm_failure_generations = {"pathology_analysis": 5}
        event = asyncio.Event()
        workflow._llm_failure_events["pathology_analysis"] = event

        result = workflow.signal_llm_failure_action("pathology_analysis", "skip", 5)

        assert result is True
        assert workflow._llm_failure_actions["pathology_analysis"] == "skip"
        assert event.is_set()

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

    def test_rejects_missing_event(self, workflow):
        """signal returns False when no event exists for the node."""
        workflow._llm_failure_generations = {"pathology_analysis": 5}
        # No event registered for this node

        result = workflow.signal_llm_failure_action("pathology_analysis", "skip", 5)
        assert result is False


# ── Test 3: UpstreamExhaustedError context fields ─────────────────────


class TestUpstreamExhaustedError:
    """Test UpstreamExhaustedError with new context fields."""

    def test_default_values(self):
        error = UpstreamExhaustedError("test message")
        assert error.agent_name == "unknown"
        assert error.model == ""
        assert error.llm_server == ""
        assert str(error) == "test message"

    def test_custom_values(self):
        error = UpstreamExhaustedError(
            "test error",
            agent_name="pathology",
            model="gpt-4",
            llm_server="https://api.openai.com/v1",
        )
        assert error.agent_name == "pathology"
        assert error.model == "gpt-4"
        assert error.llm_server == "https://api.openai.com/v1"

    def test_backward_compatible(self):
        """Existing code that only passes message should still work."""
        error = UpstreamExhaustedError("timeout occurred")
        assert str(error) == "timeout occurred"
        assert error.agent_name == "unknown"


# ── Test 4: Bioinformatics _bioinformatics_analysis propagation ────


class TestBioinformaticsNodeUpstreamPropagation:
    """Verify that UpstreamExhaustedError from bioinformatics_agent
    propagates through _bioinformatics_analysis (the @with_retry-decorated
    workflow node) to trigger LlmTimeoutDialog.

    This is a regression test: if someone re-adds 'except
    UpstreamExhaustedError: break' inside the agent loop or 'except
    UpstreamExhaustedError: return' in the workflow caller, these
    tests will fail because the error won't reach @with_retry.
    """

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow with the real _bioinformatics_analysis method bound."""
        from backend.workflow import SingleCellAnalysisWorkflow

        mock = MagicMock(spec=SingleCellAnalysisWorkflow)
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
        mock._settings_provider._client_cache = {}
        mock._collect_current_round_results = MagicMock(return_value={})
        mock._build_previous_findings_context = MagicMock(return_value="")
        mock._build_code_history_summary = MagicMock(return_value="")
        mock._figure_review_callback = None
        mock._supervisor_callback = None
        mock.supervisor_agent = None
        mock.config = MagicMock()
        mock.config.bioinfo_max_steps = 10

        # Mock the bioinformatics agent so run_autonomous_analysis raises UpstreamExhaustedError
        mock.bioinformatics_agent = MagicMock()
        # has_resumable_checkpoint is awaited in _bioinformatics_analysis BEFORE
        # run_autonomous_analysis. A plain MagicMock attribute returns a
        # non-awaitable MagicMock, which raises TypeError and diverts the flow
        # away from the UpstreamExhaustedError path the test exercises. Mock it
        # as an async no-checkpoint so the run reaches run_autonomous_analysis.
        mock.bioinformatics_agent.has_resumable_checkpoint = AsyncMock(return_value=False)
        mock.bioinformatics_agent.run_autonomous_analysis = AsyncMock(
            side_effect=UpstreamExhaustedError(
                "No available channel for model claude-sonnet-4.6",
                agent_name="bioinformatics",
                model="claude-sonnet-4.6",
                llm_server="https://api.n1n.ai/v1",
            )
        )

        # Mock pi_agent for the PI planning section of _bioinformatics_analysis
        mock.pi_agent = MagicMock()
        mock.pi_agent.plan_agent_tasks = AsyncMock(return_value=["Perform standard scRNA-seq analysis"])

        # Bind real async methods — MagicMock auto-creates a non-awaitable stub
        mock._wait_for_llm_intervention = SingleCellAnalysisWorkflow._wait_for_llm_intervention.__get__(mock, type(mock))
        mock._drive_hitl_gate = SingleCellAnalysisWorkflow._drive_hitl_gate.__get__(mock, type(mock))
        mock._require_core_agent = SingleCellAnalysisWorkflow._require_core_agent.__get__(mock, type(mock))
        mock._next_generation = SingleCellAnalysisWorkflow._next_generation.__get__(mock, type(mock))
        mock._generation_for = SingleCellAnalysisWorkflow._generation_for.__get__(mock, type(mock))

        # Bind the real method
        mock._bioinformatics_analysis = SingleCellAnalysisWorkflow._bioinformatics_analysis.__get__(mock, type(mock))

        return mock

    @pytest.fixture
    def bioinfo_state(self):
        """Create a workflow state that will reach the run_autonomous_analysis call."""
        from backend.models.schemas import DataFile
        analysis_state = AnalysisState(user_id="test-user")
        # has_sequencing_data() checks data_files (not data_sections);
        # without a sequencing DataFile, _bioinformatics_analysis returns
        # early and never calls run_autonomous_analysis.
        analysis_state.data_files = [
            DataFile(
                filename="test.h5ad",
                file_path="/tmp/test.h5ad",
                section="sequencing",
                data_type="scrnaseq",
            )
        ]
        analysis_state.current_round = 1
        return {
            "analysis_state": analysis_state,
            "skip_nodes": [],
            "pi_instructions": {
                "bioinformatics_tasks": ["Perform standard scRNA-seq analysis"],
                "overview": "Test analysis",
            },
            "human_guidance": "",
            "pathology_result": None,
            "biochemistry_result": None,
        }

    @pytest.mark.asyncio
    async def test_upstream_error_from_agent_reaches_with_retry(self, mock_workflow, bioinfo_state):
        # Signal SKIP so the test doesn't hang on event.wait()
        task = asyncio.create_task(
            _signal_action_once(mock_workflow, "bioinformatics_analysis", LlmFailureAction.SKIP.value)
        )
        result = await mock_workflow._bioinformatics_analysis(bioinfo_state)
        await task

        # If the error was swallowed (old behavior), @with_retry would NOT
        # have been triggered and these assertions would fail:
        # 1. generation incremented => @with_retry handled the error
        assert mock_workflow._generation_for("bioinformatics_analysis") >= 1
        # 2. _send_state_sync called with LLM_FAILURE => popup triggered
        # Note: decorator sends LLM_FAILURE first (line 164), then LLM_RETRY
        # after the user action is processed (line 180). call_args references
        # the last call, so use assert_any_call to check LLM_FAILURE.
        mock_workflow._send_state_sync.assert_any_call(
            bioinfo_state["analysis_state"], StateSyncChangeType.LLM_FAILURE.value
        )
        # 3. Result has failed_node => @with_retry returned proper state dict
        assert result.get("failed_node") == "bioinformatics_analysis"
        assert result.get("bioinformatics_result", {}).get("status") == "failed"

    @pytest.mark.asyncio
    async def test_upstream_error_populates_llm_failure_info(self, mock_workflow, bioinfo_state):
        captured_info = {}
        original_sync = mock_workflow._send_state_sync

        async def capture_and_forward(*args, **kwargs):
            info = bioinfo_state["analysis_state"].llm_failure_info
            if info:
                captured_info.update(info)
            await original_sync(*args, **kwargs)

        mock_workflow._send_state_sync = capture_and_forward

        task = asyncio.create_task(
            _signal_action_once(mock_workflow, "bioinformatics_analysis", LlmFailureAction.SKIP.value)
        )
        await mock_workflow._bioinformatics_analysis(bioinfo_state)
        await task

        # llm_failure_info must contain the UpstreamExhaustedError context
        assert captured_info.get("agent") == "bioinformatics"
        assert captured_info.get("model") == "claude-sonnet-4.6"
        assert captured_info.get("llm_server") == "https://api.n1n.ai/v1"
        assert captured_info.get("node_name") == "bioinformatics_analysis"

    @pytest.mark.asyncio
    async def test_upstream_error_not_swallowed_by_agent_break(self, mock_workflow, bioinfo_state):
        """Regression guard: if run_autonomous_analysis raises UpstreamExhaustedError,
        it must NOT be silently caught and turned into a normal result (the old
        'except ... break' pattern that returned status='max_steps_reached')."""
        task = asyncio.create_task(
            _signal_action_once(mock_workflow, "bioinformatics_analysis", LlmFailureAction.SKIP.value)
        )
        result = await mock_workflow._bioinformatics_analysis(bioinfo_state)
        await task

        # Old bug: agent break => status='max_steps_reached', no failed_node
        # Fix: error propagates => @with_retry sets failed_node
        assert result.get("failed_node") == "bioinformatics_analysis"
        # Must NOT be a "normal" result like max_steps_reached
        bioinfo_result = result.get("bioinformatics_result", {})
        assert bioinfo_result.get("status") == "failed"
        assert bioinfo_result.get("status") != "max_steps_reached"

    @pytest.mark.asyncio
    async def test_runtime_discard_unwinds_superseded_bioinformatics_node(
        self, mock_workflow, bioinfo_state,
    ):
        from backend.services.runtime_persistence_gate import (
            RuntimePersistenceBlocked,
        )
        from backend.workflow.helpers import WorkflowSupersededError

        mock_workflow.bioinformatics_agent.run_autonomous_analysis = AsyncMock(
            side_effect=RuntimePersistenceBlocked("runtime save failed")
        )
        mock_workflow._wait_for_runtime_persistence_intervention = AsyncMock(
            return_value="discard"
        )

        with pytest.raises(WorkflowSupersededError):
            await mock_workflow._bioinformatics_analysis(bioinfo_state)

        assert (
            mock_workflow.bioinformatics_agent.run_autonomous_analysis.await_count
            == 1
        )
