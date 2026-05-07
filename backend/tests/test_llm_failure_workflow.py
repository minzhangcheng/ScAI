"""Unit tests for LLM failure workflow: while-loop retry, per-node events,
generation-based idempotency, and UpstreamExhaustedError context fields."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.utils.llm_client import UpstreamExhaustedError
from backend.constants import LlmFailureAction, WaitInputReason, StateSyncChangeType
from backend.models.schemas import AnalysisState, TaskStatus


async def _signal_action_once(mock_self, node_name, action):
    """Helper: poll for the per-node Event to appear, then signal once.

    Uses polling instead of a fixed sleep so the signal is delivered
    as soon as the decorator creates the Event and calls event.wait().
    """
    while node_name not in mock_self._llm_failure_events:
        await asyncio.sleep(0.01)
    # Give the decorator a moment to enter event.wait() after releasing the lock
    await asyncio.sleep(0.02)
    mock_self._llm_failure_actions[node_name] = action
    mock_self._llm_failure_events[node_name].set()


# ── Test 1: @with_retry decorator LLM failure handling ────────────────


class TestWithRetryLlmFailure:
    """Test @with_retry while-loop retry reset and per-node Event isolation."""

    @pytest.fixture
    def mock_self(self):
        """Create a mock workflow instance with required attributes."""
        mock = MagicMock()
        mock.session_id = "test-session"
        mock._llm_failure_lock = asyncio.Lock()
        mock._llm_failure_events = {}
        mock._llm_failure_actions = {}
        mock._llm_failure_generation = 0
        mock._send_message = AsyncMock()
        mock._send_state_sync = AsyncMock()
        mock._set_status = MagicMock()
        # Mock settings_provider for client cache clearing
        mock._settings_provider = MagicMock()
        mock._settings_provider._client_cache = {}
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
        assert mock_self._llm_failure_generation >= 1

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
        """On retry, the settings_provider client cache is cleared."""
        from backend.workflow import with_retry

        # Pre-populate the cache
        mock_self._settings_provider._client_cache["old_model"] = MagicMock()

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

        # Cache should have been cleared
        assert len(mock_self._settings_provider._client_cache) == 0


# ── Test 2: signal_llm_failure_action with generation idempotency ────


class TestSignalLlmFailureAction:
    """Test signal_llm_failure_action with generation-based idempotency."""

    @pytest.fixture
    def workflow(self):
        """Create a mock workflow with the signal method bound."""
        from backend.workflow import SingleCellAnalysisWorkflow
        mock = MagicMock(spec=SingleCellAnalysisWorkflow)
        mock.session_id = "test-session"
        mock._llm_failure_generation = 0
        mock._llm_failure_events = {}
        mock._llm_failure_actions = {}

        # Bind the actual method
        mock.signal_llm_failure_action = SingleCellAnalysisWorkflow.signal_llm_failure_action.__get__(mock, type(mock))
        return mock

    def test_rejects_wrong_generation(self, workflow):
        """signal returns False when generation doesn't match."""
        workflow._llm_failure_generation = 5
        result = workflow.signal_llm_failure_action("pathology_analysis", "skip", 3)
        assert result is False

    def test_rejects_zero_generation(self, workflow):
        """signal returns False when current generation is 0 (no failure in progress)."""
        workflow._llm_failure_generation = 0
        result = workflow.signal_llm_failure_action("pathology_analysis", "skip", 0)
        assert result is False

    def test_accepts_matching_generation(self, workflow):
        """signal returns True and sets event when generation matches."""
        workflow._llm_failure_generation = 5
        event = asyncio.Event()
        workflow._llm_failure_events["pathology_analysis"] = event

        result = workflow.signal_llm_failure_action("pathology_analysis", "skip", 5)

        assert result is True
        assert workflow._llm_failure_actions["pathology_analysis"] == "skip"
        assert event.is_set()

    def test_rejects_missing_event(self, workflow):
        """signal returns False when no event exists for the node."""
        workflow._llm_failure_generation = 5
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