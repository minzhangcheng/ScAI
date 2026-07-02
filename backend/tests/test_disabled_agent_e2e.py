"""End-to-end: clear PI model, verify dialog raised; signal retry
after reconfiguring; verify workflow resumes."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.workflow.core import SingleCellAnalysisWorkflow, WorkflowTerminatedError
from backend.constants import LlmFailureAction, WaitInputReason
from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.agents import AgentNodesMixin


@pytest.mark.asyncio
async def test_disabled_pi_raises_dialog_and_resumes_on_retry():
    """Start with empty PI model, verify agent_disabled state,
    reconfigure model, signal retry, verify proceeds."""
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-e2e"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generations = {}
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = lambda s, st: setattr(s, 'status', st)

    wf._settings_provider = MagicMock()
    wf._settings_provider.get_model = MagicMock(return_value="")
    wf._settings_provider._client_cache = {}

    analysis_state = AnalysisState(session_id="test-e2e", user_id="test")
    pi_instructions = {"bioinformatics_tasks": ["task1"]}
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": pi_instructions,
        "skip_nodes": [],
    }

    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf, type(wf))
    wf._wait_for_disabled_agent_intervention = (
        SingleCellAnalysisWorkflow._wait_for_disabled_agent_intervention.__get__(wf, type(wf))
    )

    async def run_guard():
        return await wf._require_core_agent(state, "pi", "pi_planning")

    guard_task = asyncio.create_task(run_guard())
    await asyncio.sleep(0.05)

    assert analysis_state.wait_input_reason == WaitInputReason.AGENT_DISABLED.value
    assert analysis_state.agent_disabled_info is not None
    assert analysis_state.agent_disabled_info["agent"] == "pi"
    assert analysis_state.agent_disabled_info["node_name"] == "pi_planning"
    assert analysis_state.agent_disabled_info["generation"] >= 1
    assert analysis_state.status == TaskStatus.WAITING_INPUT
    wf._send_state_sync.assert_called()

    # Reconfigure model
    wf._settings_provider.get_model = MagicMock(return_value="some/model")

    gen = analysis_state.agent_disabled_info["generation"]
    ok = wf.signal_llm_failure_action("pi_planning", "retry", gen)
    assert ok, "signal_llm_failure_action should accept matching generation"

    result = await asyncio.wait_for(guard_task, timeout=1.0)
    assert result is None, "_require_core_agent should return None when model is set"
    assert analysis_state.wait_input_reason is None
    assert analysis_state.agent_disabled_info is None


@pytest.mark.asyncio
async def test_disabled_pi_skip_adds_to_skip_nodes():
    """When user chooses SKIP for a disabled core agent,
    the node is added to skip_nodes and a skip-result is returned."""
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-e2e-skip"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generations = {}
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = lambda s, st: setattr(s, 'status', st)

    wf._settings_provider = MagicMock()
    wf._settings_provider.get_model = MagicMock(return_value="")
    wf._settings_provider._client_cache = {}

    analysis_state = AnalysisState(session_id="test-e2e-skip", user_id="test")
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"bioinformatics_tasks": ["task1"]},
        "skip_nodes": [],
    }

    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf, type(wf))
    wf._wait_for_disabled_agent_intervention = (
        SingleCellAnalysisWorkflow._wait_for_disabled_agent_intervention.__get__(wf, type(wf))
    )

    async def run_guard():
        return await wf._require_core_agent(state, "pi", "pi_planning")

    guard_task = asyncio.create_task(run_guard())
    await asyncio.sleep(0.05)

    gen = analysis_state.agent_disabled_info["generation"]
    ok = wf.signal_llm_failure_action("pi_planning", "skip", gen)
    assert ok

    result = await asyncio.wait_for(guard_task, timeout=1.0)

    assert result is not None
    assert "pi_planning" in result.get("skip_nodes", [])
    assert result.get("pi_instructions", {}).get("status") == "skipped"


@pytest.mark.asyncio
async def test_disabled_pi_terminate_raises_error():
    """When user chooses TERMINATE for a disabled core agent,
    WorkflowTerminatedError is raised."""
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-e2e-term"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generations = {}
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = lambda s, st: setattr(s, 'status', st)

    wf._settings_provider = MagicMock()
    wf._settings_provider.get_model = MagicMock(return_value="")
    wf._settings_provider._client_cache = {}

    analysis_state = AnalysisState(session_id="test-e2e-term", user_id="test")
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"bioinformatics_tasks": ["task1"]},
        "skip_nodes": [],
    }

    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf, type(wf))
    wf._wait_for_disabled_agent_intervention = (
        SingleCellAnalysisWorkflow._wait_for_disabled_agent_intervention.__get__(wf, type(wf))
    )

    async def run_guard():
        return await wf._require_core_agent(state, "pi", "pi_planning")

    guard_task = asyncio.create_task(run_guard())
    await asyncio.sleep(0.05)

    gen = analysis_state.agent_disabled_info["generation"]
    ok = wf.signal_llm_failure_action("pi_planning", "terminate", gen)
    assert ok

    with pytest.raises(WorkflowTerminatedError):
        await asyncio.wait_for(guard_task, timeout=1.0)


@pytest.mark.asyncio
async def test_planning_dialogue_blocks_when_bioinformatics_disabled():
    """_planning_dialogue blocks on bioinformatics being disabled
    (pi passes the guard, bioinformatics triggers agent_disabled)."""
    from backend.workflow.planning import PlanningMixin

    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-dialogue-block"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generations = {}
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = lambda s, st: setattr(s, 'status', st)

    wf._settings_provider = MagicMock()
    wf._settings_provider.get_model = MagicMock(side_effect=lambda a: {
        "pi": "some/model",
        "bioinformatics": "",
    }.get(a, ""))
    wf._settings_provider._client_cache = {}

    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf, type(wf))
    wf._wait_for_disabled_agent_intervention = (
        SingleCellAnalysisWorkflow._wait_for_disabled_agent_intervention.__get__(wf, type(wf))
    )
    wf._build_planning_context = MagicMock(return_value=None)
    wf.pi_agent = MagicMock()
    wf.bioinformatics_agent = MagicMock()

    analysis_state = AnalysisState(session_id="t", user_id="u")
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"bioinformatics_tasks": ["task1"], "overview": "test"},
        "planning_stage": "full",
        "skip_nodes": [],
    }

    guard_task = asyncio.create_task(wf._planning_dialogue(state))
    await asyncio.sleep(0.05)

    assert analysis_state.wait_input_reason == WaitInputReason.AGENT_DISABLED.value
    assert analysis_state.agent_disabled_info is not None
    assert analysis_state.agent_disabled_info["agent"] == "bioinformatics"

    gen = analysis_state.agent_disabled_info["generation"]
    ok = wf.signal_llm_failure_action("planning_dialogue", "skip", gen)
    assert ok
    result = await asyncio.wait_for(guard_task, timeout=1.0)
    assert result is not None


@pytest.mark.asyncio
async def test_send_state_sync_includes_agent_disabled_info():
    """_send_state_sync payload must include agent_disabled_info so the
    frontend can display the disabled-agent dialog (regression guard)."""
    from backend.workflow.messaging import MessagingMixin
    from backend.models.schemas import WebSocketMessage

    messages: list[WebSocketMessage] = []

    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "t"
    wf.state_callback = None
    wf.message_callback = messages.append
    wf._send_message = AsyncMock()

    analysis_state = AnalysisState(session_id="t", user_id="u")
    analysis_state.agent_disabled_info = {
        "agent": "pi",
        "node_name": "pi_planning",
        "generation": 1,
    }

    wf._send_state_sync = MessagingMixin._send_state_sync.__get__(wf, type(wf))
    await wf._send_state_sync(analysis_state, "agent_disabled")

    assert len(messages) == 1
    content = messages[0].content
    assert content.get("agent_disabled_info") is not None
    assert content["agent_disabled_info"]["agent"] == "pi"
    assert content["agent_disabled_info"]["node_name"] == "pi_planning"
    assert content["agent_disabled_info"]["generation"] == 1
    assert content.get("change_type") == "agent_disabled"
