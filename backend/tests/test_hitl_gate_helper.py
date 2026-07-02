import asyncio
import pytest
from backend.constants import LlmFailureAction


def _bare_workflow():
    from backend.workflow.core import SingleCellAnalysisWorkflow
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "s1"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generations = {}
    wf._settings_provider = None
    async def _noop(*a, **k): pass
    wf._send_message = _noop
    wf._send_state_sync = _noop
    wf._set_status = lambda s, st: setattr(s, "status", st)
    return wf


class _State:
    wait_input_reason = None
    llm_failure_info = None
    agent_disabled_info = None
    status = None


@pytest.mark.asyncio
async def test_gate_retry_clears_cache_when_requested():
    wf = _bare_workflow()
    state = _State()

    async def resolve():
        await asyncio.sleep(0)
        wf._llm_failure_actions["nodeX"] = LlmFailureAction.RETRY.value
        wf._llm_failure_events["nodeX"].set()
    asyncio.create_task(resolve())
    action = await wf._drive_hitl_gate(
        "nodeX", state, reason="llm_failure",
        info_attr="llm_failure_info", info_payload={"node_name": "nodeX"},
        enter_change="llm_failure", exit_change="llm_retry",
        clear_cache_on_retry=True,
    )
    assert action == LlmFailureAction.RETRY


@pytest.mark.asyncio
async def test_gate_skip_returns_skip():
    wf = _bare_workflow()
    state = _State()

    async def resolve():
        await asyncio.sleep(0)
        wf._llm_failure_actions["nodeY"] = LlmFailureAction.SKIP.value
        wf._llm_failure_events["nodeY"].set()
    asyncio.create_task(resolve())
    action = await wf._drive_hitl_gate(
        "nodeY", state, reason="agent_disabled",
        info_attr="agent_disabled_info", info_payload={"agent": "pi"},
        enter_change="agent_disabled", exit_change="agent_resumed",
        clear_cache_on_retry=False,
    )
    assert action == LlmFailureAction.SKIP


@pytest.mark.asyncio
async def test_gate_terminate_returns_terminate():
    wf = _bare_workflow()
    state = _State()

    async def resolve():
        await asyncio.sleep(0)
        wf._llm_failure_actions["nodeZ"] = LlmFailureAction.TERMINATE.value
        wf._llm_failure_events["nodeZ"].set()
    asyncio.create_task(resolve())
    action = await wf._drive_hitl_gate(
        "nodeZ", state, reason="marker_db_missing",
        info_attr="llm_failure_info", info_payload={},
        enter_change="llm_failure", exit_change="llm_retry",
        clear_cache_on_retry=False,
    )
    assert action == LlmFailureAction.TERMINATE


@pytest.mark.asyncio
async def test_gate_sets_and_clears_state():
    wf = _bare_workflow()
    state = _State()

    async def resolve():
        await asyncio.sleep(0)
        wf._llm_failure_actions["nodeA"] = LlmFailureAction.SKIP.value
        wf._llm_failure_events["nodeA"].set()
    asyncio.create_task(resolve())
    await wf._drive_hitl_gate(
        "nodeA", state, reason="test_reason",
        info_attr="llm_failure_info", info_payload={"k": "v"},
        enter_change="llm_failure", exit_change="llm_retry",
        clear_cache_on_retry=False,
    )
    assert state.wait_input_reason is None
    assert state.llm_failure_info is None
