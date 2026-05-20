import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.constants import LlmFailureAction, WaitInputReason
from backend.models.schemas import AnalysisState


@pytest.mark.asyncio
async def test_wait_for_disabled_agent_returns_retry_on_signal():
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "s"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generation = 0
    wf._settings_provider = MagicMock()
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = MagicMock()
    state = AnalysisState(session_id="s", user_id="u")

    async def signaller():
        await asyncio.sleep(0.02)
        gen = wf._llm_failure_generation
        wf.signal_llm_failure_action("pi_planning", "retry", gen)

    waiter = asyncio.create_task(
        wf._wait_for_disabled_agent_intervention("pi", "pi_planning", state)
    )
    asyncio.create_task(signaller())
    action = await asyncio.wait_for(waiter, timeout=1.0)
    assert action == LlmFailureAction.RETRY


@pytest.mark.asyncio
async def test_wait_for_disabled_agent_returns_skip_on_signal():
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "s"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generation = 0
    wf._settings_provider = MagicMock()
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = MagicMock()
    state = AnalysisState(session_id="s", user_id="u")

    async def signaller():
        await asyncio.sleep(0.02)
        gen = wf._llm_failure_generation
        wf.signal_llm_failure_action("pi_planning", "skip", gen)

    waiter = asyncio.create_task(
        wf._wait_for_disabled_agent_intervention("pi", "pi_planning", state)
    )
    asyncio.create_task(signaller())
    action = await asyncio.wait_for(waiter, timeout=1.0)
    assert action == LlmFailureAction.SKIP
