"""Regression tests: _user_review_gate must not consume a stale _plan_review_action
left behind by a previous gate invocation or a previous iteration of the same gate.

Each test builds a minimal stub that mixes in PlanningMixin and supplies the
collaborators the gate touches (send_message, send_state_sync, set_status,
_build_plan_dialogue_html, pi_agent.process_user_plan_feedback, _approve_plan).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_stub():
    from backend.workflow.planning import PlanningMixin

    class _GateStub(PlanningMixin):
        def __init__(self):
            self.session_id = "test-session"
            self._plan_review_event = asyncio.Event()
            self._plan_review_action = None
            self._sent = []
            self._syncs = []
            self._approve_calls = []
            self.pi_agent = SimpleNamespace(
                process_user_plan_feedback=AsyncMock(),
            )

        async def _send_message(self, msg_type, content, agent=None):
            self._sent.append((msg_type, content))

        async def _send_state_sync(self, state, change_type):
            self._syncs.append((state.status, change_type))

        def _set_status(self, state, status):
            state.status = status

        def _build_plan_dialogue_html(self, *args, **kwargs):
            return "<html></html>"

        async def _approve_plan(self, state, plan, analysis_state, action):
            self._approve_calls.append(action)
            plan.status = "approved"
            state["plan_approved"] = True

        def signal_plan_review(self, action, user_input=None):
            self._plan_review_action = {"action": action, "user_input": user_input}
            self._plan_review_event.set()

    return _GateStub()


def _make_state():
    plan = MagicMock()
    plan.status = "pi_reviewing"
    plan.model_dump.return_value = {"id": "plan-1"}
    analysis_state = MagicMock()
    analysis_state.current_plan = plan
    analysis_state.plan_dialogue_history = []
    analysis_state.analysis_phase = SimpleNamespace(value="exploration")
    analysis_state.current_round = 1
    analysis_state.round_wait_seconds = -1
    analysis_state.auto_approve_plans = False
    analysis_state.plan_review_wait_seconds = -1
    state = {
        "analysis_state": analysis_state,
        "planning_stage": "stage1",
    }
    return state, plan


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", ["retry", "skip", "terminate"])
async def test_feedback_failure_preserves_input_and_never_implies_approval(resolution):
    from backend.constants import LlmFailureAction
    from backend.utils.llm_client import UpstreamExhaustedError
    from backend.workflow.helpers import WorkflowTerminatedError

    stub = _make_stub()
    state, _ = _make_state()
    feedback = "Add a comparison before execution"
    stub.pi_agent.process_user_plan_feedback.side_effect = [
        UpstreamExhaustedError("timeout", agent_name="pi"),
        {"action": "revise_with_bioinfo", "guidance_for_bioinfo": feedback},
    ]

    async def intervene(*args, **kwargs):
        assert state["user_plan_feedback"] == feedback
        return LlmFailureAction(resolution)

    stub._wait_for_llm_intervention = AsyncMock(side_effect=intervene)
    task = asyncio.create_task(stub._user_review_gate(state))
    await asyncio.sleep(0)
    stub.signal_plan_review("modify", feedback)
    if resolution == "terminate":
        with pytest.raises(WorkflowTerminatedError):
            await asyncio.wait_for(task, 1)
    elif resolution == "retry":
        result = await asyncio.wait_for(task, 1)
        assert result["user_plan_feedback"] == feedback
        assert stub.pi_agent.process_user_plan_feedback.await_count == 2
    else:
        for _ in range(20):
            if sum(kind == "plan_review" for kind, _ in stub._sent) == 2:
                break
            await asyncio.sleep(0)
        assert not task.done()
        assert sum(kind == "plan_review" for kind, _ in stub._sent) == 2
        assert state["user_plan_feedback"] == feedback
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stub._approve_calls == []
    stub._wait_for_llm_intervention.assert_awaited_once()


@pytest.mark.asyncio
async def test_gate_does_not_reuse_action_from_previous_invocation():
    """Regression: stage1→stage2 leak. When gate #1 consumes an approve that
    arrived during its await (the normal runtime path), _plan_review_action is
    left populated after the break. Gate #2 must wait for a fresh user signal,
    not auto-approve via the stale action."""
    stub = _make_stub()
    state1, _ = _make_state()

    task1 = asyncio.create_task(stub._user_review_gate(state1))
    await asyncio.sleep(0.05)
    assert not task1.done(), "gate #1 must wait for user input"

    stub.signal_plan_review("approve")
    await asyncio.wait_for(task1, timeout=1.0)
    assert stub._approve_calls == ["approved"]

    state2, _ = _make_state()
    state2["planning_stage"] = "stage2"
    task2 = asyncio.create_task(stub._user_review_gate(state2))
    await asyncio.sleep(0.05)
    assert not task2.done(), (
        "gate #2 must wait for user; instead it consumed stale approve from gate #1"
    )

    stub.signal_plan_review("skip")
    await asyncio.wait_for(task2, timeout=1.0)
    assert stub._approve_calls == ["approved", "skipped"]
