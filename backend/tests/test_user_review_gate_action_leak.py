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
            self._approve_calls = []
            self.pi_agent = SimpleNamespace(
                process_user_plan_feedback=AsyncMock(),
            )

        async def _send_message(self, msg_type, content, agent=None):
            self._sent.append((msg_type, content))

        async def _send_state_sync(self, *args, **kwargs):
            pass

        def _set_status(self, *args, **kwargs):
            pass

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
async def test_gate_waits_when_no_action_preset():
    """Fresh gate entry with no pre-set action must block on the event."""
    stub = _make_stub()
    state, _ = _make_state()

    task = asyncio.create_task(stub._user_review_gate(state))
    await asyncio.sleep(0.05)
    assert not task.done(), "gate must wait for user input"

    stub.signal_plan_review("approve")
    await asyncio.wait_for(task, timeout=1.0)
    assert stub._approve_calls == ["approved"]


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


@pytest.mark.asyncio
async def test_clarify_continue_does_not_reuse_modify_action():
    """Regression: the clarify_with_user branch uses `continue`. When a modify
    action arrived during the await (the normal path), _plan_review_action is
    left populated. The next loop iteration must wait for a NEW user signal,
    not replay the previous modify."""
    stub = _make_stub()
    state, _ = _make_state()

    stub.pi_agent.process_user_plan_feedback.side_effect = [
        {"action": "clarify_with_user", "clarification_question": "which subset?"},
        {"action": "proceed_to_execution"},
    ]

    task = asyncio.create_task(stub._user_review_gate(state))
    await asyncio.sleep(0.05)
    assert not task.done(), "gate must wait for user input"

    stub.signal_plan_review("modify", user_input="please refine step 3")
    await asyncio.sleep(0.05)

    assert not task.done(), "gate must wait after clarify_with_user continue"
    assert stub.pi_agent.process_user_plan_feedback.await_count == 1

    stub.signal_plan_review("modify", user_input="merge clusters 4 and 13")
    await asyncio.wait_for(task, timeout=1.0)

    assert stub.pi_agent.process_user_plan_feedback.await_count == 2
    assert stub._approve_calls == ["approved_after_clarification"]
