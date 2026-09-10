"""Objective guards for malformed and identical execute-code requests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents import bioinformatics_agent as bio_mod
from backend.agents.bioinformatics import executor_execute, executor_loop
from backend.models.schemas import AnalysisPlan, AnalysisState, CodeExecution, PlanStep


def _state(plan_id: str = "plan-current") -> AnalysisState:
    return AnalysisState(
        id="no-progress-session",
        user_id="user",
        current_plan=AnalysisPlan(
            id=plan_id,
            phase="exploration",
            status="approved",
            steps=[PlanStep(id=1, title="Current task", method="Run it")],
        ),
    )


def _execution(
    code: str, *, plan_id: str = "plan-current", status="succeeded", **values
):
    return CodeExecution(
        code=code,
        plan_id=plan_id,
        plan_step_ids=[1],
        execution_purpose="planned",
        plan_binding_status="candidate",
        execution_status=status,
        **values,
    )


@pytest.mark.parametrize("args", [{}, {"code": "   \n"}, {"code": "# comment only"}])
def test_execute_code_requires_an_executable_statement(args):
    _, error = executor_execute._execution_request_error(_state(), args, [])

    assert error is not None
    assert error.startswith("InvalidToolArgumentsError:")


def test_identical_successful_code_is_bounded_per_approved_plan():
    code = "result = 1\nprint(result)"
    history = [_execution(code) for _ in range(3)]

    _, error = executor_execute._execution_request_error(
        _state(), {"code": "result = 1  \nprint(result)\n"}, history
    )

    assert error is not None
    assert error.startswith("RepeatedExecutionError:")
    assert "already succeeded 3 times" in error


def test_failed_or_other_plan_executions_do_not_consume_replay_ceiling():
    code = "print('bounded')"
    history = [
        _execution(code, status="failed"),
        _execution(code, plan_id="plan-replaced"),
        _execution(code),
        _execution(code),
    ]

    _, error = executor_execute._execution_request_error(
        _state(), {"code": code}, history
    )

    assert error is None


def test_prior_cohort_executions_do_not_consume_replay_ceiling():
    code = "print('new cohort')"
    state = _state()
    state.sandbox_state = {"cohort_identity_v1": {
        "cohort_generation": 2,
        "cohort_digest": "b" * 64,
    }}
    history = [
        _execution(
            code,
            cohort_generation=1,
            cohort_digest="a" * 64,
        )
        for _ in range(3)
    ]

    _, error = executor_execute._execution_request_error(
        state, {"code": code}, history
    )

    assert error is None


@pytest.mark.asyncio
async def test_rejected_replay_creates_no_execution_receipt():
    executor_execute.bind_facade(bio_mod)
    code = "print('same')"
    agent = SimpleNamespace(
        code_history=[_execution(code) for _ in range(3)],
        _notify=AsyncMock(),
    )
    context = SimpleNamespace(
        agent=agent,
        state=_state(),
        messages=[],
        analysis_log=[],
        step_count=4,
        agent_reasoning="repeat it",
        has_substantive_work=False,
        consecutive_errors=0,
        total_errors=0,
    )
    tool_call = SimpleNamespace(
        id="call-repeat",
        arguments={"code": code, "description": "same cell"},
    )

    action = await executor_execute.handle_execute_code(context, tool_call)

    assert action is executor_execute.ToolAction.CONTINUE
    assert len(agent.code_history) == 3
    assert context.has_substantive_work is False
    assert context.consecutive_errors == 1
    assert context.total_errors == 1
    assert context.analysis_log[-1].code_error.startswith(
        "RepeatedExecutionError:"
    )
    assert context.messages[-1]["content"].startswith("Execution blocked:")


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [{}, {"code": "\n  \n"}])
async def test_malformed_code_never_reaches_the_sandbox_or_receipt(args):
    executor_execute.bind_facade(bio_mod)
    agent = SimpleNamespace(code_history=[], _notify=AsyncMock())
    context = SimpleNamespace(
        agent=agent,
        state=_state(),
        messages=[],
        analysis_log=[],
        step_count=1,
        agent_reasoning=None,
        has_substantive_work=False,
        consecutive_errors=0,
        total_errors=0,
    )
    tool_call = SimpleNamespace(id="call-empty", arguments=args)

    await executor_execute.handle_execute_code(context, tool_call)

    assert agent.code_history == []
    assert context.has_substantive_work is False
    assert context.analysis_log[-1].code_error.startswith(
        "InvalidToolArgumentsError:"
    )


def test_supervisor_feedback_dedup_uses_the_complete_guidance():
    common = "The result is skewed. " * 10

    first = executor_loop._supervisor_feedback_key(common + "Use a robust index.")
    revised = executor_loop._supervisor_feedback_key(
        common + "PI guidance: use log-normalized expression."
    )

    assert first != revised
