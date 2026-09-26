"""Supervisor should escalate to PI when agent is stuck, regardless of error type.

Also tests force_skip flag: when stuck AND PI already intervened, the supervisor
signals that the current step must be force-skipped.
"""
import inspect
import logging
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.bioinformatics import executor_loop
from backend.agents.bioinformatics.executor_auxiliary import LoopExit, TurnEffects
from backend.agents.bioinformatics.executor_context import ExecutorLoopContext
from backend.agents.bioinformatics.executor_report_gates import evidence_gate_record
from backend.agents.supervisor_agent import SupervisorAgent, SupervisorEvalResult
from backend.models.schemas import AnalysisStep
from backend.utils.error_parsing import extract_error_type


STUCK_ERROR = "SyntaxError: unterminated string literal (Cell In[18], line 124)"


def _make_step(code_error: str, step_number: int = 1) -> AnalysisStep:
    return AnalysisStep(
        step_number=step_number,
        step_type="execute_code",
        code="some_broken_code",
        code_error=code_error,
    )


def _mk_supervisor(recent_errors=None):
    """Build a minimal SupervisorAgent for unit testing stuck detection."""
    sup = SupervisorAgent.__new__(SupervisorAgent)
    sup._recent_errors = list(recent_errors or [])
    sup._recent_rule_feedback_hashes = set()
    sup._step_count = 0
    sup._pi_question_hashes = set()
    sup.session_id = "test-session"
    sup._provider = MagicMock()
    sup._last_config_revision = sup._provider.get_config_revision.return_value
    sup.supervision_log = []
    sup.pi_callback = AsyncMock(return_value="Try a different gene list.")
    sup.technique_agent = None
    sup._sandbox = None
    sup._notify = AsyncMock()
    sup._llm_evaluate = AsyncMock(return_value={
        "severity": "warning",
        "message": "SyntaxError in code",
        "issues": ["SyntaxError: unterminated string literal"],
        "issue_type": "technical",
    })
    return sup


@pytest.mark.asyncio
async def test_stuck_agent_escalates_to_pi_regardless_of_severity():
    """When the same error type repeats (is_stuck=True), PI must be escalated
    even when LLM severity is 'warning' (not 'critical')."""
    sup = _mk_supervisor(recent_errors=[
        "SyntaxError: unterminated string literal (test.py, line 10)\nTraceback...",
        "SyntaxError: unterminated string literal (test.py, line 12)\nTraceback...",
    ])
    step = _make_step(
        code_error="SyntaxError: unterminated string literal (test.py, line 14)\nTraceback...",
        step_number=3,
    )

    with patch.object(sup, '_is_technical_error', return_value=True):
        result = await sup.evaluate_step(step, [])

    # PI escalation MUST have happened
    assert sup.pi_callback.called, (
        f"Expected PI escalation for stuck agent, but pi_callback was not called. "
        f"Recent errors: {sup._recent_errors}"
    )

    # Verify the escalation question mentions "stuck"
    call_args = sup.pi_callback.call_args
    assert call_args is not None
    question = call_args[0][0]
    assert "stuck" in question.lower(), f"Expected 'stuck' in question, got: {question}"

    # Current-step PI guidance is advisory; a prior guided executor turn is
    # required before force-skip.
    assert isinstance(result, SupervisorEvalResult)
    assert result.force_skip is False


@pytest.mark.asyncio
async def test_platform_fault_force_skips_immediately_with_failure_class():
    sup = _mk_supervisor(recent_errors=[])
    step = AnalysisStep(
        step_number=1,
        step_type="execute_code",
        code="x = (v for v in values)",
        code_error=(
            "SyntaxError: Generator expression must be parenthesized "
            "(_scai_expression_contract_call)"
        ),
    )
    result = await sup.evaluate_step(step, [])
    assert result.force_skip is True
    assert result.failure_class == "platform_fault"


def _force_stop_context(*, prior_gates, findings):
    """Minimal loop context for the third force-skip terminal-settlement path."""
    agent = SimpleNamespace(
        _recent_sup_feedback=set(),
        _force_skip_error_type="RepeatedExecutionError",
        _force_skip_counter=2,
    )
    return ExecutorLoopContext(
        agent=agent,
        analysis_state=None,
        messages=[],
        findings=findings,
        figures=[],
        analysis_log=[
            SimpleNamespace(
                code_error="RepeatedExecutionError: repeated execution requested",
            )
        ],
        mechanism_programs=[],
        checkpoint_manager=None,
        save_checkpoint=AsyncMock(),
        review_callback=None,
        supervisor_callback=AsyncMock(
            return_value=SimpleNamespace(feedback="", force_skip=True)
        ),
        planning_stage="full",
        max_steps=10,
        attempt_start_step=0,
        step_count=1,
        evidence_gate_findings=list(prior_gates),
    ), agent


@pytest.mark.asyncio
async def test_force_skip_truthfully_settles_degraded_without_polluting_findings(
    monkeypatch, caplog,
):
    """The third ignored force-skip settles the loop; it must not claim or make
    a report_findings call, and the only durable trace is the structured gate."""
    monkeypatch.setattr(
        executor_loop, "extract_error_type", extract_error_type, raising=False,
    )
    monkeypatch.setattr(
        executor_loop, "_evidence_gate_record", evidence_gate_record,
        raising=False,
    )
    monkeypatch.setattr(
        executor_loop, "logger", logging.getLogger("test.executor.force_stop"),
        raising=False,
    )
    caplog.set_level(logging.WARNING, logger="test.executor.force_stop")

    prior_gate = {
        "gate": "prior-evidence-gate",
        "rule_id": "scai-native/prior-gate",
        "detail": "A prior observation that must survive.",
        "severity": "advise",
        "forced_release": False,
    }
    findings: list = []
    context, agent = _force_stop_context(prior_gates=[prior_gate], findings=findings)
    effects = TurnEffects(
        response=None,
        step_figures=[],
        has_substantive_work=False,
        has_figure_work=False,
        has_register_score=False,
    )

    exit_status = await executor_loop.finish_turn(context, effects)

    assert exit_status is LoopExit.TERMINAL
    assert context.findings is findings
    assert context.findings == [], (
        "the forced stop must not inject a technical note into scientific findings"
    )
    assert context.evidence_gate_degraded is True
    assert context.force_break is True
    assert context.completed is True
    assert context.evidence_gate_findings[0] == prior_gate
    forced = [
        record for record in context.evidence_gate_findings
        if record.get("gate") == "supervisor-force-stop"
    ]
    assert len(forced) == 1
    assert forced[0]["rule_id"] == "scai-native/supervisor-force-stop"
    assert forced[0]["detail"] == (
        "Analysis was forcibly stopped after repeated execution failures."
    )
    assert "forcing a degraded terminal settlement" in caplog.text
    assert "report_findings was not invoked" in caplog.text
    assert "forcing report_findings" not in caplog.text
    system_messages = [
        message["content"] for message in context.messages
        if message["role"] == "user" and message["content"].startswith("[System]")
    ]
    assert len(system_messages) == 1
    assert "You MUST call report_findings now" in system_messages[0]

    # Re-entering the same branch must not duplicate the forced-stop record or
    # alter scientific findings, while preserving the prior observation.
    agent._force_skip_counter = 2
    agent._force_skip_error_type = "RepeatedExecutionError"
    context.force_break = False
    second_exit = await executor_loop.finish_turn(context, effects)

    assert second_exit is LoopExit.TERMINAL
    assert context.findings == []
    assert context.evidence_gate_findings.count(prior_gate) == 1

    # Existing scientific findings are equally untouched.
    existing_context, _agent = _force_stop_context(
        prior_gates=[], findings=["existing scientific finding"]
    )
    await executor_loop.finish_turn(existing_context, effects)
    assert existing_context.findings == ["existing scientific finding"]
    forced_after = [
        record for record in existing_context.evidence_gate_findings
        if record.get("gate") == "supervisor-force-stop"
    ]
    assert len(forced_after) == 1
    assert len(existing_context.evidence_gate_findings) == 1

    # Source-level supplements: the old false claim is gone and the
    # transcript-only nature of the [System] message is documented.
    source = inspect.getsource(executor_loop.finish_turn)
    assert "by injecting report_findings and breaking" not in source
    assert "forcing report_findings" not in source
    assert "all_findings.append" not in source
    assert "transcript-only" in source
