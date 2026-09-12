"""Supervisor should escalate to PI when agent is stuck, regardless of error type.

Also tests force_skip flag: when stuck AND PI already intervened, the supervisor
signals that the current step must be force-skipped.
"""
from datetime import datetime
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
from backend.agents.supervisor.shared import SupervisorRecord
from backend.models.schemas import AnalysisStep
from backend.utils.error_parsing import extract_error_type


STUCK_ERROR = "SyntaxError: unterminated string literal (Cell In[18], line 124)"


def _prior_step(code_error: str, *, pi_escalated: bool, step_number: int = 1):
    return SupervisorRecord(
        step_number=step_number,
        step_type="execute_code",
        timestamp=datetime(2026, 8, 7, 12, 0, 0),
        code_error=code_error,
        pi_escalated=pi_escalated,
    )


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
async def test_first_technical_error_not_stuck_skips_pi():
    """First-occurrence technical error should NOT escalate to PI (agent self-corrects)."""
    sup = _mk_supervisor(recent_errors=[])
    step = _make_step(
        code_error="KeyError: 'nonexistent_column'\nTraceback...",
        step_number=1,
    )

    with patch.object(sup, '_is_technical_error', return_value=True):
        result = await sup.evaluate_step(step, [])

    assert not sup.pi_callback.called, (
        "PI should NOT be escalated for first technical error"
    )
    assert isinstance(result, SupervisorEvalResult)
    assert result.force_skip is False


@pytest.mark.asyncio
async def test_syntax_error_stuck_triggers_pi():
    """Repeated SyntaxError from the bioinformatics agent's LLM code generation
    is a stuck scenario. PI must intervene with strategic guidance."""
    sup = _mk_supervisor(recent_errors=[
        "SyntaxError: unterminated string literal (Cell In[14], line 129)",
        "SyntaxError: unterminated string literal (Cell In[16], line 129)",
    ])
    step = _make_step(
        code_error="SyntaxError: unterminated string literal (Cell In[18], line 124)",
        step_number=3,
    )

    with patch.object(sup, '_is_technical_error', return_value=True):
        result = await sup.evaluate_step(step, [])

    assert sup.pi_callback.called, (
        "Repeated SyntaxError should trigger PI escalation (is_stuck=True)"
    )
    assert isinstance(result, SupervisorEvalResult)
    assert result.force_skip is False


@pytest.mark.asyncio
async def test_stuck_block_skipped_when_pi_already_escalated():
    """If the critical-severity branch already escalated PI for this step,
    the stuck block must not fire a second time (record-level dedup)."""
    sup = _mk_supervisor(recent_errors=[
        "SyntaxError: unterminated string literal (Cell In[14], line 129)",
        "SyntaxError: unterminated string literal (Cell In[16], line 129)",
    ])
    step = _make_step(
        code_error="SyntaxError: unterminated string literal (Cell In[18], line 124)",
        step_number=3,
    )

    # Simulate: critical branch already escalated PI
    sup._llm_evaluate = AsyncMock(return_value={
        "severity": "critical",
        "message": "Critical syntax error",
        "issues": ["SyntaxError: unterminated string literal"],
        "issue_type": "ambiguous",  # NOT "technical" → forces else branch to fire
    })

    with patch.object(sup, '_is_technical_error', return_value=False):
        result = await sup.evaluate_step(step, [])

    # PI should be called exactly ONCE (from the critical branch, not the stuck block)
    assert sup.pi_callback.call_count == 1, (
        f"Expected 1 PI escalation (critical branch), got {sup.pi_callback.call_count}. "
        "Stuck block should be skipped when record.pi_escalated is already True."
    )
    assert isinstance(result, SupervisorEvalResult)
    # Current-step critical guidance does not itself authorize force-skip.
    assert result.force_skip is False


@pytest.mark.asyncio
async def test_force_skip_false_when_not_stuck():
    """When the agent is not stuck, force_skip should be False regardless of
    PI escalation status."""
    sup = _mk_supervisor(recent_errors=[])
    step = _make_step(
        code_error="KeyError: 'column_name'\nTraceback...",
        step_number=1,
    )

    with patch.object(sup, '_is_technical_error', return_value=True):
        result = await sup.evaluate_step(step, [])

    assert isinstance(result, SupervisorEvalResult)
    assert result.force_skip is False


@pytest.mark.asyncio
async def test_force_skip_false_when_pi_escalation_fails():
    """When stuck detection fires but PI callback returns None (escalation failed),
    record.pi_escalated stays False, so force_skip should also be False."""
    sup = _mk_supervisor(recent_errors=[
        "SyntaxError: unterminated string literal (Cell In[14], line 129)",
        "SyntaxError: unterminated string literal (Cell In[16], line 129)",
    ])
    step = _make_step(
        code_error="SyntaxError: unterminated string literal (Cell In[18], line 124)",
        step_number=3,
    )

    # PI callback returns None (escalation failed)
    sup.pi_callback = AsyncMock(return_value=None)

    with patch.object(sup, '_is_technical_error', return_value=True):
        result = await sup.evaluate_step(step, [])

    assert isinstance(result, SupervisorEvalResult)
    # PI was called but returned None, so pi_escalated stays False
    assert result.force_skip is False


@pytest.mark.asyncio
async def test_same_fingerprint_stops_only_after_prior_guidance_attempt():
    current = _mk_supervisor(recent_errors=[STUCK_ERROR, STUCK_ERROR])
    current_result = await current.evaluate_step(_make_step(STUCK_ERROR, 3), [])
    assert current_result.force_skip is False

    prior = _mk_supervisor(recent_errors=[STUCK_ERROR, STUCK_ERROR])
    prior.supervision_log = [_prior_step(STUCK_ERROR, pi_escalated=True)]
    prior_result = await prior.evaluate_step(_make_step(STUCK_ERROR, 5), [])
    assert prior_result.force_skip is True


@pytest.mark.asyncio
async def test_current_or_different_fingerprint_guidance_does_not_stop():
    different = _mk_supervisor(recent_errors=[STUCK_ERROR, STUCK_ERROR])
    different.supervision_log = [
        _prior_step("ValueError: could not convert string to float", pi_escalated=True),
        SupervisorRecord(
            step_number=2,
            step_type="annotate_clusters",
            timestamp=datetime(2026, 8, 7, 12, 0, 0),
            pi_escalated=True,
        ),
    ]
    result = await different.evaluate_step(_make_step(STUCK_ERROR, 5), [])
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
