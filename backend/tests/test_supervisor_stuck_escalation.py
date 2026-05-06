"""Supervisor should escalate to PI when agent is stuck, regardless of error type."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import AnalysisStep


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
        feedback = await sup.evaluate_step(step, [])

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


@pytest.mark.asyncio
async def test_first_technical_error_not_stuck_skips_pi():
    """First-occurrence technical error should NOT escalate to PI (agent self-corrects)."""
    sup = _mk_supervisor(recent_errors=[])
    step = _make_step(
        code_error="KeyError: 'nonexistent_column'\nTraceback...",
        step_number=1,
    )

    with patch.object(sup, '_is_technical_error', return_value=True):
        feedback = await sup.evaluate_step(step, [])

    assert not sup.pi_callback.called, (
        "PI should NOT be escalated for first technical error"
    )


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
        feedback = await sup.evaluate_step(step, [])

    assert sup.pi_callback.called, (
        "Repeated SyntaxError should trigger PI escalation (is_stuck=True)"
    )


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
        feedback = await sup.evaluate_step(step, [])

    # PI should be called exactly ONCE (from the critical branch, not the stuck block)
    assert sup.pi_callback.call_count == 1, (
        f"Expected 1 PI escalation (critical branch), got {sup.pi_callback.call_count}. "
        "Stuck block should be skipped when record.pi_escalated is already True."
    )
