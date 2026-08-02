"""Traceback consumer budgets must retain the final exception."""

from pathlib import Path

from backend.agents.supervisor.report_content import ReportContentMixin
from backend.models.schemas import AnalysisState, AnalysisStep, CodeExecution
from backend.services.report.analysis_log import AnalysisLogMixin
from backend.services.report.regenerator import _build_analysis_log_from_report
from backend.workflow.context import ContextMixin


def _long_traceback() -> str:
    frames = "\n".join(
        f'  File "/usr/lib/dependency_{index}.py", line {index}, in helper'
        for index in range(100)
    )
    return (
        "Traceback (most recent call last):\n"
        "  Cell In[7], line 12, in <module>\n"
        "    run_analysis()\n"
        f"{frames}\n"
        "TypeError: final consumer failure"
    )


def _state() -> AnalysisState:
    return AnalysisState(
        id="traceback-consumers",
        name="Traceback consumers",
        initial_instruction="test",
        user_id="user",
    )


def test_workflow_context_retains_final_exception_under_budget():
    state = _state()
    state.code_history = [CodeExecution(code="run_analysis()", error=_long_traceback())]

    context = ContextMixin()._build_code_history_summary(state)

    assert "FAILED" in context
    assert "TypeError: final consumer failure" in context


def test_supervisor_report_context_retains_final_exception_under_budget():
    state = _state()
    narrative = ReportContentMixin()._build_analysis_narrative(
        [{
            "step_number": 1,
            "step_type": "execute_code",
            "code_error": _long_traceback(),
        }],
        state,
        None,
    )
    step_context = ReportContentMixin()._build_step_context(
        AnalysisStep(
            step_number=1,
            step_type="execute_code",
            code_error=_long_traceback(),
        )
    )

    assert "TypeError: final consumer failure" in narrative
    assert "TypeError: final consumer failure" in step_context


def test_regenerated_report_retains_final_exception_under_budget(tmp_path: Path):
    html = _build_analysis_log_from_report(
        {
            "analysis_log": [{
                "step_number": 1,
                "step_type": "execute_code",
                "timestamp": "2026-08-02T00:00:00",
                "round_number": 1,
                "code_error": _long_traceback(),
            }],
        },
        "en",
        tmp_path,
    )

    assert "TypeError: final consumer failure" in html


def test_persisted_analysis_log_compacts_before_consumer_slicing():
    state = _state()
    state.analysis_log = [
        AnalysisStep(
            step_number=1,
            step_type="execute_code",
            code="run_analysis()",
            code_error=_long_traceback(),
        )
    ]

    log = AnalysisLogMixin()._build_full_analysis_log(state)

    assert "TypeError: final consumer failure" in log[0]["code_error"]
    assert len(log[0]["code_error"]) <= 3000
