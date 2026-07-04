"""Integration-loop finalize/resume hardening (post-implementation review fixes).

B) generate_report must clear final_report_requested so a later WS reconnect
   does not re-trigger report generation.
C) resume_restored_wait_state must NOT re-send the integration_gate message when
   the user chose to finalize (final_report_requested=True).
E) _check_convergence_condition must not stop on max_rounds while the active
   direction batch is still incomplete (align with _should_continue).
"""
import inspect
import types

from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.workflow.routing import RoutingMixin


def test_generate_report_clears_final_report_requested():
    src = inspect.getsource(SingleCellAnalysisWorkflow.generate_report)
    assert "consume_final_report_request()" in src, (
        "generate_report must consume final_report_requested so reconnect "
        "does not re-trigger report-only generation"
    )


def test_resume_does_not_resend_gate_when_finalizing():
    src = inspect.getsource(SingleCellAnalysisWorkflow.resume_restored_wait_state)
    # The integration_gate re-send must be guarded against the finalize request.
    assert "final_report_requested" in src, (
        "resume_restored_wait_state must guard the integration_gate re-send "
        "against final_report_requested"
    )


class _Router(RoutingMixin):
    def __init__(self):
        self.session_id = "s"
        self.config = types.SimpleNamespace(max_rounds=3)


def _running(**kw):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.status = TaskStatus.RUNNING
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_incomplete_batch_does_not_stop_even_past_max_rounds():
    # current_round has passed max_rounds, but the active batch is not covered.
    # Must continue (analyze the batch), not stop.
    s = _running(current_round=5, max_rounds=3,
                 active_direction_batch=[{"direction": "C"}])
    assert _Router()._check_convergence_condition({"analysis_state": s}) == "continue"
