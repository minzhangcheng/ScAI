import inspect
import backend.routes.websockets as ws
from backend.routes.websocket_parts import analysis_connection, state_sync
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus


def test_needs_report_only_guards_integration_loop_source():
    src = inspect.getsource(state_sync._needs_report_only)
    assert "in_integration_loop" in src
    assert "not in_integration_loop" in src
    helper = inspect.getsource(state_sync._in_integration_loop)
    assert "pending_integration" in helper
    assert "INTEGRATION" in helper or "integration" in helper
    assert "active_direction_batch" in helper or "integration_gate_candidates" in helper


def test_covered_batch_no_pending_allows_report_only():
    from backend.routes.websockets import _in_integration_loop
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.active_direction_batch = [{"direction": "A"}]
    s.pending_integration = False
    s.integration_gate_candidates = []
    from backend.models.schemas import RoundResult
    rr = RoundResult(round_number=3); rr.phase = "innovation"; rr.direction = "A"
    rr.pi_summary = "x"; rr.round_status = "completed"; rr.evidence_status = "substantive"
    rr.bioinformatics_findings = ["f"]
    s.round_results = [rr]
    assert _in_integration_loop(s) is False

def _final_round_state(**overrides):
    from backend.models.schemas import RoundResult

    state = AnalysisState(id="ws-report-only", user_id="u", name="t")
    state.status = TaskStatus.RUNNING
    state.max_rounds = 2
    state.current_round = 2
    state.round_results = [
        RoundResult(round_number=n, direction="d", findings="f") for n in (1, 2)
    ]
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_report_only_fires_for_an_interrupted_report():
    """The case the predicate exists for: analysis done, report generation cut off."""
    assert ws._needs_report_only(_final_round_state(), 2, False) is True


def test_an_explicit_start_intent_is_not_answered_with_a_report():
    """Re-run means execute. Reporting instead fabricates a final report.

    POST /run arms `explicit_start` as the "re-run from your data" the
    blocked-resume message promises, and it also writes RUNNING — which is what
    made this predicate fire on a final-round import. The report then completed the
    session, cleared the runtime warning and left `runtime_recompute_pending`
    unconsumed, so nothing was left saying the kernel is empty.
    """
    state = _final_round_state(restore_start_intent="explicit_start")

    assert ws._needs_report_only(state, 2, False) is False


def test_an_explicit_finalize_still_wins():
    """`final_report_requested` is the explicit "write the report" intent."""
    state = _final_round_state(
        restore_start_intent="explicit_start", final_report_requested=True
    )

    assert ws._needs_report_only(state, 2, False) is True


def test_explicit_start_dispatches_to_fresh_run_even_after_rounds_exist():
    """The WebSocket runner must not send a fresh intent into resume()."""
    from backend.routes.websockets import should_resume_workflow

    state = _final_round_state(restore_start_intent="explicit_start")
    assert should_resume_workflow(state) is False
