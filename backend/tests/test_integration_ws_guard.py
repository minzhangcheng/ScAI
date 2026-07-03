import inspect
import backend.routes.websockets as ws
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus


def test_needs_report_only_guards_integration_loop_source():
    src = open(ws.__file__).read()
    assert "in_integration_loop" in src
    assert "not in_integration_loop" in src
    helper = inspect.getsource(ws._in_integration_loop)
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
