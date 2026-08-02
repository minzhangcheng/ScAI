import asyncio, types
from unittest.mock import patch
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus
from backend.workflow.integration import IntegrationMixin


class _Harness(IntegrationMixin):
    def __init__(self, candidates):
        self.session_id = "sess"
        self._candidates = candidates
        self.sent, self.snapshots, self.segments = [], [], []
        self._continue_event = types.SimpleNamespace(cleared=False)
        self._continue_event.clear = lambda: setattr(self._continue_event, "cleared", True)
        self.config = types.SimpleNamespace(max_rounds=10)

        pi = types.SimpleNamespace()
        async def _analyze(state, dirs): return "INTEGRATION ANALYSIS TEXT"
        async def _gen(state, cumulative_rounds, integration_report, new_literature): return list(self._candidates)
        async def _score(state, cands, evidence): return list(cands)
        pi.analyze_direction_correlations = _analyze
        pi.generate_integration_direction_candidates = _gen
        pi.score_integration_directions_with_evidence = _score
        self.pi_agent = pi

        lit = types.SimpleNamespace()
        async def _open(state, report): return "NEW LIT"
        async def _novelty(cands, state=None): return [{"direction_id": c.get("direction_id", "")} for c in cands]
        lit.integration_literature_search = _open
        lit.evaluate_direction_novelty_candidates = _novelty
        self.literature_agent = lit

    async def _send_message(self, t, c, agent=None): self.sent.append((t, c))
    async def _send_state_sync(self, state, change_type): self.sent.append(("state_sync", change_type))
    async def _create_auto_snapshot(self, state, phase): self.snapshots.append(phase)
    async def _build_and_save_detailed_segment(self, state, phase, title, graph_state=None, round_result=None):
        self.segments.append(phase)
    def _set_status(self, state, status): state.status = status


def _state_with_batch():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.current_round = 3
    s.active_direction_batch = [{"direction": "A"}, {"direction": "B"}]
    s.pending_integration = True
    return s


def test_integration_with_candidates_opens_gate():
    h = _Harness([{"direction": "NEW", "direction_id": "int_001"}])
    s = _state_with_batch()
    out = asyncio.run(h._integration_synthesis({"analysis_state": s}))
    assert out["integration_next"] == "gate"
    assert s.analysis_phase == AnalysisPhase.INTEGRATION
    assert s.status == TaskStatus.WAITING_INPUT
    assert s.integration_cycle == 1
    assert s.pending_integration is False
    assert s.integration_gate_candidates
    assert s.integration_reports[-1] == "INTEGRATION ANALYSIS TEXT"
    assert s.multi_direction_correlation == "INTEGRATION ANALYSIS TEXT"
    assert "pre_integration_gate" in h.snapshots
    assert h._continue_event.cleared is True
    assert any(t == "integration_gate" for t, _ in h.sent)
    assert ("state_sync", "integration_gate") in h.sent


def test_integration_empty_candidates_short_circuits():
    h = _Harness([])
    s = _state_with_batch()
    out = asyncio.run(h._integration_synthesis({"analysis_state": s}))
    assert out["integration_next"] == "stop"
    assert out["should_continue"] is False
    assert s.pending_integration is False
    assert s.analysis_phase == AnalysisPhase.INTEGRATION
    assert "pre_integration_gate" not in h.snapshots


def test_integration_cap_short_circuits_even_with_candidates():
    h = _Harness([{"direction": "NEW"}])
    s = _state_with_batch(); s.integration_cycle = 2; s.max_integration_cycles = 3
    out = asyncio.run(h._integration_synthesis({"analysis_state": s}))
    assert out["integration_next"] == "stop"
    assert out["should_continue"] is False
    assert s.integration_cycle == 3
    assert s.analysis_phase == AnalysisPhase.INTEGRATION


def test_after_integration_routes_on_flag():
    h = _Harness([])
    assert h._after_integration({"integration_next": "gate"}) == "gate"
    assert h._after_integration({"integration_next": "stop"}) == "stop"
    assert h._after_integration({}) == "stop"


def test_integration_creates_round_result():
    """Integration synthesis should create a RoundResult with phase=integration."""
    h = _Harness([{"direction": "NEW", "direction_id": "int_001"}])
    s = _state_with_batch()
    asyncio.run(h._integration_synthesis({"analysis_state": s}))
    integration_rrs = [rr for rr in s.round_results if getattr(rr, "phase", None) == "integration"]
    assert len(integration_rrs) == 1
    rr = integration_rrs[0]
    assert rr.round_number == s.current_round
    assert rr.phase == "integration"
    assert rr.pi_summary
    assert rr.integration_report_summary == "INTEGRATION ANALYSIS TEXT"
    assert len(rr.integration_candidates) == 1
    assert rr.integration_candidates[0]["direction"] == "NEW"
    assert rr.round_status == "completed"
    assert rr.evidence_status == "substantive"
    assert rr.direction_coverage_eligible is None


def test_integration_sends_round_summary_message():
    """Integration synthesis should send a round_summary WebSocket message."""
    h = _Harness([{"direction": "NEW", "direction_id": "int_001"}])
    s = _state_with_batch()
    asyncio.run(h._integration_synthesis({"analysis_state": s}))
    round_summary_msgs = [c for t, c in h.sent if t == "round_summary"]
    assert len(round_summary_msgs) == 1
    msg = round_summary_msgs[0]
    assert msg["round"] == s.current_round
    assert msg["phase"] == "integration"
    assert msg.get("integration_candidates")
    assert len(msg["integration_candidates"]) == 1


def test_integration_synthesis_generates_report_stop_path():
    """Verify _integration_synthesis generates integration report HTML and
    sets integration_report_html_path (empty-candidate stop path)."""
    h = _Harness([])
    s = _state_with_batch()
    with patch("backend.services.report.generate_integration_report_html",
               return_value="/tmp/test/integration_report.html") as mock_gen:
        out = asyncio.run(h._integration_synthesis({"analysis_state": s}))
    assert out["integration_next"] == "stop"
    mock_gen.assert_called_once()
    assert s.integration_report_html_path == "/tmp/test/integration_report.html"


def test_integration_synthesis_generates_report_gate_path():
    """Verify _integration_synthesis generates integration report HTML
    even when candidates exist (gate path)."""
    h = _Harness([{"direction": "NEW", "direction_id": "int_001"}])
    s = _state_with_batch()
    with patch("backend.services.report.generate_integration_report_html",
               return_value="/tmp/test/integration_report.html") as mock_gen:
        out = asyncio.run(h._integration_synthesis({"analysis_state": s}))
    assert out["integration_next"] == "gate"
    mock_gen.assert_called_once()
    assert s.integration_report_html_path == "/tmp/test/integration_report.html"


def test_integration_complete_payload_includes_html_url():
    """Verify the stop-path integration_complete WS payload includes html_url."""
    h = _Harness([])
    s = _state_with_batch()
    with patch("backend.services.report.generate_integration_report_html",
               return_value="/data/sessions/sess/reports/integration_report.html"):
        asyncio.run(h._integration_synthesis({"analysis_state": s}))
    complete_msgs = [c for t, c in h.sent if t == "integration_complete"]
    assert len(complete_msgs) == 1
    assert complete_msgs[0].get("html_url") == "/api/session/sess/integration-report/html"


def test_integration_gate_payload_includes_html_url():
    """Verify the gate-path integration_gate WS payload includes html_url."""
    h = _Harness([{"direction": "NEW", "direction_id": "int_001"}])
    s = _state_with_batch()
    with patch("backend.services.report.generate_integration_report_html",
               return_value="/data/sessions/sess/reports/integration_report.html"):
        asyncio.run(h._integration_synthesis({"analysis_state": s}))
    gate_msgs = [c for t, c in h.sent if t == "integration_gate"]
    assert len(gate_msgs) == 1
    assert gate_msgs[0].get("html_url") == "/api/session/sess/integration-report/html"
