import asyncio, types
from unittest.mock import patch
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus, RoundResult
from backend.workflow.integration import IntegrationMixin


class _Harness(IntegrationMixin):
    def __init__(self, candidates):
        self.session_id = "sess"
        self._candidates = candidates
        self.sent, self.snapshots, self.segments = [], [], []
        self.analyzed_directions = []
        self._continue_event = types.SimpleNamespace(cleared=False)
        self._continue_event.clear = lambda: setattr(self._continue_event, "cleared", True)
        self.config = types.SimpleNamespace(max_rounds=10)

        pi = types.SimpleNamespace()
        async def _analyze(state, dirs):
            self.analyzed_directions = list(dirs)
            return "INTEGRATION ANALYSIS TEXT"
        async def _gen(state, cumulative_rounds, integration_report, new_literature): return list(self._candidates)
        async def _score(state, cands, evidence): return list(cands)
        pi.analyze_direction_correlations = _analyze
        pi.generate_integration_direction_candidates = _gen
        pi.score_integration_directions_with_evidence = _score
        pi.plan_direction_literature_search = lambda *args, **kwargs: None
        pi.review_direction_literature_search = lambda *args, **kwargs: None
        self.pi_agent = pi

        lit = types.SimpleNamespace()
        async def _open(state, report): return "NEW LIT"
        async def _novelty(cands, state=None, **kwargs): return [{"direction_id": c.get("direction_id", "")} for c in cands]
        lit.integration_literature_search = _open
        lit.evaluate_direction_literature_search = _novelty
        self.literature_agent = lit

    async def _send_message(self, t, c, agent=None): self.sent.append((t, c))
    async def _send_state_sync(self, state, change_type): self.sent.append(("state_sync", change_type))
    async def _wait_for_llm_intervention(self, error, node, state):
        raise error
    async def _build_and_save_detailed_segment(self, state, phase, title, graph_state=None, round_result=None):
        self.segments.append(phase)
    def _set_status(self, state, status): state.status = status


def _state_with_batch():
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.current_round = 3
    s.active_direction_batch = [{"direction": "A"}, {"direction": "B"}]
    s.user_chosen_directions = list(s.active_direction_batch)
    s.round_results = [
        RoundResult(
            round_number=2, phase="innovation", direction="A",
            pi_summary="A summary", bioinformatics_findings=["A finding"],
            round_status="completed", evidence_status="substantive",
            direction_coverage_eligible=True,
            bioinformatics_evidence_verified=True,
        ),
        RoundResult(
            round_number=3, phase="innovation", direction="B",
            pi_summary="B summary", bioinformatics_findings=["B finding"],
            round_status="completed", evidence_status="substantive",
            direction_coverage_eligible=True,
            bioinformatics_evidence_verified=True,
        ),
    ]
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
    assert h._continue_event.cleared is True
    assert any(t == "integration_gate" for t, _ in h.sent)
    assert ("state_sync", "integration_gate") in h.sent


def test_integration_cap_short_circuits_even_with_candidates():
    h = _Harness([{"direction": "NEW"}])
    s = _state_with_batch(); s.integration_cycle = 2; s.max_integration_cycles = 3
    out = asyncio.run(h._integration_synthesis({"analysis_state": s}))
    assert out["integration_next"] == "stop"
    assert out["should_continue"] is False
    assert s.integration_cycle == 3
    assert s.analysis_phase == AnalysisPhase.INTEGRATION


def test_partial_direction_inputs_are_retained_and_mark_integration_partial():
    h = _Harness([{"direction": "NEW", "direction_id": "int_001"}])
    s = _state_with_batch()
    s.round_results[1].round_status = "partial"
    s.round_results[1].direction_coverage_eligible = False
    s.round_results[1].bioinformatics_evidence_verified = False
    s.round_results[1].evidence_notes = ["Replication was unavailable."]

    out = asyncio.run(h._integration_synthesis({"analysis_state": s}))

    assert out["integration_next"] == "gate"
    assert [item["direction"] for item in h.analyzed_directions] == ["A", "B"]
    assert h.analyzed_directions[1]["available_findings"] == ["B finding"]
    assert "Replication was unavailable." in h.analyzed_directions[1]["evidence_gaps"]
    integration_result = next(rr for rr in s.round_results if rr.phase == "integration")
    assert integration_result.round_status == "partial"
    assert integration_result.evidence_status == "substantive"
    assert any("partial direction attempts" in note for note in integration_result.evidence_notes)


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
