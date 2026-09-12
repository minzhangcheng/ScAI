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
        self.pi_agent = pi

        lit = types.SimpleNamespace()
        async def _open(state, report): return "NEW LIT"
        async def _novelty(cands, state=None): return [{"direction_id": c.get("direction_id", "")} for c in cands]
        lit.integration_literature_search = _open
        lit.evaluate_direction_novelty_candidates = _novelty
        self.literature_agent = lit

    async def _send_message(self, t, c, agent=None): self.sent.append((t, c))
    async def _send_state_sync(self, state, change_type): self.sent.append(("state_sync", change_type))
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


def test_integration_empty_candidates_short_circuits():
    h = _Harness([])
    s = _state_with_batch()
    out = asyncio.run(h._integration_synthesis({"analysis_state": s}))
    assert out["integration_next"] == "stop"
    assert out["should_continue"] is False
    assert s.pending_integration is False
    assert s.analysis_phase == AnalysisPhase.INTEGRATION


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


def test_integration_retry_preserves_restore_origin_on_same_identity():
    """The production integration node keeps first-result branch provenance."""
    h = _Harness([])
    s = _state_with_batch()
    origin = "source/r001"
    s.restore_branch_id = "restore-source-r001"
    s.restore_branch_label = "Restored direction"
    s.last_restored_snapshot_id = origin
    s.restore_provenance_snapshot_id = origin

    with patch(
        "backend.services.report.generate_integration_report_html",
        return_value="/tmp/integration.html",
    ):
        asyncio.run(h._integration_synthesis({"analysis_state": s}))
        first = next(rr for rr in s.round_results if rr.phase == "integration")
        assert first.round_number == 4
        assert first.restored_from_snapshot_id == origin
        assert first.lineage_id == s.restore_branch_id

        # Model a replay of the same integration identity after the one-shot
        # origin has already been consumed and the prior result was retained.
        s.current_round = 3
        s.integration_cycle = 0
        s.pending_integration = True
        asyncio.run(h._integration_synthesis({"analysis_state": s}))
        retries = [rr for rr in s.round_results if rr.phase == "integration"]
        assert len(retries) == 1
        assert retries[0].restored_from_snapshot_id == origin

        asyncio.run(h._integration_synthesis({"analysis_state": s}))

    integration_results = [rr for rr in s.round_results if rr.phase == "integration"]
    assert [rr.round_number for rr in integration_results] == [4, 5]
    assert integration_results[1].restored_from_snapshot_id is None
    assert sum(
        rr.restored_from_snapshot_id == origin for rr in integration_results
    ) == 1


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
    from backend.services.round_snapshots.restore import _strip_unavailable_report_urls
    assert _strip_unavailable_report_urls(complete_msgs[0])["html_url"] is None


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
    from backend.services.round_snapshots.restore import _strip_unavailable_report_urls
    assert _strip_unavailable_report_urls(gate_msgs[0])["html_url"] is None
