from backend.models.schemas import AnalysisState, RoundResult
from backend.workflow.synthesis import build_round_summary_payload


def test_round_result_accepts_integration_phase():
    rr = RoundResult(round_number=4, phase="integration", pi_summary="Integration synthesis")
    assert rr.phase == "integration"


def test_round_result_integration_fields_default():
    rr = RoundResult(round_number=4, phase="integration")
    assert rr.integration_report_summary == ""
    assert rr.integration_candidates == []


def test_round_result_integration_fields_populated():
    rr = RoundResult(
        round_number=4,
        phase="integration",
        integration_report_summary="Shared: copper deficiency. Complementary: Dir1+Dir2.",
        integration_candidates=[
            {"direction": "New axis X", "score": 0.85},
        ],
    )
    assert "copper deficiency" in rr.integration_report_summary
    assert len(rr.integration_candidates) == 1
    assert rr.integration_candidates[0]["direction"] == "New axis X"


def test_round_result_existing_phases_still_valid():
    rr_explore = RoundResult(round_number=1, phase="exploration")
    rr_innovate = RoundResult(round_number=2, phase="innovation")
    assert rr_explore.phase == "exploration"
    assert rr_innovate.phase == "innovation"


def test_round_summary_payload_preserves_integration_report_summary():
    rr = RoundResult(
        round_number=3,
        phase="integration",
        pi_summary="Cross-direction",
        integration_report_summary="Shared: copper deficiency.",
        integration_candidates=[{"direction": "New axis X", "score": 0.85}],
    )

    payload = build_round_summary_payload(rr, AnalysisState(user_id="test-user"), effective_max_rounds=3)

    assert payload["phase"] == "integration"
    assert payload["integration_report_summary"] == "Shared: copper deficiency."
    assert payload["integration_candidates"] == [{"direction": "New axis X", "score": 0.85}]
