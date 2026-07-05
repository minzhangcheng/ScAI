from backend.models.schemas import RoundResult


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
