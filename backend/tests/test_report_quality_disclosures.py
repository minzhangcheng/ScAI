"""Contracts for deterministic report completeness and quality disclosures."""
from types import SimpleNamespace


from backend.models.schemas import (
    AnalysisState,
    RoundResult,
)
from backend.services.report.generator import (
    report_completion_fields,
    report_partial_reason,
)
from backend.services.report.quality import (
    build_quality_disclosures,
    render_report_limitations_html,
)


def _round(**overrides):
    values = {
        "round_number": 1,
        "evidence_schema_version": 1,
        "round_status": "completed",
        "evidence_status": "substantive",
        "direction_coverage_eligible": True,
        "bioinformatics_evidence_verified": True,
        "evidence_execution_ids": ["exec-1"],
    }
    values.update(overrides)
    return RoundResult(**values)


def test_legacy_and_current_evidence_limits_are_nonblocking_cautions():
    legacy = AnalysisState(
        user_id="u",
        round_results=[RoundResult(round_number=1, pi_summary="Historical report")],
    )
    current = AnalysisState(
        user_id="u",
        round_results=[_round(
            evidence_status="empty",
            direction_coverage_eligible=False,
            bioinformatics_evidence_verified=False,
        )],
    )

    legacy_quality = build_quality_disclosures(legacy)
    current_quality = build_quality_disclosures(current)

    assert legacy_quality["status"] == "qualified"
    assert legacy_quality["blocking_issues"] == []
    assert any("predates" in item for item in legacy_quality["cautions"])
    assert current_quality["status"] == "qualified"
    assert current_quality["blocking_issues"] == []
    assert any("empty evidence" in item for item in current_quality["cautions"])

    future = AnalysisState(
        user_id="u",
        round_results=[_round(evidence_schema_version=2, evidence_status="empty")],
    )
    assert build_quality_disclosures(future)["status"] == "incomplete"


def test_pi_no_action_decisions_close_host_linkage_review():
    review = {
        "round_number": 1,
        "review_status": "no_action",
        "review_plan": {
            "verdict": "no_action",
            "step_decisions": [
                {"plan_step_id": 1, "status": "completed"},
                {"plan_step_id": 2, "status": "skipped"},
            ],
        },
        "plan_audit": {
            "n_steps": 2,
            "n_unmet": 0,
            "n_unverifiable": 0,
            "n_awaiting_pi_decision": 2,
        },
        "review_result": {},
        "assessment": {"remaining_gaps": []},
    }
    state = SimpleNamespace(
        round_results=[], post_execution_reviews=[review],
        annotation_record=None, annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "complete"
    assert quality["post_execution_reviews"][0]["final_gap_status"] == "not_applicable"


def test_unverified_round_and_unmet_post_review_remain_nonblocking_observations():
    state = AnalysisState(
        user_id="u",
        round_results=[_round(
            direction_coverage_eligible=False,
            bioinformatics_evidence_verified=False,
        )],
        post_execution_reviews=[{
            "review_attempt_id": "review-1",
            "round_number": 1,
            "review_status": "partial",
            "plan_audit": {
                "n_steps": 2,
                "n_completed": 1,
                "n_unmet": 1,
                "unmet_titles": ["Cell communication"],
            },
            "review_result": {"status": "max_steps_reached", "warnings": ["Budget exhausted"]},
            "assessment": {"verdict": "partial", "remaining_gaps": ["Validate interaction"]},
        }],
    )

    quality = build_quality_disclosures(state)
    assert quality["status"] == "qualified"
    assert quality["blocking_issues"] == []
    assert any(
        "not linked to a current successful execution receipt" in issue
        for issue in quality["cautions"]
    )
    retained_review = quality["post_execution_reviews"][0]
    assert retained_review["plan_audit"]["unmet_titles"] == ["Cell communication"]
    assert retained_review["assessment"]["remaining_gaps"] == ["Validate interaction"]
    assert render_report_limitations_html({"quality_disclosures": quality}) == ""
    assert "max_steps_reached" in " ".join(quality["cautions"])

    neutral = SimpleNamespace(allow_partial_report=False)
    assert report_completion_fields(neutral, quality_disclosures=quality) == (
        "completed",
        "Complete",
    )
    assert report_partial_reason(neutral, quality_disclosures=quality) is None
