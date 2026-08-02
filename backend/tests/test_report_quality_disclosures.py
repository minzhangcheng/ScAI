"""Contracts for deterministic report completeness and quality disclosures."""
from types import SimpleNamespace

import pytest

from backend.models.schemas import (
    AnalysisState,
    AnnotationClusterEntry,
    AnnotationRecord,
    RoundResult,
)
from backend.services.report.generator import (
    figureless_outline_sections,
    report_completion_fields,
    report_partial_reason,
)
from backend.services.report.quality import (
    build_quality_disclosures,
    render_quality_disclosures_html,
    render_report_status_banner_html,
)
from backend.workflow.reporting_parts.final_report import (
    analysis_completion_message,
    detailed_report_event_payload,
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


def test_legacy_round_defaults_are_cautions_but_current_empty_evidence_blocks():
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
    assert current_quality["status"] == "incomplete"
    assert any("empty evidence" in item for item in current_quality["blocking_issues"])

    future = AnalysisState(
        user_id="u",
        round_results=[_round(evidence_schema_version=2, evidence_status="empty")],
    )
    assert build_quality_disclosures(future)["status"] == "incomplete"


@pytest.mark.parametrize(
    "round_payload",
    [
        {
            "evidence_schema_version": 2,
            "round_status": "completed",
            "evidence_status": "substantive",
            "direction_coverage_eligible": True,
            "bioinformatics_evidence_verified": True,
        },
        {
            "evidence_schema_version": 1,
            "round_status": "future_status",
            "evidence_status": "substantive",
            "direction_coverage_eligible": True,
            "bioinformatics_evidence_verified": True,
        },
        {
            "evidence_schema_version": 1,
            "round_status": "completed",
            "evidence_status": "unverified",
            "direction_coverage_eligible": True,
            "bioinformatics_evidence_verified": True,
        },
    ],
)
def test_future_schema_and_unknown_round_evidence_statuses_fail_closed(
    round_payload,
):
    state = SimpleNamespace(
        round_results=[{"round_number": 1, **round_payload}],
        post_execution_reviews=[],
        annotation_record=None,
        annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "incomplete"
    assert quality["blocking_issues"]


@pytest.mark.parametrize(
    "malformed_field",
    ["direction_coverage_eligible", "bioinformatics_evidence_verified"],
)
def test_current_round_verification_fields_require_literal_true(
    malformed_field,
):
    payload = {
        "round_number": 1,
        "phase": "exploration",
        "evidence_schema_version": 1,
        "round_status": "completed",
        "evidence_status": "substantive",
        "direction_coverage_eligible": True,
        "bioinformatics_evidence_verified": True,
    }
    payload[malformed_field] = "false"
    state = SimpleNamespace(
        round_results=[payload],
        post_execution_reviews=[],
        annotation_record=None,
        annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "incomplete"
    assert quality["blocking_issues"]


@pytest.mark.parametrize(
    ("review_override", "expected_gap_status"),
    [
        ({"review_status": "failed"}, "failed"),
        ({"review_result": {"status": "failed"}}, "failed"),
        ({"review_result": {"status": "unverifiable"}}, "unverifiable"),
        ({"assessment": {"verdict": "unverifiable", "remaining_gaps": []}}, "unverifiable"),
        ({"assessment": {"verdict": "resolved", "remaining_gaps": ["open"]}}, "unresolved"),
    ],
)
def test_failure_unverifiable_and_remaining_gaps_override_resolved_review(
    review_override, expected_gap_status,
):
    review = {
        "round_number": 1,
        "review_status": "completed",
        "plan_audit": {"n_unmet": 1, "n_unverifiable": 1},
        "review_result": {"status": "completed"},
        "assessment": {"verdict": "resolved", "remaining_gaps": []},
    }
    review.update(review_override)
    state = SimpleNamespace(
        round_results=[],
        post_execution_reviews=[review],
        annotation_record=None,
        annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "incomplete"
    assert quality["post_execution_reviews"][0]["final_gap_status"] == expected_gap_status


def test_only_successful_execution_and_verifiable_resolved_assessment_closes_review():
    review = {
        "round_number": 1,
        "review_status": "completed",
        "plan_audit": {"n_unmet": 2, "n_unverifiable": 1},
        "review_result": {"status": "completed"},
        "assessment": {"verdict": "resolved", "remaining_gaps": []},
    }
    state = SimpleNamespace(
        round_results=[],
        post_execution_reviews=[review],
        annotation_record=None,
        annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "complete"
    assert quality["blocking_issues"] == []
    assert quality["post_execution_reviews"][0]["final_gap_status"] == "resolved"


@pytest.mark.parametrize("review_status", ["no_action", "skipped"])
@pytest.mark.parametrize("unverifiable_source", ["execution", "assessment"])
def test_declined_review_cannot_hide_explicit_unverifiable_result(
    review_status, unverifiable_source,
):
    review = {
        "round_number": 1,
        "review_status": review_status,
        "review_plan": {"verdict": "no_action"},
        "plan_audit": {"n_unmet": 0, "n_unverifiable": 0},
        "review_result": {},
        "assessment": {"remaining_gaps": []},
    }
    if unverifiable_source == "execution":
        review["review_result"] = {"status": "unverifiable"}
    else:
        review["assessment"]["verdict"] = "unverifiable"
    state = SimpleNamespace(
        round_results=[],
        post_execution_reviews=[review],
        annotation_record=None,
        annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "incomplete"
    assert quality["post_execution_reviews"][0]["final_gap_status"] == "unverifiable"


def test_malformed_retained_review_is_unverifiable_not_silently_ignored():
    state = SimpleNamespace(
        round_results=[],
        post_execution_reviews=["not-a-review-record"],
        annotation_record=None,
        annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "incomplete"
    assert quality["post_execution_reviews"][0]["final_gap_status"] == "unverifiable"


def test_quality_record_keeps_confirmation_validation_and_unknown_counts_separate():
    state = AnalysisState(
        user_id="u",
        round_results=[_round(round_status="partial", evidence_notes=["Optional modality unavailable."])],
        annotation_record=AnnotationRecord(
            auto_confirmed=True,
            validation_status="partial",
            validation_summary="One low-confidence cluster remains.",
            clusters=[
                AnnotationClusterEntry(
                    cluster_id="0",
                    final_cell_type="Type A",
                    n_cells=17,
                    n_cells_known=True,
                    n_cells_source="cluster_group_proportions",
                ),
                AnnotationClusterEntry(
                    cluster_id="1",
                    final_cell_type="Type B",
                    n_cells=0,
                    n_cells_known=False,
                    n_cells_source="unknown",
                ),
            ],
        ),
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "qualified"
    assert quality["annotation"]["confirmation"]["status"] == "auto_confirmed"
    assert quality["annotation"]["validation"]["status"] == "partial"
    counts = quality["annotation"]["cell_counts"]
    assert counts["known_cell_total"] == 17
    assert counts["unknown_clusters"] == 1
    assert counts["clusters"][1] == {
        "cluster_id": "1",
        "final_cell_type": "Type B",
        "n_cells": None,
        "n_cells_known": False,
        "n_cells_source": "unknown",
    }
    rendered = render_quality_disclosures_html(quality)
    assert "One low-confidence cluster remains" in rendered
    assert "1 known / 1 unknown" in rendered


def test_unverified_round_and_unmet_post_review_make_report_quality_incomplete():
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
    assert quality["status"] == "incomplete"
    assert any("unverified" in issue for issue in quality["blocking_issues"])
    html = render_quality_disclosures_html(quality)
    assert "Cell communication" in html
    assert "Validate interaction" in html
    assert "max_steps_reached" in " ".join(quality["cautions"])

    neutral = SimpleNamespace(allow_partial_report=False)
    assert report_completion_fields(neutral, quality_disclosures=quality) == (
        "partial",
        "Complete with unresolved analysis-quality gaps",
    )
    assert report_partial_reason(neutral, quality_disclosures=quality) == "analysis_quality"


@pytest.mark.parametrize(
    ("reason", "extra", "expected"),
    [
        ("figure_coverage", {"figureless_sections": ["direction_1"]}, "direction_1"),
        ("degraded_fallback", {}, "degraded summary"),
        ("approved_partial", {}, "explicitly approved"),
        ("analysis_quality", {}, "quality gaps"),
    ],
)
def test_partial_banner_covers_every_machine_readable_reason(reason, extra, expected):
    report = {
        "status": "partial",
        "partial_reason": reason,
        "partial_reasons": [reason],
        **extra,
    }
    html = render_report_status_banner_html(report)
    assert "Partial report" in html
    assert expected in html


def test_detailed_report_event_carries_partial_and_quality_contracts():
    quality = {"schema_version": 1, "status": "incomplete"}
    payload = detailed_report_event_payload({
        "id": "r1",
        "status": "partial",
        "partial_reason": "figure_coverage",
        "partial_reasons": ["figure_coverage", "analysis_quality"],
        "figureless_sections": ["direction_1"],
        "unreported_figures": ["missing.jpg"],
        "quality_disclosures": quality,
        "sections": [],
        "figures": [],
        "tables": [],
    })

    assert payload["partial_reason"] == "figure_coverage"
    assert payload["partial_reasons"] == ["figure_coverage", "analysis_quality"]
    assert payload["figureless_sections"] == ["direction_1"]
    assert payload["unreported_figures"] == ["missing.jpg"]
    assert payload["quality_disclosures"] == quality
    assert analysis_completion_message({
        "status": "partial",
        "current_step": "Figure coverage incomplete",
    }) == "Analysis completed; report is partial: Figure coverage incomplete"
    assert analysis_completion_message({"status": "completed"}) == "Analysis completed"


def test_fallback_figureless_sections_require_an_embed_ready_file(tmp_path):
    state = AnalysisState(user_id="u", round_results=[_round()])
    missing_all = figureless_outline_sections(state, [])
    assert missing_all

    ready = tmp_path / "ready.jpg"
    ready.write_bytes(b"render")
    first = missing_all[0]
    sections = [{"id": first, "figures": [{"png_path": str(ready)}]}]
    missing_after = figureless_outline_sections(state, sections)

    assert first not in missing_after
    assert len(missing_after) == len(missing_all) - 1
