"""Contracts for deterministic report completeness and quality disclosures."""
from types import SimpleNamespace

import pytest

from backend.models.schemas import (
    AnalysisPhase,
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
    render_figure_reservation_notice_html,
    render_report_limitations_html,
    render_technical_review_notice_html,
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

    assert quality["status"] == "qualified"
    assert quality["blocking_issues"] == []
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


def test_pi_no_action_with_missing_step_decision_does_not_close_review():
    review = {
        "round_number": 1,
        "review_status": "no_action",
        "review_plan": {
            "verdict": "no_action",
            "step_decisions": [{"plan_step_id": 1, "status": "completed"}],
        },
        "plan_audit": {"n_steps": 2, "n_unmet": 0, "n_unverifiable": 0},
        "review_result": {},
        "assessment": {"remaining_gaps": []},
    }
    state = SimpleNamespace(
        round_results=[], post_execution_reviews=[review],
        annotation_record=None, annotation_state=None,
    )

    quality = build_quality_disclosures(state)

    assert quality["status"] == "qualified"
    disclosure = quality["post_execution_reviews"][0]
    assert disclosure["final_gap_status"] == "pending_pi_decision"
    assert disclosure["pi_completion"]["missing_decision_count"] == 1
    assert any(
        "no recorded PI completion decision for 1 plan step" in caution
        for caution in quality["cautions"]
    )


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

    assert quality["status"] == "qualified"
    assert quality["blocking_issues"] == []
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
    assert quality["annotation"]["validation"]["summary"] == (
        "One low-confidence cluster remains."
    )
    assert render_report_limitations_html({"quality_disclosures": quality}) == ""


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


def test_integration_round_does_not_require_bioinformatics_verification():
    state = AnalysisState(
        user_id="u",
        round_results=[_round(
            phase=AnalysisPhase.INTEGRATION,
            direction_coverage_eligible=None,
            bioinformatics_evidence_verified=None,
            integration_report_summary="Cross-direction synthesis",
        )],
    )

    quality = build_quality_disclosures(state)

    assert quality["rounds"][0]["phase"] == "integration"
    assert not any("bioinformatics" in item.lower() for item in quality["blocking_issues"])
    assert not any("direction-coverage" in item for item in quality["blocking_issues"])


@pytest.mark.parametrize(
    ("reason", "extra"),
    [
        ("figure_coverage", {"figureless_sections": ["direction_1"]}),
        ("degraded_fallback", {}),
        ("approved_partial", {}),
        ("analysis_quality", {}),
    ],
)
def test_deterministic_qualification_block_is_omitted_for_every_reason(reason, extra):
    report = {
        "status": "partial",
        "partial_reason": reason,
        "partial_reasons": [reason],
        **extra,
    }
    assert render_report_limitations_html(report) == ""
    assert report["partial_reason"] == reason


def test_report_limitations_hook_omits_blocks_without_mutating_details():
    report = {
        "status": "partial",
        "partial_reasons": ["approved_partial", "analysis_quality"],
        "quality_disclosures": {
            "blocking_issues": ["Round 2 evidence is unverified."],
            "cautions": ["Optional modality unavailable."],
        },
    }

    assert render_report_limitations_html(report) == ""
    assert report["partial_reasons"] == ["approved_partial", "analysis_quality"]
    assert report["quality_disclosures"]["blocking_issues"] == [
        "Round 2 evidence is unverified."
    ]


def test_technical_review_notice_is_selective_and_escaped():
    report = {
        "status": "partial",
        "partial_reasons": ["analysis_quality"],
        "figures": [{
            "id": "figure-unavailable",
            "filename": "panel<&.png",
            "is_publication": True,
            "review_unavailable": True,
            "review_status": "approved",
        }, {
            "id": "figure-pending",
            "filename": "pending.png",
            "is_publication": True,
            "review_status": "pending",
        }],
        "graphic_abstract": {
            "review_available": None,
            "selection_available": False,
        },
        "quality_disclosures": {
            "cautions": ["Scientific evidence is partial."],
            "post_execution_reviews": [{
                "review_status": "pending",
                "lifecycle": "reserved",
                "lifecycle_status": "reserved",
                "supplementary_execution": {"status": "unknown"},
            }, {
                "review_status": "failed",
                "review_reason": "Review planning failed: API <down & retry>",
                "supplementary_execution": {"status": "unknown"},
            }, {
                "review_status": "failed",
                "persistence_status": "failed",
                "persistence_failure": "Persistence <error & retry>",
                "supplementary_execution": {"status": "failed"},
            }],
        },
    }

    notice = render_technical_review_notice_html(report)

    assert notice.count("technical-review-notice") == 1
    assert "panel&lt;&amp;.png" in notice
    assert "pending.png" in notice
    assert "Graphic abstract candidate selection was unavailable" in notice
    assert "Technical review remains pending" in notice
    assert "API &lt;down &amp; retry&gt;" in notice
    assert "Persistence &lt;error &amp; retry&gt;" in notice
    assert "Scientific evidence is partial" not in notice
    assert "<script" not in notice


def test_figure_reservation_notice_is_deterministic_and_escaped_once():
    rendered = render_figure_reservation_notice_html({
        "summary": "PI accepted A & B < C with reservations.",
        "details": ["Labels A & B < C still overlap."],
    })

    assert "figure-reservation-notice" in rendered
    assert "A &amp; B &lt; C" in rendered
    assert "&amp;amp;" not in rendered
    assert "&amp;lt;" not in rendered


def test_quality_disclosure_separates_review_time_and_final_figure_state():
    state = AnalysisState(id="session", user_id="owner")
    state.post_execution_reviews = [{
        "review_attempt_id": "attempt-1",
        "round_number": 9,
        "review_status": "completed",
        "review_result": {"status": "completed", "warnings": []},
        "review_plan": {"step_decisions": []},
        "assessment": {
            "verdict": "partial",
            "remaining_gaps": ["A scientific limitation remains."],
        },
        "plan_audit": {"n_figure_unmet": 1},
        "technical_snapshots": {
            "at_review": {
                "figure_contract": {
                    "passed": False, "n_requirements": 1, "n_missing": 1,
                },
            },
            "after_figure_followup": {
                "figure_contract": {
                    "passed": True, "n_requirements": 1, "n_missing": 0,
                    "eligible_figure_names": ["round9_umap"],
                },
            },
        },
    }]

    review = build_quality_disclosures(state)["post_execution_reviews"][0]

    assert review["plan_audit"]["n_figure_unmet"] == 1
    assert review["figure_contract_at_review"]["passed"] is False
    assert review["final_figure_contract"]["passed"] is True
    assert review["figure_gap_status"] == "fulfilled"
    assert review["final_gap_status"] == "unresolved"


def test_technical_notice_ignores_legacy_graphic_metadata_and_generic_partial():
    report = {
        "status": "partial",
        "partial_reason": "analysis_quality",
        "graphic_abstract": {
            "review_available": None,
            "selection_available": None,
        },
        "quality_disclosures": {
            "post_execution_reviews": [{
                "review_status": "completed",
                "lifecycle": "completed",
                "review_reason": "A scientific caveat remains.",
            }],
        },
    }

    assert render_technical_review_notice_html(report) == ""


def test_technical_notice_suppresses_scientific_trigger_after_supplementary_failure():
    report = {
        "quality_disclosures": {
            "post_execution_reviews": [{
                "review_status": "failed",
                "lifecycle": "failed",
                "lifecycle_status": "failed",
                "review_reason": "Scientific trigger: missing interaction evidence",
                "supplementary_execution": {"status": "failed"},
            }],
        },
    }

    notice = render_technical_review_notice_html(report)

    assert "technical-review-notice" in notice
    assert "Supplementary technical review execution failed" in notice
    assert "Scientific trigger" not in notice


def test_technical_notice_normalizer_keeps_api_failure_reason_without_execution():
    state = SimpleNamespace(
        round_results=[],
        annotation_record=None,
        annotation_state=None,
        post_execution_reviews=[{
            "review_status": "failed",
            "review_reason": "Review planning failed: API <down & retry>",
        }],
    )

    quality = build_quality_disclosures(state)
    notice = render_technical_review_notice_html({
        "quality_disclosures": quality,
    })

    review = quality["post_execution_reviews"][0]
    assert review["supplementary_execution"]["status"] == "unknown"
    assert "API &lt;down &amp; retry&gt;" in notice


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
