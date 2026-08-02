from backend.models.schemas import RoundResult
from backend.workflow.direction_coverage import (
    active_batch_complete,
    completed_innovation_direction_keys,
    normalize_legacy_round_result,
)
from backend.workflow.synthesis import derive_round_completion_status
from backend.models.schemas import AnalysisPhase, AnalysisState


def _round(direction: str = "A") -> RoundResult:
    return RoundResult(
        round_number=2,
        phase="innovation",
        direction=direction,
        pi_summary="summary",
        bioinformatics_findings=["finding"],
    )


def test_failed_node_marks_round_skipped_and_not_complete():
    result = _round()
    derive_round_completion_status(
        {"failed_node": "bioinformatics_analysis", "bioinformatics_result": {"status": "failed", "error": "upstream"}},
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "partial"
    assert result.evidence_status == "substantive"
    assert completed_innovation_direction_keys(state) == set()


def test_agent_error_with_findings_marks_partial_not_complete():
    result = _round()
    derive_round_completion_status(
        {
            "bioinformatics_result": {"status": "failed", "error": "execution failed"},
            "literature_result": {"status": "completed", "findings": ["paper"]},
        },
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "partial"
    assert result.evidence_status == "substantive"
    assert completed_innovation_direction_keys(state) == set()


def test_unverified_execution_evidence_marks_round_partial():
    result = _round()
    derive_round_completion_status(
        {
            "bioinformatics_result": {
                "status": "completed",
                "findings": ["unsupported claim"],
                "evidence_verified": False,
            },
            "literature_result": {"status": "completed", "findings": ["paper"]},
        },
        result,
    )

    assert result.round_status == "partial"
    assert result.direction_coverage_eligible is False


def test_completed_results_with_findings_mark_substantive():
    result = _round()
    derive_round_completion_status(
        {
            "bioinformatics_result": {"status": "completed", "findings": ["finding"]},
            "literature_result": {"status": "completed", "findings": ["paper"]},
        },
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "completed"
    assert result.evidence_status == "substantive"
    assert completed_innovation_direction_keys(state) == {"a"}


def test_optional_skipped_agents_do_not_block_completed_direction():
    result = _round()
    derive_round_completion_status(
        {
            "pathology_result": {"status": "skipped", "findings": []},
            "biochemistry_result": {"status": "no_data", "findings": []},
            "bioinformatics_result": {"status": "completed", "findings": ["finding"]},
            "literature_result": {"status": "completed", "findings": ["paper"]},
        },
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "completed"
    assert result.evidence_status == "substantive"
    assert completed_innovation_direction_keys(state) == {"a"}


def test_core_skipped_agent_blocks_completed_direction_even_with_evidence():
    result = _round()
    derive_round_completion_status(
        {
            "bioinformatics_result": {"status": "completed", "findings": ["finding"]},
            "literature_result": {"status": "skipped", "findings": []},
        },
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "partial"
    assert result.evidence_status == "substantive"
    assert completed_innovation_direction_keys(state) == set()


def test_optional_failed_agent_marks_partial_not_completed():
    result = _round()
    derive_round_completion_status(
        {
            "pathology_result": {"status": "failed", "error": "pathology failed", "findings": []},
            "bioinformatics_result": {"status": "completed", "findings": ["finding"]},
            "literature_result": {"status": "completed", "findings": ["paper"]},
        },
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "partial"
    assert result.evidence_status == "substantive"
    assert result.direction_coverage_eligible is True
    assert completed_innovation_direction_keys(state) == {"a"}


def test_optional_partial_agent_degrades_round_but_covers_direction():
    result = _round()
    derive_round_completion_status(
        {
            "pathology_result": {"status": "partial", "findings": ["limited image coverage"]},
            "bioinformatics_result": {"status": "completed", "findings": ["finding"]},
            "literature_result": {"status": "completed", "findings": ["paper"]},
        },
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "partial"
    assert result.direction_coverage_eligible is True
    assert completed_innovation_direction_keys(state) == {"a"}


def test_optional_only_evidence_does_not_cover_direction():
    result = RoundResult(
        round_number=2,
        phase="innovation",
        direction="A",
        pi_summary="summary",
        pathology_findings=["limited image finding"],
    )
    derive_round_completion_status(
        {
            "pathology_result": {
                "status": "partial",
                "findings": ["limited image finding"],
            },
            "bioinformatics_result": {"status": "completed", "findings": []},
            "literature_result": {"status": "completed", "findings": []},
        },
        result,
    )

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}],
        round_results=[result],
    )

    assert result.round_status == "partial"
    assert result.evidence_status == "substantive"
    assert result.direction_coverage_eligible is False
    assert completed_innovation_direction_keys(state) == set()


def test_optional_partial_results_allow_multi_direction_batch_to_converge():
    results = []
    for round_number, direction in enumerate(("A", "B"), start=2):
        result = _round(direction)
        result.round_number = round_number
        derive_round_completion_status(
            {
                "pathology_result": {
                    "status": "partial",
                    "findings": ["limited image coverage"],
                },
                "bioinformatics_result": {
                    "status": "completed",
                    "findings": [f"finding for {direction}"],
                },
                "literature_result": {
                    "status": "completed",
                    "findings": [f"paper for {direction}"],
                },
            },
            result,
        )
        results.append(result)

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[
            {"direction": "A", "notes": ""},
            {"direction": "B", "notes": ""},
        ],
        active_direction_batch=[
            {"direction": "A", "notes": ""},
            {"direction": "B", "notes": ""},
        ],
        round_results=results,
    )

    assert all(result.round_status == "partial" for result in results)
    assert completed_innovation_direction_keys(state) == {"a", "b"}
    assert active_batch_complete(state) is True


def test_missing_core_results_mark_stopped_round_partial():
    """Stop-and-summarize may have optional evidence before core agents run."""
    result = RoundResult(
        round_number=7,
        phase="innovation",
        direction="A",
        pi_summary="summary from available evidence",
        biochemistry_findings=["available finding"],
    )
    derive_round_completion_status(
        {
            "biochemistry_result": {
                "status": "completed",
                "findings": ["available finding"],
            },
            "bioinformatics_result": None,
            "literature_result": None,
        },
        result,
    )

    assert result.round_status == "partial"
    assert result.evidence_status == "substantive"


def test_missing_core_results_without_evidence_mark_round_skipped():
    result = RoundResult(
        round_number=7,
        phase="innovation",
        direction="A",
        pi_summary="No analysis completed",
    )
    derive_round_completion_status({}, result)

    assert result.round_status == "skipped"
    assert result.evidence_status == "failed"


def test_integration_completion_uses_integration_evidence_not_core_agent_slots():
    result = RoundResult(
        round_number=4,
        phase="integration",
        integration_report_summary="Cross-direction evidence synthesis",
        integration_candidates=[{"direction": "New direction"}],
    )

    derive_round_completion_status({"analysis_state": AnalysisState(id="s", user_id="u")}, result)

    assert result.round_status == "completed"
    assert result.evidence_status == "substantive"
    assert result.direction_coverage_eligible is None


def test_integration_record_never_satisfies_innovation_direction_coverage():
    result = RoundResult(
        round_number=4,
        phase="integration",
        direction="A",
        integration_report_summary="Cross-direction evidence synthesis",
        direction_coverage_eligible=True,
    )
    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INTEGRATION,
        user_chosen_directions=[{"direction": "A"}],
        round_results=[result],
    )

    assert completed_innovation_direction_keys(state) == set()


def test_legacy_integration_false_failure_is_normalized_from_own_evidence():
    result = RoundResult(
        round_number=4,
        phase="integration",
        integration_report_summary="Persisted integration synthesis",
        round_status="skipped",
        evidence_status="failed",
        direction_coverage_eligible=False,
    )

    result = normalize_legacy_round_result(result)

    assert result.round_status == "completed"
    assert result.evidence_status == "substantive"
    assert result.direction_coverage_eligible is None
