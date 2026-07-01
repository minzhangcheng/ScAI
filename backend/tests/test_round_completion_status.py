from backend.models.schemas import RoundResult
from backend.workflow.direction_coverage import completed_innovation_direction_keys
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
    assert completed_innovation_direction_keys(state) == set()
