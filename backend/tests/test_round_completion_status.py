from backend.models.schemas import RoundResult
import pytest
from backend.workflow.direction_coverage import (
    active_batch_complete,
    completed_innovation_direction_keys,
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


@pytest.mark.parametrize("linkage_status", [False, None])
def test_execution_linkage_does_not_set_scientific_round_status(linkage_status):
    result = _round()
    derive_round_completion_status(
        {
            "bioinformatics_result": {
                "status": "completed",
                "findings": ["unsupported claim"],
                "evidence_verified": linkage_status,
            },
            "literature_result": {"status": "completed", "findings": ["paper"]},
        },
        result,
    )

    assert result.round_status == "completed"
    assert result.direction_coverage_eligible is True


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
