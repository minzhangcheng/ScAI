from backend.models.schemas import AnalysisPhase, AnalysisState, RoundResult
from backend.workflow.reporting import report_coverage_observation
from backend.workflow.core import SingleCellAnalysisWorkflow


def test_missing_direction_coverage_is_a_pi_review_observation():
    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "A", "notes": ""}, {"direction": "B", "notes": ""}],
        round_results=[
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="A",
                pi_summary="summary",
                bioinformatics_findings=["finding"],
                evidence_status="substantive",
            )
        ],
    )

    message = report_coverage_observation(state)

    assert "b" in message.lower()
    assert "PI must decide" in message


def test_partial_flag_does_not_hide_direction_coverage_fact():
    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        allow_partial_report=True,
        user_chosen_directions=[{"direction": "A", "notes": ""}, {"direction": "B", "notes": ""}],
        round_results=[],
    )

    message = report_coverage_observation(state)

    assert message
    assert "PI must decide" in message


def test_reporting_has_no_host_direction_coverage_acceptance_router():
    assert not hasattr(SingleCellAnalysisWorkflow, "_after_final_report")
