from backend.models.schemas import AnalysisPhase, AnalysisState, RoundResult
from backend.workflow.reporting import validate_complete_report_allowed
from backend.workflow.core import SingleCellAnalysisWorkflow


def test_missing_direction_coverage_blocks_complete_report():
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

    allowed, message = validate_complete_report_allowed(state)

    assert allowed is False
    assert "b" in message.lower()


def test_partial_report_requires_explicit_state_flag():
    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        allow_partial_report=True,
        user_chosen_directions=[{"direction": "A", "notes": ""}, {"direction": "B", "notes": ""}],
        round_results=[],
    )

    allowed, message = validate_complete_report_allowed(state)

    assert allowed is True
    assert message == ""


def test_after_final_report_routes_blocked_state_to_terminal():
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    assert workflow._after_final_report({"report_blocked": True}) == "blocked"
    assert workflow._after_final_report({"report_blocked": False}) == "continue"
