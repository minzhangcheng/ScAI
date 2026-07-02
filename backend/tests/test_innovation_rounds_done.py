from backend.models.schemas import AnalysisPhase, AnalysisState, RoundResult
from backend.workflow.direction_coverage import normalize_direction_key
from backend.workflow.synthesis import count_innovation_rounds_done


def test_failed_innovation_round_not_counted():
    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        round_results=[
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction A",
                direction_key=normalize_direction_key("Direction A"),
                result_identity_key="main:innovation:2:direction a",
                pi_summary="summary",
                round_status="failed",
                evidence_status="failed",
                bioinformatics_findings=["finding"],
            ),
        ],
        user_chosen_directions=[{"direction": "Direction A", "notes": ""}],
    )
    assert count_innovation_rounds_done(state) == 0
