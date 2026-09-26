"""X1 contract tests for retained pathology coverage observations."""



from backend.models.schemas import AnalysisState, RoundResult
from backend.services.report.quality import (
    build_quality_disclosures,
)
from backend.workflow.synthesis import build_round_summary_payload


def test_missing_pathology_coverage_is_neutral_and_nonblocking():
    state = AnalysisState(
        user_id="owner",
        round_results=[RoundResult(round_number=1)],
    )
    quality = build_quality_disclosures(state)
    assert quality["blocking_issues"] == []
    assert not any("pathology" in item.lower() for item in quality["cautions"])

    payload = build_round_summary_payload(
        state.round_results[0], state, effective_max_rounds=3,
    )
    assert payload["pathology_coverage"] == []
