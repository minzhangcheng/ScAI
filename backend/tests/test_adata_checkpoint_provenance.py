"""F1: artifact provenance follows the cited execution receipt."""
from __future__ import annotations

from backend.agents.bioinformatics.receipt_lookup import artifact_provenance
from backend.models.schemas import AnalysisState, CodeExecution


def _state() -> AnalysisState:
    state = AnalysisState(user_id="u")
    state.sandbox_state = {
        "analysis_generation": 9,
        "cohort_identity_v1": {
            "cohort_generation": 9,
            "cohort_digest": "9" * 64,
        },
    }
    state.code_history = [
        CodeExecution(
            id="exec-old",
            code="old_analysis()",
            output="completed old analysis",
            round_number=2,
            plan_id="plan-old",
            lineage_id="main",
            evidence_schema_version=2,
            execution_status="succeeded",
            analysis_generation=2,
            cohort_generation=3,
            cohort_digest="a" * 64,
        )
    ]
    return state


def test_explicit_source_receipt_wins_over_active_cohort():
    state = _state()
    provenance, error = artifact_provenance(
        state,
        "exec-old",
        fallback_lineage="branch-new",
        fallback_round=7,
        fallback_plan_id="plan-new",
    )
    assert error is None
    assert provenance["analysis_generation"] == 2
    assert provenance["cohort_generation"] == 3
    assert provenance["cohort_digest"] == "a" * 64
    assert provenance["round_number"] == 2
    assert provenance["plan_id"] == "plan-old"
    assert provenance["legacy_provenance"] is False


def test_explicit_missing_source_is_technical_error_not_active_fallback():
    state = _state()
    provenance, error = artifact_provenance(
        state,
        "exec-missing",
        fallback_lineage="branch-new",
        fallback_round=7,
        fallback_plan_id="plan-new",
    )
    assert provenance == {}
    assert "does not resolve" in str(error)


def test_explicit_failed_source_is_technical_error_not_active_fallback():
    state = _state()
    state.code_history[0].execution_status = "failed"
    provenance, error = artifact_provenance(
        state,
        "exec-old",
        fallback_lineage="branch-new",
        fallback_round=7,
        fallback_plan_id="plan-new",
    )
    assert provenance == {}
    assert "not a successful evidence receipt" in str(error)


def test_no_source_id_is_labeled_legacy_fallback():
    state = _state()
    provenance, error = artifact_provenance(
        state,
        None,
        fallback_lineage="branch-new",
        fallback_round=7,
        fallback_plan_id="plan-new",
    )
    assert error is None
    assert provenance["legacy_provenance"] is True
    assert provenance["analysis_generation"] == 9
