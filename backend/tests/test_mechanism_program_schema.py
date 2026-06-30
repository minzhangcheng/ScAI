from backend.models.schemas import (
    DataDrivenMechanismProgram, RoundResult, AnalysisState,
)


def test_program_roundtrip_minimal():
    p = DataDrivenMechanismProgram(label="iron-redox-lipid hepatocyte program")
    dumped = p.model_dump()
    restored = DataDrivenMechanismProgram(**dumped)
    assert restored.label == "iron-redox-lipid hepatocyte program"
    assert restored.evidence_strength == "weak"
    assert restored.specificity == "low"
    assert restored.bulk_support == "unknown"
    assert restored.source_agent == "bioinformatics"
    assert restored.supporting_genes == []


def test_roundresult_default_empty_programs():
    rr = RoundResult(round_number=1)
    assert rr.data_driven_mechanism_programs == []


def test_analysisstate_default_empty_programs():
    # NOTE: AnalysisState.user_id is required (Field(..., min_length=1)).
    st = AnalysisState(user_id="u1")
    assert st.data_driven_mechanism_programs == []


def test_old_session_without_field_still_loads():
    # AnalysisState must accept dicts that omit the new field entirely
    st = AnalysisState.model_validate({"user_id": "u1", "current_round": 2})
    assert st.data_driven_mechanism_programs == []
