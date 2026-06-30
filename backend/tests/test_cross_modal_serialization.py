import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    AnalysisState, RoundResult, CrossModalSynthesis,
    CrossModalMechanismCandidate, CrossModalEvidenceItem,
)


def _candidate():
    return CrossModalMechanismCandidate(
        theme="iron oxidative injury",
        mechanism_statement="Iron drives lipid peroxidation",
        supporting_evidence=[CrossModalEvidenceItem(source="biochemistry", evidence="Fe 2.2x", strength="moderate")],
        confidence="medium",
        relation_to_current_direction="uncovered",
        report_recommendation="secondary",
    )


def test_serialize_includes_simplified_candidates():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=2)
    rr.cross_modal_synthesis = CrossModalSynthesis(
        matrix_markdown="| a | b |\n|---|---|\n| 1 | 2 |",
        candidates=[_candidate()],
        excluded_or_low_support_themes=["something"],
    )
    state.round_results.append(rr)
    dumped = state.model_dump(mode="json")  # triggers serializer path used by load API

    serialized = AnalysisState._serialize_round_results([
        {**state.round_results[0].model_dump(mode="json")}
    ])
    cands = serialized[0]["cross_modal_candidates"]
    assert len(cands) == 1
    c = cands[0]
    assert set(c.keys()) == {
        "theme", "mechanism_statement", "confidence",
        "relation_to_current_direction", "report_recommendation",
    }
    assert "matrix_markdown" not in c
    assert "supporting_evidence" not in c


def test_serialize_no_candidates_when_absent():
    serialized = AnalysisState._serialize_round_results([
        {**RoundResult(round_number=1).model_dump(mode="json")}
    ])
    assert serialized[0]["cross_modal_candidates"] == []
