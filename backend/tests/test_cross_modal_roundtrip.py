import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    AnalysisState, RoundResult, CrossModalSynthesis, CrossModalMechanismCandidate,
    CrossModalEvidenceItem, LiteratureReport, LiteratureMechanismSignal,
)


def test_state_model_dump_and_reload_preserves_audit():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=2)
    rr.cross_modal_synthesis = CrossModalSynthesis(
        matrix_markdown="| m |",
        candidates=[CrossModalMechanismCandidate(
            theme="iron", mechanism_statement="m",
            supporting_evidence=[CrossModalEvidenceItem(source="biochemistry", evidence="Fe", strength="moderate")],
        )],
    )
    state.round_results.append(rr)
    lr = LiteratureReport(topic="t", report="r")
    lr.mechanism_signals = [LiteratureMechanismSignal(mechanism="ferroptosis", supporting_pmids=["1"])]
    state.literature_reports.append(lr)

    dumped = state.model_dump(mode="json")
    restored = AnalysisState(**dumped)

    assert restored.round_results[0].cross_modal_synthesis.candidates[0].theme == "iron"
    assert restored.literature_reports[0].mechanism_signals[0].mechanism == "ferroptosis"


def test_old_session_json_without_new_fields_loads():
    # Simulate an old session.json that predates the new optional fields
    old = {
        "user_id": "u1", "name": "n", "initial_instruction": "i",
        "round_results": [{"round_number": 1}],
        "literature_reports": [{"topic": "t", "report": "r"}],
    }
    restored = AnalysisState(**old)
    assert restored.round_results[0].cross_modal_synthesis is None
    assert restored.literature_reports[0].mechanism_signals is None


def test_load_serializer_emits_candidates_after_reload():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=2)
    rr.cross_modal_synthesis = CrossModalSynthesis(
        candidates=[CrossModalMechanismCandidate(theme="iron", mechanism_statement="m")],
    )
    state.round_results.append(rr)
    serialized = AnalysisState._serialize_round_results(
        [r for r in state.model_dump(mode="json")["round_results"]]
    )
    assert serialized[0]["cross_modal_candidates"][0]["theme"] == "iron"
