import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    AnalysisState, RoundResult, CrossModalSynthesis, CrossModalMechanismCandidate,
)
from backend.services.report.generator import build_cross_modal_appendix


def test_appendix_includes_synthesis_and_excluded():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=2)
    rr.cross_modal_synthesis = CrossModalSynthesis(
        matrix_markdown="| m |",
        candidates=[CrossModalMechanismCandidate(theme="iron", mechanism_statement="m")],
        excluded_or_low_support_themes=["rejected theme"],
    )
    state.round_results.append(rr)
    appendix = build_cross_modal_appendix(state)
    assert appendix[0]["round_number"] == 2
    assert appendix[0]["matrix_markdown"] == "| m |"
    assert appendix[0]["excluded_or_low_support_themes"] == ["rejected theme"]
    assert appendix[0]["candidates"][0]["theme"] == "iron"


def test_appendix_empty_when_no_synthesis():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    state.round_results.append(RoundResult(round_number=1))
    assert build_cross_modal_appendix(state) == []
