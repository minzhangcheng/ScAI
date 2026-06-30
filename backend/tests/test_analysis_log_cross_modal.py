import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    AnalysisState, RoundResult, CrossModalSynthesis, CrossModalMechanismCandidate,
)


def test_round_summaries_include_cross_modal_candidates():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rr = RoundResult(round_number=2, pi_summary="s")
    rr.cross_modal_synthesis = CrossModalSynthesis(
        matrix_markdown="| m |",
        candidates=[CrossModalMechanismCandidate(
            theme="iron injury", mechanism_statement="m", confidence="medium",
            relation_to_current_direction="uncovered", report_recommendation="secondary",
        )],
    )
    state.round_results.append(rr)

    from backend.services.report.analysis_log import build_round_summaries
    summaries = build_round_summaries(state)
    assert summaries[0]["cross_modal_candidates"][0]["theme"] == "iron injury"
    assert "matrix_markdown" not in summaries[0]["cross_modal_candidates"][0]


def test_analysis_log_mixin_methods_still_exist():
    """Regression: build_round_summaries must be module-level, NOT inside
    AnalysisLogMixin, and the mixin must retain all its original methods."""
    from backend.services.report.analysis_log import AnalysisLogMixin, build_round_summaries
    # build_round_summaries is a plain function, not a method
    assert callable(build_round_summaries)
    # AnalysisLogMixin still has its methods
    for method in ("_extract_findings", "_build_analysis_record",
                   "_build_full_analysis_log", "_read_notebook"):
        assert hasattr(AnalysisLogMixin, method), f"AnalysisLogMixin missing {method}"


def test_detailed_report_generator_has_mixin_methods():
    """Regression: DetailedReportGenerator must inherit all AnalysisLogMixin methods
    after build_round_summaries was moved to module level."""
    from backend.services.report.generator import DetailedReportGenerator
    for method in ("_extract_findings", "_build_analysis_record",
                   "_build_full_analysis_log", "_read_notebook"):
        assert hasattr(DetailedReportGenerator, method), f"DetailedReportGenerator missing {method}"

