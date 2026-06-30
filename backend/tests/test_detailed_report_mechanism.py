"""Tests: mechanism programs surfaced in detailed report producer + consumer."""
from backend.models.schemas import AnalysisState, RoundResult, DataDrivenMechanismProgram
from backend.services.report.analysis_log import build_round_summaries
from backend.services.report.planning import insert_cross_modal_section


def _state_with_program():
    st = AnalysisState(user_id="u1")
    rr = RoundResult(round_number=1)
    rr.data_driven_mechanism_programs = [DataDrivenMechanismProgram(
        label="iron-redox-lipid hepatocyte program",
        triggering_observations=["GPX4 down", "ACSL4 up"],
        evidence_strength="moderate", specificity="medium",
        interpretation="Candidate, not confirmed.",
    )]
    st.round_results = [rr]
    return st


def test_build_round_summaries_exposes_programs():
    summaries = build_round_summaries(_state_with_program())
    assert summaries[0]["mechanism_programs"]
    assert summaries[0]["mechanism_programs"][0]["label"].startswith("iron-redox-lipid")


def test_insert_section_renders_programs_without_cross_modal():
    summaries = build_round_summaries(_state_with_program())
    plan = {"sections": [{"title": "Conclusion"}]}
    out = insert_cross_modal_section(plan, summaries)
    titles = [s.get("title", "") for s in out["sections"]]
    assert any("Mechanism" in t for t in titles)
