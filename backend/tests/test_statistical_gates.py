"""Statistical observations run in code, not only in the prompt.

The prompt already asks for most of this. Over a 200-step loop prompted
guidance drifts and nothing observes whether it was followed, which is how a
negative conclusion came to rest on a five-gene panel and a q-value came to
depend on which other sets shared its call.

These checks read structured state -- registered scores, frozen plans, declared
sample counts -- so they see what the analysis actually did rather than what its
code text mentions. Each observation carries the rule that explains it, and no
observation decides acceptance.
"""
from types import SimpleNamespace


import backend.agents.bioinformatics_agent as bio_mod
from backend.services.statistical_gates import (
    check_analysis_unit,
    check_declared_testing_family,
    check_underpowered_negative_claims,
    run_gates,
)


def _score(name, *, method="gseapy.prerank", underpowered=False, plan_id="",
           effective_n=5):
    return SimpleNamespace(
        score_name=name,
        method=method,
        plan_id=plan_id,
        power={"underpowered": underpowered, "effective_n": effective_n,
               "basis": "matched"},
    )


def _plan(plan_id, *, family=()):
    return SimpleNamespace(id=plan_id, testing_family=list(family))


def test_underpowered_negative_claim_is_observed():
    """The real failure: 'not significant' from a five-gene panel."""
    scores = [_score("panel_x", underpowered=True, effective_n=5)]
    findings = ["panel_x was not significant, arguing against this mechanism"]
    out = check_underpowered_negative_claims(findings, scores)
    assert len(out) == 1
    assert out[0].rule_id == "no-negative-claim-from-an-underpowered-panel"
    assert "5 usable gene" in out[0].detail


def test_enrichment_without_a_declared_family_is_observed():
    scores = [_score("enrich_a", method="gseapy.prerank", plan_id="p1")]
    plans = [_plan("p1", family=[])]
    out = check_declared_testing_family(scores, plans)
    assert len(out) == 1
    assert out[0].rule_id == "testing-family-must-be-declared"


def test_single_sample_per_condition_advises_on_an_inferential_claim():
    """Advisory since the count reaching this check is a cross-modality
    minimum; see test_analysis_unit_advisory.py."""
    findings = ["the effect of condition on this program is significant"]
    out = check_analysis_unit(findings, {"A": 1, "B": 1})
    assert len(out) == 1
    assert out[0].rule_id == "one-sample-per-condition-is-unidentifiable"
    assert out[0].severity == "advise"


#: A per-cluster test of proportions, which triggers the simplex observation.
_SIMPLEX_CLAIM = "The proportion of cluster 3 rose from 12% to 31% (chi-square p = 0.001)."


def test_a_registered_joint_model_keeps_the_observation_as_a_note():
    """run_gates held score_definitions and did not forward them, so the
    exemption this check declares was unreachable."""
    scores = [_score("composition", method="pertpy.tools.Sccoda")]
    report = run_gates([_SIMPLEX_CLAIM], score_definitions=scores,
                       scoring_plans=[], samples_per_group={"A": 3, "B": 3})
    assert report.passed
    assert [f.gate for f in report.advisories] == ["simplex-ignored"], (
        "the caveat must still travel: method is free text the model writes, so "
        "the exemption may not delete the finding"
    )


def test_statistical_observation_message_is_actionable_and_non_blocking():
    scores = [_score("panel_x", underpowered=True, plan_id="p1")]
    plans = [_plan("p1", family=[])]
    report = run_gates(["panel_x was not significant"],
                       score_definitions=scores, scoring_plans=plans,
                       samples_per_group={"A": 3, "B": 3})
    assert report.passed is True
    assert report.blocking == []
    assert report.advisories == report.findings
    text = report.message()
    assert "[PI REVIEW OBSERVATION]" in text
    assert "method rule:" in text
    assert "[REJECTED]" not in text


# A gate is only as good as the state it is given. These pin the two
# projections the report_findings handler builds, because narrowing either of
# them is indistinguishable from deleting the checks that read it.

def _live_state(**kw):
    from backend.models.schemas import AnalysisState

    state = AnalysisState(session_id="t", user_id="u", research_purpose="t")
    for key, value in kw.items():
        setattr(state, key, value)
    return state


def _registered(name, method, round_number, *, plan_id="", parameters=None,
                background_source=""):
    from backend.models.schemas import ScoreDefinition

    return ScoreDefinition(
        score_name=name, method=method, description="",
        parameters=parameters or {}, plan_id=plan_id,
        background_source=background_source, round_number=round_number,
    )


def _frozen_plan(plan_id, name, method, *, family=()):
    from backend.models.schemas import ScoringPlan

    return ScoringPlan(id=plan_id, score_name=name, method=method,
                       testing_family=list(family))


def test_an_earlier_rounds_defective_record_remains_available_to_pi_review():
    state = _live_state(
        score_definitions=[
            _registered("orphan_ora", "gseapy.enrich", 2),
            _registered("stale_gsea", "gseapy.prerank", 2, plan_id="p3",
                        parameters={"permutation_num": 1000}),
        ],
        scoring_plans=[_frozen_plan("p3", "stale_gsea", "gseapy.prerank")],
        current_round=3,
    )
    carried = bio_mod._gate_score_definitions(state)
    assert [score.score_name for score in carried] == [
        "orphan_ora", "stale_gsea",
    ]
    report = run_gates(["a positive observation about cluster identity"],
                       score_definitions=bio_mod._gate_score_definitions(state),
                       scoring_plans=state.scoring_plans,
                       samples_per_group={"A": 8, "B": 8})
    assert report.passed


def test_the_current_rounds_scores_are_never_withheld():
    state = _live_state(
        score_definitions=[_registered("fresh_ora", "gseapy.enrich", 3)],
        scoring_plans=[],
        current_round=3,
    )
    carried = bio_mod._gate_score_definitions(state)
    assert [s.score_name for s in carried] == ["fresh_ora"]
    report = run_gates(["a positive observation"], score_definitions=carried,
                       scoring_plans=[], samples_per_group={"A": 8, "B": 8})
    assert "ora-universe-undeclared" in [f.gate for f in report.advisories]


# The resampling floor is a property of one method's draws. These pin what it
# may be applied to, because a floor aimed at the wrong number refuses a correct
# result and the only rewording that satisfies it overstates the uncertainty.
