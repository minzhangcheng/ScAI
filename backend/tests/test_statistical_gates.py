"""Statistical gates run in code, not in the prompt.

The prompt already asks for most of this. Over a 200-step loop prompted
guidance drifts and nothing observes whether it was followed, which is how a
negative conclusion came to rest on a five-gene panel and a q-value came to
depend on which other sets shared its call.

These gates read structured state -- registered scores, frozen plans, declared
sample counts -- so they see what the analysis actually did rather than what its
code text mentions. Each refusal carries the rule that explains it.
"""
import inspect
from types import SimpleNamespace

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.services.statistical_gates import (
    MIN_SAMPLES_PER_GROUP_FOR_PERMUTATION,
    GateFinding,
    asserts_absence,
    check_analysis_unit,
    check_declared_testing_family,
    check_permutation_at_small_n,
    check_underpowered_negative_claims,
    run_gates,
    score_kind,
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


# ------------------------------------------------- negative-claim detection

@pytest.mark.parametrize("text", [
    "the pathway was not significant in bulk",
    "no evidence of activation",
    "this mechanism is ruled out",
    "the program does not support a role",
])
def test_absence_claims_are_detected(text):
    assert asserts_absence(text)


@pytest.mark.parametrize("text", [
    "the program is elevated",
    "scores increase relative to the comparator",
    "this is compatible with the observed pattern",
])
def test_positive_claims_are_not_flagged(text):
    assert not asserts_absence(text)


def test_underpowered_negative_claim_is_refused():
    """The real failure: 'not significant' from a five-gene panel."""
    scores = [_score("panel_x", underpowered=True, effective_n=5)]
    findings = ["panel_x was not significant, arguing against this mechanism"]
    out = check_underpowered_negative_claims(findings, scores)
    assert len(out) == 1
    assert out[0].rule_id == "no-negative-claim-from-an-underpowered-panel"
    assert "5 usable gene" in out[0].detail


def test_underpowered_positive_claim_is_allowed():
    """Significance despite low power is a stronger signal, not a weaker one."""
    scores = [_score("panel_x", underpowered=True)]
    findings = ["panel_x is elevated relative to the comparator"]
    assert check_underpowered_negative_claims(findings, scores) == []


def test_well_powered_negative_claim_is_allowed():
    """A properly powered null result is a legitimate finding."""
    scores = [_score("panel_x", underpowered=False)]
    findings = ["panel_x was not significant"]
    assert check_underpowered_negative_claims(findings, scores) == []


def test_negative_claim_about_a_different_score_is_not_flagged():
    scores = [_score("panel_x", underpowered=True)]
    findings = ["panel_y was not significant"]
    assert check_underpowered_negative_claims(findings, scores) == []


# ------------------------------------------------------------ testing family

def test_enrichment_without_a_declared_family_is_refused():
    scores = [_score("enrich_a", method="gseapy.prerank", plan_id="p1")]
    plans = [_plan("p1", family=[])]
    out = check_declared_testing_family(scores, plans)
    assert len(out) == 1
    assert out[0].rule_id == "testing-family-must-be-declared"


def test_enrichment_with_a_declared_family_passes():
    scores = [_score("enrich_a", method="gseapy.prerank", plan_id="p1")]
    plans = [_plan("p1", family=["enrich_a", "enrich_b"])]
    assert check_declared_testing_family(scores, plans) == []


def test_non_enrichment_scores_need_no_family():
    """A module score is not corrected across a family, so the rule does not apply."""
    scores = [_score("sig_a", method="sc.tl.score_genes", plan_id="p1")]
    assert check_declared_testing_family(scores, [_plan("p1")]) == []


def test_enrichment_with_no_plan_at_all_is_refused():
    scores = [_score("enrich_a", method="gseapy.prerank", plan_id="")]
    assert len(check_declared_testing_family(scores, [])) == 1


# -------------------------------------------------------- permutation at n

def test_permutation_qvalue_below_viable_n_is_refused():
    """3 vs 3 admits only 10 label partitions; nothing below ~0.1 is reachable."""
    scores = [_score("enrich_a", method="gseapy.prerank")]
    findings = ["enrich_a reached FDR 0.045, supporting the mechanism"]
    out = check_permutation_at_small_n(findings, scores, {"A": 3, "B": 3})
    assert len(out) == 1
    assert out[0].rule_id == "gene-permutation-is-anticonservative"


def test_permutation_qvalue_at_adequate_n_passes():
    scores = [_score("enrich_a", method="gseapy.prerank")]
    findings = ["enrich_a reached FDR 0.045"]
    n = MIN_SAMPLES_PER_GROUP_FOR_PERMUTATION
    assert check_permutation_at_small_n(findings, scores, {"A": n, "B": n}) == []


def test_effect_size_claim_at_small_n_is_allowed():
    """The rule targets q-values, not the effect sizes it recommends instead."""
    scores = [_score("enrich_a", method="gseapy.prerank")]
    findings = ["enrich_a shows a large positive effect size, direction concordant"]
    assert check_permutation_at_small_n(findings, scores, {"A": 3, "B": 3}) == []


def test_unknown_sample_counts_do_not_trigger_the_gate():
    """Absent information must not be read as a violation."""
    scores = [_score("enrich_a", method="gseapy.prerank")]
    findings = ["enrich_a reached FDR 0.045"]
    assert check_permutation_at_small_n(findings, scores, None) == []


# --------------------------------------------------------------- design

def test_single_sample_per_condition_blocks_an_inferential_claim():
    findings = ["the effect of condition on this program is significant"]
    out = check_analysis_unit(findings, {"A": 1, "B": 1})
    assert len(out) == 1
    assert out[0].rule_id == "one-sample-per-condition-is-unidentifiable"


def test_descriptive_wording_passes_at_n1():
    """Correctly labelled contrasts are exactly what the rule asks for."""
    findings = [
        "this contrast is descriptive: condition is confounded with specimen, "
        "so no condition-level effect is estimable"
    ]
    assert check_analysis_unit(findings, {"A": 1, "B": 1}) == []


def test_adequate_replication_passes():
    findings = ["the effect of condition is significant"]
    assert check_analysis_unit(findings, {"A": 3, "B": 3}) == []


# ----------------------------------------------------------- composition

#: A per-cluster test of proportions, which is what the simplex gate refuses.
_SIMPLEX_CLAIM = "The proportion of cluster 3 rose from 12% to 31% (chi-square p = 0.001)."


def test_composition_claim_without_a_joint_model_is_refused():
    report = run_gates([_SIMPLEX_CLAIM], score_definitions=[], scoring_plans=[],
                       samples_per_group={"A": 3, "B": 3})
    assert [f.gate for f in report.blocking] == ["simplex-ignored"]


def test_a_registered_joint_model_downgrades_the_refusal_to_a_note():
    """run_gates held score_definitions and did not forward them, so the
    exemption this check declares was unreachable."""
    scores = [_score("composition", method="pertpy.tools.Sccoda")]
    report = run_gates([_SIMPLEX_CLAIM], score_definitions=scores,
                       scoring_plans=[], samples_per_group={"A": 3, "B": 3})
    assert report.passed, "a registered joint model must not be refused"
    assert [f.gate for f in report.advisories] == ["simplex-ignored"], (
        "the caveat must still travel: method is free text the model writes, so "
        "the exemption may not delete the finding"
    )


def test_an_unrelated_score_does_not_open_the_composition_exemption():
    scores = [_score("panel_x", method="gseapy.prerank")]
    report = run_gates([_SIMPLEX_CLAIM], score_definitions=scores,
                       scoring_plans=[_plan("", family=["ranked_enrichment"])],
                       samples_per_group={"A": 3, "B": 3})
    assert "simplex-ignored" in [f.gate for f in report.blocking]


def test_the_composition_family_covers_the_spellings_a_score_records():
    """Whole-token matching, so a code spelling is not implied by a prose one."""
    for method in ("scanpro.scanpro", "miloR::testNhoods", "pertpy.tools.Sccoda"):
        assert score_kind(SimpleNamespace(method=method)) == "differential_abundance", (
            f"{method} should classify as a joint compositional model"
        )
    for method in ("scipy.stats.betabinom", "statsmodels.glm(family=quasibinomial)"):
        assert score_kind(SimpleNamespace(method=method)) != "differential_abundance", (
            f"{method} fits one population at a time -- it is the design the gate "
            f"refuses, not an escape from it"
        )


# ------------------------------------------------------------- integration

def test_run_gates_aggregates_and_reports_clean():
    report = run_gates(["a positive observation"],
                       score_definitions=[], scoring_plans=[],
                       samples_per_group={"A": 3, "B": 3})
    assert report.passed
    assert report.message() == ""


def test_run_gates_message_is_actionable():
    scores = [_score("panel_x", underpowered=True, plan_id="p1")]
    plans = [_plan("p1", family=[])]
    report = run_gates(["panel_x was not significant"],
                       score_definitions=scores, scoring_plans=plans,
                       samples_per_group={"A": 3, "B": 3})
    assert not report.passed
    text = report.message()
    assert "[REJECTED]" in text
    assert "method rule:" in text, "a refusal must carry the rule that explains it"
    assert "call report_findings again" in text


def test_gate_finding_message_includes_rule_guidance():
    finding = GateFinding(
        gate="g", rule_id="minimum-gene-set-size", detail="detail here",
        subject="panel_x",
    )
    text = finding.message()
    assert "panel_x" in text and "detail here" in text
    assert "[method rule: minimum-gene-set-size]" in text


# ----------------------------------------------------------------- wiring

def test_gate_is_wired_into_report_findings():
    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "report_findings":')
    block = src[start:start + 20000]
    assert "run_statistical_gates" in block
    assert "stats_gate_reject_count" in block


def test_gate_is_bounded_and_records_its_escape():
    """An unsatisfiable check must not trap the loop, but must not vanish either.

    The ceiling is now per-violation rather than fixed. A defect the model can
    repair -- an undeclared testing family, an unattributed panel -- is worth
    asking about up to MAX_STATS_GATE_REJECTIONS times. A design defect it
    cannot repair mid-run gets one attempt: every session measured has one
    cell-resolved library per condition, so the only satisfiable response to a
    cell-level-inference rejection is to reword the claim, and asking twice more
    trains the phrasing rather than the analysis.
    """
    src = inspect.getsource(bio_mod)
    assert "MAX_STATS_GATE_REJECTIONS" in src
    assert "_unsatisfiable" in src, "the per-violation ceiling is missing"
    assert "cell-level-inference" in src
    start = src.index("if stats_gate_reject_count < _cap:")
    block = src[start:start + 3000]
    assert "_escaped_notes.append" in block, (
        "a force-released gate must record the escape in the report"
    )


def test_advisories_reach_the_report_without_blocking():
    """A narrow check earns a note, not a rejection."""
    src = inspect.getsource(bio_mod)
    assert "advisory_message()" in src
    idx = src.index("advisory_message()")
    window = src[idx:idx + 400]
    assert "_escaped_notes.append" in window, (
        "advisories must be recorded in the report rather than only logged"
    )


def test_gates_reference_real_rule_ids():
    """A gate citing a missing rule would emit an empty explanation."""
    from backend.services.method_rules import get_rule

    src = inspect.getsource(
        __import__("backend.services.statistical_gates", fromlist=["x"])
    )
    import re

    for rule_id in set(re.findall(r'rule_id="([a-z0-9-]+)"', src)):
        assert get_rule(rule_id) is not None, f"gate cites unknown rule {rule_id!r}"


def test_gates_inspect_structure_not_biology():
    import re

    from backend.services import statistical_gates

    src = inspect.getsource(statistical_gates).lower()
    for term in ("ferroptos", "apoptos", "nrf2", "hepatocyte"):
        assert term not in src
    for word in ("liver", "mouse", "cancer", "copper"):
        assert not re.search(rf"\b{word}\b", src)
