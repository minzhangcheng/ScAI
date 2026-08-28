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
from backend.agents.bioinformatics import executor_report_gates
from backend.services import statistical_gates
from backend.services.statistical_gates import (
    MIN_SAMPLES_PER_GROUP_FOR_PERMUTATION,
    GateFinding,
    asserts_absence,
    check_analysis_unit,
    check_declared_testing_family,
    check_permutation_at_small_n,
    check_resampling_resolution,
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


def test_partially_unknown_sample_counts_do_not_raise():
    scores = [_score("enrich_a", method="gseapy.prerank")]
    findings = ["enrich_a reached FDR 0.045"]
    out = check_permutation_at_small_n(
        findings, scores, {"A": None, "B": 3},
    )
    assert len(out) == 1


# --------------------------------------------------------------- design

def test_single_sample_per_condition_advises_on_an_inferential_claim():
    """Advisory since the count reaching this check is a cross-modality
    minimum; see test_analysis_unit_advisory.py."""
    findings = ["the effect of condition on this program is significant"]
    out = check_analysis_unit(findings, {"A": 1, "B": 1})
    assert len(out) == 1
    assert out[0].rule_id == "one-sample-per-condition-is-unidentifiable"
    assert out[0].severity == "advise"


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


def test_run_gates_calls_families_in_frozen_order(monkeypatch):
    calls = []
    names = [
        "check_underpowered_negative_claims",
        "check_declared_testing_family",
        "check_permutation_at_small_n",
        "check_resampling_resolution",
        "check_panel_redundancy",
        "check_analysis_unit",
        "check_cell_level_inference",
        "check_ora_background",
        "check_composition_claims",
        "check_magnitude_claims",
    ]
    for name in names:
        monkeypatch.setattr(
            statistical_gates,
            name,
            lambda *args, _name=name, **kwargs: calls.append(_name) or [],
        )

    statistical_gates.run_gates([], score_definitions=[], scoring_plans=[])

    assert calls == names


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
    block = inspect.getsource(executor_report_gates.prepare_report_attempt)
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
    src = inspect.getsource(executor_report_gates.prepare_report_attempt)
    assert "max_stats_gate_rejections" in src
    assert "_unsatisfiable" in src, "the per-violation ceiling is missing"
    assert "cell-level-inference" in src
    start = src.index("if context.stats_gate_reject_count < _cap:")
    block = src[start:start + 3000]
    assert "_escaped_notes.append" in block, (
        "a force-released gate must record the escape in the report"
    )


def test_advisories_reach_the_report_without_blocking():
    """A narrow check earns a note, not a rejection."""
    src = inspect.getsource(executor_report_gates.prepare_report_attempt)
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


# -------------------------------------------- what the agent hands the gates
# A gate is only as good as the state it is given. These pin the two
# projections the report_findings handler builds, because narrowing either of
# them is indistinguishable from deleting the checks that read it.

def _live_state(**kw):
    from backend.models.schemas import AnalysisState

    state = AnalysisState(session_id="t", user_id="u", research_purpose="t")
    for key, value in kw.items():
        setattr(state, key, value)
    return state


def _group(group_id, name, sample_count):
    from backend.models.schemas import TreatmentGroup

    return TreatmentGroup(id=group_id, name=name, sample_count=sample_count)


def _uploaded(group_id, sample_number):
    from backend.models.schemas import DataFile

    return DataFile(filename="f", file_path="/f", section="sequencing",
                    data_type="scRNA-seq", group_id=group_id,
                    sample_number=sample_number)


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


def test_declarations_without_profiling_fail_closed_as_unresolved():
    """Group declarations alone cannot establish biological replication."""
    state = _live_state(
        treatment_groups=[_group("ga", "Vehicle", 1), _group("gb", "Treated", 1)],
        data_files=[_uploaded("ga", None), _uploaded("gb", None)],
        current_round=2,
    )
    samples = bio_mod._gate_samples_per_group_by_modality(state)
    assert samples == {"scRNA-seq": {}}
    report = run_gates(["Treatment significantly increases Il6 (p = 0.003)."],
                       score_definitions=[], scoring_plans=[],
                       samples_per_group_by_modality=samples)
    assert report.advisories == []
    assert [f.gate for f in report.blocking] == ["sample-design-unresolved"]


def test_numbered_uploads_raise_a_count_left_at_the_schema_default():
    """sample_count defaults to 1, so per-sample files are the harder evidence.

    They raise a stale declaration rather than deciding whether the group is
    looked at, which is what keeps the previous test's design visible without
    refusing a study whose upload plainly holds replicates.
    """
    state = _live_state(
        treatment_groups=[_group("gc", "Control", 1), _group("gd", "KO", 1)],
        data_files=[_uploaded(g, n) for g in ("gc", "gd") for n in (1, 2, 3)],
        current_round=2,
    )
    samples = bio_mod._gate_samples_per_group_by_modality(state)
    assert samples == {"scRNA-seq": {"Control": 3, "KO": 3}}
    report = run_gates(["the effect of condition is significant (p = 0.01)"],
                       score_definitions=[], scoring_plans=[],
                       samples_per_group_by_modality=samples)
    assert report.passed


def test_an_earlier_rounds_resample_count_still_bounds_a_later_p_value():
    """Evidence does not expire at a round boundary.

    The count was declared when the score was registered; the value it makes
    unattainable is quoted a round later, which is the normal shape of a
    multi-round analysis.
    """
    state = _live_state(
        score_definitions=[_registered("hallmark_gsea", "gseapy.prerank", 2,
                                       plan_id="p1",
                                       parameters={"permutation_num": 500})],
        scoring_plans=[_frozen_plan("p1", "hallmark_gsea", "gseapy.prerank",
                                   family=["hallmark"])],
        current_round=3,
    )
    carried = bio_mod._gate_score_definitions(state)
    assert [s.score_name for s in carried] == ["hallmark_gsea"]
    report = run_gates(["GSEA gives NES = 2.1, p = 0.0002."],
                       score_definitions=carried,
                       scoring_plans=state.scoring_plans,
                       samples_per_group={"A": 8, "B": 8})
    assert [f.gate for f in report.blocking] == ["p-below-resampling-resolution"]


def test_an_earlier_rounds_joint_model_still_downgrades_the_simplex_refusal():
    """The exemption travels with the evidence, or the gate refuses good work.

    check_composition_claims runs unconditionally and reads score_definitions
    for the stand-down. Dropping an earlier round's scCODA registration turns a
    correctly modelled composition claim back into a hard block, and there is
    no way to re-register it from the round that is being refused.
    """
    state = _live_state(
        score_definitions=[_registered("composition", "pertpy.tools.Sccoda", 2,
                                       plan_id="p2")],
        scoring_plans=[_frozen_plan("p2", "composition",
                                   "pertpy.tools.Sccoda")],
        current_round=3,
    )
    carried = bio_mod._gate_score_definitions(state)
    assert [s.score_name for s in carried] == ["composition"]
    report = run_gates([_SIMPLEX_CLAIM], score_definitions=carried,
                       scoring_plans=state.scoring_plans,
                       samples_per_group={"A": 8, "B": 8})
    assert report.passed
    assert [f.gate for f in report.advisories] == ["simplex-ignored"]


def test_an_earlier_rounds_defective_record_does_not_refuse_forever():
    """The one thing a stale score may not do is block a round it has nothing
    to do with.

    An undeclared testing family and an unnamed ORA universe are refused
    against the record itself, and register_score only replaces a score of the
    same name, so a round working on something else can never clear them. Those
    records, and only those, are left out.
    """
    state = _live_state(
        score_definitions=[
            _registered("orphan_ora", "gseapy.enrich", 2),
            _registered("stale_gsea", "gseapy.prerank", 2, plan_id="p3",
                        parameters={"permutation_num": 1000}),
        ],
        scoring_plans=[_frozen_plan("p3", "stale_gsea", "gseapy.prerank")],
        current_round=3,
    )
    assert bio_mod._gate_score_definitions(state) == []
    report = run_gates(["a positive observation about cluster identity"],
                       score_definitions=bio_mod._gate_score_definitions(state),
                       scoring_plans=state.scoring_plans,
                       samples_per_group={"A": 8, "B": 8})
    assert report.passed


def test_the_current_rounds_scores_are_never_withheld():
    """A defective record registered this round must still be refused now."""
    state = _live_state(
        score_definitions=[_registered("fresh_ora", "gseapy.enrich", 3)],
        scoring_plans=[],
        current_round=3,
    )
    carried = bio_mod._gate_score_definitions(state)
    assert [s.score_name for s in carried] == ["fresh_ora"]
    report = run_gates(["a positive observation"], score_definitions=carried,
                       scoring_plans=[], samples_per_group={"A": 8, "B": 8})
    assert "ora-universe-undeclared" in [f.gate for f in report.blocking]


# ------------------------------------------- which number the floor applies to
# The resampling floor is a property of one method's draws. These pin what it
# may be applied to, because a floor aimed at the wrong number refuses a correct
# result and the only rewording that satisfies it overstates the uncertainty.

def _floor_gates(findings, scores):
    return [f.gate for f in check_resampling_resolution(findings, scores)
            if f.gate == "p-below-resampling-resolution"]


def test_an_analytic_p_value_is_not_held_to_the_enrichment_floor():
    """A model-fit p-value has no resolution floor, whatever shares its paragraph.

    Reporting a differential-expression result and an enrichment result in one
    paragraph is the ordinary shape of a finding. Deciding the whole text is a
    resampling result because one clause mentions GSEA, then applying the floor
    to every number in it, refuses a Wald p-value that no number of draws
    constrains.
    """
    scores = [_registered("hallmark_gsea", "gseapy.prerank", 1,
                          parameters={"permutation_num": 1000})]
    mixed = ("Differential expression with pydeseq2 identified GENE1 as "
             "strongly upregulated (log2FC 3.4, adjusted p = 1.2e-08). GSEA "
             "preranked on the same contrast showed enrichment of the "
             "interferon module (NES 2.1).")
    assert _floor_gates([mixed], scores) == []

    # The floor still reaches a resampling value stated in its own sentence.
    assert _floor_gates(["Preranked GSEA gives NES = 2.1, p = 0.0002."],
                        scores) == ["p-below-resampling-resolution"]


def test_a_coarse_probe_does_not_bound_a_finer_runs_p_value():
    """The floor follows the score the finding names, not the worst registered.

    A cheap 100-draw probe beside a 10,000-draw run is the normal shape of an
    analysis. Taking the minimum count across every registered score applies the
    probe's floor to the run's own p-values.
    """
    scores = [_registered("quick_check", "gseapy.prerank", 1,
                          parameters={"permutation_num": 100}),
              _registered("main_gsea", "gseapy.prerank", 1,
                          parameters={"permutation_num": 10000})]
    fine = ("GSEA preranked analysis of the main contrast: NES = 2.31, "
            "FDR q = 0.0005 for the top pathway.")
    assert _floor_gates([fine], scores) == []

    # Named coarse score, so its own count binds and the value is unattainable.
    coarse = "The quick check panel gives NES = 1.9, p = 0.0004."
    assert _floor_gates([coarse], scores) == ["p-below-resampling-resolution"]


def test_a_conventional_cutoff_is_not_a_resolution_floor_violation():
    """0.001 quoted against gseapy's default 1000 draws is a threshold restated.

    1/1001 rounds to the same three digits as the conventional cutoff, so
    without the carve-out the most standard sentence the method produces is
    refused -- and the refusal reads "states a p-value of 0.001 ... the smallest
    value is 0.001", which the model cannot act on.
    """
    scores = [_registered("hallmark_gsea", "gseapy.prerank", 1,
                          parameters={"permutation_num": 1000})]
    for text in ("Preranked GSEA identified HALLMARK_INFLAMMATORY_RESPONSE "
                 "(NES 2.4, FDR q = 0.001).",
                 "Preranked GSEA showed the module significant at padj = 0.001."):
        assert _floor_gates([text], scores) == []

    # A value that is not a conventional cutoff and is under the floor still goes.
    assert _floor_gates(["Preranked GSEA: NES = 2.4, q = 0.0004."],
                        scores) == ["p-below-resampling-resolution"]


def test_a_detected_singleton_reaches_the_analysis_unit_check():
    """The end of the measured failure.

    Declared 3, uploaded 1, every DataFile carrying sample_number=None: the old
    resolver computed max(3, 0) = 3 and check_analysis_unit -- which exists for
    exactly this design -- returned nothing.

    Single-modality on purpose: this pins that the singleton is still seen. The
    multi-modality case, where the same collapse produces a count that describes
    a different arm than the finding, is test_analysis_unit_advisory.py.
    """
    from backend.models.schemas import SampleCountRecord

    state = _live_state(
        treatment_groups=[_group("g1", "Con", 3), _group("g2", "DKO", 3)],
        data_files=[_uploaded("g1", None), _uploaded("g2", None)],
        current_round=1,
    )
    state.detected_sample_counts = [
        SampleCountRecord(group_id=gid, group_name=name, section="sequencing",
                          data_type="scRNA-seq", detected_count=1,
                          declared_count=3, method="obs_column:sample",
                          evidence="single value", conflicts=True)
        for gid, name in (("g1", "Con"), ("g2", "DKO"))
    ]
    samples = bio_mod._gate_samples_per_group_by_modality(state)
    assert samples == {"scRNA-seq": {"Con": 1, "DKO": 1}}
    report = run_gates(["DKO significantly increases Mt1 (p = 0.004)."],
                       score_definitions=[], scoring_plans=[],
                       samples_per_group_by_modality=samples)
    assert [f.gate for f in report.advisories] == ["unidentifiable-design"]
    assert [f.gate for f in report.blocking] == [
        "unreplicated-single-cell-inference"
    ]
