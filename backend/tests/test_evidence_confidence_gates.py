"""Checks that keep a reported confidence tied to the evidence behind it.

Each test here corresponds to a way a real run overstated its own certainty
without any single step being wrong:

* a resampling q-value quoted at its arithmetic floor, shared verbatim by eight
  unrelated panels, and read as though it ordered them;
* two panels sharing most of their members, reported side by side as though
  their agreement were corroboration;
* a mechanism graded `strong` on instruments that existed only inside that run.

None of these is detectable from a single number. The platform records the
structure as PI review observations; it does not change a scientific grade or
decide acceptance itself.
"""
from types import SimpleNamespace

from backend.services.analysis_plan import panel_overlap
from backend.services.statistical_gates import (
    check_panel_redundancy,
    check_resampling_resolution,
    run_gates,
)


def _score(name, method="gseapy.prerank", genes=(), **params):
    return SimpleNamespace(
        score_name=name, method=method, gene_list=list(genes),
        parameters=dict(params), plan_id="p", power={},
    )


# --------------------------------------------------------------- containment

def test_containment_sees_a_subset_that_jaccard_misses():
    """The case the check exists for: small panel wholly inside a large one."""
    small = ["A", "B", "C", "D"]
    large = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    overlap = panel_overlap(small, large)
    assert overlap["containment"] == 1.0
    assert overlap["jaccard"] < 0.4, (
        "Jaccard is low precisely because the large panel has other members, "
        "which is why it is the wrong statistic for redundancy"
    )
    assert overlap["n_shared"] == 4


def test_overlap_is_symmetric_and_case_insensitive():
    assert panel_overlap(["a", "B"], ["A", "b"])["containment"] == 1.0
    assert (panel_overlap(["A", "B", "C"], ["B", "C", "D"])["containment"]
            == panel_overlap(["B", "C", "D"], ["A", "B", "C"])["containment"])


def test_disjoint_panels_have_no_overlap():
    overlap = panel_overlap(["A", "B"], ["C", "D"])
    assert overlap["containment"] == 0.0
    assert overlap["shared"] == []


# ------------------------------------------------------- panel redundancy

def test_near_duplicate_panels_are_disclosed():
    shared = [f"G{i}" for i in range(13)]
    a = _score("panel_a", genes=shared + ["X1", "X2", "X3", "X4"])
    b = _score("panel_b", genes=shared + ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"])
    out = check_panel_redundancy([a, b], [])
    assert len(out) == 1
    assert out[0].severity == "advise", "overlap is disclosed, never refused"
    assert "panel_a" in out[0].detail and "panel_b" in out[0].detail
    assert "13" in out[0].detail


def test_panels_with_distinct_content_are_left_alone():
    a = _score("panel_a", genes=[f"A{i}" for i in range(12)])
    b = _score("panel_b", genes=[f"B{i}" for i in range(12)])
    assert check_panel_redundancy([a, b], []) == []


def test_a_single_panel_cannot_be_redundant():
    assert check_panel_redundancy([_score("only", genes=[f"G{i}" for i in range(12)])], []) == []


def test_redundancy_travels_as_an_advisory_through_run_gates():
    shared = [f"G{i}" for i in range(15)]
    report = run_gates(
        ["both panels moved in the same direction"],
        score_definitions=[_score("a", genes=shared), _score("b", genes=shared + ["Z"])],
    )
    assert any(f.gate == "overlapping-panels" for f in report.advisories)
    assert not any(f.gate == "overlapping-panels" for f in report.blocking), (
        "overlap is disclosed, never refused: legitimate designs test a broad set "
        "and a component of it together"
    )
    assert "share" in report.advisory_message()


# --------------------------------------------------- resampling resolution

def test_a_p_value_at_the_floor_is_recorded_for_pi_review():
    """500 resamples put the floor at 1/501; a value there is a bound.

    Quoted rounded, as it always is in prose: 1/501 reads "0.002".
    """
    score = _score("ranked_panel", n_resamples=500)
    out = check_resampling_resolution(
        ["The GSEA result for ranked_panel was strongly enriched, FDR = 0.002."],
        [score],
    )
    assert len(out) == 1
    assert out[0].severity == "advise"
    assert out[0].rule_id == "resampling-p-has-a-resolution-floor"
    assert "0.002" in out[0].detail


def test_the_next_attainable_value_up_has_no_resolution_observation():
    """The rounding tolerance must not reach a genuinely larger p-value."""
    score = _score("ranked_panel", n_resamples=500)
    out = check_resampling_resolution(
        ["The GSEA result for ranked_panel gave FDR = 0.004."], [score])
    assert out == []


def test_the_same_value_reported_as_a_bound_is_accepted():
    """Reporting the floor as an inequality is the correct reporting."""
    score = _score("ranked_panel", n_resamples=500)
    out = check_resampling_resolution(
        ["The GSEA result for ranked_panel reached FDR < 0.002."], [score])
    assert out == []


def test_an_analytic_p_value_is_not_subject_to_a_resampling_floor():
    """A model-fit adjusted p-value may legitimately be arbitrarily small."""
    score = _score("ranked_panel", n_resamples=500)
    out = check_resampling_resolution(
        ["The differential expression test gave padj = 2.2e-43 for this feature."],
        [score],
    )
    assert out == []


def test_an_undeclared_resample_count_advises_but_does_not_block():
    out = check_resampling_resolution(
        ["ranked_panel was enriched."], [_score("ranked_panel")])
    assert len(out) == 1
    assert out[0].severity == "advise"
    assert "resample count" in out[0].detail


def test_one_value_shared_across_several_results_is_flagged():
    out = check_resampling_resolution(
        [
            "panel one reached FDR = 0.00617.",
            "panel two reached FDR = 0.00617.",
            "panel three reached FDR = 0.00617.",
        ],
        [],
    )
    assert len(out) == 1
    assert out[0].gate == "adjusted-p-tied-across-results"
    assert out[0].severity == "advise"


def test_a_repeated_conventional_cutoff_is_not_a_resolution_floor():
    """Restating the alpha level is not the same as sharing a floor."""
    out = check_resampling_resolution(
        ["A was significant at FDR = 0.05.",
         "B was significant at FDR = 0.05.",
         "C was significant at FDR = 0.05."],
        [],
    )
    assert out == []


def test_module_scores_do_not_trigger_the_resampling_checks():
    """Only resampling-based methods have a resolution floor."""
    module = _score("module_panel", method="scanpy.tl.score_genes")
    assert check_resampling_resolution(["module_panel was higher."], [module]) == []


# -------------------------------------------------- evidence-grade ceiling

def _program(strength, scores):
    return {
        "programs": [{
            "label": "candidate program",
            "triggering_observations": ["an observation"],
            "evidence_strength": strength,
            "specificity": "high",
            "interpretation": "a candidate interpretation",
            "evidence_execution_ids": ["exec-1"],
            "supporting_scores": list(scores),
        }]
    }


def test_analysis_specific_instruments_are_observed_without_changing_grade():
    from backend.agents.bioinformatics.mechanism_validation import (
        validate_mechanism_program_payload,
    )

    progs, err = validate_mechanism_program_payload(
        _program("strong", ["hand_built"]),
        score_provenance={"hand_built": {"kind": "bespoke"}},
    )
    assert err is None
    assert progs[0].evidence_strength == "strong"
    assert any("cannot be retrieved independently" in m
               for m in progs[0].missing_evidence)


def test_a_versioned_source_leaves_the_grade_alone():
    from backend.agents.bioinformatics.mechanism_validation import (
        validate_mechanism_program_payload,
    )

    progs, err = validate_mechanism_program_payload(
        _program("strong", ["from_a_database"]),
        score_provenance={"from_a_database": {"kind": "database"}},
    )
    assert err is None
    assert progs[0].evidence_strength == "strong"


def test_one_reproducible_source_avoids_the_analysis_specific_observation():
    from backend.agents.bioinformatics.mechanism_validation import (
        validate_mechanism_program_payload,
    )

    progs, _ = validate_mechanism_program_payload(
        _program("strong", ["hand_built", "from_a_database"]),
        score_provenance={"hand_built": {"kind": "bespoke"},
                          "from_a_database": {"kind": "database"}},
    )
    assert progs[0].evidence_strength == "strong"
    assert not any("cannot be retrieved independently" in item
                   for item in progs[0].missing_evidence)


def test_an_unresolvable_citation_does_not_lower_the_grade():
    """A bookkeeping gap is not evidence about reproducibility."""
    from backend.agents.bioinformatics.mechanism_validation import (
        validate_mechanism_program_payload,
    )

    progs, _ = validate_mechanism_program_payload(
        _program("strong", ["never_registered"]),
        score_provenance={"something_else": {"kind": "bespoke"}},
    )
    assert progs[0].evidence_strength == "strong"


def test_no_provenance_map_leaves_the_grade_alone():
    """Existing callers must keep their behaviour exactly."""
    from backend.agents.bioinformatics.mechanism_validation import (
        validate_mechanism_program_payload,
    )

    progs, _ = validate_mechanism_program_payload(_program("strong", ["hand_built"]))
    assert progs[0].evidence_strength == "strong"


def test_a_weak_declared_grade_is_left_alone():
    from backend.agents.bioinformatics.mechanism_validation import (
        validate_mechanism_program_payload,
    )

    progs, _ = validate_mechanism_program_payload(
        _program("weak", ["hand_built"]),
        score_provenance={"hand_built": {"kind": "bespoke"}},
    )
    assert progs[0].evidence_strength == "weak"
