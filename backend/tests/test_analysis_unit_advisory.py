"""check_analysis_unit advises; it does not refuse.

Measured on session 3d675631-8c5f-4a6a-ad66-ef4c8996bb44 (scai-2, 0.48.1). The
session holds two modalities: bulk RNA-seq with three replicates per group
(merged_counts.csv, six sample columns) and single-nucleus RNA-seq with one
library per group. data_profiling measured both correctly and wrote a
SampleCountRecord for each.

_gate_samples_per_group then collapsed them. The bulk record carries
group_id=None -- one matrix spanning both groups cannot belong to either -- so
it is skipped outright, and same-group modalities are reduced with min(). The
gates were handed {Con: 1, DKO: 1}.

check_analysis_unit refused report_findings on this finding:

    Bulk RNA-seq identifies 564 significant DEGs including Slc7a11, Gsta13,
    Nupr1 upregulated and Mup family downregulated

which rests on n=3 per group and is perfectly estimable. The refusal was not
actionable: the only edit that satisfies it -- restating a bulk contrast as
descriptive -- is wrong, so the executor did not make it. It reworded the
single-cell findings instead (its own words: "I understand the statistical
validity concern. The scRNA-seq data has n=1 sample per condition, so condition
effects cannot be statistically estimated"), resubmitted, and was refused again
because the gate scans findings one at a time and the bulk one had not changed.
The rejection ceiling then force-released, marking the round's evidence
degraded.

The rule itself is sound and stays in the corpus: with one library per
condition a condition-level claim is not estimable, and that is arithmetic, not
convention. What changes is the channel. n=1 single-cell work is ordinary and
publishable as descriptive analysis, so a note the model cannot suppress is the
proportionate instrument; a refusal is not, because the gate cannot tell which
modality a finding rests on and a wrong refusal has no satisfiable response.

The collapse in _gate_samples_per_group is NOT fixed here -- widening it means
changing the run_gates parameter shape. Until then the advisory's numbers are
still the per-group minimum across modalities, which is why its wording must
not assert unconditional inestimability.
"""
import re

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.services.method_rules import ENFORCEMENT_ADVISORY, get_rule
from backend.services.statistical_gates import check_analysis_unit, run_gates

INFERENTIAL = "DKO significantly increases Mt1 (p = 0.004)."
BULK = ("Bulk RNA-seq identifies 564 significant DEGs including Slc7a11 "
        "upregulated and the Mup family downregulated")


def _multimodal_state():
    """The measured session: bulk n=3/group ungrouped, single-cell n=1/group."""
    from backend.models.schemas import SampleCountRecord

    state = _live_state(
        treatment_groups=[_group("g1", "Con", 3), _group("g2", "DKO", 3)],
        current_round=1,
    )
    state.detected_sample_counts = [
        SampleCountRecord(
            group_id=gid, group_name=name, section="sequencing",
            data_type="scRNA-seq", detected_count=1, declared_count=3,
            method="barcode_suffix", evidence="single library", conflicts=True)
        for gid, name in (("g1", "Con"), ("g2", "DKO"))
    ] + [
        SampleCountRecord(
            group_id=None, group_name="", section="sequencing",
            data_type="bulk RNA-seq", detected_count=6, declared_count=None,
            method="matrix_columns",
            evidence="six sample columns, three per group", conflicts=False),
    ]
    return state


def _live_state(**kw):
    from backend.models.schemas import AnalysisState

    state = AnalysisState(session_id="t", user_id="u", research_purpose="t")
    for key, value in kw.items():
        setattr(state, key, value)
    return state


def _group(group_id, name, sample_count):
    from backend.models.schemas import TreatmentGroup

    return TreatmentGroup(id=group_id, name=name, sample_count=sample_count)


# ------------------------------------------------------------- the channel

def test_analysis_unit_finding_is_advisory():
    out = check_analysis_unit([INFERENTIAL], {"Con": 1, "DKO": 1})
    assert len(out) == 1
    assert out[0].rule_id == "one-sample-per-condition-is-unidentifiable"
    assert out[0].severity == "advise", (
        "the check still fires; it must ride the advisory channel"
    )


def test_run_gates_does_not_refuse_at_n1():
    report = run_gates([INFERENTIAL], score_definitions=[], scoring_plans=[],
                       samples_per_group={"Con": 1, "DKO": 1})
    assert report.blocking == []
    assert report.passed is True
    assert [f.gate for f in report.advisories] == ["unidentifiable-design"]


def test_advisory_message_still_reaches_the_model():
    """An advisory the model cannot suppress is the whole point of the change."""
    report = run_gates([INFERENTIAL], score_definitions=[], scoring_plans=[],
                       samples_per_group={"Con": 1, "DKO": 1})
    assert "unidentifiable-design" in report.advisory_message()


# ----------------------------------------------------- the measured session

def test_measured_multimodal_session_is_not_refused():
    """The bulk finding that was refused at n=3 per group."""
    state = _multimodal_state()
    samples = bio_mod._gate_samples_per_group(state)
    # Unchanged on purpose: the collapse is a separate defect. The bulk record
    # is group_id=None so it is skipped, and the declaration loses to the
    # single-cell measurement.
    assert samples == {"Con": 1, "DKO": 1}

    report = run_gates([BULK], score_definitions=[], scoring_plans=[],
                       samples_per_group=samples)
    assert report.passed is True, (
        "a bulk contrast at n=3 per group must not be refused by a count "
        "taken from another modality"
    )


def test_descriptive_wording_still_produces_nothing():
    """Correctly labelled contrasts were never the target, gate or advisory."""
    findings = ["this contrast is descriptive: condition is confounded with "
                "specimen, so no condition-level effect is estimable"]
    assert check_analysis_unit(findings, {"Con": 1, "DKO": 1}) == []


# ------------------------------------------------------------- the wording

def test_detail_does_not_assert_unconditional_inestimability():
    """While the count is a cross-modality minimum, the note must not state as
    fact that no effect is estimable -- in the measured session that is false
    for the modality the refused finding actually rested on."""
    detail = check_analysis_unit([INFERENTIAL], {"Con": 1, "DKO": 1})[0].detail
    assert "not estimable at all" not in detail
    lowered = detail.lower()
    assert "modalit" in lowered, (
        "the note must say the count is the lowest across modalities"
    )
    assert "descriptive" in lowered, "the actionable instruction must survive"


def test_detail_names_the_counts_it_was_given():
    detail = check_analysis_unit([INFERENTIAL], {"Con": 1, "DKO": 0})[0].detail
    assert "Con" in detail and "DKO" in detail


# -------------------------------------------------------- corpus agreement

def test_corpus_marks_the_rule_advisory():
    """enforcement=gate promises a refusal in code; this rule no longer is one.

    test_rule_reachability's gate parametrizations select on this field, so the
    corpus and the implementation cannot drift apart silently.
    """
    rule = get_rule("one-sample-per-condition-is-unidentifiable")
    assert rule is not None
    assert rule["enforcement"] == ENFORCEMENT_ADVISORY


def test_unsatisfiable_ceiling_no_longer_lists_the_rule():
    """The 1-rejection ceiling exists for gates a retry cannot satisfy. This one
    no longer reaches `blocking`, so listing it there is a dead branch."""
    src = inspect_source()
    m = re.search(r"_unsatisfiable\s*=\s*\{[^}]*\}", src)
    assert m, "could not locate the _unsatisfiable set"
    assert "cell-level-inference" in m.group(0), "the other entry must remain"
    assert "unidentifiable-design" not in m.group(0)


def inspect_source():
    import inspect

    return inspect.getsource(bio_mod)


@pytest.mark.parametrize("bad", ["liver", "mouse", "cancer", "copper"])
def test_new_wording_stays_domain_neutral(bad):
    detail = check_analysis_unit([INFERENTIAL], {"Con": 1, "DKO": 1})[0].detail
    assert not re.search(rf"\b{bad}\b", detail, re.I)
