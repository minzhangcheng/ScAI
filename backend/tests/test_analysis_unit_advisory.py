"""Analysis-unit advisories use the finding's own modality sample design.

The measured mixed-modality session has replicated bulk RNA-seq and one
single-cell library per condition. Bulk replication must not be lowered by the
single-cell arm, and the single-cell arm must not borrow the bulk sample size.
"""


import backend.agents.bioinformatics_agent as bio_mod
from backend.services.statistical_gates import run_gates

INFERENTIAL = "DKO significantly increases Mt1 (p = 0.004)."
BULK = ("Bulk RNA-seq identifies 564 significant DEGs including Slc7a11 "
        "upregulated and the Mup family downregulated")


def _multimodal_state():
    """Verified bulk n=3/group and single-cell n=1/group."""
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
            group_id=gid, group_name=name, section="sequencing",
            data_type="bulk RNA-seq", detected_count=3, declared_count=3,
            method="matrix_columns",
            evidence=f"three verified columns for {name}", conflicts=False)
        for gid, name in (("g1", "Con"), ("g2", "DKO"))
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

def test_run_gates_does_not_refuse_at_n1():
    report = run_gates([INFERENTIAL], score_definitions=[], scoring_plans=[],
                       samples_per_group={"Con": 1, "DKO": 1})
    assert report.blocking == []
    assert report.passed is True
    assert [f.gate for f in report.advisories] == ["unidentifiable-design"]


# ----------------------------------------------------- the measured session

def test_measured_multimodal_session_is_not_refused():
    """The bulk finding that was refused at n=3 per group."""
    state = _multimodal_state()
    samples = bio_mod._gate_samples_per_group_by_modality(state)
    assert samples == {
        "bulk RNA-seq": {"Con": 3, "DKO": 3},
        "scRNA-seq": {"Con": 1, "DKO": 1},
    }

    report = run_gates(
        [{"text": BULK, "modality": "bulk_rna_seq"}],
        score_definitions=[], scoring_plans=[],
                       samples_per_group_by_modality=samples)
    assert report.passed is True, (
        "a bulk contrast at n=3 per group must not be refused by a count "
        "taken from another modality"
    )
    assert not any(f.gate == "unidentifiable-design" for f in report.findings)


def test_single_cell_finding_at_n1_surfaces_both_observations():
    samples = bio_mod._gate_samples_per_group_by_modality(_multimodal_state())
    report = run_gates(
        [{"text": "condition effect is significant (p = 0.004)",
          "modality": "single_cell"}],
        score_definitions=[], scoring_plans=[],
        samples_per_group_by_modality=samples,
    )
    assert report.blocking == []
    assert [f.gate for f in report.advisories] == [
        "unreplicated-single-cell-inference", "unidentifiable-design",
    ]


def test_finding_text_cannot_override_structured_single_cell_modality():
    samples = bio_mod._gate_samples_per_group_by_modality(_multimodal_state())
    report = run_gates(
        [{"text": "Bulk RNA-seq condition effect is significant (p = 0.004)",
          "modality": "single_cell"}],
        score_definitions=[], scoring_plans=[],
        samples_per_group_by_modality=samples,
    )
    assert "unreplicated-single-cell-inference" in [
        f.gate for f in report.advisories
    ]
    assert report.blocking == []


def test_multimodal_inferential_finding_must_name_its_source():
    samples = bio_mod._gate_samples_per_group_by_modality(_multimodal_state())
    report = run_gates(
        [{"text": "The condition effect is significant (p = 0.004)"}],
        score_definitions=[], scoring_plans=[],
        samples_per_group_by_modality=samples,
    )
    assert "modality-unresolved" in [f.gate for f in report.advisories]
    assert report.blocking == []


# ------------------------------------------------------------- the wording

# -------------------------------------------------------- corpus agreement
