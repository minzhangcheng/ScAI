"""Schema tests for cross-modal mechanism audit models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    CrossModalEvidenceItem,
    CrossModalMechanismCandidate,
    CrossModalSynthesis,
    LiteratureMechanismSignal,
    LiteratureReport,
    RoundResult,
)


def test_evidence_item_defaults():
    item = CrossModalEvidenceItem(source="biochemistry", evidence="Cu -80%", strength="strong")
    assert item.limitation == ""


def test_candidate_round_trips():
    cand = CrossModalMechanismCandidate(
        theme="iron-associated oxidative injury",
        mechanism_statement="Iron accumulation drives lipid peroxidation",
        supporting_evidence=[
            CrossModalEvidenceItem(source="biochemistry", evidence="Fe 2.2x", strength="moderate"),
            CrossModalEvidenceItem(source="bioinformatics", evidence="Hmox1 up", strength="weak"),
        ],
        evidence_consistency="medium",
        confidence="low",
        novelty_relative_to_current_direction="high",
        relation_to_current_direction="uncovered",
        missing_evidence=["4-HNE not measured"],
        report_recommendation="future_direction",
    )
    dumped = cand.model_dump()
    assert dumped["relation_to_current_direction"] == "uncovered"
    assert len(dumped["supporting_evidence"]) == 2


def test_synthesis_defaults_empty():
    cms = CrossModalSynthesis(matrix_markdown="", candidates=[], excluded_or_low_support_themes=[])
    assert cms.candidates == []


def test_round_result_cross_modal_optional():
    rr = RoundResult(round_number=1)
    assert rr.cross_modal_synthesis is None


def test_literature_report_mechanism_signals_optional():
    lr = LiteratureReport(topic="t", report="r")
    assert lr.mechanism_signals is None


def test_literature_mechanism_signal():
    sig = LiteratureMechanismSignal(
        mechanism="ferroptosis",
        supporting_pmids=["38963329"],
        evidence_type="correlational",
        relevance_to_current_study="links Cu deficiency to ferroptosis",
        observed_data_links=["iron accumulation"],
        limitations=["no direct lipid peroxidation assay"],
    )
    assert sig.evidence_type == "correlational"
