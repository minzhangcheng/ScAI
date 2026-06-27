"""Tests for Direction Evidence Models (Task 1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from datetime import datetime
from pydantic import ValidationError

from pydantic import BaseModel
from backend.models.schemas import (
    ExplorationReport,
    PubMedArticle,
    PubMedEvidenceItem,
    DirectionSearchQuery,
    DirectionLiteratureEvidence,
    ResearchDirectionCandidate,
    DirectionSelectionDecision,
    SpeciesTissueQualification,
)


class _ValidateQualification(BaseModel):
    q: SpeciesTissueQualification


class TestSpeciesTissueQualification:
    def test_valid_literals(self):
        assert _ValidateQualification(q="").q == ""
        assert _ValidateQualification(q="matched_species_tissue_available").q == "matched_species_tissue_available"
        assert _ValidateQualification(q="no_matched_species_tissue_evidence").q == "no_matched_species_tissue_evidence"
        assert _ValidateQualification(q="cross_tissue_only").q == "cross_tissue_only"
        assert _ValidateQualification(q="cross_species_only").q == "cross_species_only"

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            _ValidateQualification(q="invalid_value")


class TestPubMedArticle:
    def test_minimal_construction(self):
        article = PubMedArticle(
            pubmed_id="12345",
            title="Test Article",
            journal="Test Journal",
            publication_date="2024-01-01",
        )
        assert article.pubmed_id == "12345"
        assert article.title == "Test Article"
        assert article.journal == "Test Journal"
        assert article.publication_date == "2024-01-01"
        assert article.authors == []
        assert article.abstract == ""
        assert article.url == ""

    def test_full_construction(self):
        article = PubMedArticle(
            pubmed_id="67890",
            title="Full Article",
            authors=["Author A", "Author B"],
            journal="Good Journal",
            publication_date="2024-06-15",
            abstract="This is an abstract.",
            url="https://pubmed.ncbi.nlm.nih.gov/67890/",
        )
        assert len(article.authors) == 2
        assert article.abstract == "This is an abstract."
        assert article.url.startswith("https://")

    def test_json_serializable(self):
        article = PubMedArticle(
            pubmed_id="12345",
            title="Test",
            journal="J",
            publication_date="2024-01-01",
        )
        d = article.model_dump(mode="json")
        json.dumps(d)
        assert d["pubmed_id"] == "12345"


class TestPubMedEvidenceItem:
    def test_required_fields_only(self):
        item = PubMedEvidenceItem(
            pmid="12345",
            title="Test Study",
            relevance="high",
            supports="direct",
            why_it_matters="Key evidence for mechanism",
        )
        assert item.pmid == "12345"
        assert item.title == "Test Study"
        assert item.relevance == "high"
        assert item.supports == "direct"
        assert item.why_it_matters == "Key evidence for mechanism"
        assert item.year is None
        assert item.url is None
        assert item.source == "new_search"
        assert item.species_context == ""
        assert item.tissue_context == ""

    def test_all_fields(self):
        item = PubMedEvidenceItem(
            pmid="67890",
            title="Full Study",
            year="2024",
            url="https://pubmed.ncbi.nlm.nih.gov/67890/",
            relevance="medium",
            supports="analogous",
            source="session_cache",
            species_context="Mus musculus",
            tissue_context="liver",
            why_it_matters="Analogous mechanism in mice",
        )
        assert item.year == "2024"
        assert item.source == "session_cache"
        assert item.species_context == "Mus musculus"

    def test_invalid_relevance_raises(self):
        with pytest.raises(ValidationError):
            PubMedEvidenceItem(
                pmid="1", title="T", relevance="invalid", supports="direct",
                why_it_matters="test",
            )

    def test_invalid_supports_raises(self):
        with pytest.raises(ValidationError):
            PubMedEvidenceItem(
                pmid="1", title="T", relevance="high", supports="invalid",
                why_it_matters="test",
            )

    def test_json_serializable(self):
        item = PubMedEvidenceItem(
            pmid="12345",
            title="Test",
            relevance="low",
            supports="methodological",
            why_it_matters="Methods paper",
        )
        d = item.model_dump(mode="json")
        json.dumps(d)
        assert d["pmid"] == "12345"
        assert d["relevance"] == "low"


class TestDirectionSearchQuery:
    def test_required_fields_only(self):
        q = DirectionSearchQuery(category="direct_prior_work", query="test query")
        assert q.category == "direct_prior_work"
        assert q.query == "test query"
        assert q.result_count == 0
        assert q.relevant_count == 0
        assert q.degraded is False
        assert q.notes == ""

    def test_all_fields(self):
        q = DirectionSearchQuery(
            category="pathway_focused",
            query="TGF-beta signaling",
            result_count=50,
            relevant_count=10,
            degraded=True,
            notes="Some results not directly relevant",
        )
        assert q.result_count == 50
        assert q.relevant_count == 10
        assert q.degraded is True
        assert "not directly relevant" in q.notes

    def test_all_categories_accepted(self):
        for cat in [
            "direct_prior_work",
            "analogous_mechanism",
            "contradictory_expectedness",
            "methodological_precedent",
            "gene_model_relaxed",
            "cell_type_focused",
            "pathway_focused",
            "cross_species_or_tissue",
        ]:
            q = DirectionSearchQuery(category=cat, query="test")
            assert q.category == cat

    def test_invalid_category_raises(self):
        with pytest.raises(ValidationError):
            DirectionSearchQuery(category="invalid_cat", query="test")

    def test_json_serializable(self):
        q = DirectionSearchQuery(category="direct_prior_work", query="test")
        d = q.model_dump(mode="json")
        json.dumps(d)
        assert d["category"] == "direct_prior_work"
        assert d["query"] == "test"


class TestDirectionLiteratureEvidence:
    def test_minimal_construction(self):
        ev = DirectionLiteratureEvidence(direction_id="dir_1", direction="Test Direction")
        assert ev.direction_id == "dir_1"
        assert ev.direction == "Test Direction"
        assert ev.status == "partial"
        assert ev.search_tier == "standard"
        assert ev.literature_novelty == "unknown"
        assert ev.confidence == "low"
        assert ev.direct_evidence == []
        assert ev.analogous_evidence == []
        assert ev.contradictory_evidence == []
        assert ev.methodological_precedent == []
        assert ev.queries == []
        assert ev.gap_assessment == ""
        assert ev.summary == ""
        assert ev.limitations == []
        assert ev.timeout_seconds is None
        assert ev.timed_out is False
        assert ev.species_tissue_qualification == ""

    def test_with_evidence_items(self):
        item = PubMedEvidenceItem(
            pmid="12345",
            title="Relevant Study",
            relevance="high",
            supports="direct",
            why_it_matters="Direct evidence",
        )
        ev = DirectionLiteratureEvidence(
            direction_id="dir_1",
            direction="Test Direction",
            status="completed",
            search_tier="deep",
            literature_novelty="high",
            confidence="high",
            direct_evidence=[item],
            species_tissue_qualification="matched_species_tissue_available",
            gap_assessment="No major gaps",
            summary="Strong evidence",
            limitations=["Small sample size"],
            timeout_seconds=30.5,
            timed_out=False,
        )
        assert ev.status == "completed"
        assert ev.search_tier == "deep"
        assert ev.literature_novelty == "high"
        assert ev.confidence == "high"
        assert len(ev.direct_evidence) == 1
        assert ev.direct_evidence[0].pmid == "12345"
        assert ev.species_tissue_qualification == "matched_species_tissue_available"
        assert ev.limitations == ["Small sample size"]
        assert ev.timeout_seconds == 30.5
        assert ev.timed_out is False

    def test_negative_validation_axes(self):
        ev = DirectionLiteratureEvidence(
            direction_id="dir_1",
            direction="Test",
            negative_validation_axes={"species_matched": True, "tissue_matched": False},
        )
        assert ev.negative_validation_axes["species_matched"] is True
        assert ev.negative_validation_axes["tissue_matched"] is False

    def test_all_evidence_lists(self):
        item = PubMedEvidenceItem(
            pmid="1", title="A", relevance="high", supports="direct",
            why_it_matters="E1",
        )
        ev = DirectionLiteratureEvidence(
            direction_id="d1",
            direction="D",
            direct_evidence=[item],
            analogous_evidence=[PubMedEvidenceItem(
                pmid="2", title="B", relevance="medium", supports="analogous",
                why_it_matters="E2",
            )],
            contradictory_evidence=[PubMedEvidenceItem(
                pmid="3", title="C", relevance="low", supports="contradictory",
                why_it_matters="E3",
            )],
            methodological_precedent=[PubMedEvidenceItem(
                pmid="4", title="D", relevance="medium", supports="methodological",
                why_it_matters="E4",
            )],
        )
        assert len(ev.direct_evidence) == 1
        assert len(ev.analogous_evidence) == 1
        assert len(ev.contradictory_evidence) == 1
        assert len(ev.methodological_precedent) == 1

    def test_with_queries(self):
        q = DirectionSearchQuery(category="direct_prior_work", query="test query")
        ev = DirectionLiteratureEvidence(
            direction_id="d1",
            direction="D",
            queries=[q],
        )
        assert len(ev.queries) == 1
        assert ev.queries[0].query == "test query"

    def test_count_fields(self):
        ev = DirectionLiteratureEvidence(
            direction_id="d1",
            direction="D",
            direct_evidence_count=5,
            analogous_evidence_count=3,
            contradictory_evidence_count=1,
            methodological_precedent_count=2,
        )
        assert ev.direct_evidence_count == 5
        assert ev.analogous_evidence_count == 3
        assert ev.contradictory_evidence_count == 1
        assert ev.methodological_precedent_count == 2

    def test_json_serializable(self):
        ev = DirectionLiteratureEvidence(direction_id="dir_1", direction="Test")
        d = ev.model_dump(mode="json")
        json.dumps(d)
        assert d["direction_id"] == "dir_1"
        assert d["direction"] == "Test"


class TestResearchDirectionCandidate:
    def test_required_fields_only(self):
        cand = ResearchDirectionCandidate(direction_id="dir_1", direction="Test Direction")
        assert cand.direction_id == "dir_1"
        assert cand.direction == "Test Direction"
        assert cand.rationale == ""
        assert cand.risk_flags == []
        assert cand.prerequisite_dependencies == []
        assert cand.evidence_ref is None

    def test_all_fields(self):
        cand = ResearchDirectionCandidate(
            direction_id="dir_1",
            direction="Test Direction",
            rationale="Key pathway alteration",
            data_support_summary="DEG analysis shows significance",
            literature_support_summary="Several papers support this",
            data_reliability="high",
            literature_novelty="medium",
            biology_expectedness="expected",
            data_novelty="high",
            mechanistic_importance="high",
            functional_impact="medium",
            data_sufficiency="sufficient",
            methodological_gap="none",
            overall_priority="high",
            risk_flags=["small sample size", "batch effect"],
            prerequisite_dependencies=["Must validate marker genes first"],
            evidence_ref="dir_lit_1",
        )
        assert cand.overall_priority == "high"
        assert len(cand.risk_flags) == 2
        assert cand.evidence_ref == "dir_lit_1"

    def test_json_serializable(self):
        cand = ResearchDirectionCandidate(direction_id="dir_1", direction="Test")
        d = cand.model_dump(mode="json")
        json.dumps(d)
        assert d["direction_id"] == "dir_1"
        assert d["direction"] == "Test"


def test_direction_literature_evidence_enhanced_fields_round_trip():
    evidence = DirectionLiteratureEvidence(
        direction_id="dir_001",
        direction="Resolve copper paradox",
        support_level="weak",
        contradiction_level="moderate",
        prior_art_level="low",
        novelty_status="genuine_novelty",
        similar_published_directions=[{"pmid": "123", "title": "Related mechanism"}],
        knowledge_gap_summary="No direct prior work in matched liver DKO context.",
        fallback_tier="tier3_broad",
        scheduler_limitation="",
    )

    dumped = evidence.model_dump()

    assert dumped["support_level"] == "weak"
    assert dumped["contradiction_level"] == "moderate"
    assert dumped["prior_art_level"] == "low"
    assert dumped["novelty_status"] == "genuine_novelty"
    assert dumped["similar_published_directions"] == [{"pmid": "123", "title": "Related mechanism"}]
    assert dumped["knowledge_gap_summary"] == "No direct prior work in matched liver DKO context."
    assert dumped["fallback_tier"] == "tier3_broad"


class TestDirectionSelectionDecision:
    def test_empty_lists(self):
        decision = DirectionSelectionDecision()
        assert decision.selected_directions == []
        assert decision.rejected_directions == []
        assert decision.selection_rationale == ""

    def test_with_directions(self):
        decision = DirectionSelectionDecision(
            selected_directions=[{"direction": "A", "notes": "good"}, {"direction": "B", "notes": "also good"}],
            rejected_directions=[{"direction": "C", "notes": "weak evidence"}],
            selection_rationale="User selected A and B based on data support",
        )
        assert len(decision.selected_directions) == 2
        assert len(decision.rejected_directions) == 1
        assert decision.selected_directions[0]["direction"] == "A"
        assert decision.selection_rationale == "User selected A and B based on data support"

    def test_json_serializable(self):
        decision = DirectionSelectionDecision()
        d = decision.model_dump(mode="json")
        json.dumps(d)
        assert "selected_directions" in d
        assert "selection_rationale" in d


class TestExplorationReportExtension:
    def test_legacy_research_direction_candidates_still_works(self):
        """Legacy sessions use List[Dict[str, Any]] — must still load."""
        report = ExplorationReport(
            research_direction_candidates=[
                {"direction": "Legacy Dir", "rationale": "Old", "expected_innovation": "", "expected_novelty": "", "data_reliability": ""}
            ]
        )
        assert len(report.research_direction_candidates) == 1
        assert report.research_direction_candidates[0]["direction"] == "Legacy Dir"

    def test_new_fields_default(self):
        report = ExplorationReport()
        assert report.raw_direction_candidates == []
        assert report.direction_literature_evidence == []
        assert report.direction_dependency_graph == {}
        assert report.direction_selection_summary == {}

    def test_can_set_new_fields(self):
        report = ExplorationReport(
            raw_direction_candidates=[
                {"direction_id": "d1", "direction": "D1", "rationale": "test"}
            ],
            direction_literature_evidence=[
                {"direction_id": "d1", "direction": "D1"}
            ],
            direction_dependency_graph={"d1": ["d2"]},
            direction_selection_summary={
                "selected": ["d1"],
                "rationale": "Strong evidence"
            },
        )
        assert len(report.raw_direction_candidates) == 1
        assert len(report.direction_literature_evidence) == 1
        assert report.direction_dependency_graph["d1"] == ["d2"]
        assert report.direction_selection_summary["selected"] == ["d1"]

    def test_legacy_loading_with_minimal_dict(self):
        """Test that legacy format with just a direction field still loads."""
        report = ExplorationReport(
            research_direction_candidates=[{"direction": "Legacy"}]
        )
        assert len(report.research_direction_candidates) == 1

    def test_full_report_json_serializable(self):
        report = ExplorationReport(
            research_direction_candidates=[
                {"direction": "L", "rationale": "", "expected_innovation": "", "expected_novelty": "", "data_reliability": ""}
            ],
            raw_direction_candidates=[
                {"direction_id": "d1", "direction": "D1"}
            ],
            direction_literature_evidence=[
                {"direction_id": "d1", "direction": "D1"}
            ],
        )
        d = report.model_dump(mode="json")
        json.dumps(d)
        assert len(d["research_direction_candidates"]) == 1
        assert len(d["raw_direction_candidates"]) == 1
        assert len(d["direction_literature_evidence"]) == 1


def test_direction_search_query_accepts_new_evidence_categories():
    categories = ["challenge_contradiction", "prior_art", "tier3_support", "tier3_challenge"]
    for category in categories:
        q = DirectionSearchQuery(category=category, query="test")
        assert q.category == category
