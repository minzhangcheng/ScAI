from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models.schemas import ExplorationReport
from backend.services.report.variants import _render_direction_evidence_section


def test_direction_evidence_section_contains_complete_detail_fields():
    report = ExplorationReport(
        research_direction_candidates=[{
            "direction_id": "dir_001",
            "direction": "IFN network",
            "overall_priority": "high",
            "data_reliability": "high",
            "literature_novelty": "high",
            "biology_expectedness": "partially_expected",
            "data_novelty": "high",
            "mechanistic_importance": "high",
            "functional_impact": "high",
            "data_sufficiency": "partial",
            "methodological_gap": "minor",
            "prerequisite_dependencies": ["dir_000"],
            "evidence_ref": "dir_001",
        }],
        direction_literature_evidence=[{
            "direction_id": "dir_001",
            "direction": "IFN network",
            "status": "completed",
            "search_tier": "deep",
            "literature_novelty": "high",
            "confidence": "medium",
            "negative_validation_confidence": "medium",
            "species_tissue_qualification": "cross_tissue_only",
            "methodological_precedent_count": 1,
            "analogous_evidence": [{
                "pmid": "123", "title": "Trace metal innate immunity",
                "relevance": "medium", "supports": "analogous",
                "why_it_matters": "Analogous mechanism.",
            }],
            "queries": [{"category": "analogous_mechanism", "query": "trace metal immunity", "result_count": 5, "relevant_count": 1}],
            "limitations": ["No direct matched-species evidence."],
            "direct_evidence": [],
            "contradictory_evidence": [],
            "methodological_precedent": [],
            "direct_evidence_count": 0,
            "analogous_evidence_count": 1,
            "contradictory_evidence_count": 0,
        }],
    )
    html = _render_direction_evidence_section(report)
    for text in ["Literature-Gated Direction Evaluation", "IFN network",
                 "biology expectedness", "functional impact",
                 "negative validation", "cross_tissue_only",
                 "Cross-tissue evidence only", "PMID: 123",
                 "No direct matched-species evidence", "dir_000"]:
        assert text.lower() in html.lower(), f"Missing: {text}"
    assert "Literature-Gated Direction Evaluation" in html


def test_direction_evidence_html_marks_timed_out_zero_reference_search():
    report = ExplorationReport(
        research_direction_candidates=[{
            "direction_id": "dir_001",
            "direction": "Resolve copper redistribution",
            "rationale": "Copper data are contradictory.",
            "overall_priority": "high",
            "data_reliability": "medium",
            "literature_novelty": "unknown",
            "evidence_ref": "dir_001",
        }],
        direction_literature_evidence=[{
            "direction_id": "dir_001",
            "direction": "Resolve copper redistribution",
            "status": "partial",
            "timed_out": True,
            "timeout_seconds": 180,
            "direct_evidence_count": 0,
            "analogous_evidence_count": 0,
            "contradictory_evidence_count": 0,
            "methodological_precedent_count": 0,
            "direct_evidence": [],
            "analogous_evidence": [],
            "contradictory_evidence": [],
            "methodological_precedent": [],
            "limitations": ["Direction-level timeout exceeded"],
        }],
    )

    html = _render_direction_evidence_section(report)

    assert "Timed out" in html
    assert "Literature search timed out before references could be retained." in html
    assert "Direction-level timeout exceeded" in html


def test_direction_evidence_html_keeps_partial_references_after_timeout():
    report = ExplorationReport(
        research_direction_candidates=[{
            "direction_id": "dir_003",
            "direction": "Investigate metabolic collapse",
            "rationale": "Metabolic pathways are downregulated.",
            "overall_priority": "high",
            "data_reliability": "high",
            "literature_novelty": "medium",
            "evidence_ref": "dir_003",
        }],
        direction_literature_evidence=[{
            "direction_id": "dir_003",
            "direction": "Investigate metabolic collapse",
            "status": "partial",
            "timed_out": True,
            "timeout_seconds": 180,
            "direct_evidence_count": 1,
            "analogous_evidence_count": 0,
            "contradictory_evidence_count": 0,
            "methodological_precedent_count": 0,
            "direct_evidence": [{
                "pmid": "12345678",
                "title": "Mouse liver metabolism and copper stress",
                "relevance": "high",
                "supports": "direct",
                "why_it_matters": "Retained before timeout.",
            }],
            "analogous_evidence": [],
            "contradictory_evidence": [],
            "methodological_precedent": [],
            "limitations": ["Direction-level timeout exceeded"],
        }],
    )

    html = _render_direction_evidence_section(report)

    assert "Partial" in html
    assert "Literature evidence search only partially completed; listed references are partial." in html
    assert "PMID: 12345678" in html
