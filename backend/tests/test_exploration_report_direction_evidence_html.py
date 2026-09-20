from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models.schemas import ExplorationReport
from backend.services.report.variants import _render_direction_evidence_section
from backend.services.report.variants_parts.integration import (
    _render_integration_evidence_section,
)


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


def test_new_policy_timeout_renders_retained_inventory_and_execution_limitation():
    report = ExplorationReport(
        research_direction_candidates=[{
            "direction_id": "dir_new", "direction": "Retained direction",
            "evidence_ref": "dir_new",
        }],
        direction_literature_evidence=[{
            "direction_id": "dir_new", "direction": "Retained direction",
            "status": "partial", "timed_out": True,
            "policy_version": "direction_search_v1",
            "assessment_status": "unavailable",
            "article_inventory": [{
                "pmid": "9988", "title": "Retained before extraction timeout",
                "source": "new_search", "task_ids": ["task-1"],
            }],
            "task_executions": [{
                "task_id": "task-1", "direction_id": "dir_new",
                "compiled_query": "query", "status": "timeout",
            }],
            "assessment_limitations": ["Extraction timed out; raw inventory retained."],
        }],
    )

    html = _render_direction_evidence_section(report)

    assert "Retained PMID:</strong> 9988" in html
    assert "Task task-1:</strong> timeout" in html
    assert "Extraction timed out; raw inventory retained." in html
    assert "before references could be retained" not in html


def test_new_policy_assessment_renders_source_passage_and_pi_uncertainty():
    report = ExplorationReport(
        research_direction_candidates=[{
            "direction_id": "dir_assessed", "direction": "Assessed direction",
            "evidence_ref": "dir_assessed",
        }],
        direction_literature_evidence=[{
            "direction_id": "dir_assessed", "direction": "Assessed direction",
            "status": "completed", "policy_version": "direction_search_v1",
            "assessment_status": "uncertain",
            "assessment_uncertainty": ["Context remains unmatched"],
            "article_inventory": [{
                "pmid": "1122", "title": "Source study", "source": "new_search",
                "task_ids": ["task-1"],
            }],
            "extractions": [{
                "pmid": "1122", "task_ids": ["task-1"],
                "source_passage": "Exact source passage.",
            }],
            "task_executions": [{
                "task_id": "task-1", "direction_id": "dir_assessed",
                "compiled_query": "query", "status": "completed",
            }],
        }],
    )

    html = _render_direction_evidence_section(report)

    assert "PI Assessment:</strong> uncertain" in html
    assert "Context remains unmatched" in html
    assert "Source passage (PMID 1122):</strong> Exact source passage." in html


def test_integration_pi_assessment_details_are_gated_by_policy_version():
    candidates = [{"direction_id": "d1", "direction": "Direction"}]
    legacy = [{
        "direction_id": "d1", "status": "completed", "summary": "Legacy summary",
        "assessment_status": "assessed", "assessment_limitations": ["New-only detail"],
    }]
    current = [{**legacy[0], "policy_version": "direction_search_v1"}]

    legacy_html = _render_integration_evidence_section(candidates, legacy)
    current_html = _render_integration_evidence_section(candidates, current)

    assert "PI assessment" not in legacy_html
    assert "New-only detail" not in legacy_html
    assert "Legacy summary</td>" in legacy_html
    assert "Legacy summary<br>" not in legacy_html
    assert "PI assessment: assessed" in current_html
    assert "New-only detail" in current_html


def test_integration_new_policy_keeps_raw_source_and_partial_execution_context():
    candidates = [{"direction_id": "d1", "direction": "Direction"}]
    evidence = [{
        "direction_id": "d1",
        "status": "partial",
        "summary": "Extraction was incomplete.",
        "policy_version": "direction_search_v1",
        "assessment_status": "uncertain",
        "assessment_uncertainty": ["The source could not be fully extracted."],
        "article_inventory": [{"pmid": "445566", "title": "Retained raw source"}],
        "extractions": [],
        "task_executions": [{"task_id": "task-1", "status": "partial"}],
        "assessment_limitations": ["Raw inventory retained after extraction failure."],
    }]

    rendered = _render_integration_evidence_section(candidates, evidence)

    assert "Source PMIDs: 445566" in rendered
    assert "Task status: task-1=partial" in rendered
    assert "PI uncertainty: The source could not be fully extracted." in rendered
    assert "Raw inventory retained after extraction failure." in rendered
