from backend.models.schemas import ExplorationReport
from backend.services.report.variants import _render_direction_evidence_section


def test_evidence_linked_by_evidence_ref_when_direction_id_absent():
    report = ExplorationReport(
        research_direction_candidates=[{
            "direction": "IFN network",
            "overall_priority": "high",
            "evidence_ref": "dir_001",
        }],
        direction_literature_evidence=[{
            "direction_id": "dir_001",
            "direction": "IFN network",
            "status": "completed",
            "analogous_evidence": [{"pmid": "777", "title": "T", "relevance": "medium",
                                    "supports": "analogous", "why_it_matters": "w"}],
        }],
    )
    html = _render_direction_evidence_section(report)
    assert "777" in html
    assert "no_evidence" not in html.lower()
