import os
from pathlib import Path

import pytest
from backend.services.report.variants import _render_direction_evidence_section_from_lists
from backend.services.report.variants import _deduplicate_detailed_figure_blocks
from backend.tests.split_facade import setattr_facade
from backend.utils.blob_store import externalize_blobs


def test_render_direction_evidence_section_from_lists_empty():
    result = _render_direction_evidence_section_from_lists([], [])
    assert result == ""


def test_combined_detailed_report_deduplicates_keyed_figures():
    block = (
        '<div class="figure-item" data-figure-key="same">'
        '<img src="data:image/jpeg;base64,x" /><p>Caption</p></div>'
    )
    result = _deduplicate_detailed_figure_blocks(block + "<hr>" + block)
    assert result.count('data-figure-key="same"') == 1


def test_render_direction_evidence_section_from_lists_with_candidates():
    candidates = [{"direction": "Test dir", "direction_id": "d1", "overall_priority": "high"}]
    evidence = [{"direction_id": "d1", "status": "completed", "direct_evidence_count": 2}]
    result = _render_direction_evidence_section_from_lists(candidates, evidence)
    assert "Test dir" in result
    assert "Literature-Gated Direction Evaluation" in result


def test_direction_evidence_renderer_treats_scalar_lists_as_single_items():
    result = _render_direction_evidence_section_from_lists([{
        "direction": "Scalar-safe direction",
        "direction_id": "dir_001",
        "prerequisite_dependencies": "dir_000",
        "risk_flags": "batch_confounding",
    }], [])

    assert "dir_000" in result
    assert "batch_confounding" in result
    assert "d, i, r" not in result
    assert "b, a, t" not in result


def _make_integration_state():
    from backend.models.schemas import AnalysisState, RoundResult
    state = AnalysisState(id="test-session", name="Test Project", user_id="test", integration_cycle=1)
    state.multi_direction_correlation = "Cross-direction analysis found correlation between X and Y."
    state.round_results = [
        RoundResult(round_number=2, phase="innovation", direction="Direction A",
                    pi_summary="Found significant results in pathway X.",
                    bioinformatics_findings=["Gene X upregulated"], pathology_findings=[],
                    biochemistry_findings=[], literature_findings=[]),
    ]
    return state


def _patch_report_paths(monkeypatch, tmp_path):
    """Redirect the session-path helpers for every module that resolves them.

    The generators live in backend/services/report/variants_parts/ and read these
    names from their own globals, so the whole split has to be rebound.
    """
    session_root = tmp_path / "sessions" / "test-session"
    reports_dir = session_root / "reports"

    facade = "backend.services.report.variants"
    setattr_facade(monkeypatch, facade, "get_session_dir", lambda _session_id: session_root)
    setattr_facade(monkeypatch, facade, "get_session_reports", lambda _session_id: reports_dir)
    return session_root, reports_dir


def test_generate_integration_report_html_with_candidates(tmp_path, monkeypatch):
    from backend.services.report.variants import generate_integration_report_html

    state = _make_integration_state()
    _patch_report_paths(monkeypatch, tmp_path)
    candidates = [
        {
            "direction": "Novel pathway Z",
            "rationale": "Emerges from cross-direction correlation",
            "priority_rank": 1,
            "novelty_score": 8,
            "evidence_strength": "strong",
            "confidence": "high",
            "report_recommendation": "headline",
            "direction_id": "d1",
        },
    ]
    evidence = [
        {
            "direction_id": "d1",
            "status": "completed",
            "direct_evidence_count": 3,
            "analogous_evidence_count": 1,
            "contradictory_evidence_count": 0,
            "summary": "Direct support exists.",
        }
    ]

    path = generate_integration_report_html(state, candidates, evidence)

    assert os.path.exists(path)
    html = Path(path).read_text(encoding="utf-8")
    assert "Integration Report" in html
    assert "Novel pathway Z" in html
    assert "Cross-Direction Correlation" in html
    assert "8.0/10" in html
    assert "Rank 1" in html
    assert "Direct support exists." in html
    assert "Evidence strength" in html


def test_generate_integration_report_html_empty_candidates(tmp_path, monkeypatch):
    from backend.services.report.variants import generate_integration_report_html

    state = _make_integration_state()
    _patch_report_paths(monkeypatch, tmp_path)

    path = generate_integration_report_html(state, [], [])

    assert os.path.exists(path)
    html = Path(path).read_text(encoding="utf-8")
    assert "Integration Report" in html
    assert "Cross-Direction Correlation" in html
    assert ">2. Innovation Round Summaries<" in html
    assert "Research Direction Candidates" not in html


def test_generate_integration_report_html_inlines_blob_correlation(tmp_path, monkeypatch):
    from backend.services.report.variants import generate_integration_report_html

    state = _make_integration_state()
    session_root, _reports_dir = _patch_report_paths(monkeypatch, tmp_path)
    long_markdown = "## Shared mechanism\n\n" + ("A repeated cross-direction mechanism. " * 80)
    state.multi_direction_correlation = externalize_blobs(
        long_markdown,
        session_root / "blobs",
    )

    path = generate_integration_report_html(state, [], [])

    html = Path(path).read_text(encoding="utf-8")
    assert "@scai-blob:" not in html
    assert "Shared mechanism" in html
    assert "repeated cross-direction mechanism" in html


def test_generate_integration_report_html_handles_non_numeric_novelty_score(tmp_path, monkeypatch):
    from backend.services.report.variants import generate_integration_report_html

    state = _make_integration_state()
    _patch_report_paths(monkeypatch, tmp_path)
    candidates = [
        {
            "direction": "Qualitative novelty",
            "rationale": "LLM returned a qualitative novelty score.",
            "priority_rank": 2,
            "novelty_score": "high",
            "evidence_strength": "moderate",
            "confidence": "medium",
        }
    ]

    path = generate_integration_report_html(state, candidates, [])

    html = Path(path).read_text(encoding="utf-8")
    assert "Qualitative novelty" in html
    assert "N/A" in html
    assert "high/10" not in html


def test_generate_integration_report_html_renumbers_when_correlation_missing(tmp_path, monkeypatch):
    from backend.services.report.variants import generate_integration_report_html

    state = _make_integration_state()
    state.multi_direction_correlation = ""
    _patch_report_paths(monkeypatch, tmp_path)

    path = generate_integration_report_html(state, [], [])

    html = Path(path).read_text(encoding="utf-8")
    assert ">1. Innovation Round Summaries<" in html
    assert ">2. Innovation Round Summaries<" not in html
