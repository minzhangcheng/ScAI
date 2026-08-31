"""Tests for regenerator graphic abstract and reproducibility appendix."""
import pytest
from pathlib import Path

from backend.constants import VERSION


def test_regenerator_includes_graphic_abstract_and_reproducibility(tmp_path, monkeypatch):
    from backend.services.report.regenerator import _build_graphic_abstract_html, _build_appendix_from_report
    from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
    from backend.config import get_session_workspace

    # ---- Setup: graphic abstract image ----
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    ga_path = figures_dir / "graphic_abstract.jpg"
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(str(ga_path), "JPEG")

    # ---- Setup: reproducibility directory with checksums ----
    def mock_get_session_workspace(session_id):
        return tmp_path
    monkeypatch.setattr(
        "backend.services.report.regenerator.get_session_workspace",
        mock_get_session_workspace,
    )

    repro_dir = tmp_path / REPRODUCIBILITY_DIR_NAME
    repro_dir.mkdir(parents=True)
    checksum_path = repro_dir / "checksums.sha256"
    checksum_path.write_text(
        "abc123def456abc123def456abc123def456abc123def456abc123def456abc1  data/file1.h5ad\n"
        "def789abc456def789abc456def789abc456def789abc456def789abc456def789a  data/file2.csv\n"
    )

    report = {
        "session_id": "test-session",
        "graphic_abstract_path": str(ga_path),
        "appendix": {"project_name": "Test"},
    }

    ga = _build_graphic_abstract_html(report)
    assert ga != ""
    assert "Graphic Abstract" in ga

    appendix = _build_appendix_from_report(report, "en", "test-session")
    assert "Reproducibility" in appendix or "Checksum" in appendix


def test_regenerator_renders_enriched_direction_history():
    from backend.services.report.regenerator import _build_appendix_from_report

    report = {
        "appendix": {
            "project_name": "Test",
            "user_interactions": {
                "direction_selection": [{
                    "selection_event": "Integration Cycle 2 · integration",
                    "decision_mode": "PI-selected",
                    "direction": "Cross-direction mechanism",
                    "notes": "Strong shared evidence",
                }],
            },
        },
    }

    html = _build_appendix_from_report(report, "en")

    assert "Research Direction Selections" in html
    assert "Selection Event" in html
    assert "Integration Cycle 2" in html
    assert "PI-selected" in html
    assert "Cross-direction mechanism" in html


def test_regenerator_reports_persisted_session_scai_version():
    from backend.services.report.regenerator import _build_appendix_from_report

    html = _build_appendix_from_report(
        {
            "generated_at": "2026-08-31T12:00:00",
            "appendix": {
                "project_name": "Historical analysis",
                "scai_version": "0.49.9",
            },
        },
        "en",
    )

    assert "ScAI Version (Session Created With)" in html
    assert "0.49.9" in html
    assert VERSION != "0.49.9"
    assert VERSION not in html


def test_regenerator_marks_legacy_session_version_unknown():
    from backend.services.report.regenerator import _build_appendix_from_report

    html = _build_appendix_from_report(
        {"appendix": {"project_name": "Legacy analysis"}},
        "en",
    )

    assert "Unknown (legacy session)" in html


def test_fallback_report_reports_persisted_session_scai_version(
    tmp_path, monkeypatch,
):
    from backend.models.schemas import AnalysisState
    from backend.services.report.sections import SectionsMixin

    monkeypatch.setattr(
        "backend.services.report.sections.get_session_workspace",
        lambda _session_id: tmp_path,
    )
    monkeypatch.setattr(
        "backend.services.report.regenerator.get_agent_llm_stats",
        lambda _session_id: {},
    )
    state = AnalysisState(
        id="fallback-versioned-session",
        name="Historical analysis",
        initial_instruction="Test",
        user_id="u1",
        scai_version="0.49.9",
    )

    html = SectionsMixin()._build_appendix_html(
        state,
        {"generated_at": "2026-08-31T12:00:00"},
        "en",
    )

    assert "ScAI Version (Session Created With)" in html
    assert "0.49.9" in html
