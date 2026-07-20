"""Tests for regenerator graphic abstract and reproducibility appendix."""
import pytest
from pathlib import Path


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
