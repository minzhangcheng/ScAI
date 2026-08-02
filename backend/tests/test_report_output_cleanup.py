"""Regression tests for stale report-owned derivatives on regeneration."""

from backend.services.report.generator import DetailedReportGenerator
from backend.services.report.output_cleanup import reset_report_asset_directories


def _seed_stale_outputs(report_dir):
    (report_dir / "figures" / "nested").mkdir(parents=True)
    (report_dir / "figures" / "nested" / "stale.png").write_bytes(b"stale")
    (report_dir / "tables").mkdir()
    (report_dir / "tables" / "stale.csv").write_text("old\n", encoding="utf-8")
    (report_dir / "report.json").write_text("{}", encoding="utf-8")
    (report_dir / "previous_report.zip").write_bytes(b"archive")


def test_reset_removes_only_report_owned_asset_directories(tmp_path):
    report_dir = tmp_path / "report"
    _seed_stale_outputs(report_dir)

    reset_report_asset_directories(report_dir)

    assert list((report_dir / "figures").iterdir()) == []
    assert list((report_dir / "tables").iterdir()) == []
    assert (report_dir / "report.json").exists()
    assert (report_dir / "previous_report.zip").exists()


def test_fallback_generator_resets_assets_during_construction(tmp_path):
    report_dir = tmp_path / "report"
    _seed_stale_outputs(report_dir)

    DetailedReportGenerator(
        llm_client=object(),
        pi_model="pi",
        bio_model="bio",
        output_dir=str(report_dir),
    )

    assert list((report_dir / "figures").iterdir()) == []
    assert list((report_dir / "tables").iterdir()) == []
