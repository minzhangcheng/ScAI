from pathlib import Path

from backend.routes.reports import _should_skip_in_report_zip


def test_tar_gz_not_skipped():
    assert _should_skip_in_report_zip(Path("/reports/session_export.tar.gz")) is False
    assert _should_skip_in_report_zip(Path("/reports/foo.tar.gz")) is False


def test_skip_previous_report_zip():
    assert _should_skip_in_report_zip(Path("/reports/previous_report_v1.zip")) is True
    assert _should_skip_in_report_zip(Path("/reports/previous_report_v99.zip")) is True


def test_skip_zip_tmp_orphan():
    assert _should_skip_in_report_zip(Path("/reports/previous_report_v3.zip.tmp")) is True


def test_allow_report_files():
    assert _should_skip_in_report_zip(Path("/reports/figures/volcano.png")) is False
    assert _should_skip_in_report_zip(Path("/reports/tables/degs.csv")) is False
    assert _should_skip_in_report_zip(Path("/reports/report.html")) is False
    assert _should_skip_in_report_zip(Path("/reports/report.json")) is False