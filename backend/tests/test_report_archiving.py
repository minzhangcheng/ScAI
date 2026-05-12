import pytest
import zipfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_archive_version_calculation():
    from backend.routes.reports import _next_archive_version
    
    reports_dir = Path("/tmp/test_reports")
    assert _next_archive_version(reports_dir) == 1
    
    with patch.object(Path, 'glob') as mock_glob:
        mock_glob.return_value = [
            Path("/tmp/test_reports/previous_report_v1.zip"),
            Path("/tmp/test_reports/previous_report_v3.zip"),
        ]
        with patch.object(Path, 'exists', return_value=True):
            assert _next_archive_version(reports_dir) == 4


def test_archive_excludes_existing_zips():
    from backend.routes.reports import _create_report_archive
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir) / "reports"
        reports_dir.mkdir()
        (reports_dir / "report.json").write_text('{"test": true}')
        (reports_dir / "previous_report_v1.zip").write_bytes(b"old archive")
        
        archive_path = _create_report_archive(reports_dir, version=2)
        assert archive_path.exists()
        
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            assert any("report.json" in n for n in names)
            assert not any("previous_report_v1.zip" in n for n in names)


def test_archive_uses_zip_stored_compression():
    from backend.routes.reports import _create_report_archive
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir) / "reports"
        reports_dir.mkdir()
        (reports_dir / "report.json").write_text('{"test": true}')

        archive_path = _create_report_archive(reports_dir, version=1)
        assert archive_path.exists()

        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED, (
                    f"{info.filename} should use ZIP_STORED, got {info.compress_type}"
                )


def test_archive_retention_keeps_most_recent():
    from backend.routes.reports import _enforce_archive_retention
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir) / "reports"
        reports_dir.mkdir()

        archives = []
        for i in range(7):
            path = reports_dir / f"previous_report_v{i + 1}.zip"
            path.write_bytes(b"archive content")
            time.sleep(0.05)
            archives.append(path)

        _enforce_archive_retention(reports_dir, max_archives=5)

        remaining = sorted(reports_dir.glob("previous_report_v*.zip"))
        assert len(remaining) == 5
        assert remaining[0].name == "previous_report_v3.zip"
        assert remaining[-1].name == "previous_report_v7.zip"

        assert not (reports_dir / "previous_report_v1.zip").exists()
        assert not (reports_dir / "previous_report_v2.zip").exists()


def test_archive_atomic_write_no_partial_files():
    from backend.routes.reports import _create_report_archive
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir) / "reports"
        reports_dir.mkdir()
        (reports_dir / "report.json").write_text('{"test": true}')

        archive_path = _create_report_archive(reports_dir, version=1)
        tmp_path = archive_path.with_suffix(".zip.tmp")
        assert not tmp_path.exists(), "Temp file should be cleaned up after atomic write"
        assert archive_path.exists()


def test_download_full_report_uses_slugified_name():
    from backend.utils.files import slugify_session_name

    names_with_special_chars = [
        ("My Session/Report", "My_SessionReport"),
        ('Test "Quoted" Name', "Test_Quoted_Name"),
        ("Session\nWith\nNewlines", "SessionWithNewlines"),
        ("  Spaces  Everywhere  ", "Spaces_Everywhere"),
        ("", "session"),
    ]

    for raw, expected_prefix in names_with_special_chars:
        result = slugify_session_name(raw)
        assert expected_prefix == result, f"slugify_session_name({raw!r}) = {result!r}, expected {expected_prefix!r}"