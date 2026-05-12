import inspect
from pathlib import Path
import tempfile

from backend.routes.reports import _should_skip_in_report_zip


def test_rpts_dir_does_not_contain_export_filename():
    """The export tar builder iterates rpts_dir which is
    separate from UPLOAD_TEMP_DIR, so the filename guard was dead code.
    This test confirms that files from UPLOAD_TEMP_DIR never appear
    in a reports directory listing."""
    export_filename = "export_abcd1234_Session.tar.gz"
    with tempfile.TemporaryDirectory() as tmpdir:
        rpts_dir = Path(tmpdir) / "reports" / "sess123"
        rpts_dir.mkdir(parents=True)
        (rpts_dir / "report.json").write_text('{"test": true}')

        upload_temp = Path(tmpdir) / "upload_temp"
        upload_temp.mkdir()
        (upload_temp / export_filename).write_bytes(b"fake tar data")

        rpts_files = [f.name for f in rpts_dir.rglob("*") if f.is_file()]
        assert export_filename not in rpts_files, \
            f"UPLOAD_TEMP_DIR file {export_filename} should not appear in rpts_dir"


def test_should_skip_does_not_check_tar_gz():
    """The .tar.gz check was dead code because export files are written
    to UPLOAD_TEMP_DIR, not the reports directory. It should not appear
    in _should_skip_in_report_zip's source."""
    source = inspect.getsource(_should_skip_in_report_zip)
    assert ".tar.gz" not in source, (
        "Dead .tar.gz check should be removed from _should_skip_in_report_zip"
    )