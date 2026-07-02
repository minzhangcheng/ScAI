"""Tests for report subdirectory file serving (H14).

Ensures _resolve_report_file preserves subdirectory paths
while still blocking path traversal.
"""
from pathlib import Path
import pytest


def test_subdir_file_path_preserved(tmp_path):
    from backend.routes.reports import _resolve_report_file
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.html").write_text("ok")
    resolved = _resolve_report_file(tmp_path, "sub/f.html")
    assert resolved.read_text() == "ok"


def test_traversal_still_blocked(tmp_path):
    from backend.routes.reports import _resolve_report_file
    with pytest.raises(Exception):
        _resolve_report_file(tmp_path, "../../etc/passwd")


def test_traversal_with_subdir_still_blocked(tmp_path):
    from backend.routes.reports import _resolve_report_file
    (tmp_path / "sub").mkdir()
    with pytest.raises(Exception):
        _resolve_report_file(tmp_path, "sub/../../../etc/passwd")
