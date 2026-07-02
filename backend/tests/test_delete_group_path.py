from pathlib import Path

from backend.routes.sessions import _safe_remove_session_file


def test_out_of_session_file_not_removed(tmp_path):
    session_dir = tmp_path / "sessions" / "sid"
    session_dir.mkdir(parents=True)
    outside = tmp_path / "victim.txt"
    outside.write_text("keep me")
    _safe_remove_session_file(str(outside), session_dir)
    assert outside.exists()


def test_in_session_file_removed(tmp_path):
    session_dir = tmp_path / "sessions" / "sid"
    (session_dir / "uploads").mkdir(parents=True)
    inside = session_dir / "uploads" / "f.csv"
    inside.write_text("data")
    _safe_remove_session_file(str(inside), session_dir)
    assert not inside.exists()
