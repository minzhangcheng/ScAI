from pathlib import Path


def test_start_script_uses_backend_venv_python_for_marker_fetch():
    script = Path(__file__).resolve().parents[1] / ".." / "start.sh"
    text = script.resolve().read_text()
    assert '"$VENV_DIR/bin/python" scripts/fetch_markers.py' in text
