"""Pin the full route registry so refactors cannot silently drop/rename URLs."""
import json
from pathlib import Path

from backend.main import app


SNAPSHOT_PATH = Path(__file__).parent / "route_snapshot.json"


def _current_routes():
    return sorted(
        (r.path, sorted(getattr(r, "methods", {"WS"}) or {"WS"}))
        for r in app.routes
        if hasattr(r, "path")
    )


def test_route_registry_matches_snapshot():
    expected = json.loads(SNAPSHOT_PATH.read_text())
    expected = [(p, m) for p, m in expected]
    actual = _current_routes()
    assert actual == expected, (
        f"Route registry drift. "
        f"Missing: {set(map(tuple, expected)) - set(map(tuple, actual))}. "
        f"Added: {set(map(tuple, actual)) - set(map(tuple, expected))}."
    )
