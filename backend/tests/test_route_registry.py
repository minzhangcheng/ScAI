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
    # JSON deserializes methods as lists; convert to tuples for set operations
    expected_set = {(p, tuple(m)) for p, m in expected}
    actual_set = {(p, tuple(m)) for p, m in _current_routes()}
    missing = expected_set - actual_set
    added = actual_set - expected_set
    assert not missing and not added, (
        f"Route registry drift. Missing: {missing}. Added: {added}."
    )
