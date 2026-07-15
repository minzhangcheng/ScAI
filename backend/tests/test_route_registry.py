"""Pin the full route registry so refactors cannot silently drop/rename URLs."""
import json
from pathlib import Path

from backend.main import app


SNAPSHOT_PATH = Path(__file__).parent / "route_snapshot.json"


def _iter_routes(routes):
    """Yield (path, methods) for every reachable route.

    FastAPI >= ~0.13x no longer flattens `include_router` children into
    `app.routes`; it parks a lazy `_IncludedRouter` wrapper there instead, and
    that wrapper has no `.path`. Filtering on `hasattr(r, "path")` therefore
    silently reduced this test to FastAPI's four built-in doc routes. Expand the
    wrapper when present, and stay compatible with versions that do flatten.
    """
    for route in routes:
        candidates = getattr(route, "effective_candidates", None)
        if callable(candidates):
            yield from _iter_routes(candidates())
            low_priority = getattr(route, "effective_low_priority_routes", None)
            if callable(low_priority):
                yield from _iter_routes(low_priority())
            continue

        # An expanded route carries the effective (prefixed) path, while its
        # methods live on the route it wraps. WebSocket routes leave that path
        # empty and resolve it on the wrapped route instead.
        source = getattr(route, "original_route", route)
        path = (
            getattr(route, "path", None)
            or getattr(getattr(route, "starlette_route", None), "path", None)
            or getattr(source, "path", None)
        )
        if not path:
            continue
        yield path, tuple(sorted(getattr(source, "methods", None) or {"WS"}))


def _current_routes():
    return sorted(set(_iter_routes(app.routes)))


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


def test_route_enumeration_sees_the_mounted_routers():
    """Guard the enumeration itself: if it ever collapses to FastAPI's built-in
    doc routes again, the snapshot check above silently stops protecting URLs."""
    paths = {p for p, _ in _current_routes()}

    assert "/api/session/{session_id}/load" in paths
    assert len(paths) > 50
