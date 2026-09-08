"""The Python client keeps all retained server paths in one registry."""

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest


PACKAGE = Path(importlib.import_module("scai_client").__file__).parent


def _load_endpoints():
    if importlib.util.find_spec("scai_client.endpoints") is None:
        pytest.fail("scai_client.endpoints is required as the sole path registry")
    return importlib.import_module("scai_client.endpoints")


def _code_strings(tree: ast.AST):
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.value


def test_no_path_literals_outside_endpoints():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "endpoints.py":
            continue
        for text in _code_strings(ast.parse(path.read_text())):
            if "/api/" in text:
                offenders.append(f"{path.relative_to(PACKAGE)}: {text!r}")
    assert not offenders, "server paths must come from scai_client.endpoints:\n" + "\n".join(offenders)


def test_registry_matches_retained_routes():
    endpoints = _load_endpoints()
    expected = {
        "create_analysis", "analysis", "start_analysis", "analysis_export",
        "analysis_config", "analysis_artifact", "upload_init", "upload_chunk",
        "upload_complete", "health", "login", "api_keys", "sessions",
        "session_import", "import_status", "import_cancel", "session_export",
        "stop_session", "session_snapshots", "session_snapshot",
        "restore_snapshot",
    }
    builders = {
        name for name, value in vars(endpoints).items()
        if not name.startswith("_")
        and inspect.isfunction(value)
        and value.__module__ == endpoints.__name__
    }
    assert builders == expected
    assert set(endpoints.OUTSIDE_V1) == {
        "health", "login", "api_keys", "sessions", "session_import",
        "import_status", "import_cancel", "session_export", "stop_session",
        "session_snapshots", "session_snapshot", "restore_snapshot",
    }
    assert endpoints.restore_snapshot("s", "r003-round-g57") == (
        "/api/session/s/snapshots/r003-round-g57/restore"
    )
    assert endpoints.session_snapshot("s", "..") == "/api/session/s/snapshots/%2E%2E"
    assert endpoints.analysis("a/b") == "/api/v1/analyses/a%2Fb"
    assert endpoints.analysis("..") == "/api/v1/analyses/%2E%2E"
    assert endpoints.analysis_artifact("a/b", "figures/x y.png") == (
        "/api/v1/analyses/a%2Fb/artifacts/figures/x%20y.png"
    )
    assert endpoints.analysis_artifact("job", "figures/../x.png") == (
        "/api/v1/analyses/job/artifacts/figures/%2E%2E/x.png"
    )
