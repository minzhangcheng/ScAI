"""Structural boundary tests for Snapshot V2 fail-closed producers."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend.tests.split_facade import source_for_path


def _repo_root() -> Path:
    return Path(__file__).parents[2]


def _read(rel: str) -> str:
    return source_for_path(rel)


def test_retry_py_rethrows_required_snapshot_blocked():
    """retry.py must re-raise RequiredSnapshotBlocked before generic Exception."""
    source = _read("backend/workflow/retry.py")
    assert "RequiredSnapshotBlocked" in source
    tree = ast.parse(source)
    # Find except handlers in order within with_retry wrapper
    found_required = False
    found_generic_after = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            type_name = ""
            if isinstance(node.type, ast.Name):
                type_name = node.type.id
            elif isinstance(node.type, ast.Attribute):
                type_name = node.type.attr
            if type_name == "RequiredSnapshotBlocked":
                found_required = True
            if found_required and type_name == "Exception":
                found_generic_after = True
    assert found_required, "retry.py must have except RequiredSnapshotBlocked"
    assert found_generic_after, (
        "except RequiredSnapshotBlocked must appear before generic Exception"
    )


def test_core_run_catches_required_snapshot_blocked():
    source = _read("backend/workflow/core.py")
    assert "RequiredSnapshotBlocked" in source
    # run() should catch it (not only import)
    assert "except RequiredSnapshotBlocked" in source


def test_create_auto_snapshot_uses_capture_required_snapshot():
    source = _read("backend/workflow/core.py")
    assert "capture_required_snapshot" in source
    # Must not swallow failures with bare log-and-return for required path
    # (fail closed raises RequiredSnapshotBlocked)
    assert "RequiredSnapshotBlocked" in source


def test_reporting_uses_idempotency_key():
    source = _read("backend/workflow/reporting.py")
    assert "idempotency_key" in source or "report_generation_id" in source
    assert "capture_required_snapshot" in source or "_capture_required_snapshot" in source


def test_snapshot_v2_failure_skip_is_producer_aware():
    """Safe producers offer Skip while pre-restore remains fail-closed."""
    errors = _read("backend/services/snapshot_v2/errors.py")
    gate = _read("backend/services/snapshot_v2/failure_gate.py")
    assert "SNAPSHOT_FAILURE_ACTIONS_SKIPPABLE" in errors
    assert '"workflow_boundary", "post_report", "chat"' in gate
    assert "pre_restore (or any unknown producer): skip is unsafe" in gate


def test_safe_point_module_exports_process_wide_getter():
    source = _read("backend/services/snapshot_v2/safe_point.py")
    assert "def get_snapshot_safe_point_manager" in source



# ── Batch 3 Task 4: V2 route management boundary ──────────────────────────


_V2_FORBIDDEN_IN_ROUTES = (
    "ManifestStore",
    "DeletionService",
    "SnapshotGarbageCollector",
)


def _route_sources() -> dict[str, str]:
    return {
        "backend/routes/snapshots.py": _read("backend/routes/snapshots.py"),
        "backend/routes/export_import.py": _read("backend/routes/export_import.py"),
    }


def test_v2_routes_do_not_instantiate_store_helpers_directly():
    """V2 routes must not import/instantiate ManifestStore/DeletionService/GC.

    V1 restore is HTTP 410 (no SnapshotManager). V1 list/delete/preview may
    still use SnapshotManager readers/deleters; that must not spread to V2.
    """
    offenders = []
    for rel, source in _route_sources().items():
        tree = ast.parse(source)
        # Collect import names and Call names
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    name = alias.name
                    if name in _V2_FORBIDDEN_IN_ROUTES:
                        offenders.append(f"{rel}: import {name} from {mod}")
                    if "deletion" in mod and name == "DeletionService":
                        offenders.append(f"{rel}: import DeletionService from {mod}")
                    if "gc" in mod.split(".") and name == "SnapshotGarbageCollector":
                        offenders.append(f"{rel}: import SnapshotGarbageCollector")
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in _V2_FORBIDDEN_IN_ROUTES:
                    offenders.append(f"{rel}: call {name}()")
    assert not offenders, "V2 routes must not use store helpers directly:\n" + "\n".join(offenders)


def test_v2_routes_delegate_mutations_to_snapshot_service():
    """Delete/export/import V2 routes call SnapshotService methods."""
    snaps = _read("backend/routes/snapshots.py")
    exp = _read("backend/routes/export_import.py")
    assert "SnapshotService" in snaps
    assert "delete_leaf" in snaps or "delete_branch" in snaps
    assert "list_snapshots" in snaps
    assert "preview_snapshot" in snaps
    assert "SnapshotService" in exp
    assert "export_snapshots" in exp
    assert "import_snapshots" in exp
    # No direct DeletionService in snapshots
    assert "DeletionService(" not in snaps
    assert "ManifestStore(" not in snaps
    assert "SnapshotExporter(" not in exp
    assert "SnapshotImporter(" not in exp


def test_legacy_v1_restore_is_http_410_cutover():
    """Batch 4: V1 restore is HTTP 410 with no SnapshotManager restore body."""
    source = _read("backend/routes/snapshots.py")
    tree = ast.parse(source)
    v1_restore_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "restore_snapshot":
            v1_restore_fn = node
            break
    assert v1_restore_fn is not None, "legacy restore_snapshot endpoint must remain (HTTP 410)"
    text = ast.get_source_segment(source, v1_restore_fn) or ""
    assert "410" in text
    assert "SnapshotManager" not in text
    assert "teardown_running_workflow" not in text

    # V2 delete must not use SnapshotManager
    v2_delete = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "delete_snapshot_v2":
            v2_delete = node
            break
    assert v2_delete is not None
    for node in ast.walk(v2_delete):
        if isinstance(node, ast.Name) and node.id in ("SnapshotManager", "DeletionService", "ManifestStore"):
            pytest.fail(f"delete_snapshot_v2 must not reference {node.id}")


def test_v2_service_mutations_enter_session_lock():
    """delete_leaf/branch/export/import must acquire get_session_lock once each."""
    source = _read("backend/services/snapshot_v2/service.py")
    tree = ast.parse(source)
    required = {
        "delete_leaf": False,
        "delete_branch": False,
        "export_snapshots": False,
        "import_snapshots": False,
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in required:
            body = ast.dump(node)
            assert "get_session_lock" in body, f"{node.name} must call get_session_lock"
            required[node.name] = True
    missing = [k for k, v in required.items() if not v]
    assert not missing, f"Missing service methods: {missing}"
