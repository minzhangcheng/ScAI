"""Batch 4 cutover: V1 restore always HTTP 410; V1 inventory retained non-restorable."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.config._config import SystemConfig
from backend.constants import VERSION
from backend.models.schemas import AnalysisState, Snapshot


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_snapshot_v1_restore_cutover_defaults_on():
    """No config combination should leave V1 restore enabled after cutover."""
    cfg = SystemConfig()
    assert cfg.snapshot_v1_restore_cutover is True
    # Dual-path flag removed or forced authoritative for V2 lifecycle.
    assert not hasattr(cfg, "snapshot_v2_enabled") or cfg.snapshot_v2_enabled is True


def test_should_reject_v1_restore_always():
    from backend.services.snapshot_v2.legacy import should_reject_v1_restore

    msg = should_reject_v1_restore(deprecation_version=VERSION)
    assert msg is not None
    assert "V1" in msg or "legacy" in msg.lower()
    assert VERSION in msg


def test_build_legacy_inventory_marks_non_restorable():
    from backend.services.snapshot_v2.legacy import build_legacy_inventory

    state = AnalysisState(session_id="s-v1", user_id="u1", initial_instruction="x")
    state.snapshots = [
        Snapshot(name="legacy-a", phase="pre_round_wait", format_version=None),
        Snapshot(name="legacy-b", phase="pre_direction", format_version=1),
        Snapshot(name="v2-proj", phase="post_report", format_version=2),
    ]
    inv = build_legacy_inventory(state, warning_version=VERSION)
    assert inv.snapshot_count == 2  # V2 projection excluded from legacy inventory
    assert inv.warning_version == VERSION
    assert inv.restorable is False
    for item in inv.snapshots:
        assert item.get("restorable") is False
        assert item.get("format_version", 1) in (None, 1) or item.get("format_version") == 1


def test_v2_list_excludes_v1_records():
    """V1 file-backed snapshots must never appear in V2 list projection."""
    from backend.services.snapshot_v2.legacy import filter_v1_from_v2_list

    mixed = [
        {"snapshot_id": "v1-a", "manifest": {"format_version": 1}},
        {"snapshot_id": "v2-b", "manifest": {"format_version": 2}},
        {"snapshot_id": "legacy", "manifest": None},
    ]
    filtered = filter_v1_from_v2_list(mixed)
    assert [s["snapshot_id"] for s in filtered] == ["v2-b"]


@pytest.mark.asyncio
async def test_v1_restore_returns_410_before_teardown(monkeypatch):
    """POST V1 restore must 410 before any workflow teardown or state mutation."""
    import backend.routes.snapshots as snaps
    import backend.routes._helpers as helpers

    sid = "sess-v1-cutover-410"
    state = AnalysisState(session_id=sid, user_id="u1")
    snap = Snapshot(name="legacy", phase="pre_round_wait")
    state.snapshots.append(snap)

    async def fake_owner(session_id, user_id):
        return state

    teardown_called = {"v": False}

    async def fake_teardown(session_id):
        teardown_called["v"] = True

    monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)
    monkeypatch.setattr(helpers, "teardown_running_workflow", fake_teardown)

    class Req:
        state = type("S", (), {"user_id": "u1"})()

        async def json(self):
            return {"runtime_choice": "reset"}

    with pytest.raises(HTTPException) as ei:
        await snaps.restore_snapshot(sid, snap.id, Req())

    assert ei.value.status_code == 410
    detail = str(ei.value.detail)
    assert "V1" in detail or "legacy" in detail.lower() or "unsupported" in detail.lower()
    assert VERSION in detail
    assert teardown_called["v"] is False
    assert not hasattr(snaps, "teardown_running_workflow")


@pytest.mark.asyncio
async def test_v1_restore_410_even_when_snapshot_missing_still_no_teardown(monkeypatch):
    """Unknown snapshot id still 410 (cutover is unconditional; no teardown path)."""
    import backend.routes.snapshots as snaps
    import backend.routes._helpers as helpers

    sid = "sess-v1-missing"
    state = AnalysisState(session_id=sid, user_id="u1")

    async def fake_owner(session_id, user_id):
        return state

    teardown = AsyncMock()
    monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)
    monkeypatch.setattr(helpers, "teardown_running_workflow", teardown)

    class Req:
        state = type("S", (), {"user_id": "u1"})()

        async def json(self):
            return {}

    with pytest.raises(HTTPException) as ei:
        await snaps.restore_snapshot(sid, "does-not-exist", Req())

    assert ei.value.status_code == 410
    teardown.assert_not_awaited()


def test_v1_restore_body_has_no_snapshot_manager_restore():
    """After cutover, restore_snapshot must not call SnapshotManager restore paths."""
    source = (REPO_ROOT / "backend/routes/snapshots.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    restore_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "restore_snapshot":
            restore_fn = node
            break
    assert restore_fn is not None
    text = ast.get_source_segment(source, restore_fn) or ""
    assert "teardown_running_workflow" not in text
    assert "mgr.restore_snapshot" not in text
    assert "SnapshotManager" not in text
    assert "410" in text


def test_no_snapshot_v2_enabled_branching_in_routes():
    """Dead dual-path flags must not gate restore/list behavior."""
    snaps = (REPO_ROOT / "backend/routes/snapshots.py").read_text(encoding="utf-8")
    assert "snapshot_v2_enabled" not in snaps
    cfg = (REPO_ROOT / "backend/config/_config.py").read_text(encoding="utf-8")
    # Flag either removed or permanently True without consumers.
    if "snapshot_v2_enabled" in cfg:
        assert "snapshot_v2_enabled: bool = True" in cfg or "snapshot_v2_enabled: bool = Field" in cfg
