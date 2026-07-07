"""Tests for snapshot restore validation fix: type widening, normalization, fail-closed."""

import json
import uuid

import pytest
from fastapi import HTTPException

import backend.routes.snapshots as snaps
from backend.models.schemas import AnalysisState, Snapshot
from backend.state import initializing_sessions
from backend.utils.snapshot import SnapshotManager, SnapshotRestoreError


@pytest.fixture
def mock_config(monkeypatch, tmp_path):
    session_id = "restore-val-001"
    session_dir = tmp_path / "sessions" / session_id
    (session_dir / "snapshots").mkdir(parents=True)
    (session_dir / "states").mkdir(parents=True)
    (session_dir / "workspace" / "figures").mkdir(parents=True)
    (session_dir / "workspace" / "tables").mkdir(parents=True)
    (session_dir / "workspace" / "reproducibility").mkdir(parents=True)
    (session_dir / "reports" / "figures").mkdir(parents=True)
    (session_dir / "reports" / "tables").mkdir(parents=True)
    # NOTE: deliberately no kernel pkl. These tests exercise the state-JSON schema /
    # validation / fallback path only. Writing a pkl would drag every case through the
    # zstd compress/decompress path, which is unnecessary here and hangs/times out in
    # some sandboxed environments. The pkl round-trip has dedicated coverage in
    # test_snapshot_rollback.py / test_sandbox_save_restore_streaming.py.

    monkeypatch.setattr(
        "backend.utils.snapshot.get_session_dir",
        lambda sid: tmp_path / "sessions" / sid,
    )
    monkeypatch.setattr(
        "backend.utils.snapshot.get_session_states",
        lambda sid: tmp_path / "sessions" / sid / "states",
    )
    return session_id, session_dir, tmp_path


def _make_state(session_id: str) -> AnalysisState:
    return AnalysisState(
        session_id=session_id, user_id="u", data_files=[], current_round=2,
    )


def _write_snapshot_state(mgr: SnapshotManager, state: AnalysisState, patch: dict | None = None) -> Snapshot:
    """Write a minimal snapshot state JSON without exercising create_snapshot I/O."""
    snap = Snapshot(name="T", phase="pre_round_wait", round_number=state.current_round)
    snap.state_file = f"{snap.id}.json"
    state_dict = state.model_dump(mode="json")
    state_dict.pop("snapshots", None)
    state_dict.pop("realtime_messages", None)
    if patch:
        state_dict.update(patch)
    state_path = mgr.snapshots_dir / snap.state_file
    state_path.write_text(json.dumps(state_dict), encoding="utf-8")
    return snap


@pytest.mark.asyncio
async def test_restore_preserves_is_custom_bool(mock_config):
    session_id, _session_dir, _tmp = mock_config
    mgr = SnapshotManager(session_id)
    state = _make_state(session_id)
    # Mirror production: direct assignment bypasses validation (validate_assignment is off).
    dirs = [{"direction": "D1", "notes": "", "is_custom": False}]
    state.user_chosen_directions = list(dirs)
    state.directions_queue = list(dirs)
    state.active_direction_batch = list(dirs)

    snap = _write_snapshot_state(mgr, state)
    restored = await mgr.restore_snapshot(snap)  # pre-fix: raises ValidationError -> 500

    assert restored.active_direction_batch[0]["is_custom"] is False
    assert restored.directions_queue[0]["direction"] == "D1"


@pytest.mark.asyncio
async def test_restore_normalizes_is_custom_string(mock_config):
    session_id, _session_dir, _tmp = mock_config
    mgr = SnapshotManager(session_id)
    state = _make_state(session_id)
    # Drifted string value that Dict[str, Any] would accept verbatim.
    state.directions_queue = [
        {"direction": "D1", "notes": "n", "is_custom": "false", "candidate_index": "2"}
    ]
    snap = _write_snapshot_state(mgr, state)

    restored = await mgr.restore_snapshot(snap)

    item = restored.directions_queue[0]
    assert item["is_custom"] is False          # coerced from "false"
    assert item["candidate_index"] == 2        # coerced from "2"
    assert item["direction"] == "D1"           # preserved
    assert len(restored.directions_queue) == 1 # queue not emptied


@pytest.mark.asyncio
async def test_restore_degrades_recoverable_field(mock_config, caplog):
    session_id, _session_dir, _tmp = mock_config
    mgr = SnapshotManager(session_id)
    # integration_cycle is `int`; a non-numeric string cannot be coerced -> ValidationError.
    snap = _write_snapshot_state(
        mgr,
        _make_state(session_id),
        {"integration_cycle": "not-a-number"},
    )

    with caplog.at_level("WARNING"):
        restored = await mgr.restore_snapshot(snap)

    assert restored.integration_cycle == 0  # degraded to model default
    assert any("integration_cycle" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_restore_fails_closed_on_non_allowlist_scalar(mock_config):
    session_id, _session_dir, _tmp = mock_config
    mgr = SnapshotManager(session_id)
    # current_round is an `int` NOT in the recoverable allowlist; drift must fail
    # closed, not blank to default. ("not-a-number" is non-coercible — do NOT use a
    # numeric string like "2", which Pydantic would silently coerce.)
    snap = _write_snapshot_state(
        mgr,
        _make_state(session_id),
        {"current_round": "not-a-number"},
    )

    with pytest.raises(SnapshotRestoreError):
        await mgr.restore_snapshot(snap)


@pytest.mark.asyncio
async def test_restore_fails_closed_on_collection_field(mock_config):
    session_id, _session_dir, _tmp = mock_config
    mgr = SnapshotManager(session_id)
    # round_results is List[RoundResult] carrying workflow meaning. Inject an element
    # with a non-coercible field so the error is unambiguous: loc = ("round_results", 0,
    # "round_number"), loc[0] = "round_results" -> not allowlisted -> fail closed. The
    # whole list must NEVER be silently popped.
    snap = _write_snapshot_state(
        mgr,
        _make_state(session_id),
        {"round_results": [{"round_number": "not-a-number"}]},
    )

    with pytest.raises(SnapshotRestoreError):
        await mgr.restore_snapshot(snap)


class _FakeReqState:
    user_id = "u"


class _FakeRequest:
    state = _FakeReqState()


@pytest.mark.asyncio
async def test_restore_route_maps_failure_to_422_and_clears_flag(monkeypatch):
    session_id = str(uuid.uuid4())
    snap = Snapshot(name="n", phase="pre_round_wait")
    state = AnalysisState(session_id=session_id, user_id="u", data_files=[], current_round=1)
    state.snapshots = [snap]

    async def fake_owner(session_id, user_id):
        return state

    async def fake_teardown(session_id):
        return None

    async def fake_restore(self, snapshot):
        raise snaps.SnapshotRestoreError("current_round type mismatch")

    monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)
    monkeypatch.setattr(snaps, "teardown_running_workflow", fake_teardown)
    monkeypatch.setattr(snaps.SnapshotManager, "restore_snapshot", fake_restore)

    initializing_sessions.discard(session_id)          # ensure clean precondition
    assert session_id not in initializing_sessions
    with pytest.raises(HTTPException) as excinfo:
        await snaps.restore_snapshot(session_id, snap.id, _FakeRequest())

    assert excinfo.value.status_code == 422
    assert isinstance(excinfo.value.detail, str)
    assert "Rollback failed" in excinfo.value.detail
    assert "current_round" in excinfo.value.detail  # detail names the offending field (spec §3)
    assert session_id not in initializing_sessions  # no leaked flag


@pytest.mark.asyncio
async def test_restore_route_clears_flag_on_late_failure(monkeypatch):
    """A failure AFTER restore_snapshot (e.g. notebook restore) must still clear the flag."""
    session_id = str(uuid.uuid4())
    snap = Snapshot(name="n", phase="pre_round_wait")
    state = AnalysisState(session_id=session_id, user_id="u", data_files=[], current_round=1)
    state.snapshots = [snap]

    async def fake_owner(session_id, user_id):
        return state

    async def fake_teardown(session_id):
        return None

    async def fake_restore(self, snapshot):
        return state  # restore itself succeeds

    async def boom_notebook(self, snapshot):
        raise RuntimeError("late failure restoring notebook")

    monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)
    monkeypatch.setattr(snaps, "teardown_running_workflow", fake_teardown)
    monkeypatch.setattr(snaps.SnapshotManager, "restore_snapshot", fake_restore)
    monkeypatch.setattr(snaps.SnapshotManager, "restore_notebook", boom_notebook)

    initializing_sessions.discard(session_id)
    with pytest.raises(RuntimeError):  # late failure is NOT mapped to 422, but flag still cleared
        await snaps.restore_snapshot(session_id, snap.id, _FakeRequest())

    assert session_id not in initializing_sessions  # finally cleared it
