"""Startup recovery barrier: unresolved APPLYING journals block forward execution."""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.services.snapshot_v2.recovery import RecoveryResult


# Certifies Snapshot V2-enabled behaviour. The operator gate ships off, so pin it
# on here instead of inheriting it from the environment: these assertions describe
# the enabled lifecycle and are meaningless (and were failing) with it off.
pytestmark = pytest.mark.requires_snapshot_v2

def _prepare_session_on_disk(tmp_path, monkeypatch, *, status: str = "running"):
    """Write a minimal session.json and patch session path helpers."""
    import backend.services.session_store as ss_mod
    from backend.state import sessions

    session_id = f"sess-rec-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

    def _sess_dir(sid):
        return tmp_path / "data" / "sessions" / sid

    monkeypatch.setattr("backend.config._config.get_session_dir", _sess_dir)
    monkeypatch.setattr(
        "backend.config._config.get_session_workspace",
        lambda sid: _sess_dir(sid) / "workspace",
    )
    monkeypatch.setattr(
        "backend.config._config.get_session_states",
        lambda sid: _sess_dir(sid) / "states",
    )
    monkeypatch.setattr(
        "backend.services.snapshot_v2.paths.get_session_dir", _sess_dir,
    )
    # load_session_from_disk / save paths also use get_session_metadata_path
    monkeypatch.setattr(
        "backend.services.session_store.get_session_metadata_path",
        lambda sid: _sess_dir(sid) / "session.json",
    )
    monkeypatch.setattr(
        "backend.services.session_store.resolve_session_metadata_path",
        lambda sid: _sess_dir(sid) / "session.json",
    )

    root = _sess_dir(session_id)
    root.mkdir(parents=True)
    (root / "workspace").mkdir()
    (root / "states").mkdir()
    (root / "session.json").write_text(json.dumps({
        "id": session_id,
        "user_id": "u1",
        "name": "blocked",
        "status": status,
        "current_round": 1,
        "max_rounds": 4,
        "initial_instruction": "t",
    }))

    sessions.pop(session_id, None)
    return session_id, ss_mod, sessions


@pytest.mark.asyncio
async def test_ensure_session_loaded_sets_recovery_blocked_on_failed_recovery(
    tmp_path, monkeypatch,
):
    """When journal recovery fails, session load sets recovery_blocked and PAUSED."""
    session_id, ss_mod, sessions = _prepare_session_on_disk(tmp_path, monkeypatch)

    async def fake_recover(**kw):
        return RecoveryResult(
            recovered=False,
            recovered_transactions=[],
            failed_transactions=["restore-applying-dead"],
            errors=["Journal integrity failure: HMAC mismatch"],
            mutations_blocked=True,
        )

    monkeypatch.setattr(
        "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
        fake_recover,
    )

    state = await ss_mod.ensure_session_loaded(session_id)
    assert state is not None
    assert state.recovery_blocked is True
    assert state.status == TaskStatus.PAUSED

    sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_unexpected_recovery_exception_blocks_session(tmp_path, monkeypatch):
    """Any unexpected recovery exception must fail closed (recovery_blocked)."""
    from backend.services.session_store import assert_recovery_clear
    from backend.services.snapshot_v2.errors import SnapshotTransactionError

    session_id, ss_mod, sessions = _prepare_session_on_disk(
        tmp_path, monkeypatch, status="running",
    )

    monkeypatch.setattr(
        "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
        AsyncMock(side_effect=OSError("journal directory unavailable")),
    )

    state = await ss_mod.ensure_session_loaded(session_id)
    assert state is not None
    assert state.recovery_blocked is True
    assert state.status == TaskStatus.PAUSED
    err = (state.snapshot_transaction or {}).get("recovery_error") or {}
    assert err.get("code") == "RECOVERY_UNEXPECTED"
    with pytest.raises(SnapshotTransactionError, match="RECOVERY_BLOCKED"):
        assert_recovery_clear(state)

    sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_clean_recovery_clears_prior_block(tmp_path, monkeypatch):
    """Blocked session can resume after a later clean recovery succeeds."""
    from backend.services.session_store import assert_recovery_clear

    session_id, ss_mod, sessions = _prepare_session_on_disk(
        tmp_path, monkeypatch, status="running",
    )

    monkeypatch.setattr(
        "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
        AsyncMock(side_effect=OSError("journal directory unavailable")),
    )
    blocked = await ss_mod.ensure_session_loaded(session_id)
    assert blocked is not None
    assert blocked.recovery_blocked is True
    assert session_id in ss_mod._recovery_blocked_sessions

    async def clean_recover(**kw):
        return RecoveryResult(
            recovered=True,
            recovered_transactions=["restore-ok"],
            failed_transactions=[],
            errors=[],
            mutations_blocked=False,
        )

    monkeypatch.setattr(
        "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
        clean_recover,
    )
    cleared = await ss_mod.ensure_session_loaded(session_id)
    assert cleared is not None
    assert cleared.recovery_blocked is False
    assert session_id not in ss_mod._recovery_blocked_sessions
    tx = cleared.snapshot_transaction or {}
    assert "recovery_error" not in tx
    assert_recovery_clear(cleared)
    assert_recovery_clear(session_id=session_id)

    sessions.pop(session_id, None)


def test_mark_recovery_blocked_merges_snapshot_transaction():
    """mark_recovery_blocked must not wipe restore/capture fields."""
    from backend.services.session_store import (
        mark_recovery_blocked,
        _recovery_blocked_sessions,
    )

    sid = f"sess-merge-{uuid.uuid4().hex[:8]}"
    state = AnalysisState(
        id=sid, user_id="u1", name="t", initial_instruction="t",
        status=TaskStatus.WAITING_INPUT,
        snapshot_transaction={
            "transaction_id": "tx-restore-1",
            "status": "applying",
            "snapshot_id": "snap-9",
        },
    )
    try:
        mark_recovery_blocked(
            sid, state,
            {"code": "RECOVERY_BLOCKED", "message": "HMAC mismatch"},
        )
        tx = state.snapshot_transaction
        assert tx["transaction_id"] == "tx-restore-1"
        assert tx["status"] == "applying"
        assert tx["snapshot_id"] == "snap-9"
        assert tx["recovery_error"]["code"] == "RECOVERY_BLOCKED"
        assert state.recovery_blocked is True
        assert state.status == TaskStatus.PAUSED
    finally:
        _recovery_blocked_sessions.discard(sid)


def test_mark_recovery_blocked_preserves_terminal_status():
    """Do not force PAUSED when session is already COMPLETED/FAILED."""
    from backend.services.session_store import (
        mark_recovery_blocked,
        _recovery_blocked_sessions,
    )

    for terminal in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        sid = f"sess-term-{uuid.uuid4().hex[:8]}"
        state = AnalysisState(
            id=sid, user_id="u1", name="t", initial_instruction="t",
            status=terminal,
        )
        try:
            mark_recovery_blocked(
                sid, state,
                {"code": "RECOVERY_UNEXPECTED", "message": "boom"},
            )
            assert state.status == terminal
            assert state.recovery_blocked is True
            assert state.snapshot_transaction["recovery_error"]["code"] == (
                "RECOVERY_UNEXPECTED"
            )
        finally:
            _recovery_blocked_sessions.discard(sid)


@pytest.mark.asyncio
async def test_second_cache_branch_persists_recovery_marker(tmp_path, monkeypatch):
    """Second in-lock cache hit must persist recovery_blocked like the first branch."""
    session_id, ss_mod, sessions = _prepare_session_on_disk(
        tmp_path, monkeypatch, status="running",
    )

    live = AnalysisState(
        id=session_id, user_id="u1", name="blocked",
        initial_instruction="t", status=TaskStatus.RUNNING,
        snapshot_transaction={"transaction_id": "keep-me", "status": "running"},
    )
    sessions[session_id] = live

    # Outer `if session_id in sessions` misses once; in-lock check hits.
    real_sessions = sessions
    first_check = {"done": False}

    class _SessionsProxy(dict):
        def __contains__(self, key):
            if key == session_id and not first_check["done"]:
                first_check["done"] = True
                return False
            return key in real_sessions

        def get(self, key, default=None):
            return real_sessions.get(key, default)

        def __getitem__(self, key):
            return real_sessions[key]

        def pop(self, key, *args):
            return real_sessions.pop(key, *args)

    monkeypatch.setattr(ss_mod, "sessions", _SessionsProxy())
    monkeypatch.setattr(
        "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
        AsyncMock(side_effect=OSError("boom")),
    )

    state = await ss_mod.ensure_session_loaded(session_id)
    assert state is not None
    assert state.recovery_blocked is True
    assert state.snapshot_transaction.get("transaction_id") == "keep-me"
    assert state.snapshot_transaction["recovery_error"]["code"] == "RECOVERY_UNEXPECTED"

    disk = ss_mod.load_session_from_disk(session_id)
    assert disk is not None
    assert disk.recovery_blocked is True
    assert disk.snapshot_transaction["recovery_error"]["code"] == "RECOVERY_UNEXPECTED"
    assert disk.snapshot_transaction.get("transaction_id") == "keep-me"

    real_sessions.pop(session_id, None)
    ss_mod._recovery_blocked_sessions.discard(session_id)


@pytest.mark.asyncio
async def test_assert_recovery_clear_blocks_workflow_and_kernel(tmp_path, monkeypatch):
    """Unresolved APPLYING / recovery_blocked raises before workflow/kernel start."""
    from backend.services.session_store import assert_recovery_clear
    from backend.services.snapshot_v2.errors import SnapshotTransactionError

    state = AnalysisState(
        id="s1", user_id="u1", name="t",
        initial_instruction="t", recovery_blocked=True,
    )
    with pytest.raises(SnapshotTransactionError) as exc_info:
        assert_recovery_clear(state)
    assert exc_info.value.code in (
        "RECOVERY_BLOCKED",
        "MUTATIONS_BLOCKED",
        "APPLYING_UNRESOLVED",
    )


@pytest.mark.asyncio
async def test_lifecycle_run_and_resume_refuse_when_recovery_blocked():
    """Lifecycle start/resume must call assert_recovery_clear."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from fastapi import HTTPException
    from backend.routes import lifecycle
    from backend.services.session_store import mark_recovery_blocked, _recovery_blocked_sessions

    sid = f"sess-life-guard-{uuid.uuid4().hex[:8]}"
    state = AnalysisState(
        id=sid, user_id="u1", name="t",
        initial_instruction="t", status=TaskStatus.PAUSED,
        recovery_blocked=True,
    )
    mark_recovery_blocked(sid, state, {"code": "RECOVERY_UNEXPECTED", "message": "x"})

    req = MagicMock()
    req.state = MagicMock(user_id="u1")
    try:
        with patch.object(
            lifecycle, "require_session_ownership", new=AsyncMock(return_value=state)
        ):
            with patch.object(lifecycle, "validate_session_id"):
                with pytest.raises(HTTPException) as ei:
                    await lifecycle.run_analysis(sid, req)
                assert ei.value.status_code == 409

                resp = await lifecycle._resume_analysis_locked(sid, "u1")
                assert resp.success is False
                assert resp.data.get("recovery_blocked") is True
    finally:
        _recovery_blocked_sessions.discard(sid)


@pytest.mark.asyncio
async def test_workflow_construction_and_kernel_start_refuse_when_recovery_blocked(
    tmp_path, monkeypatch,
):
    """Workflow __init__ and sandbox.start guard via assert_recovery_clear."""
    from backend.services.session_store import mark_recovery_blocked, _recovery_blocked_sessions
    from backend.services.snapshot_v2.errors import SnapshotTransactionError
    from backend.utils.sandbox import LocalJupyterSandbox

    sid = f"sess-wf-guard-{uuid.uuid4().hex[:8]}"
    mark_recovery_blocked(sid, None, {"code": "RECOVERY_UNEXPECTED", "message": "x"})
    try:
        from backend.workflow.core import SingleCellAnalysisWorkflow
        from backend.config._config import SystemConfig

        with pytest.raises(SnapshotTransactionError, match="RECOVERY_BLOCKED"):
            SingleCellAnalysisWorkflow(config=SystemConfig(), session_id=sid)

        sb = LocalJupyterSandbox(
            session_id=sid,
            state_dir=str(tmp_path / "states"),
            working_dir=str(tmp_path),
        )
        with pytest.raises(SnapshotTransactionError, match="RECOVERY_BLOCKED"):
            await sb.start()
    finally:
        _recovery_blocked_sessions.discard(sid)


@pytest.mark.asyncio
async def test_lifespan_recovery_surfaces_mutations_blocked(tmp_path, monkeypatch):
    """Startup recover_for_all_sessions propagates mutations_blocked."""
    from backend.config._config import SESSION_DIR
    from backend.services.snapshot_v2.recovery import recover_for_all_sessions

    sid = f"sess-life-{uuid.uuid4().hex[:8]}"
    sd = SESSION_DIR / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "session.json").write_text(json.dumps({
        "id": sid, "user_id": "u1", "name": "t",
    }))

    async def fake_recover(**kw):
        return RecoveryResult(
            recovered=False,
            recovered_transactions=[],
            failed_transactions=["restore-tampered"],
            errors=["tampered APPLYING journal"],
            mutations_blocked=True,
        )

    monkeypatch.setattr(
        "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
        fake_recover,
    )

    combined = await recover_for_all_sessions()
    assert combined.mutations_blocked is True
    assert "restore-tampered" in combined.failed_transactions
