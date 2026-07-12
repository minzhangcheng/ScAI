import json
import shutil
from pathlib import Path

import pytest

from backend.config import get_session_states
from backend.models.schemas import Snapshot, AnalysisState
from backend.utils.snapshot import SnapshotManager
from backend.utils import runtime_apply


def _integrity_dir_path(sid):
    from backend.utils.sandbox import _integrity_dir
    return _integrity_dir(sid)


def _snapshot(runtime_included: bool) -> Snapshot:
    return Snapshot(name="t", phase="manual", runtime_state_included=runtime_included)


def _write_zst(path: Path, payload: bytes = b"hello-runtime") -> None:
    import zstandard as zstd
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(zstd.ZstdCompressor().compress(payload))


def _write_v1_state_json(mgr: SnapshotManager, state: AnalysisState, snap: Snapshot) -> None:
    """Hand-written V1 state JSON for restore tests (no create_snapshot)."""
    mgr._ensure_dir()
    snap.state_file = f"{snap.id}.json"
    state_dict = state.model_dump(mode="json")
    state_dict.pop("snapshots", None)
    state_dict.pop("realtime_messages", None)
    (mgr.snapshots_dir / snap.state_file).write_text(
        json.dumps(state_dict, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def test_status_state_only_when_flag_false():
    snap = _snapshot(False)
    assert runtime_apply.compute_runtime_status(snap, "sess-status-a") == "state_only"


def test_status_missing_when_flag_true_but_no_file():
    snap = _snapshot(True)
    assert runtime_apply.compute_runtime_status(snap, "sess-status-b") == "missing"


def test_status_complete_when_valid_zst_present():
    sid = "sess-status-c"
    snap = _snapshot(True)
    zst = SnapshotManager(sid).snapshots_dir / f"{snap.id}_session.pkl.zst"
    _write_zst(zst)
    try:
        assert runtime_apply.compute_runtime_status(snap, sid) == "complete"
    finally:
        shutil.rmtree(zst.parent, ignore_errors=True)


def test_status_corrupt_when_zst_magic_broken():
    sid = "sess-status-d"
    snap = _snapshot(True)
    zst = SnapshotManager(sid).snapshots_dir / f"{snap.id}_session.pkl.zst"
    _write_zst(zst)
    data = bytearray(zst.read_bytes())
    data[0] ^= 0xFF  # break the zstd magic
    zst.write_bytes(bytes(data))
    try:
        assert runtime_apply.compute_runtime_status(snap, sid) == "corrupt"
    finally:
        shutil.rmtree(zst.parent, ignore_errors=True)


def test_resolve_complete_ignores_choice():
    assert runtime_apply.resolve_runtime_choice("complete", None) == ("apply", "complete")
    assert runtime_apply.resolve_runtime_choice("complete", "keep") == ("apply", "complete")


def test_resolve_unavailable_without_choice_raises():
    for status in ("missing", "state_only", "corrupt"):
        with pytest.raises(runtime_apply.RuntimeChoiceRequired):
            runtime_apply.resolve_runtime_choice(status, None)


def test_resolve_reset_and_keep():
    assert runtime_apply.resolve_runtime_choice("missing", "reset") == ("reset", "unavailable")
    assert runtime_apply.resolve_runtime_choice("missing", "keep") == ("keep", "intentionally_kept")


def test_resolve_rejects_unknown_choice():
    with pytest.raises(ValueError):
        runtime_apply.resolve_runtime_choice("missing", "bogus")


def _make_snapshot_zst(sid: str, payload: bytes) -> Path:
    import zstandard as zstd
    zst = SnapshotManager(sid).snapshots_dir / "src_session.pkl.zst"
    zst.parent.mkdir(parents=True, exist_ok=True)
    with open(zst, "wb") as f:
        f.write(zstd.ZstdCompressor().compress(payload))
    return zst


def test_apply_materializes_pkl_and_signature_passes():
    from backend.utils.sandbox import _verify_pkl_integrity, _integrity_dir
    sid = "sess-apply-a"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    payload = b"restored-kernel-bytes"
    zst = _make_snapshot_zst(sid, payload)
    try:
        runtime_apply.apply_runtime_pkl(sid, zst)
        live = states / f"{sid}_session.pkl"
        assert live.read_bytes() == payload
        assert _verify_pkl_integrity(live, _integrity_dir(sid)) is True
        assert not (states / "runtime_apply.journal.json").exists()
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


def test_recover_rolls_back_interrupted_apply():
    from backend.utils.sandbox import _write_pkl_signature, _verify_pkl_integrity, _integrity_dir
    sid = "sess-apply-b"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"ORIGINAL")
    _write_pkl_signature(live, _integrity_dir(sid))
    sig = _integrity_dir(sid) / (live.name + ".sig")
    bak_pkl = live.with_suffix(live.suffix + ".bak")
    bak_sig = sig.with_suffix(sig.suffix + ".bak")
    shutil.copy2(live, bak_pkl)
    shutil.copy2(sig, bak_sig)
    live.write_bytes(b"HALF-APPLIED")
    sig.write_text("stale-new-signature")
    journal = {
        "state": "APPLYING", "had_live": True,
        "live": str(live),
        "sig": str(sig),
        "bak_pkl": str(bak_pkl), "bak_sig": str(bak_sig),
    }
    jp = _integrity_dir(sid) / "runtime_apply.journal.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(journal))
    try:
        runtime_apply.recover_runtime_apply_journal(sid)
        assert live.read_bytes() == b"ORIGINAL"
        assert _verify_pkl_integrity(live, _integrity_dir(sid)) is True
        assert not jp.exists()
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


@pytest.mark.asyncio
async def test_restore_snapshot_apply_action_signs_live_pkl(tmp_path, monkeypatch):
    import zstandard as zstd
    from backend.utils.sandbox import _verify_pkl_integrity, _integrity_dir

    sid = "sess-restore-apply"
    mgr = SnapshotManager(sid)
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)

    state = AnalysisState(session_id=sid, user_id="u1", current_round=2)
    snap = Snapshot(name="t", phase="manual", round_number=2)
    _write_v1_state_json(mgr, state, snap)
    zst = mgr.snapshots_dir / f"{snap.id}_session.pkl.zst"
    with open(zst, "wb") as f:
        f.write(zstd.ZstdCompressor().compress(b"snap-kernel"))
    snap.runtime_state_included = True

    try:
        await mgr.restore_snapshot(snap, runtime_action="apply")
        live = states / f"{sid}_session.pkl"
        assert live.read_bytes() == b"snap-kernel"
        assert _verify_pkl_integrity(live, _integrity_dir(sid)) is True
    finally:
        shutil.rmtree(mgr.snapshots_dir.parent, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


@pytest.mark.asyncio
async def test_restore_snapshot_reset_action_quarantines_live(tmp_path):
    from backend.utils.sandbox import _write_pkl_signature, _integrity_dir

    sid = "sess-restore-reset"
    mgr = SnapshotManager(sid)
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"STALE-LIVE")
    _write_pkl_signature(live, _integrity_dir(sid))

    state = AnalysisState(session_id=sid, user_id="u1")
    snap = Snapshot(name="t", phase="manual")
    _write_v1_state_json(mgr, state, snap)
    try:
        await mgr.restore_snapshot(snap, runtime_action="reset")
        assert not live.exists()
        assert live.with_suffix(live.suffix + ".quarantine").exists()
    finally:
        shutil.rmtree(mgr.snapshots_dir.parent, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


def test_verify_live_pkl_or_raise_passes_when_absent():
    sid = "sess-verify-absent"
    get_session_states(sid).mkdir(parents=True, exist_ok=True)
    try:
        runtime_apply.verify_live_pkl_or_raise(sid)  # no pkl → no raise
    finally:
        shutil.rmtree(get_session_states(sid), ignore_errors=True)


def test_is_forward_resume_blocked():
    assert runtime_apply.is_forward_resume_blocked("unavailable") is True
    assert runtime_apply.is_forward_resume_blocked("complete") is False
    assert runtime_apply.is_forward_resume_blocked("intentionally_kept") is False


def test_verify_live_pkl_or_raise_raises_on_bad_signature():
    sid = "sess-verify-bad"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    (states / f"{sid}_session.pkl").write_bytes(b"UNSIGNED")
    try:
        with pytest.raises(runtime_apply.RuntimeApplyError):
            runtime_apply.verify_live_pkl_or_raise(sid)
    finally:
        shutil.rmtree(states, ignore_errors=True)


def test_recover_had_live_false_preserves_valid_later_pkl():
    """A stale had_live=False APPLYING journal must NOT delete a live pkl that
    currently verifies — that pkl is a legitimate later save, not apply garbage."""
    from backend.utils.sandbox import _write_pkl_signature, _verify_pkl_integrity, _integrity_dir
    sid = "sess-recover-preserve"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"LATER-GOOD")
    _write_pkl_signature(live, _integrity_dir(sid))
    sig = _integrity_dir(sid) / (live.name + ".sig")
    journal = {
        "state": "APPLYING", "had_live": False,
        "live": str(live), "sig": str(sig),
        "bak_pkl": None, "bak_sig": None,
    }
    jp = _integrity_dir(sid) / "runtime_apply.journal.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(journal))
    try:
        runtime_apply.recover_runtime_apply_journal(sid)
        assert live.read_bytes() == b"LATER-GOOD"
        assert _verify_pkl_integrity(live, _integrity_dir(sid)) is True
        assert not jp.exists()
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


@pytest.mark.asyncio
async def test_restore_session_recovers_journal_even_without_dump():
    """_restore_session must recover a lingering apply journal even when there is
    no session dump (has_session_dump() False), so the journal cannot survive to a
    later start."""
    from unittest.mock import MagicMock
    from backend.utils.sandbox import LocalJupyterSandbox

    sid = "sess-recover-nodump"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    journal = {
        "state": "APPLYING", "had_live": False,
        "live": str(states / f"{sid}_session.pkl"),
        "sig": str(_integrity_dir_path(sid) / (f"{sid}_session.pkl.sig")),
        "bak_pkl": None, "bak_sig": None,
    }
    jp = _integrity_dir_path(sid) / "runtime_apply.journal.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(journal))

    sb = LocalJupyterSandbox.__new__(LocalJupyterSandbox)
    sb.session_id = sid
    ns = MagicMock()
    ns.load = MagicMock()
    ns.has_session_dump = MagicMock(return_value=False)
    sb.notebook_state = ns
    try:
        await sb._restore_session()
        assert not jp.exists()
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir_path(sid), ignore_errors=True)


@pytest.mark.asyncio
async def test_restore_session_sets_flag_on_invalid_pkl():
    """A present pkl with a missing/invalid HMAC must abort startup AND set
    _session_restore_failed, symmetric with the dill-restore-failure path."""
    from unittest.mock import MagicMock
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.utils.runtime_apply import RuntimeApplyError

    sid = "sess-flag-invalid"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"PRESENT-BUT-UNSIGNED")

    sb = LocalJupyterSandbox.__new__(LocalJupyterSandbox)
    sb.session_id = sid
    sb._session_restore_failed = False
    ns = MagicMock()
    ns.load = MagicMock()
    ns.has_session_dump = MagicMock(return_value=True)
    ns.session_path = live
    sb.notebook_state = ns
    try:
        with pytest.raises(RuntimeApplyError):
            await sb._restore_session()
        assert sb._session_restore_failed is True
    finally:
        shutil.rmtree(states, ignore_errors=True)


def test_recover_had_live_false_deletes_unsigned_garbage():
    """Genuine rollback: had_live=False and the live pkl does NOT verify (real
    apply garbage) → it is removed."""
    sid = "sess-recover-garbage"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"HALF-APPLIED-GARBAGE")
    sig = _integrity_dir_path(sid) / (live.name + ".sig")
    journal = {
        "state": "APPLYING", "had_live": False,
        "live": str(live), "sig": str(sig),
        "bak_pkl": None, "bak_sig": None,
    }
    jp = _integrity_dir_path(sid) / "runtime_apply.journal.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(journal))
    try:
        runtime_apply.recover_runtime_apply_journal(sid)
        assert not live.exists()
        assert not jp.exists()
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir_path(sid), ignore_errors=True)


# ---- Runtime policy helpers (Batch 2 Task 3) --------------------------------


def test_unavailable_reset_sets_status_and_quarantines():
    """reset choice quarantines live pair and yields runtime_restore_status unavailable."""
    from backend.utils.sandbox import _write_pkl_signature, _integrity_dir
    sid = "sess-policy-reset"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"LIVE-TO-QUARANTINE")
    _write_pkl_signature(live, _integrity_dir(sid))
    try:
        action, status = runtime_apply.resolve_runtime_choice("missing", "reset")
        assert action == "reset"
        assert status == "unavailable"
        runtime_apply.quarantine_live_runtime(sid)
        assert not live.exists()
        assert live.with_suffix(live.suffix + ".quarantine").exists()
        assert runtime_apply.is_forward_resume_blocked("unavailable") is True
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


def test_unavailable_keep_does_not_mutate_live():
    """keep choice leaves live runtime and marks intentionally_kept."""
    from backend.utils.sandbox import _write_pkl_signature, _integrity_dir, _verify_pkl_integrity
    sid = "sess-policy-keep"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"KEEP-ME")
    _write_pkl_signature(live, _integrity_dir(sid))
    try:
        action, status = runtime_apply.resolve_runtime_choice("state_only", "keep")
        assert action == "keep"
        assert status == "intentionally_kept"
        assert live.exists()
        assert live.read_bytes() == b"KEEP-ME"
        assert _verify_pkl_integrity(live, _integrity_dir(sid)) is True
        assert runtime_apply.is_forward_resume_blocked("intentionally_kept") is False
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


def test_complete_status_ignores_runtime_choice_reset():
    """complete status always maps to apply, even if caller passes reset."""
    action, status = runtime_apply.resolve_runtime_choice("complete", "reset")
    assert action == "apply"
    assert status == "complete"
