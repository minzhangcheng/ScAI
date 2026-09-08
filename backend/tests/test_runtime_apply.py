import json
import shutil

import pytest

from backend.config import get_session_states
from backend.utils import runtime_apply


def _integrity_dir_path(sid):
    from backend.utils.sandbox import _integrity_dir
    return _integrity_dir(sid)





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
    ns.session_path = states / f"{sid}_session.pkl"
    ns.state_path = states / f"{sid}_state.json"
    ns.state_dir = states
    ns.runtime_persistence = {}
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
        assert live.exists()
        assert live.read_bytes() == b"KEEP-ME"
        assert _verify_pkl_integrity(live, _integrity_dir(sid)) is True
        assert runtime_apply.is_forward_resume_blocked("intentionally_kept") is False
    finally:
        shutil.rmtree(states, ignore_errors=True)
        shutil.rmtree(_integrity_dir(sid), ignore_errors=True)


def test_retirement_takes_the_executor_checkpoint_with_it():
    """Spec §3.11: an empty namespace under a forty-step history is not a
    fresh start.

    Retirement clears the dump and the graph position and has the agent told
    "KERNEL STATE IS EMPTY. Load the input files". The executor's own step
    checkpoint was left where it was, and `validate_checkpoint` compares only
    `round_number` and `planning_stage` -- both of which retirement
    preserves -- so the next run restored `step_number`, the whole message
    history and `code_history` on top of a kernel holding nothing they refer
    to.
    """
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    sid = "sess-retire-executor-checkpoint"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"GENERATION")
    _write_pkl_signature(live, _integrity_dir(sid))
    executor_checkpoint = states / f"{sid}_bioinfo_checkpoint.json"
    executor_checkpoint.write_text(
        json.dumps({"step_number": 40, "round_number": 2})
    )

    assert runtime_apply.retire_live_runtime(sid, include_checkpoint=True) is True

    assert not executor_checkpoint.exists()
    quarantined = states / f"{sid}_bioinfo_checkpoint.json.quarantine"
    assert json.loads(quarantined.read_text())["step_number"] == 40


def test_a_runtime_only_retirement_leaves_the_executor_checkpoint_alone():
    """`include_checkpoint=False` retires the generation, not the position.

    The two callers that pass it that way are re-running from data with the
    graph position deliberately kept; taking the executor's position with the
    dump there would change what they mean.
    """
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    sid = "sess-retire-runtime-only"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live = states / f"{sid}_session.pkl"
    live.write_bytes(b"GENERATION")
    _write_pkl_signature(live, _integrity_dir(sid))
    executor_checkpoint = states / f"{sid}_bioinfo_checkpoint.json"
    executor_checkpoint.write_text(json.dumps({"step_number": 7}))

    runtime_apply.retire_live_runtime(sid)

    assert json.loads(executor_checkpoint.read_text())["step_number"] == 7
