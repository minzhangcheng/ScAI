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


def _generation_pair(sid, revision, *, with_executor=True):
    """A committed generation's captured partner, as B2-1 writes it."""
    import sqlite3

    states = get_session_states(sid)
    pair = states / "generation" / str(revision)
    pair.mkdir(parents=True, exist_ok=True)
    (pair / "state.json").write_text(
        json.dumps({"session_id": sid, "state_version": 12, "marker": "paired"})
    )
    with sqlite3.connect(pair / "checkpoints.sqlite") as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, blob TEXT)")
        conn.execute("INSERT INTO checkpoints VALUES (?, 'paired-position')", (sid,))
    if with_executor:
        (pair / "executor_checkpoint.json").write_text(
            json.dumps({"step_number": 20, "runtime_revision": revision})
        )
    return states, pair


def test_restoring_a_pair_puts_the_positions_back_beside_the_dump():
    """Spec B2-2: the generation is healthy; what moved on is everything else.

    After a crash the state, the graph position and the executor position have
    all advanced past the newest committed dump. Restoring the pair is what
    makes that dump continuable instead of something to be retired.
    """
    sid = "sess-pair-restore"
    states, _pair = _generation_pair(sid, 44)

    live_state = states / f"{sid}_state.json"
    live_state.write_text(json.dumps({"session_id": sid, "marker": "live-newer"}))
    live_checkpoint = states / "checkpoints.sqlite"
    live_checkpoint.write_bytes(b"newer-and-unusable")
    (states / "checkpoints.sqlite-wal").write_bytes(b"wal-of-the-newer-db")
    (states / "checkpoints.sqlite-shm").write_bytes(b"shm-of-the-newer-db")
    live_executor = states / f"{sid}_bioinfo_checkpoint.json"
    live_executor.write_text(json.dumps({"step_number": 61}))

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert json.loads(live_state.read_text())["marker"] == "paired"
    assert json.loads(live_executor.read_text())["step_number"] == 20

    import sqlite3

    with sqlite3.connect(live_checkpoint) as conn:
        assert conn.execute("SELECT blob FROM checkpoints").fetchone()[0] == "paired-position"

    # The write-ahead log belonged to the database that was just replaced.
    assert not (states / "checkpoints.sqlite-wal").exists()
    assert not (states / "checkpoints.sqlite-shm").exists()


def test_a_pair_without_an_executor_position_clears_the_live_one():
    """A round boundary reached after the executor finished has no step
    checkpoint, and the live one is then newer than the generation.

    Leaving it is the §3.11 failure in a new place: an executor position with
    no namespace to match it.
    """
    sid = "sess-pair-restore-no-executor"
    states, _pair = _generation_pair(sid, 12, with_executor=False)
    (states / f"{sid}_state.json").write_text(json.dumps({"session_id": sid}))
    live_executor = states / f"{sid}_bioinfo_checkpoint.json"
    live_executor.write_text(json.dumps({"step_number": 61}))

    assert runtime_apply.restore_generation_pair(sid, revision=12) is True

    assert not live_executor.exists()
    assert json.loads(
        (states / f"{sid}_bioinfo_checkpoint.json.quarantine").read_text()
    )["step_number"] == 61


def test_a_missing_pair_changes_nothing():
    """Retirement is still the fallback, and it has to be reachable."""
    sid = "sess-pair-absent"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    live_state = states / f"{sid}_state.json"
    live_state.write_text(json.dumps({"marker": "live"}))

    assert runtime_apply.restore_generation_pair(sid, revision=99) is False
    assert json.loads(live_state.read_text())["marker"] == "live"


def test_an_incomplete_pair_is_refused_rather_than_half_applied():
    """Half a restore point is not a restore point."""
    sid = "sess-pair-incomplete"
    states, pair = _generation_pair(sid, 7)
    (pair / "checkpoints.sqlite").unlink()
    live_state = states / f"{sid}_state.json"
    live_state.write_text(json.dumps({"marker": "live"}))

    assert runtime_apply.restore_generation_pair(sid, revision=7) is False
    assert json.loads(live_state.read_text())["marker"] == "live"


def test_a_pair_restore_records_what_it_discarded():
    """Spec B2-3: continuing from an earlier generation is not free.

    The restored record reads `current` again -- current_revision equals
    committed_revision, because that is what the pair holds -- so without this
    the UI would go quiet at the exact moment the user most needs telling. The
    work between the restored generation and the interruption is gone, and the
    disclosure has to say so and from when.
    """
    sid = "sess-pair-restore-disclosure"
    states, _pair = _generation_pair(sid, 44)

    (states / f"{sid}_state.json").write_text(json.dumps({
        "session_id": sid,
        "runtime_persistence": {
            "status": "live_dirty",
            "current_revision": 61,
            "committed_revision": 44,
            "last_saved_at": "2026-09-08T08:10:19",
        },
    }))

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    record = json.loads(
        (states / f"{sid}_state.json").read_text()
    )["runtime_persistence"]
    restore = record["pair_restore"]
    assert restore["restored_revision"] == 44
    assert restore["discarded_revisions"] == 17
    assert restore["discarded_since"] == "2026-09-08T08:10:19"
    assert restore["at"]


def test_a_pair_restore_disclosure_survives_only_until_the_next_commit():
    """It describes one event, and the next successful save ends it.

    Left in place it would accumulate into the permanent warning 0.55.4 took
    out of the banner; cleared too early it would vanish before the user read
    it. The next committed generation is the point at which the session is
    genuinely current again.
    """
    from backend.utils.sandbox import NotebookState

    sid = "sess-pair-restore-cleared"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    notebook = NotebookState(sid, str(states))
    notebook.runtime_persistence["pair_restore"] = {"restored_revision": 44}

    notebook.update_runtime_persistence(
        "live_dirty", current_revision=45, saved_revision=44,
    )
    assert "pair_restore" in notebook.runtime_persistence, (
        "an ordinary dirty cell must not silence the disclosure"
    )

    notebook.update_runtime_persistence(
        "current", current_revision=45, saved_revision=45,
        commit_source="regular_save", runtime_digest="b" * 64,
    )
    assert notebook.runtime_persistence.get("pair_restore") is None
