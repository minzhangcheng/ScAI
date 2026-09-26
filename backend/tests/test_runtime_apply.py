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


def test_a_generation_pair_must_match_its_signed_checkpoint_identity():
    """A complete-looking pair may still belong to another graph position."""
    import sqlite3

    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        runtime_manifest_path,
        write_runtime_manifest,
    )
    from backend.utils.runtime_pairing import checkpoint_identity_at_path

    sid = "sess-pair-signed-identity"
    states, pair = _generation_pair(sid, 44)
    checkpoint = pair / "checkpoints.sqlite"
    checkpoint.unlink()
    with sqlite3.connect(checkpoint) as conn:
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
            "checkpoint_id TEXT, checkpoint BLOB)"
        )
        conn.execute(
            "INSERT INTO checkpoints VALUES (?, '', 'c1', X'00')", (sid,)
        )
    expected = checkpoint_identity_at_path(sid, checkpoint)
    write_runtime_manifest(
        runtime_manifest_path(sid, states_dir=states),
        build_runtime_manifest(
            revision=44,
            state_revision=12,
            pkl_digest="d" * 64,
            saved_names=[],
            skipped_names=[],
            serializer_version="dill-zstd-1",
            pairing={"state_version": 12, "checkpoint": expected},
        ),
    )
    assert runtime_apply.generation_pair_is_restorable(sid, 44) is True

    with sqlite3.connect(checkpoint) as conn:
        conn.execute(
            "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, "
            "checkpoint_id TEXT, task_id TEXT, idx INTEGER, value BLOB)"
        )
        conn.execute(
            "INSERT INTO writes VALUES (?, '', 'c1', 'task', 0, X'01')",
            (sid,),
        )
    live_state = states / f"{sid}_state.json"
    live_state.write_text(json.dumps({"session_id": sid, "marker": "live"}))

    assert runtime_apply.generation_pair_is_restorable(sid, 44) is False
    assert runtime_apply.restore_generation_pair(sid, revision=44) is False
    assert json.loads(live_state.read_text())["marker"] == "live"



# The gap this closes was found by rolling back a real crashed session: eight
# files written between the last save and the kill survived the restore, among
# them adata_annotated.h5ad. The kernel no longer knew it had made them, and a
# later step that loads an intermediate by name would have read one produced by
# work the rollback had just discarded.


def _generation_workspace(sid, pair, files, *, revision=44, generation_root=None):
    """Give a generation the workspace copy a save would have left it."""
    from backend.config import get_session_workspace
    from backend.utils.sandbox_parts import workspace_generation

    workspace = get_session_workspace(sid)
    workspace.mkdir(parents=True, exist_ok=True)
    for relative, body in files.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    root = generation_root if generation_root is not None else pair.parent
    workspace_generation.publish_workspace_generation(
        workspace, pair, root, revision,
    )
    return workspace


def test_the_rollback_never_reaches_outside_the_workspace(tmp_path):
    """A symlink is not a regular file, and is left where it is."""
    sid = "sess-pair-symlink"
    _states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, pair, {"table.csv": b"gene\n"})

    outside = tmp_path / "precious.bin"
    outside.write_bytes(b"not ours to delete")
    (workspace / "link.bin").symlink_to(outside)

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert outside.read_bytes() == b"not ours to delete"
    assert (workspace / "link.bin").is_symlink()




def test_an_interrupted_restore_still_reads_as_needing_one(monkeypatch):
    """The declaration is published last, so a second crash is recoverable.

    There is no journal here, and adding one would duplicate the commit
    transaction for three small files. The ordering does the same job: the
    record that says "this generation is the current one" goes last, so a
    crash between publishes leaves a state that is still newer than the
    generation -- coherence still reads `stale`, the next start attempts the
    restore again, and the restore is idempotent.

    The reverse order is the dangerous one: a record claiming the older
    generation is current, beside a graph position from the newer work, is a
    mismatch nothing downstream can detect.
    """
    sid = "sess-pair-restore-interrupted"
    states, _pair = _generation_pair(sid, 44)
    live_state = states / f"{sid}_state.json"
    live_state.write_text(json.dumps({
        "session_id": sid,
        "runtime_persistence": {
            "current_revision": 61,
            "committed_revision": 44,
            "last_saved_at": "2026-09-08T08:10:19",
        },
    }))
    (states / "checkpoints.sqlite").write_bytes(b"the position the crash left")

    real_replace = runtime_apply.os.replace
    published = []

    def flaky_replace(src, dst):
        published.append(str(dst))
        if len(published) == 2:
            raise OSError("interrupted between publishes")
        return real_replace(src, dst)

    monkeypatch.setattr(runtime_apply.os, "replace", flaky_replace)

    with pytest.raises(OSError):
        runtime_apply.restore_generation_pair(sid, revision=44)

    record = json.loads(live_state.read_text())["runtime_persistence"]
    assert record["current_revision"] == 61, (
        "the state record must not be the first thing published"
    )
