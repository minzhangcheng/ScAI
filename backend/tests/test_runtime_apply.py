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


# ------------------------------------------------- the workspace, on rollback

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


def test_a_rollback_puts_back_a_file_the_analysis_overwrote():
    """The case a file list cannot serve, and the reason the copy exists.

    A list can take away what was added. It cannot return the contents of a
    file that existed at the save and was rewritten afterwards, and rewriting
    in place is what h5py does when analysis code writes an .h5ad to a path it
    has used before.
    """
    sid = "sess-ws-overwrite"
    _states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, pair, {"table.csv": b"at the save\n"})

    (workspace / "table.csv").write_bytes(b"rewritten after the save\n")

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert (workspace / "table.csv").read_bytes() == b"at the save\n"


def test_a_rollback_removes_the_files_written_after_the_generation():
    sid = "sess-pair-workspace"
    _states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(
        sid, pair, {"figures/umap.png": b"at the save", "table.csv": b"gene\n"},
    )

    # What the run wrote between that save and the crash.
    (workspace / "adata_annotated.h5ad").write_bytes(b"from the discarded future")
    (workspace / "figures" / "later.png").write_bytes(b"also later")

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert (workspace / "figures" / "umap.png").read_bytes() == b"at the save"
    assert (workspace / "table.csv").is_file()
    assert not (workspace / "adata_annotated.h5ad").exists()
    assert not (workspace / "figures" / "later.png").exists()


def test_a_file_nothing_touched_is_left_where_it_is():
    """The common case, and it must cost no copy."""
    sid = "sess-ws-untouched"
    _states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, pair, {"big.bin": b"y" * 100_000})
    before = (workspace / "big.bin").lstat()

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    after = (workspace / "big.bin").lstat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_a_directory_left_empty_by_the_rollback_goes_with_it():
    sid = "sess-pair-empty-dir"
    _states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, pair, {"table.csv": b"gene\n"})
    (workspace / "scratch" / "deep").mkdir(parents=True)
    (workspace / "scratch" / "deep" / "later.bin").write_bytes(b"later")

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert not (workspace / "scratch").exists()
    assert workspace.is_dir(), "the workspace root itself is never removed"


def test_a_pair_with_no_workspace_copy_leaves_the_workspace_alone():
    """Every pair written before the copy existed has none, and must restore."""
    sid = "sess-pair-no-copy"
    _states, _pair = _generation_pair(sid, 44)
    from backend.config import get_session_workspace

    workspace = get_session_workspace(sid)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "anything.bin").write_bytes(b"untouched")

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert (workspace / "anything.bin").read_bytes() == b"untouched"


def test_a_copy_without_its_record_is_not_trusted():
    """The record is written last, so its absence means a half-built copy."""
    from backend.config import get_session_workspace
    from backend.utils.sandbox_parts.workspace_generation import (
        WORKSPACE_INVENTORY_NAME,
    )

    sid = "sess-ws-half-built"
    _states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, pair, {"table.csv": b"at the save\n"})
    (pair / WORKSPACE_INVENTORY_NAME).unlink()
    (workspace / "table.csv").write_bytes(b"newer\n")

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert (workspace / "table.csv").read_bytes() == b"newer\n"


def test_a_record_from_a_future_schema_leaves_the_workspace_alone():
    """The consequence of guessing here is deleting a file, so it does not."""
    import json as json_module

    from backend.config import get_session_workspace
    from backend.utils.sandbox_parts.workspace_generation import (
        WORKSPACE_INVENTORY_NAME,
    )

    sid = "sess-pair-future-schema"
    _states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, pair, {"table.csv": b"gene\n"})
    (pair / WORKSPACE_INVENTORY_NAME).write_text(
        json_module.dumps({"schema_version": 99, "revision": 44, "files": {}})
    )
    (workspace / "anything.bin").write_bytes(b"untouched")

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True

    assert (workspace / "anything.bin").read_bytes() == b"untouched"


def test_a_rollback_that_fails_still_leaves_the_session_restored(monkeypatch):
    """The restore is correct without the workspace; it must not undo that."""
    sid = "sess-pair-sweep-fails"
    states, pair = _generation_pair(sid, 44)
    _generation_workspace(sid, pair, {"table.csv": b"gene\n"})
    (states / f"{sid}_state.json").write_text(
        json.dumps({"session_id": sid, "marker": "live-newer"})
    )

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the volume went away")

    monkeypatch.setattr(runtime_apply, "_roll_back_workspace", _explode)

    assert runtime_apply.restore_generation_pair(sid, revision=44) is True
    assert json.loads((states / f"{sid}_state.json").read_text())["marker"] == "paired"


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


def test_a_superseded_generation_is_removed_workspace_and_all():
    """The second save is the one that finds a generation with a copy.

    Found in production, not here: a generation used to be three files and the
    prune removed them with `unlink`, which raises on the workspace directory.
    The exception left the commit, and a save that had already written its dump
    was recorded as failed. Every save after the first would have failed the
    same way, and a single-save test cannot see it because there is nothing to
    prune.
    """
    from backend.config import get_session_states
    from backend.utils.sandbox_parts.regular_save import _prune_older_generations

    sid = "sess-prune-with-workspace"
    states, first = _generation_pair(sid, 44)
    _generation_workspace(sid, first, {"big.bin": b"z" * 10_000}, revision=44)
    assert (first / "workspace").is_dir()

    second = states / "generation" / "45"
    second.mkdir(parents=True, exist_ok=True)

    _prune_older_generations(states / "generation", keep=45)

    assert not first.exists(), "the superseded generation survived the prune"
    assert second.is_dir()


def test_a_second_save_leaves_exactly_one_generation():
    """Retention is one, and the copy must not change that.

    Asserted through the two calls a save makes rather than through the commit,
    so the arithmetic of the retention is visible: publish, then prune.
    """
    from backend.config import get_session_states
    from backend.utils.sandbox_parts import workspace_generation
    from backend.utils.sandbox_parts.regular_save import _prune_older_generations

    sid = "sess-two-saves"
    states, first = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, first, {"a.bin": b"one"}, revision=44)
    _prune_older_generations(states / "generation", keep=44)

    (workspace / "b.bin").write_bytes(b"two")
    second = states / "generation" / "45"
    second.mkdir(parents=True, exist_ok=True)
    workspace_generation.publish_workspace_generation(
        workspace, second, states / "generation", 45,
    )
    _prune_older_generations(states / "generation", keep=45)

    kept = sorted(d.name for d in (states / "generation").iterdir() if d.is_dir())
    assert kept == ["45"]
    # And the surviving copy still holds the file it shared with the one that
    # was just removed: the hardlink outlives the directory it came from.
    assert (second / "workspace" / "a.bin").read_bytes() == b"one"
    assert (second / "workspace" / "b.bin").read_bytes() == b"two"


# ------------------------------------------------------ what the copy costs


def test_an_unchanged_file_is_shared_with_the_previous_generation():
    """The arithmetic this scheme exists for.

    Copying the workspace at every save moves its whole size every time. Sharing
    what has not changed moves, over a run, one workspace in total. Measured on
    a real session at 2.85 GB, 1.13 GB, then 3 MB, 10 MB and 91 MB across five
    saves of a workspace that ended at 4.08 GB.
    """
    from backend.config import get_session_workspace
    from backend.utils.sandbox_parts import workspace_generation

    sid = "sess-ws-shared"
    states, first = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, first, {"big.bin": b"z" * 50_000})

    second = states / "generation" / "45"
    second.mkdir(parents=True, exist_ok=True)
    counts = workspace_generation.stage_workspace_copy(
        workspace, second, states / "generation",
    )

    assert counts == {"linked": 1, "copied": 0, "bytes_copied": 0}
    kept = (first / "workspace" / "big.bin").lstat()
    shared = (second / "workspace" / "big.bin").lstat()
    assert kept.st_ino == shared.st_ino


def test_the_copy_is_shared_with_the_previous_generation_not_with_the_live_file():
    """The whole correctness of sharing, and it is not obvious.

    h5py opens an existing path with `w` and truncates the same inode, so a
    hardlink to a live `.h5ad` follows the rewrite. A link to the previous
    generation's own copy cannot: the live file became a different inode the
    moment it was first copied.
    """
    sid = "sess-ws-inode"
    states, pair = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, pair, {"data.bin": b"original"})

    live = (workspace / "data.bin").lstat()
    saved = (pair / "workspace" / "data.bin").lstat()
    assert live.st_ino != saved.st_ino, "the copy must not share the live inode"

    with open(workspace / "data.bin", "wb") as handle:
        handle.write(b"rewritten in place")
    assert (pair / "workspace" / "data.bin").read_bytes() == b"original"



def test_a_changed_file_is_copied_rather_than_shared():
    from backend.utils.sandbox_parts import workspace_generation

    sid = "sess-ws-changed"
    states, first = _generation_pair(sid, 44)
    workspace = _generation_workspace(sid, first, {"big.bin": b"z" * 50_000})

    import time

    time.sleep(0.01)
    (workspace / "big.bin").write_bytes(b"q" * 60_000)

    second = states / "generation" / "45"
    second.mkdir(parents=True, exist_ok=True)
    counts = workspace_generation.stage_workspace_copy(
        workspace, second, states / "generation",
    )

    assert counts["copied"] == 1 and counts["linked"] == 0
    assert (second / "workspace" / "big.bin").read_bytes() == b"q" * 60_000
    assert (first / "workspace" / "big.bin").read_bytes() == b"z" * 50_000


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


def test_pair_restore_keeps_stale_metadata_until_cleanup_can_retry(monkeypatch):
    """A cleanup failure leaves the old declaration and a retry completes."""
    sid = "sess-pair-restore-cleanup-retry"
    states, _pair = _generation_pair(sid, 12, with_executor=False)
    live_state = states / f"{sid}_state.json"
    live_state.write_text(json.dumps({
        "session_id": sid,
        "runtime_persistence": {
            "current_revision": 61,
            "committed_revision": 12,
        },
    }))
    live_checkpoint = states / "checkpoints.sqlite"
    live_checkpoint.write_bytes(b"newer-and-unusable")
    wal = live_checkpoint.with_name(live_checkpoint.name + "-wal")
    shm = live_checkpoint.with_name(live_checkpoint.name + "-shm")
    wal.write_bytes(b"stale-wal")
    shm.write_bytes(b"stale-shm")
    live_executor = states / f"{sid}_bioinfo_checkpoint.json"
    live_executor.write_text(json.dumps({"step_number": 61}))

    real_quarantine = runtime_apply._quarantine_path
    failed = False

    def fail_once(path):
        nonlocal failed
        if path == live_executor and not failed:
            failed = True
            raise OSError("cleanup interrupted")
        return real_quarantine(path)

    monkeypatch.setattr(runtime_apply, "_quarantine_path", fail_once)

    with pytest.raises(OSError, match="cleanup interrupted"):
        runtime_apply.restore_generation_pair(sid, revision=12)

    assert json.loads(live_state.read_text())["runtime_persistence"][
        "current_revision"
    ] == 61

    assert runtime_apply.restore_generation_pair(sid, revision=12) is True

    restored = json.loads(live_state.read_text())
    assert restored["runtime_persistence"]["pair_restore"]["restored_revision"] == 12
    assert not wal.exists()
    assert not shm.exists()
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
