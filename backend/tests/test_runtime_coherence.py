"""Coherence: is the newest committed generation the newest one the run needed?

Distinct from integrity (is a generation internally sound) and coverage (what it
left out). Four drafts of this were wrong, each in a way that looked right, and
the tests below pin every one of them shut:

1. the session *lifecycle generation* -- fences incarnations, normally 1 for a
   session's whole life, so the verdict could never be anything but "current";
2. saved against current *runtime* revision -- reports only whether the
   namespace is dirty between save boundaries;
3. a state revision advanced on every durable write -- which the commit's own
   state publication increments immediately afterwards, so every successful save
   reported itself stale;
4. deriving "is there a generation" by re-verifying the dump against the
   *previous* digest, which answered no for every successful commit and read a
   multi-gigabyte file twice to do it.

What survives is the save-attempt record: it moves only when a save happens and
it records whether that save succeeded.
"""
import pytest

try:
    from backend.utils.runtime_manifest import (
        COHERENCE_CURRENT,
        COHERENCE_INCONSISTENT,
        COHERENCE_STALE,
        COHERENCE_UNAVAILABLE,
        runtime_coherence,
    )
    from backend.utils.runtime_apply import (
        is_forward_resume_blocked,
        state_runtime_coherence,
    )
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


# --- the verdict -------------------------------------------------------------

def test_a_committed_generation_at_the_latest_attempt_is_current():
    assert (
        runtime_coherence(
            committed_revision=5, last_attempt_revision=5, last_attempt_status="succeeded"
        )
        == COHERENCE_CURRENT
    )


def test_a_failed_attempt_for_a_later_revision_is_stale():
    """What a failed closing save leaves behind."""
    assert (
        runtime_coherence(
            committed_revision=5, last_attempt_revision=7, last_attempt_status="failed"
        )
        == COHERENCE_STALE
    )


def test_a_generation_beyond_the_newest_attempt_is_inconsistent():
    assert (
        runtime_coherence(
            committed_revision=7, last_attempt_revision=5, last_attempt_status="succeeded"
        )
        == COHERENCE_INCONSISTENT
    )


def test_nothing_committed_is_unavailable():
    assert (
        runtime_coherence(
            committed_revision=None,
            last_attempt_revision=3,
            last_attempt_status="failed",
            has_generation=False,
        )
        == COHERENCE_UNAVAILABLE
    )


# --- read off a state --------------------------------------------------------

class _State:
    def __init__(self, persistence=None, current_round=0, state_revision=0):
        self.runtime_persistence = persistence if persistence is not None else {}
        self.current_round = current_round
        # Present only so the tests can reproduce the third draft's defect: a
        # counter that advanced on every durable state write, which the commit
        # publication incremented immediately after the commit. Nothing in the
        # verdict may read it.
        self.state_revision = state_revision


def _record(committed, attempted, status):
    return {
        "committed_source": "regular_save",
        "committed_digest": "d" * 64,
        "committed_revision": committed,
        "last_attempt_revision": attempted,
        "last_attempt_status": status,
    }


def test_a_session_whose_last_save_succeeded_is_current():
    state = _State(_record(4, 4, "succeeded"))
    assert state_runtime_coherence(state) == COHERENCE_CURRENT


def test_a_session_whose_closing_save_failed_is_stale():
    state = _State(_record(4, 6, "failed"))
    assert state_runtime_coherence(state) == COHERENCE_STALE


def test_a_session_with_no_generation_is_unavailable():
    assert state_runtime_coherence(_State({})) == COHERENCE_UNAVAILABLE


# --- what it gates -----------------------------------------------------------

def test_a_current_generation_does_not_block():
    assert is_forward_resume_blocked("complete", coherence=COHERENCE_CURRENT) is False


def test_a_stale_generation_blocks_silent_continuation():
    assert is_forward_resume_blocked("complete", coherence=COHERENCE_STALE) is True


def test_an_inconsistent_generation_blocks():
    assert (
        is_forward_resume_blocked("complete", coherence=COHERENCE_INCONSISTENT) is True
    )


def test_an_unrecorded_generation_does_not_block():
    """"unavailable" coherence is not evidence that a runtime is missing.

    It is also the state of a fresh session, of every session predating this
    contract, and of one restored by a path that installs a dump without writing
    a manifest. Whether a session that has worked still has a usable runtime is
    owned by ``runtime_restore_status``, which import and now local restore both
    set. Answering it here as well blocked nineteen legitimate resumes whose
    persistence record simply had not been populated.
    """
    assert (
        is_forward_resume_blocked("complete", coherence=COHERENCE_UNAVAILABLE) is False
    )
    assert (
        is_forward_resume_blocked("unavailable", coherence=COHERENCE_UNAVAILABLE)
        is True
    )


def test_a_pending_recompute_is_not_blocked_by_the_block_that_offered_it():
    assert (
        is_forward_resume_blocked(
            "unavailable", coherence=COHERENCE_STALE, recompute_pending=True
        )
        is False
    )


def test_the_default_keeps_every_existing_call_site_unchanged():
    assert is_forward_resume_blocked("complete") is False
    assert is_forward_resume_blocked("intentionally_kept") is False
    assert is_forward_resume_blocked("unavailable") is True


# --- the commit path ---------------------------------------------------------

def _notebook(tmp_path, name="coh"):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    states = tmp_path / "states"
    states.mkdir(exist_ok=True)
    return NotebookState(name, str(states))


def test_an_ordinary_successful_save_is_not_reported_stale(tmp_path):
    """The commit publishes its own state immediately afterwards.

    A verdict derived from any counter that advances on every durable state
    write went stale the moment that publication happened -- with no failure
    anywhere.
    """
    notebook = _notebook(tmp_path)
    snapshot = notebook.update_runtime_persistence(
        "current",
        current_revision=7,
        saved_revision=7,
        commit_source="regular_save",
        runtime_digest="e" * 64,
    )
    # The state has been written once more since the commit -- which is what
    # publishing the commit itself does. Nothing failed.
    #
    # The stale field is written in deliberately: a draft of this recorded the
    # state revision a generation committed against and compared it to the
    # session's current one, so an ordinary commit reported itself stale the
    # instant its own publication advanced the counter. The verdict must ignore
    # such a record entirely, not merely stop producing one.
    snapshot = dict(snapshot)
    snapshot["committed_state_revision"] = 7
    published = _State(snapshot, state_revision=8)
    assert state_runtime_coherence(published) == COHERENCE_CURRENT


def test_a_failed_save_after_a_good_one_is_reported_stale(tmp_path):
    notebook = _notebook(tmp_path)
    notebook.update_runtime_persistence(
        "current",
        current_revision=3,
        saved_revision=3,
        commit_source="regular_save",
        runtime_digest="a" * 64,
    )
    snapshot = notebook.update_runtime_persistence(
        "failed", current_revision=5, saved_revision=3, error_summary="closing save failed"
    )
    assert state_runtime_coherence(_State(snapshot)) == COHERENCE_STALE


def test_a_later_success_clears_the_stale_verdict(tmp_path):
    notebook = _notebook(tmp_path)
    notebook.update_runtime_persistence(
        "current", current_revision=3, saved_revision=3,
        commit_source="regular_save", runtime_digest="a" * 64,
    )
    notebook.update_runtime_persistence(
        "failed", current_revision=5, saved_revision=3, error_summary="boom"
    )
    snapshot = notebook.update_runtime_persistence(
        "current", current_revision=5, saved_revision=5,
        commit_source="regular_save", runtime_digest="b" * 64,
    )
    assert state_runtime_coherence(_State(snapshot)) == COHERENCE_CURRENT


def test_writing_metadata_never_scans_the_dump(tmp_path, monkeypatch):
    """Both full-file passes, not just the one behind has_session_dump().

    The verification form ran an HMAC pass and a SHA-256 pass over the whole
    dump on every metadata write -- several gigabytes twice per save.
    """
    from backend.utils.sandbox_parts import notebook_state as module

    notebook = _notebook(tmp_path)
    notebook.session_path.write_bytes(b"SCAI_ZSTD_DILL_V1\ndump")

    def _explode(*_args, **_kwargs):
        raise AssertionError("scanned the dump while writing metadata")

    monkeypatch.setattr(module, "_sha256_file", _explode)
    monkeypatch.setattr(module, "_verify_pkl_integrity", _explode)
    monkeypatch.setattr(type(notebook), "has_session_dump", _explode)

    snapshot = notebook.update_runtime_persistence(
        "current", current_revision=1, saved_revision=1,
        commit_source="regular_save", runtime_digest="f" * 64,
    )
    notebook.save_metadata()
    assert snapshot["committed_digest"] == "f" * 64
