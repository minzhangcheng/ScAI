"""Coherence: does the newest committed generation belong to the persisted state?

Distinct from integrity (is the generation internally sound) and coverage (what
did it leave out). Three drafts of this were wrong in instructive ways, and the
tests below pin each of them shut:

* comparing the session *lifecycle generation*, which fences incarnations and
  is normally 1 for a whole session, so the verdict was permanently "current";
* comparing saved against current *runtime* revision, which only reports whether
  the namespace is dirty and cannot see a state that advanced without any cell
  execution;
* deriving "is there a generation" by re-verifying the dump on disk against the
  *previous* digest, which reported every successful commit as having none --
  and read a multi-gigabyte file twice to do it.
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


# --- the verdict itself ------------------------------------------------------

def test_a_generation_paired_with_the_current_state_is_current():
    assert (
        runtime_coherence(committed_state_revision=5, state_revision=5)
        == COHERENCE_CURRENT
    )


def test_a_state_that_moved_on_makes_the_generation_stale():
    assert (
        runtime_coherence(committed_state_revision=5, state_revision=6)
        == COHERENCE_STALE
    )


def test_a_generation_ahead_of_the_state_is_inconsistent():
    assert (
        runtime_coherence(committed_state_revision=7, state_revision=6)
        == COHERENCE_INCONSISTENT
    )


def test_no_committed_generation_is_unavailable():
    assert (
        runtime_coherence(
            committed_state_revision=None, state_revision=3, has_generation=False
        )
        == COHERENCE_UNAVAILABLE
    )


# --- read off a state --------------------------------------------------------

class _State:
    def __init__(self, *, state_revision=0, persistence=None, current_round=0):
        self.state_revision = state_revision
        self.runtime_persistence = persistence if persistence is not None else {}
        self.current_round = current_round


def _committed(committed_state_revision):
    return {
        "committed_source": "regular_save",
        "committed_digest": "d" * 64,
        "committed_state_revision": committed_state_revision,
    }


def test_a_state_at_the_revision_its_generation_committed_against_is_current():
    state = _State(state_revision=4, persistence=_committed(4))
    assert state_runtime_coherence(state) == COHERENCE_CURRENT


def test_a_state_written_after_the_last_commit_is_stale():
    """What a failed closing save leaves behind."""
    state = _State(state_revision=7, persistence=_committed(4))
    assert state_runtime_coherence(state) == COHERENCE_STALE


def test_a_session_with_no_committed_generation_is_unavailable():
    state = _State(state_revision=2, persistence={})
    assert state_runtime_coherence(state) == COHERENCE_UNAVAILABLE


def test_a_generation_from_before_this_contract_is_not_called_stale():
    """Absence of a pairing is not evidence of a mismatch."""
    persistence = _committed(None)
    state = _State(state_revision=9, persistence=persistence)
    assert state_runtime_coherence(state) == COHERENCE_CURRENT


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
    a different question owned by ``runtime_restore_status`` -- import sets it
    to "unavailable" for exactly that case. An earlier version answered it here
    too, using ``current_round > 0``, and blocked nineteen legitimate resumes
    whose persistence record simply had not been populated.
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
            "unavailable",
            coherence=COHERENCE_STALE,
            recompute_pending=True,
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


def test_a_successful_commit_records_its_pairing(tmp_path):
    notebook = _notebook(tmp_path)
    snapshot = notebook.update_runtime_persistence(
        "current",
        current_revision=3,
        saved_revision=3,
        commit_source="regular_save",
        runtime_digest="e" * 64,
        committed_state_revision=11,
    )
    assert snapshot["committed_state_revision"] == 11
    assert snapshot["committed_digest"] == "e" * 64

    state = _State(state_revision=11, persistence=snapshot)
    assert state_runtime_coherence(state) == COHERENCE_CURRENT

    moved_on = _State(state_revision=12, persistence=snapshot)
    assert state_runtime_coherence(moved_on) == COHERENCE_STALE


def test_recording_a_commit_never_re_reads_the_dump(tmp_path, monkeypatch):
    """The pairing comes from the transaction, not from re-verifying the file.

    An earlier version called has_session_dump() here. That checks the
    already-replaced dump against the *previous* digest -- so it answered False
    for every successful commit -- and hashed a multi-gigabyte file twice to do
    it.
    """
    notebook = _notebook(tmp_path)

    def _explode(_self):
        raise AssertionError("re-verified the dump while recording metadata")

    monkeypatch.setattr(type(notebook), "has_session_dump", _explode)

    snapshot = notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest="f" * 64,
        committed_state_revision=2,
    )
    assert snapshot["committed_state_revision"] == 2


def test_a_failed_attempt_leaves_the_previous_pairing_alone(tmp_path):
    notebook = _notebook(tmp_path)
    notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest="a" * 64,
        committed_state_revision=5,
    )
    snapshot = notebook.update_runtime_persistence(
        "failed", current_revision=2, saved_revision=1, error_summary="boom"
    )
    assert snapshot["committed_state_revision"] == 5

    state = _State(state_revision=9, persistence=snapshot)
    assert state_runtime_coherence(state) == COHERENCE_STALE
