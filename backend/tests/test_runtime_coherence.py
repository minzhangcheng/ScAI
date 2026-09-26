"""Committed runtime coherence and forward resume policy."""
import pytest

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


# --- the verdict -------------------------------------------------------------


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


# --- what it gates -----------------------------------------------------------


# --- the commit path ---------------------------------------------------------

def _notebook(tmp_path, name="coh"):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    states = tmp_path / "states"
    states.mkdir(exist_ok=True)
    return NotebookState(name, str(states))


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


@pytest.mark.parametrize("committed, attempted, status, has_generation, expected", [
    (5, 5, "succeeded", True, COHERENCE_CURRENT),
    (5, 7, "failed", True, COHERENCE_STALE),
    (7, 5, "succeeded", True, COHERENCE_INCONSISTENT),
    (None, 3, "failed", False, COHERENCE_UNAVAILABLE),
])
def test_runtime_coherence_policy(committed, attempted, status, has_generation, expected):
    assert runtime_coherence(
        committed_revision=committed, last_attempt_revision=attempted,
        last_attempt_status=status, has_generation=has_generation,
    ) == expected


@pytest.mark.parametrize("runtime, coherence, recompute, expected", [
    ("complete", COHERENCE_CURRENT, False, False),
    ("complete", COHERENCE_STALE, False, True),
    ("complete", COHERENCE_INCONSISTENT, False, True),
    ("complete", COHERENCE_UNAVAILABLE, False, False),
    ("unavailable", COHERENCE_UNAVAILABLE, False, True),
    ("unavailable", COHERENCE_STALE, True, False),
])
def test_forward_resume_policy(runtime, coherence, recompute, expected):
    assert is_forward_resume_blocked(
        runtime, coherence=coherence, recompute_pending=recompute,
    ) is expected
