"""Coherence is a third axis, distinct from integrity and coverage.

Integrity asks whether a generation is internally sound; coverage asks what it
left out; coherence asks whether the newest committed generation still
describes the live namespace.

Both numbers it compares belong to the persistence layer and move with it. An
earlier draft compared the session *lifecycle generation*, which fences
incarnations rather than tracking execution: it is normally 1 for the whole life
of a session, so the verdict was permanently "current" and every gate built on
it was dead code.
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
    from backend.utils.runtime_apply import is_forward_resume_blocked
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


def test_equal_revisions_are_current():
    assert (
        runtime_coherence(saved_revision=5, current_revision=5) == COHERENCE_CURRENT
    )


def test_a_live_namespace_past_the_last_commit_is_stale():
    assert runtime_coherence(saved_revision=5, current_revision=6) == COHERENCE_STALE


def test_a_generation_ahead_of_the_live_runtime_is_inconsistent():
    """Not the benign converse of stale: a commit recorded a rolled-back revision."""
    assert (
        runtime_coherence(saved_revision=7, current_revision=6)
        == COHERENCE_INCONSISTENT
    )


def test_no_committed_generation_is_unavailable():
    assert (
        runtime_coherence(saved_revision=0, current_revision=0, has_generation=False)
        == COHERENCE_UNAVAILABLE
    )


def test_a_current_runtime_does_not_block_forward_resume():
    assert is_forward_resume_blocked("complete", coherence=COHERENCE_CURRENT) is False


def test_a_stale_runtime_blocks_silent_forward_resume():
    assert is_forward_resume_blocked("complete", coherence=COHERENCE_STALE) is True


def test_an_inconsistent_runtime_blocks_forward_resume():
    assert (
        is_forward_resume_blocked("complete", coherence=COHERENCE_INCONSISTENT) is True
    )


def test_a_fresh_session_with_no_generation_is_not_blocked():
    """"unavailable" coherence is the normal state before the first save."""
    assert (
        is_forward_resume_blocked("complete", coherence=COHERENCE_UNAVAILABLE) is False
    )


def test_an_unavailable_runtime_still_blocks_regardless_of_coherence():
    assert is_forward_resume_blocked("unavailable", coherence=COHERENCE_CURRENT) is True


def test_a_pending_recompute_is_not_blocked_by_the_block_that_offered_it():
    """The re-run the block offers must not be refused by the same block.

    The lifecycle route sets runtime_recompute_pending and clears
    runtime_restore_status; without this the graph re-checked the identical
    condition through coherence and failed the session it was restarting.
    """
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


def _notebook(tmp_path, name="coh"):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    states = tmp_path / "states"
    states.mkdir(exist_ok=True)
    return NotebookState(name, str(states))


def test_a_generation_behind_the_live_namespace_is_recorded_stale(tmp_path, monkeypatch):
    """The verdict is derived on every update, not supplied by a caller.

    Nothing in production passed a coherence value, so a version that only
    accepted one reported "current" forever -- a gate that could never fire.
    """
    notebook = _notebook(tmp_path)
    monkeypatch.setattr(type(notebook), "has_session_dump", lambda _self: True)

    snapshot = notebook.update_runtime_persistence(
        "live_dirty", current_revision=6, saved_revision=5
    )
    assert snapshot["coherence"] == COHERENCE_STALE


def test_a_successful_save_clears_a_stale_verdict(tmp_path, monkeypatch):
    """A latched verdict would outlive the condition that produced it."""
    notebook = _notebook(tmp_path)
    monkeypatch.setattr(type(notebook), "has_session_dump", lambda _self: True)

    notebook.update_runtime_persistence(
        "live_dirty", current_revision=6, saved_revision=5
    )
    snapshot = notebook.update_runtime_persistence(
        "current", current_revision=6, saved_revision=6, commit_source="regular_save"
    )
    assert snapshot["coherence"] == COHERENCE_CURRENT


def test_recording_coherence_never_changes_the_persistence_status(tmp_path, monkeypatch):
    """Coherence must not become the required set reborn."""
    notebook = _notebook(tmp_path)
    monkeypatch.setattr(type(notebook), "has_session_dump", lambda _self: True)

    snapshot = notebook.update_runtime_persistence(
        "live_dirty", current_revision=9, saved_revision=3
    )
    assert snapshot["coherence"] == COHERENCE_STALE
    assert snapshot["status"] == "live_dirty"
    assert snapshot.get("requires_recompute") is not True


def test_a_session_with_no_dump_reports_unavailable(tmp_path, monkeypatch):
    notebook = _notebook(tmp_path)
    monkeypatch.setattr(type(notebook), "has_session_dump", lambda _self: False)

    snapshot = notebook.update_runtime_persistence(
        "live_dirty", current_revision=2, saved_revision=0
    )
    assert snapshot["coherence"] == COHERENCE_UNAVAILABLE
