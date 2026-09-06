"""Coherence is a third axis, distinct from integrity and coverage.

Integrity asks whether a generation is internally sound; coverage asks what it
left out; coherence asks whether it describes the same moment as the workflow
state beside it. Only the third can be answered by comparing two numbers, and
only if both have an owner -- so the comparison is against the session lifecycle
generation the dump committed under, not an invented counter.
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
        runtime_coherence(generation_state_revision=5, persisted_state_revision=5)
        == COHERENCE_CURRENT
    )


def test_a_newer_state_makes_the_runtime_stale():
    assert (
        runtime_coherence(generation_state_revision=5, persisted_state_revision=6)
        == COHERENCE_STALE
    )


def test_a_runtime_ahead_of_the_state_is_inconsistent():
    """Not 'current': a generation cannot belong to a state that never existed."""
    assert (
        runtime_coherence(generation_state_revision=7, persisted_state_revision=6)
        == COHERENCE_INCONSISTENT
    )


def test_no_generation_at_all_is_unavailable():
    assert (
        runtime_coherence(generation_state_revision=None, persisted_state_revision=3)
        == COHERENCE_UNAVAILABLE
    )


def test_a_current_runtime_does_not_block_forward_resume():
    assert is_forward_resume_blocked("complete", coherence=COHERENCE_CURRENT) is False


def test_a_stale_runtime_blocks_silent_forward_resume():
    """Spec: exact continuation is not promised, so it is not silently taken.

    The block routes into the existing re-run affordance rather than failing the
    session: the user is offered a choice instead of being handed an older
    namespace bound to a newer workflow state.
    """
    assert is_forward_resume_blocked("complete", coherence=COHERENCE_STALE) is True


def test_an_inconsistent_runtime_blocks_forward_resume():
    assert (
        is_forward_resume_blocked("complete", coherence=COHERENCE_INCONSISTENT) is True
    )


def test_an_unavailable_runtime_still_blocks_regardless_of_coherence():
    assert is_forward_resume_blocked("unavailable", coherence=COHERENCE_CURRENT) is True


def test_the_default_keeps_every_existing_call_site_unchanged():
    """Six call sites pass only a status; none of them may start blocking."""
    assert is_forward_resume_blocked("complete") is False
    assert is_forward_resume_blocked("intentionally_kept") is False
    assert is_forward_resume_blocked("unavailable") is True


def test_coherence_is_reported_without_changing_the_persistence_status(tmp_path):
    """Coherence must not become the required set reborn.

    It is a fact about the generation, not a gate: recording it may not mark the
    runtime failed or demand a recompute, because those are what stopped the
    analysis outright in the design this replaces.
    """
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    states = tmp_path / "states"
    states.mkdir()
    notebook = NotebookState("coh", str(states))
    snapshot = notebook.update_runtime_persistence(
        "current",
        current_revision=3,
        saved_revision=3,
        commit_source="regular_save",
        coherence=COHERENCE_STALE,
    )

    assert snapshot["coherence"] == COHERENCE_STALE
    assert snapshot["status"] == "current"
    assert snapshot.get("requires_recompute") is not True
