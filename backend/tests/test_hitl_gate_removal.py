"""A failed runtime save reports and continues; it does not wait for a person.

Spec 5.1 replaces the runtime-persistence HITL gate with closing-save
semantics. One blocking case survives -- no generation ever succeeded -- and it
is answered by an explicit re-run rather than a dialog.
"""
from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"

#: Spec 4.1 row 3 lists five modules. Two of them are not this batch's:
#:
#: - ``runtime_persistence_projection.py`` is the channel that publishes the
#:   persistence record to the UI, which is exactly the disclosure spec 5.1
#:   requires. Only its one call into the gate is removed; the module stays.
#: - ``models/schema_parts/snapshot_failure.py`` and
#:   ``routes/lifecycle_parts/snapshot_failures.py`` served Snapshot V2's
#:   *capture* failure gate -- proceed/skip/terminate on a failed snapshot --
#:   not the runtime-persistence one. They are still reached from
#:   ``chat_parts/`` and the V2 restore jobs, and are deleted in A2-5 with the
#:   package.
GATE_MODULES = (
    "services/runtime_persistence_gate.py",
    "routes/lifecycle_parts/runtime_persistence.py",
)


def test_a_session_frozen_at_the_deleted_gate_is_unblocked_on_load(tmp_path):
    """The upgrade path. Spec 4.6.

    A session saved under the shipped version while the runtime-persistence
    gate was armed carries ``snapshot_failure = {"code":
    "runtime_persistence_failure", ...}`` and a matching ``wait_input_reason``.
    ``sanitize_legacy_state`` deliberately refused to clear those,
    because clearing them would have bypassed a live gate. With the gate and its
    three routes deleted, that refusal leaves the session permanently
    unstartable and the only frontend button that still reaches a backend is
    Terminate, which discards the runtime.
    """
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.legacy_state_sanitizer import (
        sanitize_legacy_state,
    )

    state = AnalysisState(id="frozen-at-the-gate", user_id="u1")
    state.status = TaskStatus.WAITING_INPUT
    state.wait_input_reason = "runtime_persistence_failure"
    state.snapshot_failure = {
        "code": "runtime_persistence_failure",
        "message": "Runtime persistence failed",
        "retry_context": {"previous_status": "running"},
    }

    changed = sanitize_legacy_state(state, workflow_active=False)

    assert changed is True
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
    assert state.status != TaskStatus.WAITING_INPUT


def test_an_unfinished_restore_journal_still_refuses(tmp_path, monkeypatch):
    """The one case where staying unstartable is correct is not weakened."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services import legacy_state_sanitizer as feature_gate

    monkeypatch.setattr(
        feature_gate, "has_unfinished_restore_journal", lambda sid: True
    )
    state = AnalysisState(id="half-applied-restore", user_id="u1")
    state.status = TaskStatus.WAITING_INPUT
    state.recovery_blocked = True

    feature_gate.sanitize_legacy_state(state, workflow_active=False)
    assert state.recovery_blocked is True


def test_an_unmarked_applying_restore_journal_sets_the_recovery_marker(
    tmp_path, monkeypatch,
):
    """The crash window is detectable before session.json has the marker."""
    from backend.models.schemas import AnalysisState
    from backend.services import legacy_state_sanitizer as feature_gate
    from backend.services import session_store

    session_root = tmp_path / "session"
    journals = session_root / "journals"
    journals.mkdir(parents=True)
    (journals / "restore.json").write_text(
        '{"kind":"restore","status":"APPLYING"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(feature_gate, "get_session_dir", lambda _sid: session_root)

    state = AnalysisState(id="unmarked-restore", user_id="u1")
    assert state.recovery_blocked is False
    assert feature_gate.sanitize_legacy_state(state, workflow_active=False) is True
    assert state.recovery_blocked is True
    with pytest.raises(feature_gate.RecoveryBlockedError):
        session_store.assert_recovery_clear(session_id=state.id)


#: Ways a component can *condition on* the retired gate code. Prose that merely
#: names it is fine and is how the remaining state-sync branch explains itself;
#: an earlier version of this guard matched any occurrence and failed on its own
#: explanatory comment.
_GATE_BRANCH_PATTERNS = (
    "startsWith('runtime_persistence_')",
    '=== \'runtime_persistence_',
    '=== "runtime_persistence_',
    "isRuntimePersistence",
)
