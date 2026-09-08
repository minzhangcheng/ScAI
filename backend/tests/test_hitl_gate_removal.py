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


def test_the_gate_modules_are_gone():
    for rel in GATE_MODULES:
        assert not (BACKEND / rel).exists(), rel


def test_the_projection_survives_because_it_is_the_disclosure_channel():
    """Spec 5.1 needs the record published; only its dialog was removed."""
    from backend.services.runtime_persistence_projection import (
        apply_runtime_persistence_projection,
        project_runtime_persistence,
    )

    assert callable(project_runtime_persistence)
    assert callable(apply_runtime_persistence_projection)
    source = (BACKEND / "services" / "runtime_persistence_projection.py").read_text()
    assert "runtime_persistence_gate" not in source


def test_no_surviving_module_imports_the_gate():
    offenders = set()
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(BACKEND.parent).as_posix()
        if rel.startswith("backend/tests/"):
            continue
        text = path.read_text()
        if "runtime_persistence_gate" in text:
            offenders.add(rel)
    assert not offenders, sorted(offenders)


def test_the_one_blocking_case_survives_under_its_own_name():
    """No generation ever succeeded is still a block; a failed save is not."""
    from backend.services.runtime_recompute import (
        RuntimeUnavailable,
        prepare_runtime_recompute,
    )

    assert issubclass(RuntimeUnavailable, RuntimeError)
    assert callable(prepare_runtime_recompute)


def test_a_stale_generation_does_not_stop_the_executor():
    """Spec 5.1: results stay usable, continuation is not promised, work goes on.

    The gate used to hold the run whenever the newest save had failed or the
    recovery was degraded. That hold is what this batch removes.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/agents/bioinformatics/executor_runtime.py")
    assert "enter_runtime_persistence_gate" not in source
    assert "degraded_recovery" not in source


def test_stopping_the_kernel_after_a_failed_save_does_not_raise_a_gate():
    """The kernel is closed; the previous committed generation is kept.

    ``stop(save=True)`` used to raise RuntimePersistenceShutdownBlocked and
    route the failure through the intervention gate, which kept the kernel
    alive waiting for a person.
    """
    from backend.tests.split_facade import split_source

    text = split_source("backend.utils.sandbox")
    assert "RuntimePersistenceShutdownBlocked" not in text
    assert "runtime_persistence_gate" not in text


@pytest.mark.asyncio
async def test_a_failed_final_save_still_closes_the_kernel(tmp_path, monkeypatch):
    """One attempt. The failure is recorded; the kernel does not stay alive."""
    from unittest.mock import AsyncMock

    from backend.utils.sandbox import LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id="failed-final-save",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(
        sb, "_save_session_locked", AsyncMock(side_effect=RuntimeError("dump failed"))
    )
    sb.kc = None
    sb.km = None

    await sb.stop(save=True)

    assert sb._initialized is False
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


def test_completed_and_non_restore_journals_remain_harmless(tmp_path, monkeypatch):
    from backend.models.schemas import AnalysisState
    from backend.services import legacy_state_sanitizer as feature_gate

    session_root = tmp_path / "session"
    journals = session_root / "journals"
    journals.mkdir(parents=True)
    monkeypatch.setattr(feature_gate, "get_session_dir", lambda _sid: session_root)
    state = AnalysisState(id="harmless-journal", user_id="u1")

    (journals / "completed.json").write_text(
        '{"kind":"restore","status":"COMPLETED"}', encoding="utf-8",
    )
    assert feature_gate.sanitize_legacy_state(state, workflow_active=False) is False
    assert state.recovery_blocked is False

    (journals / "completed.json").unlink()
    (journals / "export.json").write_text(
        '{"kind":"export","status":"APPLYING"}', encoding="utf-8",
    )
    assert feature_gate.sanitize_legacy_state(state, workflow_active=False) is False
    assert state.recovery_blocked is False


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


def test_no_frontend_component_branches_on_the_deleted_gate():
    """A dialog that points Retry at an endpoint that 409s is worse than none."""
    frontend = REPO / "frontend" / "src"
    offenders = {}
    for path in sorted(frontend.rglob("*.ts*")):
        rel = path.relative_to(REPO).as_posix()
        if "__tests__" in rel:
            continue
        text = path.read_text()
        hits = [pat for pat in _GATE_BRANCH_PATTERNS if pat in text]
        if hits:
            offenders[rel] = hits
    assert not offenders, offenders


def test_no_frontend_component_calls_a_retired_gate_endpoint():
    frontend = REPO / "frontend" / "src"
    offenders = {
        path.relative_to(REPO).as_posix()
        for path in sorted(frontend.rglob("*.ts*"))
        if "runtime-persistence/" in path.read_text()
    }
    assert not offenders, sorted(offenders)
@pytest.mark.asyncio
async def test_a_failed_closing_save_still_writes_the_notebook(tmp_path, monkeypatch):
    """Spec 5.1: "The notebook and session metadata are saved independently, so
    a failed runtime dump cannot take them with it."

    The notebook is the record of what ran. When the runtime dump fails it is
    the *only* record left, which is exactly when losing it costs most.
    """
    from unittest.mock import AsyncMock, MagicMock

    from backend.utils.sandbox import LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id="failed-save-keeps-notebook",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(
        sb, "_save_session_locked", AsyncMock(side_effect=RuntimeError("dump failed"))
    )
    notebook_save = MagicMock()
    monkeypatch.setattr(sb.notebook_state, "save", notebook_save)
    sb.kc = None
    sb.km = None

    await sb.stop(save=True)

    assert notebook_save.call_count == 1, "the notebook went down with the dump"
    assert sb._initialized is False


@pytest.mark.asyncio
async def test_a_closing_save_that_reports_no_generation_still_writes_the_notebook(
    tmp_path, monkeypatch,
):
    """The other failure shape: save_session returns False rather than raising."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.utils.sandbox import LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id="no-generation-keeps-notebook",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(sb, "_save_session_locked", AsyncMock(return_value=False))
    notebook_save = MagicMock()
    monkeypatch.setattr(sb.notebook_state, "save", notebook_save)
    sb.kc = None
    sb.km = None

    await sb.stop(save=True)

    assert notebook_save.call_count == 1
def test_a_degraded_restore_warns_the_agent_without_an_explicit_re_run():
    """The warning must hang on the record, not on the authorization.

    A restore that fell back to a verified older generation records
    ``degraded_recovery`` (notebook_state.py). The agent-facing warning used to
    be reachable because the gate's /recompute route set
    ``runtime_recompute_pending``; with the gate gone the only setters are the
    explicit re-run routes, so a degraded restore would proceed silently and
    the agent would read a round-5 notebook while holding a round-3 namespace.
    """
    from backend.agents.bioinformatics_agent import BioinformaticsAgent
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id="degraded-no-rerun", user_id="u1")
    state.runtime_recompute_pending = False
    state.runtime_persistence = {
        "status": "degraded",
        "degraded_recovery": True,
        "degraded_from_revision": 9,
        "degraded_to_revision": 4,
    }

    context = BioinformaticsAgent._build_data_context(
        BioinformaticsAgent.__new__(BioinformaticsAgent), state
    )
    assert "RUNTIME RECOVERY REQUIRED" in context


def test_an_ordinary_current_runtime_does_not_warn():
    from backend.agents.bioinformatics_agent import BioinformaticsAgent
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id="ordinary-runtime", user_id="u1")
    state.runtime_recompute_pending = False
    state.runtime_persistence = {"status": "current", "degraded_recovery": False}

    context = BioinformaticsAgent._build_data_context(
        BioinformaticsAgent.__new__(BioinformaticsAgent), state
    )
    assert "RUNTIME RECOVERY REQUIRED" not in context
