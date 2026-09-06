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
#:   ``routes/lifecycle_parts/snapshot_failures.py`` serve Snapshot V2's
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
    monkeypatch.setattr(sb, "await_pending_saves", AsyncMock(return_value=None))
    monkeypatch.setattr(
        sb, "save_session", AsyncMock(side_effect=RuntimeError("dump failed"))
    )
    sb.kc = None
    sb.km = None

    await sb.stop(save=True)

    assert sb._initialized is False
