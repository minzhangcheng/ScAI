"""Persistability is decided in the cell that binds the value, not at save time.

Promotion into ``required_outputs`` used to be a decision about names only: a
value created by one cell and read by another became runtime state the save had
to carry, with nothing ever asking whether it could be carried. The knowledge of
what dill refuses lived exclusively inside the save script, so the answer arrived
at a save boundary minutes later, addressed to the operator, about a cell the
agent could no longer change. These tests pin the probe that moves that question
to the executor, and the exclusions that keep it affordable.
"""
import asyncio
import contextlib
import io
import sqlite3

import pytest

from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker


def _sandbox(tmp_path, session_id):
    from backend.utils.sandbox import LocalJupyterSandbox

    sandbox = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sandbox._exec_lock = asyncio.Lock()
    sandbox.runtime_dependency_tracker = RuntimeDependencyTracker()
    return sandbox


def _local_kernel(namespace):
    """Run the generated probe for real against a literal namespace."""
    from backend.utils.sandbox import CodeExecutionResult

    async def _run(code, **_kwargs):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, "<probe>", "exec"), namespace)
        return CodeExecutionResult(success=True, output=stream.getvalue())

    return _run


@pytest.mark.asyncio
async def test_probe_names_the_values_that_cannot_be_carried(tmp_path):
    import pandas as pd

    namespace = {
        "conn": sqlite3.connect(":memory:"),
        "gen": (index for index in range(3)),
        "frame": pd.DataFrame({"a": [1, 2]}),
        "vals": {"n_cells": 12},
    }
    sandbox = _sandbox(tmp_path, "probe-verdicts")
    sandbox._execute_unlocked = _local_kernel(namespace)

    after, verdicts, probed = await sandbox._snapshot_globals_with_persistability(
        set(), set()
    )

    assert after == {"conn", "gen", "frame", "vals"}
    assert probed == {"conn", "gen", "frame", "vals"}
    assert set(verdicts) == {"conn", "gen"}
    assert "volatile handle" in verdicts["conn"], verdicts["conn"]
    assert "sqlite3" in verdicts["conn"]
    # A bulk container is known persistable and expensive to prove; a plain
    # value simply serializes.
    assert "frame" not in verdicts
    assert "vals" not in verdicts


@pytest.mark.asyncio
async def test_probe_leaves_values_past_its_budget_unjudged(tmp_path):
    """The large objects are the analysis data. Refusing to promote one because
    it did not fit an 8 MiB buffer would be far worse than the failure this
    prevents, so exceeding the budget yields no verdict at all."""
    namespace = {"blob": bytearray(16 * 1024 * 1024)}
    sandbox = _sandbox(tmp_path, "probe-budget")
    sandbox._execute_unlocked = _local_kernel(namespace)

    _after, verdicts, probed = await sandbox._snapshot_globals_with_persistability(
        set(), set()
    )

    assert probed == {"blob"}
    assert verdicts == {}


@pytest.mark.asyncio
async def test_probe_failure_falls_back_to_the_name_only_view(tmp_path):
    """A probe that cannot answer must not read as a clean bill of health, and
    must not stall the cell either: the save boundary stays the backstop."""
    from backend.utils.sandbox import CodeExecutionResult

    sandbox = _sandbox(tmp_path, "probe-degraded")
    calls = []

    async def _run(code, **_kwargs):
        calls.append(code)
        if "__PROBED__" in code:
            return CodeExecutionResult(success=False, output="", error="kernel busy")
        return CodeExecutionResult(success=True, output='__USER_GLOBALS__:["adata"]')

    sandbox._execute_unlocked = _run

    after, verdicts, probed = await sandbox._snapshot_globals_with_persistability(
        set(), set()
    )

    assert after == {"adata"}
    assert verdicts == {}
    assert probed == set()


def test_the_save_and_the_probe_share_one_volatile_exclusion():
    """Two copies of this list would drift, and a handle promoted by one and
    excluded by the other is exactly the state that has no recovery."""
    from backend.services.snapshot_v2.runtime_registry import (
        VOLATILE_RUNTIME_TYPE_PREFIXES,
    )
    from backend.utils.sandbox_parts.runtime_save import SaveContext, _build_save_code

    context = SaveContext(
        sandbox=None,
        restore_failed=False,
        revision=1,
        expected_lifecycle_generation=1,
        required_names=set(),
        escaped_required="[]",
        escaped_path=repr("/tmp/x_session.pkl"),
        record_generation_failure=None,
        restore_previous_persistence=None,
        SessionGenerationError=RuntimeError,
        SessionGoneError=RuntimeError,
        assert_session_generation_active=None,
        lifecycle_file_lock=None,
    )
    save_code = _build_save_code(context)
    assert repr(VOLATILE_RUNTIME_TYPE_PREFIXES) in save_code
    assert "sqlite3." in VOLATILE_RUNTIME_TYPE_PREFIXES


def test_tracker_never_promotes_a_name_the_kernel_proved_unserializable():
    tracker = RuntimeDependencyTracker()
    tracker.record_execution(
        created={"res", "counts"},
        unpersistable={"res"},
        probed={"res", "counts"},
    )
    # A later cell reads both. Only the serializable one becomes runtime state.
    tracker.record_execution(loaded={"res", "counts"})

    assert tracker.required_outputs == {"counts"}
    assert tracker.unpersistable_outputs == {"res"}


def test_tracker_releases_an_already_required_name_when_the_verdict_arrives():
    tracker = RuntimeDependencyTracker()
    tracker.merge_declared({"res"})
    assert tracker.required_outputs == {"res"}

    tracker.record_execution(created={"res"}, unpersistable={"res"}, probed={"res"})

    assert tracker.required_outputs == set()


def test_tracker_gives_a_rebound_name_a_fresh_verdict():
    """The verdict is about a value, not a name: rebinding must be able to fix
    it, or a scratch name reused later in the run stays excluded forever."""
    tracker = RuntimeDependencyTracker()
    tracker.record_execution(created={"res"}, unpersistable={"res"}, probed={"res"})
    assert tracker.unpersistable_outputs == {"res"}

    # The cell rebinds `res` to a DataFrame and reads it: probed, no verdict.
    tracker.record_execution(created={"res"}, loaded={"res"}, probed={"res"})
    tracker.record_execution(loaded={"res"})

    assert tracker.unpersistable_outputs == set()
    assert "res" in tracker.required_outputs


def test_tracker_state_round_trips_the_verdicts():
    tracker = RuntimeDependencyTracker()
    tracker.record_execution(created={"res"}, unpersistable={"res"}, probed={"res"})

    restored = RuntimeDependencyTracker.from_state_dict(tracker.to_state_dict())
    restored.record_execution(loaded={"res"})

    assert restored.unpersistable_outputs == {"res"}
    assert restored.required_outputs == set()


def _executing_kernel(namespace, creates=None):
    """Answer both the diff probes and the cell itself from one namespace.

    A non-probe cell binds *creates*, so the before/after diff sees the value
    appear exactly as a real cell would -- the probe only judges what the cell
    touched, so a fixture that pre-seeds the namespace tests nothing.
    """
    from backend.utils.sandbox import CodeExecutionResult

    async def _run(code, **_kwargs):
        if "__PROBED__" in code or "__USER_GLOBALS__" in code:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(code, "<probe>", "exec"), namespace)
            return CodeExecutionResult(success=True, output=stream.getvalue())
        namespace.update(creates or {})
        return CodeExecutionResult(success=True, output="ok")

    return _run


@pytest.mark.asyncio
async def test_execute_reports_an_unpersistable_binding_to_the_agent(tmp_path):
    sandbox = _sandbox(tmp_path, "probe-execute-note")
    namespace: dict = {}
    sandbox._execute_unlocked = _executing_kernel(
        namespace, {"conn": sqlite3.connect(":memory:")}
    )

    result = await sandbox.execute("conn = open_db()", track_globals=True)

    assert result.success
    assert "not carried across a kernel restart" in result.output.lower()
    assert "conn" in result.output
    assert "conn" not in sandbox.runtime_dependency_tracker.required_outputs


@pytest.mark.asyncio
async def test_execute_fails_a_declared_output_that_cannot_be_serialized(tmp_path):
    """A note is not enough here: the agent asked for this value to survive the
    round, and only the cell that built it can still be rewritten cheaply."""
    sandbox = _sandbox(tmp_path, "probe-execute-declared")
    namespace: dict = {}
    sandbox._execute_unlocked = _executing_kernel(
        namespace, {"conn": sqlite3.connect(":memory:")}
    )

    result = await sandbox.execute(
        "conn = open_db()", declared_outputs=["conn"], track_globals=True
    )

    assert not result.success
    assert "cannot be serialized" in (result.error or "")
    assert "conn" in (result.error or "")
    assert "conn" not in sandbox.runtime_dependency_tracker.required_outputs
