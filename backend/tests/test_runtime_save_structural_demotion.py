"""Structural-unpicklability demotion in the runtime save.

A cross-cell dependency the executor promoted into the required set can hold an
object dill refuses -- an sqlite3 connection reachable from a scratch list, a
pybiomart CachedSession behind an analysis result. Streaming it aborted the
whole generation and armed the runtime-persistence gate, and every Retry from
there repeated the identical serialization, so the run stalled on a variable the
analysis never meant to keep. These tests pin the two halves of the fix: the
in-kernel probe that proves a required value before writing any of its bytes,
and the host-side release that stops the next save from failing the same way.
"""
import contextlib
import io
import sqlite3

import pytest

from backend.utils.sandbox_parts.runtime_save import SaveContext, _build_save_code


def _save_context(tmp_path, required):
    return SaveContext(
        sandbox=None,
        restore_failed=False,
        revision=1,
        expected_lifecycle_generation=1,
        required_names=set(required),
        escaped_required=repr(sorted(required)),
        escaped_path=repr(str(tmp_path / "probe_session.pkl")),
        record_generation_failure=None,
        restore_previous_persistence=None,
        SessionGenerationError=RuntimeError,
        SessionGoneError=RuntimeError,
        assert_session_generation_active=None,
        lifecycle_file_lock=None,
    )


def _run_kernel_save(tmp_path, namespace, required):
    """Execute the generated kernel save code over a literal namespace."""
    code = _build_save_code(_save_context(tmp_path, required))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "<save_code>", "exec"), dict(namespace))
    return stream.getvalue()


def _skipped_names(output):
    return {
        line.split("\t")[1]
        for line in output.splitlines()
        if line.startswith("__SCAI_SAVE_SKIPPED__\t")
    }


def _saved_names(output):
    import json

    for line in output.splitlines():
        if line.startswith("__SCAI_SAVED_NAMES__:"):
            return set(json.loads(line.split(":", 1)[1]))
    return set()


def test_unpicklable_required_value_is_dropped_instead_of_aborting(tmp_path):
    """The observed failure: a required list holding an sqlite3 connection.

    Before the probe this raised, deleted the staged dump and left the run
    gated; the picklable required value next to it was lost with it.
    """
    namespace = {
        "namespace_items": [("conn", sqlite3.connect(":memory:"))],
        "adata_counts": {"n_obs": 12},
    }
    output = _run_kernel_save(
        tmp_path, namespace, {"namespace_items", "adata_counts"}
    )

    assert "namespace_items" in _skipped_names(output)
    assert "adata_counts" in _saved_names(output), (
        "the rest of the required tier must still reach the dump"
    )
    assert "Session saved:" in output


def test_required_anndata_still_aborts_the_generation(tmp_path):
    """The probe must not silently drop authoritative analysis state.

    Every AnnData is required no matter what the host declared, and dropping one
    is the exact silent loss the required tier exists to prevent, so it streams
    and a failure there still aborts.
    """
    class AnnData:
        def __init__(self):
            self.conn = sqlite3.connect(":memory:")

    AnnData.__module__ = "anndata._core.anndata"

    with pytest.raises(RuntimeError, match="could not be persisted: adata"):
        _run_kernel_save(tmp_path, {"adata": AnnData()}, set())


def test_probe_falls_back_to_streaming_past_its_budget(tmp_path):
    """A large required value must not be buffered whole to prove it saves."""
    namespace = {"big": bytearray(80 * 1024 * 1024)}
    output = _run_kernel_save(tmp_path, namespace, {"big"})

    assert _skipped_names(output) == set()
    assert "big" in _saved_names(output)


def _sandbox(tmp_path, required, session_id):
    import asyncio

    from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker
    from backend.utils.sandbox import LocalJupyterSandbox

    sandbox = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    tracker = RuntimeDependencyTracker()
    tracker.merge_declared(set(required))
    sandbox.runtime_dependency_tracker = tracker
    sandbox._exec_lock = asyncio.Lock()
    sandbox.runtime_revision = 1
    return sandbox


@pytest.mark.asyncio
async def test_probe_dropped_required_name_is_released_from_the_tracker(tmp_path):
    """Without the release the next save re-runs the identical failure."""
    from backend.utils.sandbox import CodeExecutionResult

    sandbox = _sandbox(tmp_path, {"res", "adata"}, "probe-release")

    async def _save(_code, **_kwargs):
        sandbox.notebook_state.session_path.write_bytes(b"DUMP")
        return CodeExecutionResult(
            success=True,
            output=(
                "__SCAI_SAVE_SKIPPED__\tres\tPrerank\tNotImplementedError\t"
                "CachedSession cannot be pickled\n"
                'Session saved: 1 variables\n__SCAI_SAVED_NAMES__:["adata"]'
            ),
        )

    sandbox._execute_unlocked = _save

    assert await sandbox.save_session() is True
    assert "res" not in sandbox.runtime_dependency_tracker.required_outputs
    assert "adata" in sandbox.runtime_dependency_tracker.required_outputs


@pytest.mark.asyncio
async def test_structural_mid_stream_failure_demotes_and_the_retry_commits(tmp_path):
    """A value too large for the probe still must not gate the run forever."""
    from backend.utils.sandbox import CodeExecutionResult

    sandbox = _sandbox(tmp_path, {"res"}, "probe-retry")
    attempts = []

    async def _save(code, **_kwargs):
        if "__USER_GLOBALS__" in code or "__PERSISTABLE_GLOBALS__" in code:
            return CodeExecutionResult(success=True, output="")
        attempts.append(code)
        if len(attempts) == 1:
            return CodeExecutionResult(
                success=False,
                output=(
                    "__SCAI_SAVE_SKIPPED__\tres\tPrerank\tNotImplementedError\t"
                    "CachedSession cannot be pickled"
                ),
                error="required runtime variable could not be persisted: res",
            )
        sandbox.notebook_state.session_path.write_bytes(b"DUMP")
        return CodeExecutionResult(
            success=True,
            output='Session saved: 0 variables\n__SCAI_SAVED_NAMES__:[]',
        )

    sandbox._execute_unlocked = _save

    assert await sandbox.save_session() is True
    assert len(attempts) == 2, "the retry must be a second, different dump"
    assert "res" not in sandbox.runtime_dependency_tracker.required_outputs
    assert sandbox.notebook_state.runtime_persistence["status"] != "failed"


@pytest.mark.asyncio
async def test_transient_failure_is_not_demoted_and_stays_fail_closed(tmp_path):
    """Only the pickler's structural refusals are deterministic.

    An OSError could genuinely succeed on the next attempt, so releasing the
    name on one would discard continuation state the run can still keep.
    """
    from backend.utils.sandbox import CodeExecutionResult

    sandbox = _sandbox(tmp_path, {"res"}, "probe-transient")

    async def _save(_code, **_kwargs):
        return CodeExecutionResult(
            success=False,
            output="__SCAI_SAVE_SKIPPED__\tres\tDataFrame\tOSError\tno space left",
            error="required runtime variable could not be persisted: res",
        )

    sandbox._execute_unlocked = _save

    with pytest.raises(RuntimeError, match="runtime save failed"):
        await sandbox.save_session()

    assert "res" in sandbox.runtime_dependency_tracker.required_outputs
    assert sandbox.notebook_state.runtime_persistence["failed_variables"] == ["res"]
