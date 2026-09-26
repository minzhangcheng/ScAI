"""The required tier is gone: nothing about *which* values fail can abort a save.

Spec 4.4 splits integrity from coverage. Integrity — corruption — must block.
Coverage — what was excluded — must only warn. The required tier was the last
mechanism that let coverage block, and it is what 0.52.8/0.52.9 deadlocked on.
"""
from __future__ import annotations

import contextlib
import io
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"

try:
    from backend.utils.sandbox_parts.runtime_save import (
        SaveContext,
        _build_save_code,
        _parse_coverage_markers,
    )
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


def _save_context(tmp_path):
    """A SaveContext with no required-set fields.

    Constructing it is itself part of the contract: after this batch those
    fields do not exist, so a test that still passes them would not compile
    against the module it is testing.
    """
    return SaveContext(
        sandbox=None,
        restore_failed=False,
        revision=1,
        expected_lifecycle_generation=1,
        escaped_path=repr(str(tmp_path / "sid_session.pkl")),
        record_generation_failure=None,
        restore_previous_persistence=None,
        SessionGenerationError=RuntimeError,
        SessionGoneError=RuntimeError,
        assert_session_generation_active=None,
        lifecycle_file_lock=None,
    )


class _Stubborn:
    """Deterministically unserialisable: a retry is the identical computation."""

    def __reduce__(self):
        raise TypeError("this value cannot be pickled, ever")


def test_an_unserialisable_value_is_skipped_and_the_rest_is_saved(tmp_path):
    """The save completes. The value is recorded as coverage, not as an abort."""
    namespace = {"adata": [1, 2, 3], "stubborn": _Stubborn()}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(
            compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"),
            dict(namespace),
        )

    parsed = _parse_coverage_markers(stream.getvalue())
    assert parsed["saved_names"] == ["adata"]
    assert "__SCAI_SAVE_SKIPPED__\tstubborn" in stream.getvalue()


def test_a_namespace_of_nothing_but_unserialisable_values_still_commits(tmp_path):
    """The pathological case the required tier turned into a wedged save."""
    namespace = {"a": _Stubborn(), "b": _Stubborn()}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(
            compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"),
            dict(namespace),
        )

    parsed = _parse_coverage_markers(stream.getvalue())
    assert parsed["saved_names"] == []


@pytest.mark.asyncio
async def test_a_cell_that_does_not_create_its_declared_output_still_fails(tmp_path, monkeypatch):
    """Spec 4.7 keeps this: a contract about the agent's own statement.

    Ported from test_sandbox_globals_diff_prune.py, which is deleted with the
    dependency tracker it was written for. The half that checked the tracker was
    not poisoned goes with the tracker; the half that checks the cell fails is
    the part that survives, and it is cheap to fix in the cell that binds it.
    """
    from unittest.mock import AsyncMock

    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id="missing-declared-output",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(sb, "_snapshot_user_globals", AsyncMock(side_effect=[set(), set()]))
    monkeypatch.setattr(
        sb,
        "_execute_unlocked",
        AsyncMock(return_value=CodeExecutionResult(success=True, output="error swallowed")),
    )

    result = await sb.execute(
        "try:\n    raise RuntimeError('failed')\nexcept Exception:\n    pass",
        declared_outputs=["interact_df"],
        track_globals=True,
    )

    assert result.success is False
    assert "Declared persistent outputs were not created: interact_df" in result.error
class _MachineFailure:
    """The machine is the problem. A retry could genuinely reach a different
    outcome, so it must not be recorded as a property of the value."""

    def __reduce__(self):
        raise OSError(28, "No space left on device")


def test_an_environment_failure_aborts_the_generation(tmp_path):
    """An ENOSPC is not a property of the value and must not commit as current.

    Spec 4.4 lists three coverage reasons -- serialisation failed, excluded by
    kind, excluded as volatile -- and every one is a fact about the value.
    Skipping a variable because the disk filled records a machine failure as a
    value failure, commits the generation as ``current`` (which resets
    ``failed_variables`` and ``error_summary``), and leaves the only trace in a
    manifest field nothing reads.
    """
    namespace = {"adata": [1, 2, 3], "unlucky": _MachineFailure()}
    stream = io.StringIO()
    with pytest.raises(Exception) as excinfo:
        with contextlib.redirect_stdout(stream):
            exec(
                compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"),
                dict(namespace),
            )
    assert "unlucky" in str(excinfo.value)
    assert not (tmp_path / "sid_session.pkl.tmp").exists()
