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


def test_the_generated_save_has_no_required_name_abort(tmp_path):
    code = _build_save_code(_save_context(tmp_path))
    for gone in (
        "_required_names",
        "_missing_required",
        "required runtime variables are missing",
        "required runtime variable could not be persisted",
        "_scai_host_required",
    ):
        assert gone not in code, gone


def test_the_persistability_probe_is_gone_and_every_frame_spools(tmp_path):
    """Spec 5.2: the spool is what keeps a serialisation failure off the stream.

    The probe existed only to let a *required* value prove itself before
    streaming. With no required set, every frame takes the spool path, which is
    what non-required values already did.
    """
    code = _build_save_code(_save_context(tmp_path))
    assert "_ScaiProbeSink" not in code
    assert "_ScaiProbeTooLarge" not in code
    assert "_scai_probe_budget" not in code
    assert "SpooledTemporaryFile" in code


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


def test_the_save_context_no_longer_carries_a_required_set():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(SaveContext)}
    assert "required_names" not in fields
    assert "escaped_required" not in fields
def test_a_declared_name_that_is_absent_no_longer_fails_the_save(tmp_path):
    """The inverse of the deleted ``test_missing_required_variable_still_fails_closed``.

    A name the agent declared and the cell did not create used to abort the
    generation at save time -- minutes later, addressed to the operator, about a
    cell the agent could no longer change. Spec 4.7 keeps the declared-output
    check in the cell that binds the value, where it is cheap to fix; the
    save-time abort is what goes.
    """
    namespace = {"something_else": 1}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(
            compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"),
            dict(namespace),
        )

    parsed = _parse_coverage_markers(stream.getvalue())
    assert parsed["saved_names"] == ["something_else"]
def test_the_sandbox_carries_no_dependency_tracker():
    """Asserted over the whole sandbox split, not one part.

    ``backend/utils/sandbox.py`` is a facade over ``sandbox_parts/``; a negative
    assertion read from the facade's own file would pass unconditionally.
    """
    from backend.tests.split_facade import split_source

    text = split_source("backend.utils.sandbox")
    for gone in (
        "runtime_dependency_tracker",
        "RuntimeDependencyTracker",
        "get_runtime_dependency_names",
        "record_declared_outputs",
        "release_runtime_dependencies",
    ):
        assert gone not in text, gone


def test_only_snapshot_v2_still_mentions_the_retired_tracker_state_key():
    """``runtime_dependency_tracker_v2`` had one writer and one reader.

    The writer was the executor; the reader is Snapshot V2's capture pipeline,
    which is deleted in A2-5. Nothing else ever consumed the key, so there is no
    read-compatibility question to answer -- only a producer to remove.
    """
    offenders = set()
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(BACKEND.parent).as_posix()
        if rel.startswith(("backend/tests/", "backend/services/snapshot_v2/")):
            continue
        if "runtime_dependency_tracker" in path.read_text():
            offenders.add(rel)
    assert not offenders, sorted(offenders)


def test_the_execution_diff_outlives_the_tracker():
    """created/loaded/removed is sandbox state in its own right and stays."""
    from backend.tests.split_facade import split_source

    text = split_source("backend.utils.sandbox")
    assert "_last_execution_diff" in text
    assert "_known_user_globals" in text
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
def test_the_cell_time_persistability_probe_is_gone():
    """Spec 4.7: the probe's verdict has no producer once the tier is deleted.

    Serialisation coverage now comes from what the save actually wrote, and it
    reaches the agent through spec section 10 -- not from a judgment made in the
    cell that bound the value.
    """
    from backend.tests.split_facade import split_source

    text = split_source("backend.utils.sandbox")
    for gone in (
        "_surface_unpersistable_outputs",
        "_snapshot_globals_with_persistability",
        "Declared persistent outputs cannot be serialized",
        "Bound but not carried across a kernel restart",
    ):
        assert gone not in text, gone


@pytest.mark.asyncio
async def test_a_cell_binding_an_unserialisable_value_succeeds(tmp_path, monkeypatch):
    """The computation succeeded. Whether it serialises is the save's business.

    Section 3.11 shows unserialisable values arise on every run, so failing a
    cell whose computation succeeded was never the right response.
    """
    from unittest.mock import AsyncMock

    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id="unserialisable-but-fine",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    # The probe is driven, not mocked away: the second _execute_unlocked is the
    # probe round trip, and it returns the exact markers the probe parses, so on
    # the unpatched tree the verdict really is produced and really does fail the
    # cell. Mocking _snapshot_globals_with_persistability instead would make
    # this pass for the wrong reason.
    probe_output = (
        '__USER_GLOBALS__:["conn"]\n'
        '__UNPERSISTABLE__:{"conn": "sqlite3.Connection"}\n'
        '__PROBED__:["conn"]\n'
    )
    # Enough entries for both shapes: with the probe present the kernel is asked
    # twice through _execute_unlocked and _snapshot_user_globals once; with it
    # gone the post-cell snapshot is a second _snapshot_user_globals call. The
    # spare entries are unused on whichever side is running.
    monkeypatch.setattr(
        sb, "_snapshot_user_globals", AsyncMock(side_effect=[set(), {"conn"}])
    )
    monkeypatch.setattr(
        sb,
        "_execute_unlocked",
        AsyncMock(side_effect=[
            CodeExecutionResult(success=True),
            CodeExecutionResult(success=True, output=probe_output),
        ]),
    )

    result = await sb.execute(
        "import sqlite3\nconn = sqlite3.connect(':memory:')",
        declared_outputs=["conn"],
        track_globals=True,
    )

    assert result.success is True, result.error
    assert "cannot be serialized" not in (result.error or "")
    assert "not carried across a kernel restart" not in (result.output or "")
