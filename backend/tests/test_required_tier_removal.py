"""The required tier is gone: nothing about *which* values fail can abort a save.

Spec 4.4 splits integrity from coverage. Integrity — corruption — must block.
Coverage — what was excluded — must only warn. The required tier was the last
mechanism that let coverage block, and it is what 0.52.8/0.52.9 deadlocked on.
"""
from __future__ import annotations

import contextlib
import io

import pytest

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
