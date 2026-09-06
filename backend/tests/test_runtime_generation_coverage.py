"""Coverage describes the dump, or the generation does not commit.

A name excluded by kind or as a volatile handle is absent from the dump exactly
like a name dill refused, but until now the two exclusion paths reported
nothing. A manifest built from the surviving channel alone would declare a
complete generation while three classes of absence went unrecorded -- so a
missing or malformed marker fails the commit rather than degrading to an empty
set.
"""
import contextlib
import io
import sqlite3

import pytest

try:
    from backend.utils.sandbox_parts.runtime_save import (
        RuntimeCoverageError,
        SaveContext,
        _build_save_code,
        _classify_skipped,
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


def test_the_three_channels_are_distinguished():
    parsed = _classify_skipped(
        [{"name": "ss", "type": "SingleSampleGSEA", "exc": "TypeError"}],
        excluded_by_kind=[{"name": "helper", "type": "function"}],
        excluded_volatile=[{"name": "fh", "type": "_io.TextIOWrapper"}],
    )
    by_name = {entry["name"]: entry for entry in parsed}
    assert by_name["ss"]["reason"] == "serialization_failed"
    assert by_name["ss"]["type"] == "SingleSampleGSEA"
    assert by_name["helper"]["reason"] == "excluded_by_kind"
    assert by_name["helper"]["type"] == "function"
    assert by_name["fh"]["reason"] == "excluded_volatile"
    assert by_name["fh"]["type"] == "_io.TextIOWrapper"


def test_every_entry_carries_a_type():
    """{name, type, reason} is the contract; an empty type does not honour it."""
    parsed = _classify_skipped(
        [{"name": "ss"}],
        excluded_by_kind=[{"name": "helper"}],
        excluded_volatile=[],
    )
    assert all(entry["type"] for entry in parsed)


def test_the_record_is_ordered_so_the_manifest_digest_is_stable():
    parsed = _classify_skipped(
        [{"name": "zeta", "type": "X"}],
        excluded_by_kind=[{"name": "alpha", "type": "function"}],
        excluded_volatile=[],
    )
    assert [entry["name"] for entry in parsed] == ["alpha", "zeta"]


def test_a_partial_record_fails_the_generation():
    """Some markers but not all means the producer broke mid-report."""
    output = '__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_EXCLUDED_KIND__:[]\n'
    with pytest.raises(RuntimeCoverageError, match="partial"):
        _parse_coverage_markers(output)


def test_no_markers_at_all_means_this_producer_reports_no_coverage():
    """Absence is not a claim of completeness, so it is not an empty record.

    A pre-transaction custom kernel reports nothing. Returning an empty record
    would assert that nothing was excluded; returning None lets the generation
    commit without a manifest and be restored as legacy, coverage unknown.
    """
    assert _parse_coverage_markers("Session saved: 3 variables") is None
    assert _parse_coverage_markers("") is None


def test_a_repeated_marker_fails_the_generation():
    output = (
        '__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_SAVED_NAMES__:["other"]\n'
        "__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]\n"
    )
    with pytest.raises(RuntimeCoverageError, match="repeated"):
        _parse_coverage_markers(output)


def test_a_malformed_marker_fails_the_generation():
    output = (
        "__SCAI_SAVED_NAMES__:not-json\n"
        "__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]\n"
    )
    with pytest.raises(RuntimeCoverageError, match="malformed"):
        _parse_coverage_markers(output)


def test_a_marker_of_the_wrong_shape_fails_the_generation():
    output = (
        '__SCAI_SAVED_NAMES__:{"adata": 1}\n'
        "__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]\n"
    )
    with pytest.raises(RuntimeCoverageError, match="malformed"):
        _parse_coverage_markers(output)


def test_a_well_formed_output_parses():
    output = (
        '__SCAI_SAVED_NAMES__:["adata"]\n'
        '__SCAI_EXCLUDED_KIND__:[{"name":"helper","type":"function"}]\n'
        "__SCAI_EXCLUDED_VOLATILE__:[]\n"
    )
    parsed = _parse_coverage_markers(output)
    assert parsed["saved_names"] == ["adata"]
    assert parsed["excluded_by_kind"] == [{"name": "helper", "type": "function"}]
    assert parsed["excluded_volatile"] == []


def _save_context(tmp_path):
    return SaveContext(
        sandbox=None,
        restore_failed=False,
        revision=1,
        expected_lifecycle_generation=1,
        required_names=set(),
        escaped_required=repr([]),
        escaped_path=repr(str(tmp_path / "sid_session.pkl")),
        record_generation_failure=None,
        restore_previous_persistence=None,
        SessionGenerationError=RuntimeError,
        SessionGoneError=RuntimeError,
        assert_session_generation_active=None,
        lifecycle_file_lock=None,
    )


def test_generated_save_reports_all_three_channels(tmp_path):
    """Drive the real generated kernel code over a literal namespace."""
    namespace = {
        "adata": [1, 2, 3],
        "helper": lambda value: value,
        "conn": sqlite3.connect(":memory:"),
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(
            compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"),
            dict(namespace),
        )

    parsed = _parse_coverage_markers(stream.getvalue())
    assert parsed["saved_names"] == ["adata"]
    assert [entry["name"] for entry in parsed["excluded_by_kind"]] == ["helper"]
    assert [entry["name"] for entry in parsed["excluded_volatile"]] == ["conn"]
    assert all(entry["type"] for entry in parsed["excluded_by_kind"])
    assert all(entry["type"] for entry in parsed["excluded_volatile"])


def test_the_save_script_does_not_clobber_user_variables(tmp_path):
    """Saving must not destroy state merely by needing an import of its own.

    The generated code runs in the kernel's own globals and used to import
    `dill`, `io`, `os`, `tempfile`, `types` and `zstd` under those names,
    rebinding any user variable that happened to share one -- before `_live`
    was collected, so the original value was gone from the dump as well as from
    the namespace.
    """
    collided = {name: f"user value for {name}" for name in
                ("dill", "io", "os", "tempfile", "types", "zstd", "json")}
    namespace = {"adata": [1, 2, 3], **collided}
    stream = io.StringIO()
    executed = dict(namespace)
    with contextlib.redirect_stdout(stream):
        exec(compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"), executed)

    for name, value in collided.items():
        assert executed[name] == value, f"the save script rebound the user variable {name!r}"

    parsed = _parse_coverage_markers(stream.getvalue())
    assert set(parsed["saved_names"]) == {"adata", *collided}
    assert parsed["excluded_by_kind"] == []
    assert parsed["excluded_volatile"] == []


def test_no_caller_passes_the_probe_flag_to_the_unlocked_executor():
    """``_execute_unlocked`` has no ``mutates_namespace`` parameter.

    A blind replace once added it to one such call, so every dill restore
    failure raised TypeError and replaced the real diagnosis with it.
    """
    import inspect
    import re
    from pathlib import Path

    from backend.utils.sandbox_parts.execution import ExecutionMixin

    signature = inspect.signature(ExecutionMixin._execute_unlocked)
    assert "mutates_namespace" not in signature.parameters

    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(
        r"_execute_unlocked\((?:[^()]|\([^()]*\))*mutates_namespace", re.S
    )
    offenders = [
        str(path.relative_to(root))
        for path in (root / "backend").rglob("*.py")
        if "tests" not in path.parts and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"_execute_unlocked called with mutates_namespace in {offenders}"
