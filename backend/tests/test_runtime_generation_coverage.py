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


def test_the_private_channel_is_distinguished_from_the_other_three():
    """A fourth way to be absent, and it must not borrow another's reason."""
    parsed = _classify_skipped(
        [{"name": "ss", "type": "SingleSampleGSEA", "exc": "TypeError"}],
        excluded_by_kind=[{"name": "helper", "type": "function"}],
        excluded_volatile=[{"name": "fh", "type": "_io.TextIOWrapper"}],
        excluded_private=[{"name": "_adata_raw", "type": "AnnData"}],
    )
    by_name = {entry["name"]: entry for entry in parsed}
    assert by_name["_adata_raw"]["reason"] == "excluded_private"
    assert by_name["_adata_raw"]["type"] == "AnnData"
    assert by_name["helper"]["reason"] == "excluded_by_kind"


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


def test_a_private_user_variable_is_recorded_rather_than_dropped(tmp_path):
    """§4.4's rule applied to the one exclusion that used to bypass it.

    Dropping underscore names is right -- IPython's history lives in the same
    namespace and is not user state. Dropping them *silently* is not: the name
    was then absent from the dump and absent from every record of the dump,
    which is exactly the outcome the two reported channels exist to prevent.
    """
    namespace = {
        "adata": [1, 2, 3],
        "_adata_raw": [4, 5, 6],
        "_i1": "print(1)",
        "_ii": "print(0)",
        "_3": "an Out value",
        "__": "second-to-last output",
        "_oh": {},
        "_dh": [],
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(
            compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"),
            dict(namespace),
        )

    parsed = _parse_coverage_markers(stream.getvalue())
    assert parsed["saved_names"] == ["adata"]

    private = {entry["name"] for entry in parsed["excluded_private"]}
    assert "_adata_raw" in private, "a user's private value must be recorded"
    assert all(entry["type"] for entry in parsed["excluded_private"])

    # The shell's own history is noise in a record meant for names a user might
    # go looking for, and `__builtins__` is the interpreter's, not theirs.
    assert not (private & {"_i1", "_ii", "_3", "__", "_oh", "_dh", "__builtins__"}), (
        f"shell history leaked into the coverage record: {sorted(private)}"
    )


def test_a_private_name_reaches_the_manifest_with_its_own_reason():
    """`build_runtime_manifest` is the gate; an unknown reason must not pass."""
    from backend.utils.runtime_manifest import SKIP_REASONS, build_runtime_manifest

    assert "excluded_private" in SKIP_REASONS

    manifest = build_runtime_manifest(
        revision=3,
        state_revision=7,
        pkl_digest="a" * 64,
        saved_names=["adata"],
        skipped_names=[
            {"name": "_adata_raw", "type": "AnnData", "reason": "excluded_private"},
        ],
        serializer_version="dill-zstd-1",
    )
    entry = manifest["skipped_names"][0]
    assert entry == {
        "name": "_adata_raw", "type": "AnnData", "reason": "excluded_private",
    }


def test_a_skipped_main_closure_is_not_claimed_as_carried(tmp_path):
    """The carried record describes saved names, not every candidate.

    A closure retaining a generator is discovered as a user definition before
    dill rejects it.  The ordinary data value must still commit, while the
    strict manifest receives no claim for the skipped helper.
    """
    from backend.utils.sandbox_parts.runtime_save import (
        _carried_definitions_for_saved_names,
    )
    from backend.utils.sandbox_parts.runtime_save import _classify_skipped
    from backend.utils.runtime_manifest import build_runtime_manifest

    namespace = {"__name__": "__main__"}
    exec(
        "def make_helper():\n"
        "    pending = (item for item in range(3))\n"
        "    def helper():\n"
        "        return next(pending)\n"
        "    return helper\n"
        "helper = make_helper()\n"
        "del make_helper\n"
        "adata = [1, 2, 3]\n",
        namespace,
    )
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(
            compile(_build_save_code(_save_context(tmp_path)), "<save_code>", "exec"),
            namespace,
        )

    parsed = _parse_coverage_markers(stream.getvalue())
    assert "adata" in parsed["saved_names"]
    assert "helper" not in parsed["saved_names"]
    assert any(item["name"] == "helper" for item in parsed["carried_definitions"])

    skipped = []
    for line in stream.getvalue().splitlines():
        if line.startswith("__SCAI_SAVE_SKIPPED__\t"):
            parts = line.split("\t")
            skipped.append({"name": parts[1], "type": parts[2]})
    skipped_record = _classify_skipped(
        skipped,
        excluded_by_kind=parsed["excluded_by_kind"],
        excluded_volatile=parsed["excluded_volatile"],
    )
    assert any(
        item["name"] == "helper" and item["reason"] == "serialization_failed"
        for item in skipped_record
    )

    carried = _carried_definitions_for_saved_names(
        parsed, {str(name) for name in parsed["saved_names"]}
    )
    assert carried is None or all(item["name"] != "helper" for item in carried)
    manifest = build_runtime_manifest(
        revision=1,
        state_revision=1,
        pkl_digest="a" * 64,
        saved_names=parsed["saved_names"],
        skipped_names=skipped_record,
        serializer_version="dill-zstd-1",
        carried_definitions=carried,
    )
    assert "carried_definitions" not in manifest


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
