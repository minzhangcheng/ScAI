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
