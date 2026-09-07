"""Spec 5.5: the user's own functions and classes survive a save/restore.

The exclusion used to be by kind -- every function, class, method and module was
dropped -- on the grounds that kernel init and notebook replay re-establish
them. That is true of an import and false of a definition: nothing
re-establishes a helper the user wrote in round two, so it was simply gone in
round three, and goals 2 and 3 promise otherwise.

The narrowing is bounded by design. A `__main__` definition dill cannot
serialise is skipped and recorded as coverage like any other value (spec 4.4);
it aborts nothing.

**The class caveat is real and is asserted here, not hidden.** Each variable is
dumped in its own frame with its own pickle memo, so dill embeds a copy of the
class in every instance it pickles by value. Carrying the class binds the name
to a *different* object from `type(instance)`. Before this change the name was
absent and the same code raised NameError, which is louder. Sharing identity
across frames would mean one shared memo, i.e. abandoning the per-frame spool,
which spec 5.2 keeps for a separate and stronger reason.
"""
from __future__ import annotations

import types

import pytest

dill = pytest.importorskip("dill")

# The namespace under test, built the way a kernel builds one: __name__ is
# "__main__", so definitions carry that module.
NAMESPACE_SOURCE = '''
class Marker:
    def __init__(self, tag):
        self.tag = tag

    def shout(self):
        return self.tag.upper()


def helper(x):
    return x * 2


import json as imported_module
from math import sqrt as imported_function

a = Marker("a")
b = Marker("b")
box = {"cls": Marker, "items": [a, b], "fn": helper}
plain = [1, 2, 3]
'''

SKIP_KINDS = (
    types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
    types.MethodType, type,
)


def build_namespace(extra: str = "") -> dict:
    ns = {"__name__": "__main__"}
    exec(NAMESPACE_SOURCE + extra, ns)
    return {k: v for k, v in ns.items() if not k.startswith("_")}


def is_own_definition(obj) -> bool:
    """The predicate the generated save script uses, mirrored.

    Mirrored rather than imported: it lives inside a source-template string that
    runs in the kernel, so there is nothing to import. The guard below asserts
    the two have not drifted.
    """
    if isinstance(obj, types.ModuleType):
        return False
    if not isinstance(obj, SKIP_KINDS):
        return False
    try:
        return getattr(obj, "__module__", None) == "__main__"
    except Exception:
        return False


def excluded_by_kind_today(obj) -> bool:
    return isinstance(obj, SKIP_KINDS)


def excluded_by_kind_narrowed(obj) -> bool:
    return isinstance(obj, SKIP_KINDS) and not is_own_definition(obj)


def round_trip(live: dict, excluded) -> tuple[dict, list[str]]:
    """One frame per variable, as `runtime_save` writes them.

    Returns the restored namespace and the names dill refused, which the real
    script records as coverage rather than raising on.
    """
    frames, skipped = [], []
    for name in sorted(live):
        obj = live[name]
        if excluded(obj):
            continue
        try:
            frames.append(dill.dumps((name, obj)))
        except Exception:
            skipped.append(name)
    restored = {}
    for blob in frames:
        name, obj = dill.loads(blob)
        restored[name] = obj
    return restored, skipped


# --- what the change is for --------------------------------------------------


def test_a_helper_defined_in_round_two_is_callable_after_restore():
    """The state goals 2 and 3 promise, lost today with no disclosure."""
    live = build_namespace()

    today, _ = round_trip(live, excluded_by_kind_today)
    assert "helper" not in today, "the premise changed; this is the old behaviour"

    narrowed, _ = round_trip(live, excluded_by_kind_narrowed)
    assert "helper" in narrowed
    assert narrowed["helper"](21) == 42


def test_a_method_on_a_carried_class_still_works():
    """Carrying the class is only worth it if its behaviour comes with it."""
    live = build_namespace()
    narrowed, _ = round_trip(live, excluded_by_kind_narrowed)

    assert "Marker" in narrowed
    assert narrowed["Marker"]("hi").shout() == "HI"


def test_an_imported_library_function_is_still_excluded():
    """`from math import sqrt` is re-established by the notebook's own import.

    Carrying it would bloat the dump and add a by-reference frame -- the single
    most likely thing to fail on restore in a different environment.
    """
    live = build_namespace()
    assert not is_own_definition(live["imported_function"])

    narrowed, _ = round_trip(live, excluded_by_kind_narrowed)
    assert "imported_function" not in narrowed


def test_a_module_object_is_still_excluded():
    """Unconditionally. Modules are the bulk the exclusion was aimed at, and a
    module is never the user's own state."""
    live = build_namespace()
    assert not is_own_definition(live["imported_module"])

    narrowed, _ = round_trip(live, excluded_by_kind_narrowed)
    assert "imported_module" not in narrowed


def test_an_instance_of_a_main_class_is_not_mistaken_for_a_definition():
    """`Marker("a").__module__` is "__main__" -- the lookup falls through to the
    class. Without the kind test first, every user object would be classified as
    a definition, which is how a predicate like this goes wrong silently."""
    live = build_namespace()

    assert getattr(live["a"], "__module__", None) == "__main__"
    assert not is_own_definition(live["a"])
    assert is_own_definition(live["Marker"])


def test_an_unserialisable_main_function_is_skipped_and_recorded():
    """Coverage, not an abort. Spec 4.4.

    This is what makes the narrowing bounded: widening what is offered to dill
    cannot turn a save that used to succeed into one that fails.
    """
    live = build_namespace(
        "\nimport threading\n"
        "def closes_over_a_lock(_l=threading.Lock()):\n"
        "    return _l\n"
    )
    assert is_own_definition(live["closes_over_a_lock"])

    restored, skipped = round_trip(live, excluded_by_kind_narrowed)

    # Whether dill refuses this particular closure is a dill detail; what must
    # hold either way is that the other values still arrived.
    assert "plain" in restored and restored["plain"] == [1, 2, 3]
    assert "a" in restored
    if skipped:
        assert "closes_over_a_lock" in skipped


# --- the divergence, pinned rather than deleted ------------------------------


def _relations(ns: dict) -> dict:
    got = {}
    a, b, box, cls = ns.get("a"), ns.get("b"), ns.get("box"), ns.get("Marker")
    if a is not None and b is not None:
        got["type(a) is type(b)"] = type(a) is type(b)
    if box is not None and a is not None:
        got["type(box.items[0]) is type(a)"] = type(box["items"][0]) is type(a)
        got["isinstance(box.items[0], box.cls)"] = isinstance(
            box["items"][0], box["cls"])
    if a is not None and cls is not None:
        got["type(a) is Marker"] = type(a) is cls
        got["isinstance(a, Marker)"] = isinstance(a, cls)
    return got


def test_class_identity_is_no_worse_than_before_the_change():
    """A2-10 Task 3 Step 3's paired comparison, as a test.

    Asserts the relation set under the narrowed exclusion is a superset of the
    one under the old exclusion -- not that identity is perfect, which it is not
    today either. "Worsens" is the question spec 5.5 asks, so a single
    `type(instance) is RestoredClass` would prove nothing either way.
    """
    live = build_namespace()

    today, _ = round_trip(live, excluded_by_kind_today)
    narrowed, _ = round_trip(live, excluded_by_kind_narrowed)

    before, after = _relations(today), _relations(narrowed)

    worsened = [k for k, v in before.items() if v is True and after.get(k) is not True]
    assert worsened == [], worsened


def test_a_carried_class_is_not_the_type_of_its_restored_instances():
    """The divergence itself, asserted so it cannot regress into a surprise.

    Each variable is its own frame with its own memo, so dill embedded a copy of
    `Marker` in `a` and another in `b`. Carrying the class adds a third object.
    Before this change `Marker` was absent and this code raised NameError.
    """
    live = build_namespace()
    narrowed, _ = round_trip(live, excluded_by_kind_narrowed)

    assert "Marker" in narrowed
    assert type(narrowed["a"]) is not narrowed["Marker"]
    assert not isinstance(narrowed["a"], narrowed["Marker"])
    # And the pre-existing divergence the narrowing does not touch:
    assert type(narrowed["a"]) is not type(narrowed["b"])
    # What does hold: objects that shared a frame shared a memo.
    assert isinstance(narrowed["box"]["items"][0], narrowed["box"]["cls"])


def test_the_save_script_discloses_what_it_carried():
    """A carried class ships with the caveat attached.

    Spec 4.4's rule is that a name absent from the dump is absent from every
    record of it, and this is the mirror: a name present under a caveat says so.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/utils/sandbox_parts/runtime_save.py")
    assert "__SCAI_CARRIED_DEFINITIONS__" in source
    assert "_carried_definitions" in source
    assert "restored class is not type(instance)" in source


def test_the_mirrored_predicate_matches_the_one_in_the_save_script():
    """The predicate above is a copy, because the original lives in a source
    template that runs in the kernel and cannot be imported. A copy that drifts
    would make every test in this file assert the wrong thing."""
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/utils/sandbox_parts/runtime_save.py")
    assert "def _scai_is_own_definition(_o):" in source
    for line in (
        "if isinstance(_o, _scai_types.ModuleType):",
        "if not isinstance(_o, _scai_skip_kinds):",
        'return getattr(_o, "__module__", None) == "__main__"',
    ):
        assert line in source, line


# --- the new record must not become a completeness requirement --------------


def _parse(output):
    from backend.utils.sandbox_parts.runtime_save import _parse_coverage_markers

    return _parse_coverage_markers(output)


THREE = (
    'Session saved: 1 variables\n'
    '__SCAI_SAVED_NAMES__:["adata"]\n'
    '__SCAI_EXCLUDED_KIND__:[]\n'
    '__SCAI_EXCLUDED_VOLATILE__:[]'
)


def test_a_producer_from_before_this_change_is_still_complete():
    """The compatibility question, and the reason the marker is optional.

    The parser treats "some markers but not all" as a broken producer whose
    generation must not commit. Putting the carried-definitions record in the
    completeness table would have reclassified every older kernel script as
    truncated -- a compatibility break wearing a disclosure's clothes.
    """
    record = _parse(THREE)

    assert record is not None
    assert record["saved_names"] == ["adata"]
    assert "carried_definitions" not in record


def test_the_carried_record_is_read_when_the_producer_emits_it():
    record = _parse(
        THREE + '\n__SCAI_CARRIED_DEFINITIONS__:'
        '[{"name": "helper", "kind": "function"}]'
    )

    assert record["carried_definitions"] == [
        {"name": "helper", "kind": "function"}
    ]


def test_an_unreadable_carried_record_is_dropped_not_fatal():
    """It describes what was brought in. A generation's completeness does not
    rest on it, so a malformed one must not stop a commit -- unlike a malformed
    exclusion marker, which must."""
    record = _parse(THREE + '\n__SCAI_CARRIED_DEFINITIONS__:{not json')

    assert record is not None
    assert "carried_definitions" not in record


def test_a_malformed_exclusion_marker_is_still_fatal():
    """The contrast that makes the previous test a decision rather than a
    leniency: the three completeness markers keep their old strictness."""
    from backend.utils.sandbox_parts.runtime_save import (
        RuntimeCoverageError,
        _parse_coverage_markers,
    )

    with pytest.raises(RuntimeCoverageError):
        _parse_coverage_markers(
            '__SCAI_SAVED_NAMES__:["adata"]\n'
            '__SCAI_EXCLUDED_KIND__:{not json\n'
            '__SCAI_EXCLUDED_VOLATILE__:[]'
        )


def test_a_partial_record_is_still_fatal():
    from backend.utils.sandbox_parts.runtime_save import (
        RuntimeCoverageError,
        _parse_coverage_markers,
    )

    with pytest.raises(RuntimeCoverageError):
        _parse_coverage_markers(
            '__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_EXCLUDED_KIND__:[]'
        )


def test_no_marker_at_all_is_still_not_a_claim_of_completeness():
    """A test double or a pre-transaction kernel. Returning an empty record
    would assert that nothing was excluded, which is the real hazard."""
    assert _parse("Session saved: 1 variables") is None


# --- the record has to reach disk, which is where the live run found it did not


def _manifest(**over):
    from backend.utils.runtime_manifest import build_runtime_manifest

    kwargs = dict(
        revision=3,
        state_revision=2,
        pkl_digest="a" * 64,
        saved_names=["adata", "helper"],
        skipped_names=[],
        serializer_version="dill-zstd-1",
    )
    kwargs.update(over)
    return build_runtime_manifest(**kwargs)


def test_the_carried_record_is_written_into_the_manifest():
    """Found by the first live run: the kernel emitted it, the parser read it,
    and `build_runtime_manifest` took only saved_names and skipped_names -- so
    nothing reached disk and a restore could not see it."""
    record = _manifest(carried_definitions=[{"name": "helper", "kind": "function"}])

    assert record["carried_definitions"] == [{"name": "helper", "kind": "function"}]


def test_a_carried_class_keeps_its_caveat_on_disk():
    """A name present under a caveat has to say so.

    The mirror of §4.4's rule that a name absent from the dump is absent from
    every record of it: a carried class is not `type(instance)` for instances
    restored from other frames, and that has to travel with it.
    """
    note = "restored class is not type(instance) for instances from other frames"
    record = _manifest(
        saved_names=["adata", "Marker"],
        carried_definitions=[{"name": "Marker", "kind": "class", "note": note}],
    )

    assert record["carried_definitions"][0]["note"] == note


def test_a_manifest_without_the_record_is_unchanged():
    """The compatibility requirement, and the reason the field is optional.

    `read_runtime_manifest` returns the *reconstruction*, so a generation
    written before §5.5's narrowing must rebuild byte-identically to what it
    does today -- the key absent, not present and empty.
    """
    assert "carried_definitions" not in _manifest()
    assert "carried_definitions" not in _manifest(carried_definitions=None)


def test_the_record_survives_a_write_and_read_round_trip(tmp_path):
    """The whole point. Writing it is not the same as reading it back: the read
    path reconstructs through the builder and drops any field it does not
    thread through, which is the same defect one level up."""
    from backend.utils.runtime_manifest import (
        read_runtime_manifest,
        write_runtime_manifest,
    )

    path = tmp_path / "session.runtime.json"
    write_runtime_manifest(
        path, _manifest(carried_definitions=[{"name": "helper", "kind": "function"}])
    )

    back = read_runtime_manifest(path)
    assert back["carried_definitions"] == [{"name": "helper", "kind": "function"}]


def test_an_old_manifest_still_reads(tmp_path):
    """A generation written before the field existed still verifies and loads."""
    from backend.utils.runtime_manifest import (
        read_runtime_manifest,
        write_runtime_manifest,
    )

    path = tmp_path / "session.runtime.json"
    write_runtime_manifest(path, _manifest())

    back = read_runtime_manifest(path)
    assert "carried_definitions" not in back
    assert back["saved_names"] == ["adata", "helper"]


def test_a_carried_name_the_dump_does_not_hold_is_refused():
    """The record describes this generation, not an aspiration for it.

    A manifest claiming to carry a definition the pkl does not contain would
    make a restore look for something that was never written -- the same class
    of lie as a saved_names entry with no frame behind it.
    """
    with pytest.raises(ValueError, match="does not hold"):
        _manifest(carried_definitions=[{"name": "absent", "kind": "function"}])


def test_an_unknown_carried_kind_is_refused():
    with pytest.raises(ValueError, match="unknown carried_definitions kind"):
        _manifest(carried_definitions=[{"name": "helper", "kind": "module"}])


def test_the_save_passes_the_record_through_to_the_manifest():
    """The three-hop path the live run showed was broken at hop three: the
    kernel prints it, the parser reads it, the save must hand it on."""
    from backend.tests.split_facade import source_for_path

    save_source = source_for_path("backend/utils/sandbox_parts/runtime_save.py")
    assert "carried_record = (coverage or {}).get(\"carried_definitions\")" in save_source
    assert '"carried_definitions": carried_record,' in save_source

    commit_source = source_for_path("backend/utils/sandbox_parts/regular_save.py")
    assert 'carried_definitions=runtime_manifest.get("carried_definitions")' in commit_source
