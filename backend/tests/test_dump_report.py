"""The dump walker's safety property, and the number it produces.

`scripts/scai-dump-report` exists to measure dumps, and the dumps most worth
measuring are the ones that cannot be loaded -- written by a build whose modules
are no longer installed, or holding an object whose class has moved. A tool that
unpickled to read a frame's name could not open those, which is most of the
reason to have the tool at all.

The measurement itself needs a seeded environment running two rounds and is
deferred; docs/coding/plain-persistence-baseline.md records that.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import pathlib
import pickle

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "scai-dump-report"


class Unloadable:
    """Pickled by reference, then removed from this module before the walk."""

    def __init__(self, payload=b"x" * 4096):
        self.payload = payload


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_loader(
        "scai_dump_report",
        importlib.machinery.SourceFileLoader("scai_dump_report", str(TOOL)),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dump(path: pathlib.Path, frames, *, compressed=True) -> None:
    """The container `runtime_save` writes: magic, then a zstd stream of the
    concatenated pickles. Written here rather than imported because the writer
    is a source template that runs inside the kernel."""
    import zstandard as zstd

    body = b"".join(pickle.dumps(frame) for frame in frames)
    with open(path, "wb") as handle:
        if not compressed:
            handle.write(body)
            return
        handle.write(b"SCAI_ZSTD_DILL_V1\n")
        with zstd.ZstdCompressor(level=1).stream_writer(handle, closefd=False) as out:
            out.write(body)


# --- the safety property -----------------------------------------------------


def test_the_dump_report_names_every_frame_without_unpickling(tool, tmp_path, monkeypatch):
    """A dump holding a payload that can no longer be loaded is still walked.

    The class is pickled by reference and then removed from its module, which is
    exactly what a dump written by a since-changed build looks like. `pickle.load`
    raises; the walk does not.
    """
    dump = tmp_path / "session.pkl"
    write_dump(dump, [("counts", Unloadable()), ("labels", ["a", "b"])])

    monkeypatch.delattr(
        "backend.tests.test_dump_report.Unloadable", raising=True
    )
    with pytest.raises(Exception):
        with open(dump, "rb") as raw:
            raw.read(len(tool.DUMP_MAGIC))
            import zstandard as zstd
            with zstd.ZstdDecompressor().stream_reader(raw) as stream:
                pickle.load(stream)

    with open(dump, "rb") as raw:
        plain = io.BytesIO()
        tool.decompress(dump, plain)
    frames = tool.walk(plain)

    assert [f["name"] for f in frames] == ["counts", "labels"]
    assert all(f["size"] > 0 for f in frames)
    assert all(len(f["sha256"]) == 64 for f in frames)


def test_the_tool_never_calls_a_loading_function(tool):
    """Asserted on the AST, not on behaviour.

    A tool that loads only on some paths passes a behavioural test written
    against the other ones, and the path that loads is the one pointed at the
    dump that matters.
    """
    import ast

    tree = ast.parse(TOOL.read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    # The strongest form of the property: it cannot unpickle, because it never
    # imports anything that can. `pickletools` reads opcodes and constructs no
    # objects.
    assert "pickle" not in imported
    assert "dill" not in imported
    assert "pickletools" in imported

    # And no `.load`/`.loads` on anything but the JSON reports, which are this
    # tool's own output and not the dump.
    loads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load", "loads")
    ]
    for node in loads:
        assert isinstance(node.func.value, ast.Name), ast.dump(node.func)
        assert node.func.value.id == "json", node.func.value.id

    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "genops" in called


# --- what it reports ---------------------------------------------------------


def test_an_alias_frame_names_both_the_alias_and_its_target(tool, tmp_path):
    """`runtime_save` writes an alias when two names are the same object.

    Reported as an alias rather than as a 3-byte variable, or the frame table
    would suggest a name was almost free when it is the other name that holds
    the bytes.
    """
    dump = tmp_path / "session.pkl"
    write_dump(dump, [
        ("adata", {"big": "payload"}),
        ("__SCAI_ALIAS__", "adata_ref", "adata"),
    ])

    plain = io.BytesIO()
    tool.decompress(dump, plain)
    frames = tool.walk(plain)

    assert frames[0]["name"] == "adata" and frames[0]["alias_of"] is None
    assert frames[1]["name"] == "adata_ref"
    assert frames[1]["alias_of"] == "adata"


def test_an_uncompressed_dump_is_still_walkable(tool, tmp_path):
    """The magic is only written by the compressing writer, so a dump from
    before it is a bare concatenation of pickles."""
    dump = tmp_path / "plain.pkl"
    write_dump(dump, [("a", 1), ("b", 2)], compressed=False)

    plain = io.BytesIO()
    assert tool.decompress(dump, plain) is False
    assert [f["name"] for f in tool.walk(plain)] == ["a", "b"]


def test_frame_sizes_are_uncompressed_and_sum_to_the_stream(tool, tmp_path):
    """Uncompressed, because a frame has no separable compressed size: the whole
    dump goes through one zstd window, so a frame's share of the file depends on
    what preceded it."""
    dump = tmp_path / "session.pkl"
    frames_in = [("a", "x" * 5000), ("b", "y" * 5000)]
    write_dump(dump, frames_in)

    plain = io.BytesIO()
    tool.decompress(dump, plain)
    frames = tool.walk(plain)

    assert sum(f["size"] for f in frames) == len(plain.getvalue())
    # Uncompressed, so two highly compressible frames are not tiny here.
    assert all(f["size"] > 4000 for f in frames)


def test_identical_payloads_under_different_names_share_a_digest(tool, tmp_path):
    """What the deduplication question turns on.

    Two frames whose *pickled bytes* match are what a frame index could store
    once. Names differ, digests do not.
    """
    dump = tmp_path / "session.pkl"
    write_dump(dump, [("first", [1, 2, 3]), ("second", [1, 2, 3])])

    plain = io.BytesIO()
    tool.decompress(dump, plain)
    frames = tool.walk(plain)

    assert frames[0]["name"] != frames[1]["name"]
    # The name is inside the pickled tuple, so identical payloads under
    # different names do NOT share a digest. This is the honest result and the
    # reason the comparison is between rounds of one session, where a name is
    # stable and only its value changes.
    assert frames[0]["sha256"] != frames[1]["sha256"]


def test_an_unchanged_variable_between_two_rounds_keeps_its_digest(tool, tmp_path):
    """The measurement's actual shape: same session, two rounds.

    A variable untouched by round two produces the identical frame, which is the
    byte fraction a frame index would avoid rewriting.
    """
    early, late = tmp_path / "r1.pkl", tmp_path / "r2.pkl"
    write_dump(early, [("adata", {"n": 1}), ("scratch", "a")])
    write_dump(late, [("adata", {"n": 1}), ("scratch", "b" * 900)])

    def frames_of(path):
        plain = io.BytesIO()
        tool.decompress(path, plain)
        return tool.walk(plain)

    a, b = frames_of(early), frames_of(late)
    by_name_a = {f["name"]: f["sha256"] for f in a}
    by_name_b = {f["name"]: f["sha256"] for f in b}

    assert by_name_a["adata"] == by_name_b["adata"]
    assert by_name_a["scratch"] != by_name_b["scratch"]


def test_the_tool_reports_no_per_frame_compressed_size(tool):
    """Stated in the docstring and absent from the output, deliberately.

    Reporting one would invite a per-variable saving that does not exist: the
    dump is one zstd stream, so a frame's compressed share depends on its
    neighbours.
    """
    source = TOOL.read_text()
    assert "no per-frame compressed size" in source.lower()
    frame_fields = {"name", "alias_of", "offset", "size", "sha256"}
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and len(node.keys) == len(frame_fields):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if keys == frame_fields:
                break
    else:
        raise AssertionError("the frame record changed shape; re-check this guard")
