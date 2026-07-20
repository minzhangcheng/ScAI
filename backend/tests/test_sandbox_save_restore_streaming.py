"""
Tests for streaming per-variable save_session/restore_session format.

Format: a single .pkl file containing a sequence of dill-pickled
(var_name: str, value: Any) tuples, written one after another. Runtime aliases
use tagged three-item frames so the target object is not serialized twice.
Read by looping `dill.load` until EOFError.

An unpicklable required variable rolls back its half-written frame and rejects
the atomic save, so the previous complete runtime remains valid.
"""
import io
import dill
import pytest


def _write_stream(path, items, skipped):
    """Helper that mimics the new save_session per-var streaming write.
    Pure function so we can unit-test the format without spinning up a kernel.
    """
    with open(path, "wb") as f:
        for name, obj in items:
            checkpoint = f.tell()
            try:
                dill.dump((name, obj), f)
            except Exception as e:
                f.seek(checkpoint)
                f.truncate()
                skipped.append((name, type(obj).__name__, type(e).__name__, str(e)[:200]))


def _read_stream(path):
    """Helper that mimics the new _restore_session streaming read."""
    out = {}
    with open(path, "rb") as f:
        while True:
            try:
                name, obj = dill.load(f)
            except EOFError:
                break
            out[name] = obj
    return out


def test_round_trip_simple_values(tmp_path):
    path = tmp_path / "session.pkl"
    items = [("a", 1), ("b", [1, 2, 3]), ("c", {"x": "y"})]
    skipped = []
    _write_stream(path, items, skipped)
    assert skipped == []
    assert _read_stream(path) == {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}


def _make_unpicklable():
    """Return an object that dill cannot pickle.

    A generator object captures frame state that dill rejects.
    """
    gen = (i for i in range(3))
    next(gen)  # advance so it's a live, non-trivial generator
    return gen


def test_unpicklable_var_is_skipped_and_file_stays_valid(tmp_path):
    path = tmp_path / "session.pkl"
    items = [
        ("good_before", {"k": 1}),
        ("bad", _make_unpicklable()),
        ("good_after", [10, 20, 30]),
    ]
    skipped = []
    _write_stream(path, items, skipped)

    # Exactly one skip, with structured detail
    assert len(skipped) == 1
    name, type_name, exc_name, _msg = skipped[0]
    assert name == "bad"
    assert exc_name  # whatever exception dill raised

    # File must still be cleanly readable — no half-written frame
    assert _read_stream(path) == {"good_before": {"k": 1}, "good_after": [10, 20, 30]}


def test_first_var_unpicklable_yields_empty_but_valid_file(tmp_path):
    path = tmp_path / "session.pkl"
    items = [("bad", _make_unpicklable())]
    skipped = []
    _write_stream(path, items, skipped)
    assert len(skipped) == 1
    # Empty pkl should be readable as empty dict (zero frames before EOF)
    assert _read_stream(path) == {}


import os


def _compress_zstd(src, dst):
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=3)
    with open(src, "rb") as sf, open(dst, "wb") as df:
        cctx.copy_stream(sf, df)


def _decompress_zstd(src, dst):
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as sf, open(dst, "wb") as df:
        dctx.copy_stream(sf, df)


def test_snapshot_round_trip_compress_decompress(tmp_path):
    """Live pkl -> compress to snapshot .zst -> decompress back -> contents identical."""
    live = tmp_path / "session.pkl"
    snapshot = tmp_path / "snap.pkl.zst"
    restored = tmp_path / "session_restored.pkl"

    items = [
        ("alpha", list(range(1000))),
        ("beta", {"nested": {"x": [1, 2, 3]}}),
    ]
    skipped = []
    _write_stream(live, items, skipped)
    assert skipped == []

    _compress_zstd(live, snapshot)
    _decompress_zstd(snapshot, restored)

    assert _read_stream(restored) == {
        "alpha": list(range(1000)),
        "beta": {"nested": {"x": [1, 2, 3]}},
    }
    # Sanity: compression saved space on this repetitive payload
    assert snapshot.stat().st_size < live.stat().st_size


@pytest.mark.asyncio
async def test_save_targets_tracked_dependencies_without_anndata_sweep(tmp_path, monkeypatch):
    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox
    from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker

    sb = LocalJupyterSandbox(
        session_id="tracked-save",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    tracker = RuntimeDependencyTracker()
    tracker.merge_declared({"analysis_result"})
    sb.runtime_dependency_tracker = tracker
    captured = {}

    async def _capture(code, **_kwargs):
        captured["code"] = code
        return CodeExecutionResult(success=True, output="Session saved: 1 variables")

    monkeypatch.setattr(sb, "_execute_unlocked", _capture)
    assert await sb.save_session()
    assert "analysis_result" in captured["code"]
    assert "isinstance(_obj, ad.AnnData)" not in captured["code"]
    assert "__SCAI_ALIAS__" in captured["code"]


@pytest.mark.asyncio
async def test_incomplete_required_runtime_save_raises(tmp_path, monkeypatch):
    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id="failed-save",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )

    async def _fail(_code, **_kwargs):
        return CodeExecutionResult(
            success=False,
            output="__SCAI_SAVE_SKIPPED__\tneeded\tThing\tTypeError\tbad state",
            error="required runtime variables could not be persisted: needed",
        )

    monkeypatch.setattr(sb, "_execute_unlocked", _fail)
    with pytest.raises(RuntimeError, match="runtime save failed"):
        await sb.save_session()
    assert sb._last_save_skipped[0]["name"] == "needed"


@pytest.mark.asyncio
async def test_real_kernel_save_restore_round_trip(tmp_path):
    """Spin up a real kernel, write a variable, save_session, kill kernel, restart, restore."""
    from backend.utils.sandbox import LocalJupyterSandbox

    session_id = "test-save-restore-roundtrip"
    states_dir = str(tmp_path / "states")
    workspace_dir = str(tmp_path / "workspace")

    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=states_dir,
        working_dir=workspace_dir,
        timeout=60,
    )
    await sb.start()
    try:
        r = await sb.execute("adata = {'cells': 1234, 'genes': ['a', 'b', 'c']}")
        assert r.success, r.error

        assert await sb.save_session()

        # Fresh sandbox, same session_id — should see the saved pkl and restore it
        sb2 = LocalJupyterSandbox(
            session_id=session_id,
            state_dir=states_dir,
            working_dir=workspace_dir,
            timeout=60,
        )
        await sb2.start()
        try:
            check = await sb2.execute("print(adata)")
            assert check.success, check.error
            assert "1234" in check.output
            assert "['a', 'b', 'c']" in check.output
        finally:
            await sb2.stop()
    finally:
        await sb.stop()
