"""
Tests for streaming per-variable save_session/restore_session format.

Format: a marker followed by a streaming zstd sequence of dill-pickled frames.
Legacy uncompressed frame streams remain readable.

An unpicklable required variable rolls back its half-written frame and rejects
the atomic save, so the previous complete runtime remains valid.
"""
import io
import dill
import pytest
from pathlib import Path

_MAGIC = b"SCAI_ZSTD_DILL_V1\n"


def _write_stream(path, items, skipped):
    """Helper that mimics the new save_session per-var streaming write.
    Pure function so we can unit-test the format without spinning up a kernel.
    """
    import zstandard as zstd
    with open(path, "wb") as raw:
        raw.write(_MAGIC)
        with zstd.ZstdCompressor(level=1).stream_writer(raw, closefd=False) as f:
            for name, obj in items:
                try:
                    dill.dump((name, obj), f)
                except Exception as e:
                    skipped.append((name, type(obj).__name__, type(e).__name__, str(e)[:200]))
                    raise


def _read_stream(path):
    """Helper that mimics the new _restore_session streaming read."""
    out = {}
    import zstandard as zstd

    def _read(f):
        while True:
            try:
                name, obj = dill.load(f)
            except EOFError:
                break
            out[name] = obj
    with open(path, "rb") as raw:
        if raw.read(len(_MAGIC)) == _MAGIC:
            with zstd.ZstdDecompressor().stream_reader(raw) as f:
                _read(f)
        else:
            raw.seek(0)
            _read(raw)
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


def test_unpicklable_var_aborts_stream(tmp_path):
    path = tmp_path / "session.pkl"
    items = [
        ("good_before", {"k": 1}),
        ("bad", _make_unpicklable()),
        ("good_after", [10, 20, 30]),
    ]
    skipped = []
    with pytest.raises(Exception):
        _write_stream(path, items, skipped)

    # Exactly one skip, with structured detail
    assert len(skipped) == 1
    name, type_name, exc_name, _msg = skipped[0]
    assert name == "bad"
    assert exc_name  # whatever exception dill raised

    # Production writes to .tmp and never publishes this incomplete stream.


def test_first_var_unpicklable_aborts_stream(tmp_path):
    path = tmp_path / "session.pkl"
    items = [("bad", _make_unpicklable())]
    skipped = []
    with pytest.raises(Exception):
        _write_stream(path, items, skipped)
    assert len(skipped) == 1


def test_reads_legacy_uncompressed_stream(tmp_path):
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as stream:
        dill.dump(("legacy", {"value": 3}), stream)
    assert _read_stream(path) == {"legacy": {"value": 3}}


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
    # Legacy snapshot code may apply an outer zstd layer to the already
    # compressed live stream. The round trip must stay lossless even when that
    # tiny outer frame is slightly larger than its input.
    assert snapshot.stat().st_size > 0


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
        # Production kernel code leaves a staged file for the host-side
        # authenticated commit.  Preserve that boundary in this code-capture
        # test instead of exercising the legacy custom-kernel fallback.
        Path(str(sb.notebook_state.session_path) + ".tmp").write_bytes(b"staged")
        return CodeExecutionResult(success=True, output="Session saved: 1 variables")

    monkeypatch.setattr(sb, "_execute_unlocked", _capture)
    assert await sb.save_session()
    assert "analysis_result" in captured["code"]
    assert "isinstance(_obj, ad.AnnData)" not in captured["code"]
    assert "__SCAI_ALIAS__" in captured["code"]
    assert "SCAI_ZSTD_DILL_V1" in captured["code"]
    assert "ZstdCompressor" in captured["code"]
    assert "SpooledTemporaryFile" in captured["code"]
    assert "_scai_stream_safe" not in captured["code"]


@pytest.mark.asyncio
async def test_optional_unpicklable_value_does_not_abort_later_frames(
    tmp_path, monkeypatch,
):
    import contextlib

    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id="optional-save-skip",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )

    async def _execute_generated(code, **_kwargs):
        namespace = {
            "a_good_before": {"value": 1},
            "b_bad_optional": _make_unpicklable(),
            "c_good_after": [2, 3],
        }
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(code, namespace, namespace)
        except Exception as exc:
            return CodeExecutionResult(
                success=False,
                output=output.getvalue(),
                error=str(exc),
            )
        return CodeExecutionResult(success=True, output=output.getvalue())

    monkeypatch.setattr(sb, "_execute_unlocked", _execute_generated)

    assert await sb.save_session() is True
    assert [item["name"] for item in sb._last_save_skipped] == [
        "b_bad_optional"
    ]
    restored = _read_stream(Path(sb.notebook_state.session_path))
    assert restored["a_good_before"] == {"value": 1}
    assert restored["c_good_after"] == [2, 3]
    assert "b_bad_optional" not in restored


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


@pytest.mark.asyncio
async def test_real_kernel_backed_anndata_materializes_with_aliases_or_fails_closed(
    tmp_path,
):
    """Regular persistence never silently drops a backed AnnData handle."""
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    from backend.utils.sandbox import LocalJupyterSandbox

    session_id = "test-backed-anndata-regular-save"
    states_dir = tmp_path / "states"
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    input_h5ad = workspace_dir / "input.h5ad"
    ad.AnnData(np.arange(24, dtype=np.float32).reshape(6, 4)).write_h5ad(
        input_h5ad
    )

    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(states_dir),
        working_dir=str(workspace_dir),
        timeout=60,
    )
    await sb.start()
    try:
        created = await sb.execute(
            "import anndata as ad\n"
            f"adata_backed = ad.read_h5ad({str(input_h5ad)!r}, backed='r')\n"
            "adata_alias = adata_backed\n"
            "print(adata_backed.isbacked, adata_alias is adata_backed)"
        )
        assert created.success, created.error
        assert "True True" in created.output
        assert await sb.save_session()

        committed = sb.notebook_state.session_path.read_bytes()
        broken = await sb.execute(
            f"broken = ad.read_h5ad({str(input_h5ad)!r}, backed='r')\n"
            "broken.file.close()\n"
            "print(broken.isbacked)"
        )
        assert broken.success, broken.error
        assert "True" in broken.output
        input_h5ad.unlink()
        with pytest.raises(RuntimeError, match="runtime save failed"):
            await sb.save_session()
        # The failed materialization cannot replace the verified generation.
        assert sb.notebook_state.session_path.read_bytes() == committed
    finally:
        # The test intentionally leaves a backed object whose source file was
        # deleted. Its failed live generation has already been verified not to
        # replace the committed dump, so test cleanup explicitly discards it.
        await sb.stop(save=False)

    restored = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(states_dir),
        working_dir=str(workspace_dir),
        timeout=60,
    )
    await restored.start()
    try:
        check = await restored.execute(
            "print(adata_backed.isbacked)\n"
            "print(adata_alias is adata_backed)\n"
            "print(adata_backed.shape, float(adata_backed.X.sum()))\n"
            "print('broken' in globals())"
        )
        assert check.success, check.error
        lines = [line.strip() for line in check.output.splitlines() if line.strip()]
        assert lines == ["False", "True", "(6, 4) 276.0", "False"]
    finally:
        await restored.stop()
