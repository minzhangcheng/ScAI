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


@pytest.mark.asyncio
async def test_real_kernel_save_restore_keeps_budgets(
    tmp_path, monkeypatch,
):
    from backend.utils.runtime_budget import RuntimeBudgetExceeded
    from backend.utils.sandbox import LocalJupyterSandbox, _sha256_file

    budgets = {
        "max_anndata_total_bytes": 256 * 1024 * 1024,
        "max_rss_bytes": 2 * 1024 * 1024 * 1024,
    }
    session_id = "real-kernel-snapshot-off-save-restore"
    states_dir = tmp_path / "states"
    workspace_dir = tmp_path / "workspace"
    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(states_dir),
        working_dir=str(workspace_dir),
        timeout=60,
        runtime_budgets=budgets,
    )
    await sb.start()
    try:
        result = await sb.execute("snapshot_off_value = {'answer': 42}")
        assert result.success, result.error
        assert await sb.save_session()
        committed_bytes = sb.notebook_state.session_path.read_bytes()
        committed_digest = _sha256_file(sb.notebook_state.session_path)
        committed_revision = sb.saved_runtime_revision

        blocked_budgets = dict(budgets)
        blocked_budgets["max_rss_bytes"] = 1
        sb.runtime_budgets = blocked_budgets
        dirty = await sb.execute("snapshot_off_value['dirty'] = True")
        assert dirty.success, dirty.error
        with pytest.raises(RuntimeBudgetExceeded) as blocked:
            await sb.save_session()
        assert any(
            item.resource == "rss_bytes" and item.limit == 1
            for item in blocked.value.violations
        )
        assert sb.notebook_state.session_path.read_bytes() == committed_bytes
        assert _sha256_file(sb.notebook_state.session_path) == committed_digest
        assert sb.saved_runtime_revision == committed_revision

        sb.runtime_budgets = dict(budgets)
        assert sb.runtime_budgets["max_rss_bytes"] == budgets["max_rss_bytes"]
    finally:
        await sb.stop(save=False)

    restored = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(states_dir),
        working_dir=str(workspace_dir),
        timeout=60,
        runtime_budgets=budgets,
    )
    await restored.start()
    try:
        check = await restored.execute("print(snapshot_off_value['answer'])")
        assert check.success, check.error
        assert "42" in check.output
    finally:
        await restored.stop(save=False)


@pytest.mark.asyncio
async def test_real_kernel_unpicklable_main_closure_commits_other_values(tmp_path):
    """An actual __main__ closure is skipped without invalidating the generation."""
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.utils.runtime_manifest import read_runtime_manifest, runtime_manifest_path

    sid = "unpicklable-main-closure"
    states = tmp_path / "states"
    sb = LocalJupyterSandbox(session_id=sid, state_dir=str(states),
                              working_dir=str(tmp_path / "workspace"), timeout=60)
    await sb.start()
    try:
        result = await sb.execute("""
adata = [1, 2, 3]
def make_helper():
    captured = (i for i in range(3))
    def helper():
        return next(captured)
    return helper
helper = make_helper()
""")
        assert result.success, result.error
        assert await sb.save_session() is True
        manifest = read_runtime_manifest(runtime_manifest_path(sid, states_dir=states))
        assert "adata" in manifest["saved_names"]
        assert "helper" not in manifest["saved_names"]
        assert any(row["name"] == "helper" and row["reason"] == "serialization_failed"
                   for row in manifest["skipped_names"])
        assert all(row["name"] != "helper" for row in manifest["carried_definitions"])
        await sb.stop(save=False)
        await sb.start()
        result = await sb.execute("assert adata == [1, 2, 3]; assert 'helper' not in globals()")
        assert result.success, result.error
    finally:
        await sb.stop(save=False)


@pytest.mark.asyncio
async def test_real_kernel_crash_and_pair_restore_round_trip(tmp_path):
    """Batch 2 end to end, on a real kernel. Spec B2-2's acceptance criterion.

    The unit tests verify the pieces in isolation, and §4b R1 is what that
    costs: the exact-revision rule was checked on its own while the signal
    carrying it never survived a resume. This drives the whole path -- save a
    generation, do more work, lose the process without a closing save, restore
    the pair, and open the session again.

    What it asserts is the promise B2-2 makes: the namespace is the one the
    dump held, the two positions are the ones committed beside it, and the
    record says what continuing from there cost.
    """
    import json
    import sqlite3

    from backend.config import get_session_states, get_session_workspace
    from backend.utils import runtime_apply
    from backend.utils.sandbox import LocalJupyterSandbox

    session_id = "test-pair-restore-roundtrip"
    states_dir = get_session_states(session_id)
    states_dir.mkdir(parents=True, exist_ok=True)
    # The configured workspace, not one of this test's own: the save scans it
    # through `get_session_workspace` and the rollback sweeps it through the
    # same call, and a test that pointed the sandbox somewhere else would
    # exercise neither.
    workspace = get_session_workspace(session_id)
    workspace.mkdir(parents=True, exist_ok=True)
    workspace_dir = str(workspace)

    checkpoint = states_dir / "checkpoints.sqlite"
    executor_checkpoint = states_dir / f"{session_id}_bioinfo_checkpoint.json"

    def write_checkpoint(marker):
        checkpoint.unlink(missing_ok=True)
        with sqlite3.connect(checkpoint) as conn:
            conn.execute("CREATE TABLE checkpoints (thread_id TEXT, blob TEXT)")
            conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?)", (session_id, marker)
            )

    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(states_dir),
        working_dir=workspace_dir,
        timeout=60,
    )
    await sb.start()
    try:
        # The work that will survive, and the two positions as they stand
        # when it is committed.
        # A first generation, so the save that matters has one to supersede.
        # A generation holds a workspace *directory* now, and superseding it is
        # a different operation from removing three files -- which is how every
        # save after the first came to fail in production, invisible to a test
        # that saves once because there is then nothing to prune.
        assert (await sb.execute("from_the_first_save = 1")).success
        (workspace / "only_in_the_first_generation.csv").write_text("a\n")
        write_checkpoint("position-at-the-first-save")
        assert await sb.save_session()

        assert (await sb.execute("adata = {'cells': 1234}")).success
        (workspace / "kept.csv").write_text("gene,logfc\nCD3E,2.1\n")
        (workspace / "rewritten.csv").write_text("gene,logfc\nAT THE SAVE,1.0\n")
        write_checkpoint("position-at-the-save")
        executor_checkpoint.write_text(json.dumps({
            "step_number": 10, "round_number": 2, "planning_stage": "stage1",
        }))

        assert await sb.save_session()
        committed = int(
            sb.notebook_state.runtime_persistence["committed_revision"]
        )
        # Retention is one, and the copy must not change that.
        generations = sorted(
            d.name for d in (states_dir / "generation").iterdir() if d.is_dir()
        )
        assert generations == [str(committed)], generations

        # The work that will be lost, and the positions moving past it.
        assert (await sb.execute("adata['cells'] = 9999")).success
        assert (await sb.execute("lost_after_the_save = True")).success
        # An intermediate written after the save. This is the shape that
        # prompted the sweep: a later step loading it by name would read a file
        # produced by work the rollback discards, and nothing would say so.
        (workspace / "adata_annotated.h5ad").write_bytes(b"from the lost work")
        # And a file that existed at the save and was rewritten afterwards.
        # A file list can take away what was added; only the copy brings this
        # one back.
        (workspace / "rewritten.csv").write_text("gene,logfc\nAFTER,9.9\n")
        write_checkpoint("position-the-crash-left")
        executor_checkpoint.write_text(json.dumps({
            "step_number": 40, "round_number": 2, "planning_stage": "stage1",
        }))

        record = sb.notebook_state.runtime_persistence
        assert int(record["current_revision"]) > committed, (
            "the kernel must be ahead of its newest generation before the crash"
        )
    finally:
        # The crash: the process goes without a closing save.
        await sb.stop(save=False)

    assert runtime_apply.restore_generation_pair(
        session_id, revision=committed
    ) is True

    # Both positions are the ones committed beside the dump.
    with sqlite3.connect(checkpoint) as conn:
        assert conn.execute(
            "SELECT blob FROM checkpoints"
        ).fetchone()[0] == "position-at-the-save"
    assert json.loads(executor_checkpoint.read_text())["step_number"] == 10

    # And the record says what continuing from there cost.
    restored = json.loads(
        (states_dir / f"{session_id}_state.json").read_text()
    )["runtime_persistence"]
    assert restored["pair_restore"]["restored_revision"] == committed
    assert restored["pair_restore"]["discarded_revisions"] >= 2

    # And the workspace holds what that generation held, which is what makes
    # the files on disk agree with the namespace and the record above them.
    assert (workspace / "kept.csv").read_text().startswith("gene,logfc")
    assert not (workspace / "adata_annotated.h5ad").exists(), (
        "a file written after the save survived the rollback"
    )
    assert "AT THE SAVE" in (workspace / "rewritten.csv").read_text(), (
        "a file overwritten after the save was not put back"
    )
    # It survived the first generation being pruned, because the second
    # generation's copy shares its inode rather than pointing into it.
    assert (workspace / "only_in_the_first_generation.csv").read_text() == "a\n"

    # The namespace is the one the dump held.
    sb2 = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(states_dir),
        working_dir=workspace_dir,
        timeout=60,
    )
    await sb2.start()
    try:
        check = await sb2.execute(
            "print(adata['cells'], 'lost_after_the_save' in globals())"
        )
        assert check.success, check.error
        assert "1234" in check.output, check.output
        assert "False" in check.output, (
            f"work done after the save must not survive it: {check.output}"
        )
    finally:
        await sb2.stop(save=False)
