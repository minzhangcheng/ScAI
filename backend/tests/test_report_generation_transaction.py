"""Atomic publication contracts for complete detailed-report generations."""

import asyncio
import json
import os
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.supervisor.report_content import ReportContentMixin
from backend.models.schemas import AnalysisState
from backend.services.report import output_generation
from backend.services.report.generator import (
    DetailedReportGenerator,
    rebind_checkpoint_asset_paths,
)
from backend.services.report import paths as report_paths
from backend.services.report.output_generation import (
    CURRENT_GENERATION_POINTER,
    GENERATION_MODE_SENTINEL,
    ReportGenerationError,
    ReportGenerationTransaction,
    cleanup_report_generations,
    resolve_current_report_dir,
    validate_and_rebind_imported_reports,
)


def _fill_generation(transaction, marker: str, *, stale_asset: bool = False):
    staging = transaction.staging_dir
    (staging / "figures").mkdir()
    (staging / "tables").mkdir()
    (staging / "report.html").write_text(
        f"<html>{marker}</html>", encoding="utf-8",
    )
    (staging / "figures" / "panel.png").write_bytes(marker.encode("ascii"))
    (staging / "tables" / "values.csv").write_text(
        f"value\n{marker}\n", encoding="utf-8",
    )
    if stale_asset:
        (staging / "figures" / "old-only.png").write_bytes(b"old-only")
    report = {
        "id": marker,
        "graphic_abstract_path": str(staging / "figures" / "panel.png"),
    }
    return report


def _published_snapshot(output_root):
    current = resolve_current_report_dir(output_root)
    report = json.loads((current / "report.json").read_text(encoding="utf-8"))
    return {
        "directory": current,
        "json": report["id"],
        "html": (current / "report.html").read_text(encoding="utf-8"),
        "figure": (current / "figures" / "panel.png").read_bytes(),
        "table": (current / "tables" / "values.csv").read_text(
            encoding="utf-8"
        ),
        "old_only": (current / "figures" / "old-only.png").exists(),
    }


async def _wait_for_thread_event(event: threading.Event, timeout: float = 2) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while not event.is_set() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    return event.is_set()


def test_legacy_flat_report_remains_readable_without_pointer(tmp_path):
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    (tmp_path / "report.html").write_text("legacy", encoding="utf-8")

    assert resolve_current_report_dir(tmp_path) == tmp_path


def test_pointer_flip_exposes_only_complete_old_or_new_tree(monkeypatch, tmp_path):
    old_transaction = ReportGenerationTransaction(tmp_path)
    old_report = _fill_generation(old_transaction, "old", stale_asset=True)
    old_dir = old_transaction.commit_report(old_report)
    assert _published_snapshot(tmp_path) == {
        "directory": old_dir,
        "json": "old",
        "html": "<html>old</html>",
        "figure": b"old",
        "table": "value\nold\n",
        "old_only": True,
    }

    new_transaction = ReportGenerationTransaction(tmp_path)
    new_report = _fill_generation(new_transaction, "new")
    original_atomic_write = output_generation.atomic_write_bytes
    observed = []

    def observe_pointer_flip(path, payload, *, mode=None):
        if path.name == CURRENT_GENERATION_POINTER:
            observed.append(_published_snapshot(tmp_path))
            original_atomic_write(path, payload, mode=mode)
            observed.append(_published_snapshot(tmp_path))
            return
        original_atomic_write(path, payload, mode=mode)

    monkeypatch.setattr(
        output_generation,
        "atomic_write_bytes",
        observe_pointer_flip,
    )

    new_dir = new_transaction.commit_report(new_report)

    assert [snapshot["json"] for snapshot in observed] == ["old", "new"]
    assert observed[0]["old_only"] is True
    assert observed[1]["old_only"] is False
    assert _published_snapshot(tmp_path) == {
        "directory": new_dir,
        "json": "new",
        "html": "<html>new</html>",
        "figure": b"new",
        "table": "value\nnew\n",
        "old_only": False,
    }
    assert (old_dir / "figures" / "old-only.png").read_bytes() == b"old-only"
    persisted = json.loads((new_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted["artifact_generation_id"] == new_transaction.generation_id
    assert persisted["html_path"] == str(new_dir / "report.html")
    assert persisted["graphic_abstract_path"] == str(
        new_dir / "figures" / "panel.png"
    )


def test_mid_generation_failure_keeps_previous_tree_and_cleans_staging(tmp_path):
    old_transaction = ReportGenerationTransaction(tmp_path)
    old_transaction.commit_report(_fill_generation(old_transaction, "old"))
    old_snapshot = _published_snapshot(tmp_path)

    failed = ReportGenerationTransaction(tmp_path)
    _fill_generation(failed, "new")
    (failed.staging_dir / "report.html").unlink()
    try:
        with pytest.raises(ReportGenerationError, match="report.html"):
            failed.commit_report({"id": "new"})
    finally:
        failed.abort()

    assert _published_snapshot(tmp_path) == old_snapshot
    assert not failed.staging_dir.exists()


def test_pointer_write_failure_keeps_previous_complete_generation(
    monkeypatch, tmp_path,
):
    old_transaction = ReportGenerationTransaction(tmp_path)
    old_dir = old_transaction.commit_report(
        _fill_generation(old_transaction, "old", stale_asset=True)
    )
    old_snapshot = _published_snapshot(tmp_path)

    failed = ReportGenerationTransaction(tmp_path)
    _fill_generation(failed, "new")
    original_atomic_write = output_generation.atomic_write_bytes

    def fail_pointer(path, payload, *, mode=None):
        if path.name == CURRENT_GENERATION_POINTER:
            raise OSError("simulated pointer publication failure")
        original_atomic_write(path, payload, mode=mode)

    monkeypatch.setattr(output_generation, "atomic_write_bytes", fail_pointer)

    with pytest.raises(OSError, match="pointer publication failure"):
        failed.commit_report({"id": "new"})
    failed.abort()

    assert resolve_current_report_dir(tmp_path) == old_dir
    assert _published_snapshot(tmp_path) == old_snapshot
    assert failed.published_dir.is_dir()


def test_checkpoint_paths_are_rebound_to_the_new_private_generation(tmp_path):
    figures_dir = tmp_path / "new" / "figures"
    tables_dir = tmp_path / "new" / "tables"
    figures_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    (figures_dir / "panel.png").write_bytes(b"panel")
    (figures_dir / "panel.pdf").write_bytes(b"pdf")
    (tables_dir / "values.csv").write_text("value\n1\n", encoding="utf-8")
    checkpoint = {
        "sections": [{
            "figures": [{
                "png_path": "/deleted/.staging-old/figures/panel.png",
                "pdf_path": "/deleted/.staging-old/figures/panel.pdf",
            }],
            "tables": [{
                "csv_path": "/deleted/.staging-old/tables/values.csv",
            }],
        }],
        "generated_figures": [{
            "png_path": "/deleted/.staging-old/figures/missing.png",
        }],
    }

    rebound = rebind_checkpoint_asset_paths(
        checkpoint,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )

    figure = rebound["sections"][0]["figures"][0]
    table = rebound["sections"][0]["tables"][0]
    assert figure["png_path"] == str(figures_dir / "panel.png")
    assert figure["pdf_path"] == str(figures_dir / "panel.pdf")
    assert table["csv_path"] == str(tables_dir / "values.csv")
    assert rebound["generated_figures"][0]["png_path"] is None
    assert checkpoint["sections"][0]["figures"][0]["png_path"].startswith(
        "/deleted/"
    )


@pytest.mark.asyncio
async def test_supervisor_wrapper_failure_does_not_publish_partial_assets(tmp_path):
    old_transaction = ReportGenerationTransaction(tmp_path)
    old_transaction.commit_report(_fill_generation(old_transaction, "old"))
    old_snapshot = _published_snapshot(tmp_path)
    staged_path = None

    class FailingReport(ReportContentMixin):
        async def _generate_detailed_report_staged(self, **kwargs):
            nonlocal staged_path
            staged_path = kwargs["output_dir"]
            staging = Path(staged_path)
            (staging / "figures").mkdir()
            (staging / "tables").mkdir()
            (staging / "figures" / "new.png").write_bytes(b"new")
            raise RuntimeError("HTML render failed")

    with pytest.raises(RuntimeError, match="HTML render failed"):
        await FailingReport().generate_detailed_report(
            analysis_state=AnalysisState(id="report-failure", user_id="owner"),
            report_model="model",
            output_dir=str(tmp_path),
        )

    assert _published_snapshot(tmp_path) == old_snapshot
    assert staged_path is not None
    assert not Path(staged_path).exists()


def test_corrupt_pointer_fails_closed_instead_of_falling_back_to_flat_mix(tmp_path):
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    (tmp_path / CURRENT_GENERATION_POINTER).write_text(
        '{"version":1,"generation":"../outside"}', encoding="utf-8",
    )

    with pytest.raises(ReportGenerationError, match="unsafe"):
        resolve_current_report_dir(tmp_path)


def test_broken_symlink_pointer_fails_closed(tmp_path):
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    try:
        (tmp_path / CURRENT_GENERATION_POINTER).symlink_to(
            tmp_path / "missing-pointer-target"
        )
    except OSError:
        pytest.skip("symlinks are not available on this platform")

    with pytest.raises(ReportGenerationError, match="unsafe"):
        resolve_current_report_dir(tmp_path)


def test_post_replace_pointer_error_exposes_only_new_tree_but_still_fails(
    monkeypatch, tmp_path, caplog,
):
    transaction = ReportGenerationTransaction(tmp_path)
    report = _fill_generation(transaction, "new")
    original_atomic_write = output_generation.atomic_write_bytes

    def write_then_fail(path, payload, *, mode=None):
        original_atomic_write(path, payload, mode=mode)
        if path.name == CURRENT_GENERATION_POINTER:
            raise OSError("pointer directory fsync failed")

    monkeypatch.setattr(output_generation, "atomic_write_bytes", write_then_fail)

    with pytest.raises(OSError, match="fsync failed"):
        transaction.commit_report(report)

    assert resolve_current_report_dir(tmp_path) == transaction.published_dir
    assert transaction._committed is False
    assert transaction._published_visible is True
    assert "durability could not be confirmed" in caplog.text


def test_generation_store_without_pointer_does_not_fall_back_to_legacy(tmp_path):
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    published = tmp_path / ".report-generations" / "generation-orphan"
    published.mkdir(parents=True)
    (tmp_path / GENERATION_MODE_SENTINEL).write_text(
        '{"version":1}', encoding="utf-8"
    )

    with pytest.raises(ReportGenerationError, match="without a current pointer"):
        resolve_current_report_dir(tmp_path)


def test_aborted_first_generation_keeps_legacy_report_readable(tmp_path):
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    (tmp_path / "report.html").write_text("legacy", encoding="utf-8")
    transaction = ReportGenerationTransaction(tmp_path)
    transaction.abort()

    assert resolve_current_report_dir(tmp_path) == tmp_path
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["id"] == "legacy"


def test_first_migration_orphan_before_pointer_keeps_complete_legacy_visible(tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    (tmp_path / "report.html").write_text("legacy", encoding="utf-8")
    transaction = ReportGenerationTransaction(tmp_path)
    _fill_generation(transaction, "new")
    os.replace(transaction.staging_dir, transaction.published_dir)

    assert resolve_current_report_dir(tmp_path) == tmp_path
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["id"] == "legacy"


@pytest.mark.asyncio
async def test_first_migration_reader_stays_on_legacy_until_pointer_publish(
    monkeypatch, tmp_path,
):
    (tmp_path / "figures").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    (tmp_path / "report.html").write_text("legacy", encoding="utf-8")
    pointer_entered = threading.Event()
    release_pointer = threading.Event()
    original_atomic_write = output_generation.atomic_write_bytes

    def block_first_pointer(path, payload, *, mode=None):
        if path.name == CURRENT_GENERATION_POINTER:
            pointer_entered.set()
            assert release_pointer.wait(timeout=5)
        original_atomic_write(path, payload, mode=mode)

    class MinimalReport(ReportContentMixin):
        async def _generate_detailed_report_staged(self, **kwargs):
            transaction_root = Path(kwargs["output_dir"])
            (transaction_root / "figures").mkdir()
            (transaction_root / "tables").mkdir()
            (transaction_root / "report.html").write_text(
                "<html>new</html>", encoding="utf-8",
            )
            return {"id": "new"}

    monkeypatch.setattr(output_generation, "atomic_write_bytes", block_first_pointer)
    task = asyncio.create_task(MinimalReport().generate_detailed_report(
        analysis_state=AnalysisState(id="migration", user_id="owner"),
        report_model="model",
        output_dir=str(tmp_path),
    ))
    try:
        assert await _wait_for_thread_event(pointer_entered)
        assert resolve_current_report_dir(tmp_path) == tmp_path
        assert json.loads(
            (tmp_path / "report.json").read_text(encoding="utf-8")
        )["id"] == "legacy"
    finally:
        release_pointer.set()
    report = await task

    assert report["id"] == "new"
    published = resolve_current_report_dir(tmp_path)
    assert published != tmp_path
    assert json.loads(
        (published / "report.json").read_text(encoding="utf-8")
    )["id"] == "new"
    assert (published / "report.html").read_text(encoding="utf-8") == "<html>new</html>"


@pytest.mark.asyncio
async def test_first_migration_pointer_failure_aborts_wrapper_and_keeps_legacy(
    monkeypatch, tmp_path,
):
    (tmp_path / "figures").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    (tmp_path / "report.html").write_text("legacy", encoding="utf-8")
    original_atomic_write = output_generation.atomic_write_bytes

    def fail_first_pointer(path, payload, *, mode=None):
        if path.name == CURRENT_GENERATION_POINTER:
            raise OSError("simulated first pointer failure")
        original_atomic_write(path, payload, mode=mode)

    class MinimalReport(ReportContentMixin):
        async def _generate_detailed_report_staged(self, **kwargs):
            transaction_root = Path(kwargs["output_dir"])
            (transaction_root / "figures").mkdir()
            (transaction_root / "tables").mkdir()
            (transaction_root / "report.html").write_text(
                "<html>new</html>", encoding="utf-8",
            )
            return {"id": "new"}

    monkeypatch.setattr(output_generation, "atomic_write_bytes", fail_first_pointer)
    with pytest.raises(OSError, match="first pointer failure"):
        await MinimalReport().generate_detailed_report(
            analysis_state=AnalysisState(id="migration", user_id="owner"),
            report_model="model",
            output_dir=str(tmp_path),
        )

    assert not (tmp_path / CURRENT_GENERATION_POINTER).exists()
    assert resolve_current_report_dir(tmp_path) == tmp_path
    assert json.loads(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )["id"] == "legacy"
    assert not list((tmp_path / ".report-generations").glob(".staging-*"))


def test_sentinel_failure_does_not_relabel_durable_pointer_as_failed(
    monkeypatch, tmp_path, caplog,
):
    transaction = ReportGenerationTransaction(tmp_path)
    report = _fill_generation(transaction, "new")
    original_atomic_write = output_generation.atomic_write_bytes

    def fail_sentinel(path, payload, *, mode=None):
        if path.name == GENERATION_MODE_SENTINEL:
            raise OSError("sentinel fsync failure")
        original_atomic_write(path, payload, mode=mode)

    monkeypatch.setattr(output_generation, "atomic_write_bytes", fail_sentinel)
    published = transaction.commit_report(report)

    assert resolve_current_report_dir(tmp_path) == published
    assert transaction._committed is True
    assert "generation-mode marker" in caplog.text


def test_generation_cache_never_matches_legacy_disk_tree(tmp_path):
    cached = {"artifact_generation_id": "generation-new"}
    assert output_generation.report_matches_published_dir(cached, tmp_path) is False


def test_report_metadata_cannot_reference_mutable_workspace_assets(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mutable = workspace / "panel.png"
    mutable.write_bytes(b"mutable")
    transaction = ReportGenerationTransaction(tmp_path / "reports")
    _fill_generation(transaction, "new")

    with pytest.raises(ReportGenerationError, match="immutable asset"):
        transaction.commit_report({
            "id": "new",
            "figures": [{"filename": "panel.png", "path": str(mutable)}],
        })
    transaction.abort()


@pytest.mark.parametrize(
    ("metadata_key", "entry"),
    [
        (
            "figures",
            {"filename": "panel.png", "png_path": None, "path": "outside.png"},
        ),
        (
            "tables",
            {"filename": "values.csv", "csv_path": None, "path": "outside.csv"},
        ),
    ],
)
def test_empty_preferred_asset_alias_cannot_hide_external_path(
    tmp_path, metadata_key, entry,
):
    outside = tmp_path / entry["path"]
    outside.write_text("mutable", encoding="utf-8")
    transaction = ReportGenerationTransaction(tmp_path / "reports")
    report = _fill_generation(transaction, "new")
    report[metadata_key] = [{**entry, "path": str(outside)}]

    with pytest.raises(ReportGenerationError, match="immutable asset"):
        transaction.commit_report(report)
    transaction.abort()


def test_generation_cleanup_is_delayed_bounded_and_never_removes_current(tmp_path):
    generations = []
    for marker in ("one", "two", "three", "four", "five"):
        transaction = ReportGenerationTransaction(tmp_path)
        generations.append(transaction.commit_report(_fill_generation(transaction, marker)))
    crashed = ReportGenerationTransaction(tmp_path)
    (crashed.staging_dir / "partial").write_text("partial", encoding="utf-8")
    old_timestamp = time.time() - 7200
    for path in [*generations, crashed.staging_dir]:
        os.utime(path, (old_timestamp, old_timestamp))

    removed = cleanup_report_generations(
        tmp_path,
        now=time.time(),
        retain_count=2,
        grace_seconds=3600,
    )

    current = resolve_current_report_dir(tmp_path)
    assert current == generations[-1]
    assert current.exists()
    assert generations[-2].exists()
    assert all(not path.exists() for path in generations[:-2])
    assert not crashed.staging_dir.exists()
    assert set(removed) == {*generations[:-2], crashed.staging_dir}


@pytest.mark.asyncio
async def test_async_supervisor_commit_keeps_event_loop_responsive(
    monkeypatch, tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    original_fsync = output_generation._fsync_staged_tree

    def blocked_fsync(staging):
        entered.set()
        assert release.wait(timeout=5)
        original_fsync(staging)

    class MinimalReport(ReportContentMixin):
        async def _generate_detailed_report_staged(self, **kwargs):
            staging = Path(kwargs["output_dir"])
            (staging / "figures").mkdir()
            (staging / "tables").mkdir()
            (staging / "report.html").write_text("<html>ok</html>", encoding="utf-8")
            return {"id": "responsive"}

    monkeypatch.setattr(output_generation, "_fsync_staged_tree", blocked_fsync)
    task = asyncio.create_task(MinimalReport().generate_detailed_report(
        analysis_state=AnalysisState(id="responsive", user_id="owner"),
        report_model="model",
        output_dir=str(tmp_path),
    ))
    assert await _wait_for_thread_event(entered)
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    report = await task
    assert report["id"] == "responsive"


@pytest.mark.asyncio
async def test_cancelled_commit_drains_atomic_publication_before_propagating(
    monkeypatch, tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    original_fsync = output_generation._fsync_staged_tree

    def blocked_fsync(staging):
        entered.set()
        assert release.wait(timeout=5)
        original_fsync(staging)

    class MinimalReport(ReportContentMixin):
        async def _generate_detailed_report_staged(self, **kwargs):
            staging = Path(kwargs["output_dir"])
            (staging / "figures").mkdir()
            (staging / "tables").mkdir()
            (staging / "report.html").write_text("<html>ok</html>", encoding="utf-8")
            return {"id": "cancelled"}

    monkeypatch.setattr(output_generation, "_fsync_staged_tree", blocked_fsync)
    task = asyncio.create_task(MinimalReport().generate_detailed_report(
        analysis_state=AnalysisState(id="cancelled", user_id="owner"),
        report_model="model",
        output_dir=str(tmp_path),
    ))
    assert await _wait_for_thread_event(entered)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Let the executor completion callback retire before pytest closes its
    # function-scoped loop (CPython 3.13 can otherwise leave its wakeup queued).
    await asyncio.sleep(0.1)

    current = resolve_current_report_dir(tmp_path)
    assert json.loads((current / "report.json").read_text(encoding="utf-8"))["id"] == "cancelled"
    assert (current / "report.html").is_file()
    pending = [
        candidate for candidate in asyncio.all_tasks()
        if candidate is not asyncio.current_task() and not candidate.done()
    ]
    assert pending == []


@pytest.mark.asyncio
async def test_fallback_generator_publishes_through_same_generation_pointer(
    monkeypatch, tmp_path,
):
    generator = DetailedReportGenerator(
        llm_client=object(),
        pi_model="pi",
        bio_model="bio",
        output_dir=str(tmp_path),
    )
    state = AnalysisState(id="fallback-generation", user_id="owner")
    cleared = []

    async def generate_staged(_state, *, resume_from_checkpoint=True):
        assert resume_from_checkpoint is True
        (generator.output_dir / "figures").mkdir(exist_ok=True)
        (generator.output_dir / "tables").mkdir(exist_ok=True)
        (generator.output_dir / "figures" / "panel.png").write_bytes(b"fallback")
        (generator.output_dir / "report.html").write_text(
            "<html>fallback</html>", encoding="utf-8",
        )
        return {"id": "fallback"}

    class Checkpoint:
        def __init__(self, **_kwargs):
            pass

        def clear(self):
            cleared.append(True)

    monkeypatch.setattr(generator, "_generate_report_staged", generate_staged)
    monkeypatch.setattr(
        "backend.services.report.generator.CheckpointManager",
        Checkpoint,
    )

    report = await generator.generate_report(state)

    current = resolve_current_report_dir(tmp_path)
    assert report["html_path"] == str(current / "report.html")
    assert generator.output_dir == current
    assert (current / "figures" / "panel.png").read_bytes() == b"fallback"
    assert cleared == [True]

    second_report = await generator.generate_report(state)
    second_current = resolve_current_report_dir(tmp_path)
    assert second_current != current
    assert second_current.parent == tmp_path / ".report-generations"
    assert second_report["html_path"] == str(second_current / "report.html")
    assert cleared == [True, True]


@pytest.mark.asyncio
async def test_force_fallback_is_forwarded_without_checkpoint_resume(
    monkeypatch, tmp_path,
):
    generator = DetailedReportGenerator(
        llm_client=object(),
        pi_model="pi",
        bio_model="bio",
        output_dir=str(tmp_path),
    )
    state = AnalysisState(id="forced-fallback-generation", user_id="owner")
    received = []

    async def generate_staged(
        _state, *, resume_from_checkpoint=True, force_fallback=False
    ):
        received.append((resume_from_checkpoint, force_fallback))
        (generator.output_dir / "figures").mkdir(exist_ok=True)
        (generator.output_dir / "tables").mkdir(exist_ok=True)
        (generator.output_dir / "report.html").write_text(
            "<html>partial</html>", encoding="utf-8",
        )
        return {"id": "partial", "status": "partial"}

    class Checkpoint:
        def __init__(self, **_kwargs):
            pass

        def clear(self):
            pass

    monkeypatch.setattr(generator, "_generate_report_staged", generate_staged)
    monkeypatch.setattr(
        "backend.services.report.generator.CheckpointManager", Checkpoint,
    )

    await generator.generate_report(state, force_fallback=True)

    assert received == [(False, True)]


@pytest.mark.asyncio
async def test_forced_fallback_staged_pipeline_never_calls_an_llm(
    monkeypatch, tmp_path,
):
    generator = DetailedReportGenerator(
        llm_client=object(),
        pi_model="pi",
        bio_model="bio",
        output_dir=str(tmp_path / "report"),
        workspace_dir=str(tmp_path / "workspace"),
    )
    state = AnalysisState(
        id="deterministic-fallback",
        user_id="owner",
        name="Retained analysis",
        initial_instruction="Summarize retained evidence.",
    )
    (tmp_path / "workspace").mkdir()

    class Checkpoint:
        def __init__(self, **_kwargs):
            pass

        async def load(self):
            raise AssertionError("forced fallback must not load an LLM checkpoint")

        async def save(self, _payload):
            pass

    monkeypatch.setattr(
        "backend.services.report.generator.CheckpointManager", Checkpoint,
    )
    monkeypatch.setattr(generator, "_report_progress", AsyncMock())
    monkeypatch.setattr(
        generator,
        "_pi_chat",
        AsyncMock(side_effect=AssertionError("LLM call in deterministic fallback")),
    )
    monkeypatch.setattr(
        generator,
        "_chat_with_vision_fallback",
        AsyncMock(side_effect=AssertionError("vision call in deterministic fallback")),
    )
    monkeypatch.setattr(generator, "_scan_existing_figures", lambda **_kwargs: [])
    monkeypatch.setattr(generator, "_scan_existing_tables", lambda **_kwargs: [])
    monkeypatch.setattr(
        generator,
        "_build_analysis_record",
        lambda _state: {
            "total_code_cells": 0,
            "analysis_steps": [],
            "total_rounds": 0,
        },
    )
    monkeypatch.setattr(
        generator,
        "_extract_findings",
        lambda _state: {
            "bioinformatics": [],
            "pathology": [],
            "biochemistry": [],
            "literature": [],
            "summary": "Retained workflow evidence only.",
            "cluster_annotations": "Not available",
        },
    )
    monkeypatch.setattr(
        generator, "_extract_cluster_annotations_from_state", lambda _state: None,
    )
    monkeypatch.setattr(
        generator,
        "_create_fallback_plan_with_existing_figures",
        lambda _state, _figures: {
            "title": "Partial retained-evidence report",
            "sections": [{
                "id": "retained",
                "title": "Retained evidence",
                "description": "Evidence retained before provider failure.",
                "analysis_type": "general",
                "existing_figures": [],
                "existing_tables": [],
            }],
        },
    )
    monkeypatch.setattr(generator, "_build_full_analysis_log", lambda _state: [])
    monkeypatch.setattr(
        generator, "_maybe_generate_reproducibility_bundle", lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "backend.services.report.generator.write_interaction_record_files",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "backend.services.report.generator.write_literature_report",
        lambda *_args, **_kwargs: None,
    )

    async def write_html(_report, _language, state=None):
        path = generator.output_dir / "report.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>partial</html>", encoding="utf-8")
        return path

    monkeypatch.setattr(generator, "_generate_html_report", write_html)
    generator.output_dir.mkdir(parents=True, exist_ok=True)
    generator.figures_dir.mkdir(exist_ok=True)
    generator.tables_dir.mkdir(exist_ok=True)

    report = await generator._generate_report_staged(
        state,
        resume_from_checkpoint=False,
        force_fallback=True,
    )

    assert report["status"] == "partial"
    assert report["degraded_fallback"] is True
    assert "deterministically" in report["abstract"]
    generator._pi_chat.assert_not_awaited()
    generator._chat_with_vision_fallback.assert_not_awaited()


def test_main_and_branch_pointers_resolve_independently(monkeypatch, tmp_path):
    monkeypatch.setattr(
        report_paths,
        "get_session_reports",
        lambda _session_id: tmp_path,
    )
    main = ReportGenerationTransaction(tmp_path)
    main_dir = main.commit_report(_fill_generation(main, "main"))
    branch_root = report_paths.resolve_report_output_dir("session", "branch/A")
    branch = ReportGenerationTransaction(branch_root)
    branch_report = _fill_generation(branch, "branch")
    branch_report["branch_id"] = "branch/A"
    branch_dir = branch.commit_report(branch_report)

    assert report_paths.resolve_published_report_dir("session", None) == main_dir
    assert (
        report_paths.resolve_published_report_dir("session", "branch/A")
        == branch_dir
    )


def test_report_archive_contains_only_the_selected_generation(tmp_path):
    from backend.routes.reports_parts.core import _create_report_archive

    old = ReportGenerationTransaction(tmp_path)
    old.commit_report(_fill_generation(old, "old", stale_asset=True))
    new = ReportGenerationTransaction(tmp_path)
    new.commit_report(_fill_generation(new, "new"))

    archive = _create_report_archive(tmp_path, version=1)

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert ".report-current.json" not in names
        assert not any(name.startswith(".report-generations/") for name in names)
        assert "figures/old-only.png" not in names
        assert json.loads(bundle.read("report.json"))["id"] == "new"
        assert bundle.read("report.html") == b"<html>new</html>"
        assert bundle.read("figures/panel.png") == b"new"


def test_session_export_captures_only_current_report_generations(tmp_path):
    from backend.routes.export_import_parts.tar_builders import _add_reports_to_tar

    old = ReportGenerationTransaction(tmp_path)
    old.commit_report(_fill_generation(old, "old", stale_asset=True))
    new = ReportGenerationTransaction(tmp_path)
    new.commit_report(_fill_generation(new, "new"))
    crashed = ReportGenerationTransaction(tmp_path)
    (crashed.staging_dir / "partial").write_text("partial", encoding="utf-8")
    (tmp_path / "exploration_report.html").write_text("variant", encoding="utf-8")
    archive = tmp_path.parent / "export.tar.gz"

    with tarfile.open(archive, "w:gz") as bundle:
        _add_reports_to_tar(bundle, tmp_path, "sid/reports/sid")

    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        prefix = f"sid/reports/sid/{output_generation.GENERATIONS_DIR_NAME}/{new.generation_id}"
        assert f"{prefix}/report.json" in names
        assert f"{prefix}/report.html" in names
        assert "sid/reports/sid/exploration_report.html" in names
        assert not any(old.generation_id in name for name in names)
        assert not any(".staging-" in name for name in names)


def test_report_export_holds_lineage_publication_lock(monkeypatch, tmp_path):
    from backend.routes.export_import_parts import tar_builders

    old = ReportGenerationTransaction(tmp_path)
    old.commit_report(_fill_generation(old, "old"))
    new = ReportGenerationTransaction(tmp_path)
    new_report = _fill_generation(new, "new")
    archive = tmp_path.parent / "concurrent-export.tar.gz"
    entered = threading.Event()
    release = threading.Event()
    original_add = tar_builders._safe_add_file

    def blocking_add(bundle, path, arcname):
        if path.name == CURRENT_GENERATION_POINTER:
            entered.set()
            assert release.wait(timeout=5)
        return original_add(bundle, path, arcname)

    def export_report():
        with tarfile.open(archive, "w:gz") as bundle:
            tar_builders._add_reports_to_tar(bundle, tmp_path, "sid/reports/sid")

    monkeypatch.setattr(tar_builders, "_safe_add_file", blocking_add)
    with ThreadPoolExecutor(max_workers=2) as executor:
        export_future = executor.submit(export_report)
        assert entered.wait(timeout=2)
        commit_future = executor.submit(new.commit_report, new_report)
        time.sleep(0.05)
        assert not commit_future.done()
        release.set()
        export_future.result(timeout=5)
        commit_future.result(timeout=5)

    with tarfile.open(archive, "r:gz") as bundle:
        report_name = next(
            name for name in bundle.getnames() if name.endswith("/report.json")
        )
        assert json.loads(bundle.extractfile(report_name).read())["id"] == "old"
        assert not any(new.generation_id in name for name in bundle.getnames())


def test_import_rebinds_legacy_report_paths_to_destination(tmp_path):
    reports = tmp_path / "reports"
    figures = reports / "figures"
    tables = reports / "tables"
    figures.mkdir(parents=True)
    tables.mkdir()
    (figures / "panel.png").write_bytes(b"panel")
    (tables / "values.csv").write_text("value\n1\n", encoding="utf-8")
    (reports / "report.html").write_text("legacy", encoding="utf-8")
    (reports / "report.json").write_text(json.dumps({
        "figures": [{"filename": "panel.png", "path": "/source/panel.png"}],
        "tables": [{"filename": "values.csv", "path": "/source/values.csv"}],
    }), encoding="utf-8")

    validate_and_rebind_imported_reports(reports)

    report = json.loads((reports / "report.json").read_text(encoding="utf-8"))
    assert report["figures"][0]["png_path"] == str(figures / "panel.png")
    assert report["figures"][0]["path"] == str(figures / "panel.png")
    assert report["tables"][0]["csv_path"] == str(tables / "values.csv")
    assert report.get("artifact_generation_id") is None


def test_import_keeps_legacy_identity_when_unselected_orphan_store_exists(
    tmp_path,
):
    figures = tmp_path / "figures"
    tables = tmp_path / "tables"
    figures.mkdir()
    tables.mkdir()
    (tmp_path / "report.html").write_text("legacy", encoding="utf-8")
    (tmp_path / "report.json").write_text(
        json.dumps({"id": "legacy"}), encoding="utf-8",
    )
    orphan = tmp_path / ".report-generations" / "generation-orphan"
    orphan.mkdir(parents=True)
    (orphan / "partial").write_text("unselected", encoding="utf-8")

    validate_and_rebind_imported_reports(tmp_path)

    report = json.loads(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )
    assert report["id"] == "legacy"
    assert report.get("artifact_generation_id") is None
    assert report["html_path"] == str(tmp_path / "report.html")
    assert not (tmp_path / GENERATION_MODE_SENTINEL).exists()


def test_import_rejects_branch_directory_that_disagrees_with_metadata(tmp_path):
    branch = tmp_path / "branches" / "wrong-component"
    transaction = ReportGenerationTransaction(branch)
    report = _fill_generation(transaction, "branch")
    report["branch_id"] = "branch/A"
    transaction.commit_report(report)

    with pytest.raises(ReportGenerationError, match="does not match"):
        validate_and_rebind_imported_reports(tmp_path)


@pytest.mark.asyncio
async def test_disk_generation_supersedes_stale_in_memory_report_cache(
    monkeypatch, tmp_path,
):
    from backend.routes.reports_parts import html_views
    from backend.state import detailed_reports

    transaction = ReportGenerationTransaction(tmp_path)
    current = transaction.commit_report(_fill_generation(transaction, "new"))
    session_id = "report-cache-generation"
    detailed_reports[session_id] = {
        "id": "old",
        "artifact_generation_id": "generation-old",
    }
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(return_value=AnalysisState(id=session_id, user_id="owner")),
    )
    monkeypatch.setattr(
        html_views,
        "resolve_published_report_dir",
        lambda _session_id, _branch_id: current,
    )

    try:
        result = await html_views.get_detailed_report(
            session_id,
            SimpleNamespace(state=SimpleNamespace(user_id="owner")),
        )
    finally:
        detailed_reports.pop(session_id, None)

    assert result["report"]["id"] == "new"
    assert result["report"]["artifact_generation_id"] == transaction.generation_id
