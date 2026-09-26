"""Atomic publication contracts for complete detailed-report generations."""

import asyncio
import hashlib
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.supervisor.report_content import ReportContentMixin
from backend.models.schemas import (
    AnalysisState, FigureReviewStatus, PublicationTable,
)
from backend.services.report.generator import DetailedReportGenerator
from backend.services.report import output_generation
from backend.services.report import paths as report_paths
from backend.services.report.output_generation import (
    CURRENT_GENERATION_POINTER,
    ReportGenerationError,
    ReportGenerationTransaction,
    resolve_current_report_dir,
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


def test_corrupt_pointer_fails_closed_instead_of_falling_back_to_flat_mix(tmp_path):
    (tmp_path / "report.json").write_text('{"id":"legacy"}', encoding="utf-8")
    (tmp_path / CURRENT_GENERATION_POINTER).write_text(
        '{"version":1,"generation":"../outside"}', encoding="utf-8",
    )

    with pytest.raises(ReportGenerationError, match="unsafe"):
        resolve_current_report_dir(tmp_path)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("basenames", [("markers.csv", "markers.csv"), ("unique.csv",)])
async def test_publication_tables_keep_content_identity_and_html_links(tmp_path, basenames):
    workspace = tmp_path / "workspace"
    records = []
    for index, basename in enumerate(basenames):
        source = workspace / str(index) / basename
        source.parent.mkdir(parents=True)
        source.write_text(f"gene,value\nG{index},{index}\n")
        records.append(PublicationTable(
            id=f"table-{index}", name=f"table-{index}", description="registered table",
            csv_path=str(source), csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            columns=["gene", "value"], preview_data=[{"gene": f"G{index}", "value": index}],
            total_rows=1, review_status=FigureReviewStatus.APPROVED,
            source_execution_id=f"execution-{index}",
        ))
    generator = DetailedReportGenerator(
        llm_client=object(), pi_model="pi", bio_model="bio",
        output_dir=str(tmp_path / "report"), workspace_dir=str(workspace),
    )
    scanned = generator._scan_existing_tables(records)
    assert len(scanned) == len(records)
    assert len({entry["filename"] for entry in scanned}) == len(records)
    for entry in scanned:
        record = next(record for record in records if record.id == entry["table_id"])
        assert entry["source_filename"] == Path(record.csv_path).name
        assert Path(entry["csv_path"]).read_bytes() == Path(record.csv_path).read_bytes()
        assert entry["csv_sha256"] == hashlib.sha256(Path(entry["csv_path"]).read_bytes()).hexdigest()
        assert ("--" in entry["filename"]) is (len(records) > 1)
    if len(records) == 1:
        assert scanned[0]["filename"] == "unique.csv"
    section = await generator._generate_section_from_existing_figures(
        AnalysisState(user_id="owner", name="Tables"),
        {"id": "registered-tables", "title": "Registered tables", "description": "",
         "analysis_type": "general", "existing_figures": [], "existing_tables": scanned,
         "narrative": "Retained table evidence."},
        {"summary": "", "bioinformatics": [], "cluster_annotations": ""},
        "en", force_fallback=True,
    )
    report_path = await generator._generate_html_report(
        {"title": "Tables", "sections": [section], "abstract": "",
         "conclusions": "", "generated_at": ""}, "en",
    )
    assert {item["filename"] for item in section["tables"]} == {item["filename"] for item in scanned}
    for entry in scanned:
        emitted = next(item for item in section["tables"] if item["filename"] == entry["filename"])
        for key in ("table_id", "csv_sha256", "source_filename"):
            assert emitted[key] == entry[key]
        assert f"tables/{entry['filename']}" in report_path.read_text()


@pytest.mark.parametrize("kind,entry", [
    ("figures", {"filename": "panel.png", "path": "panel.png"}),
    ("figures", {"filename": "panel.png", "png_path": None, "path": "panel.png"}),
    ("tables", {"filename": "values.csv", "csv_path": None, "path": "values.csv"}),
])
def test_report_metadata_rejects_mutable_asset_and_empty_alias(tmp_path, kind, entry):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mutable = workspace / entry["path"]
    mutable.write_bytes(b"mutable")
    transaction = ReportGenerationTransaction(tmp_path / "reports")
    report = _fill_generation(transaction, "new")
    report[kind] = [{**entry, "path": str(mutable)}]
    with pytest.raises(ReportGenerationError, match="immutable asset"):
        transaction.commit_report(report)
    transaction.abort()


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
