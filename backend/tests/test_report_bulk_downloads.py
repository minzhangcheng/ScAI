from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.responses import FileResponse

from backend.routes.reports_parts import downloads
from backend.models.schemas import AnalysisState, DetailedReportSegment, RoundResult
from backend.models.schema_parts.round_report import StageReviewReport
from backend.services.report.download_section import (
    build_related_reports_metadata,
    render_related_reports_downloads_html,
)


def _allocated_stage(directory: Path) -> tuple[int, Path]:
    fd, raw = tempfile.mkstemp(prefix="owned_", suffix=".zip", dir=directory)
    return fd, Path(raw)


async def _consume_response(response, *, fail_body: bool = False) -> bytes:
    body = bytearray()

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        if message["type"] == "http.response.body":
            if fail_body:
                raise RuntimeError("stream disconnected")
            body.extend(message.get("body", b""))

    await response(
        {"type": "http", "method": "GET", "headers": [], "extensions": {}},
        receive,
        send,
    )
    return bytes(body)


def test_final_report_stages_retained_stage_and_round_downloads(tmp_path):
    state = AnalysisState(restore_branch_id="child")
    for lineage in ("main", "child"):
        segment = DetailedReportSegment(
            id=f"{lineage}-round",
            phase="exploration_complete",
            title=f"Saved {lineage} round",
            round_number=1,
            content_html=f"{lineage} round content",
        )
        state.detailed_report_segments.append(segment)
        state.round_results.append(RoundResult(
            round_number=1,
            lineage_id=lineage,
            report_segment_id=segment.id,
        ))
        state.stage_review_reports.append(StageReviewReport(
            session_id=state.id,
            lineage_id=lineage,
            stage="stage1",
            revision=1,
            status="ready",
            html=f"<html>{lineage} stage content</html>",
        ))

    related = build_related_reports_metadata(
        state,
        {},
        reports_dir=tmp_path,
        lineage_id="child",
    )

    assert [item["label"] for item in related["stage_reports"]] == [
        "Round 1 Stage 1 Report — Revision 1 (ready)",
    ]
    assert [item["label"] for item in related["round_reports"]] == [
        "Round 1 Report — Saved child round",
    ]
    for item in related["stage_reports"] + related["round_reports"]:
        assert (tmp_path / item["filename"]).is_file()
    staged_html = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "history").rglob("*.html")
    )
    assert "main round content" not in staged_html
    assert "main stage content" not in staged_html
    assert "child round content" in staged_html
    assert "child stage content" in staged_html

    rendered = render_related_reports_downloads_html(related)
    assert 'href="history/stages/' in rendered
    assert 'href="history/rounds/' in rendered


@pytest.mark.asyncio
async def test_bulk_download_is_staged_on_disk_and_preserves_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    (source / "nested").mkdir(parents=True)
    (source / "root.txt").write_bytes(b"root")
    (source / "nested" / "child.txt").write_bytes(b"child")
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", staging)

    response = await downloads._disk_zip_response(
        source,
        filename="bundle.zip",
        recursive=True,
    )

    assert isinstance(response, FileResponse)
    archive = Path(response.path)
    assert archive.parent == staging
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {"root.txt", "nested/child.txt"}
        assert zf.read("nested/child.txt") == b"child"
    assert await _consume_response(response)
    assert not archive.exists()


@pytest.mark.asyncio
async def test_bulk_download_cancellation_drains_cleanup_before_cancelled_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "root.txt").write_bytes(b"root")
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", staging)
    response = await downloads._disk_zip_response(
        source,
        filename="bundle.zip",
        recursive=False,
    )
    archive = Path(response.path)
    body_started = asyncio.Event()
    hold_send = asyncio.Event()

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        if message["type"] == "http.response.body":
            body_started.set()
            await hold_send.wait()

    task = asyncio.create_task(
        response(
            {"type": "http", "method": "GET", "headers": [], "extensions": {}},
            receive,
            send,
        )
    )
    await asyncio.wait_for(body_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert not archive.exists()


@pytest.mark.asyncio
async def test_bulk_download_response_preserves_exact_range_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "root.txt").write_bytes(b"root")
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", staging)
    response = await downloads._disk_zip_response(
        source,
        filename="bundle.zip",
        recursive=False,
    )
    archive = Path(response.path)
    payload = archive.read_bytes()
    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    await response(
        {
            "type": "http",
            "method": "GET",
            "headers": [(b"range", b"bytes=1-3")],
            "extensions": {},
        },
        receive,
        send,
    )
    headers = {
        key.decode().lower(): value.decode()
        for key, value in messages[0]["headers"]
    }
    body = b"".join(message.get("body", b"") for message in messages[1:])
    assert messages[0]["status"] == 206
    assert headers["accept-ranges"] == "bytes"
    assert headers["content-range"] == f"bytes 1-3/{len(payload)}"
    assert headers["content-length"] == "3"
    assert headers["content-disposition"].startswith("attachment;")
    assert body == payload[1:4]
    assert not archive.exists()


def _six_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    metadata = path.stat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def test_report_zip_rejects_parent_directory_swap(monkeypatch, tmp_path):
    source = tmp_path / "source"
    parent = source / "nested"
    parent.mkdir(parents=True)
    target = parent / "leaf.bin"
    target.write_bytes(b"old")
    fd, destination = _allocated_stage(tmp_path)
    expected = _six_identity(parent)
    original_fstat = os.fstat
    swapped = False

    def fstat(fd):
        nonlocal swapped
        metadata = original_fstat(fd)
        if not swapped and stat.S_ISDIR(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == expected[:2]:
            old_parent = source / "nested-old"
            parent.rename(old_parent)
            parent.mkdir()
            (parent / "leaf.bin").write_bytes(b"replacement")
            swapped = True
        return metadata

    monkeypatch.setattr(os, "fstat", fstat)
    try:
        with pytest.raises(downloads._BulkZipRejected) as exc_info:
            downloads._build_disk_zip(source, destination, fd, recursive=True)

        assert exc_info.value.status_code == 409
        assert swapped
        assert not destination.exists()
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("lineage", ["main", "child"])
async def test_full_download_includes_retained_rounds_and_stage_versions(tmp_path, monkeypatch, lineage):
    state = AnalysisState(restore_branch_id="child")
    for owner in ("main", "child"):
        directory = tmp_path / owner
        (directory / "figures").mkdir(parents=True)
        (directory / "report.html").write_text(f"Published {owner}")
        (directory / "figures/retained.png").write_bytes(b"image")
        (directory / "previous_report_v1.zip").write_bytes(b"old archive")
        for number, phase in ((1, "exploration"), (2, "integration")):
            segment = DetailedReportSegment(
                id=f"{owner}-{number}", phase=phase, title=f"Saved {owner} {number}",
                round_number=number, content_html=f"Final {owner} {number}",
                decision_content_html=f"Review {owner} {number}<img src='data:image/png;base64,eA=='>",
            )
            supplement = DetailedReportSegment(
                id=f"{segment.id}-supplement", phase="supplementary", title="Supplement",
                round_number=number, content_html=f"Supplement {owner} {number}",
            )
            state.detailed_report_segments.extend([segment, supplement])
            state.round_results.append(RoundResult(
                round_number=number, phase=phase, lineage_id=owner,
                report_segment_id=segment.id, report_supplement_ids=[supplement.id],
            ))
        for stage, revision, status in (("stage1", 1, "ready"), ("stage2", 1, "partial"),
                                        ("stage2", 2, "ready"), ("stage2", 3, "failed")):
            state.stage_review_reports.append(StageReviewReport(
                session_id=state.id, lineage_id=owner, stage=stage, revision=revision,
                status=status, html=f"<html>{owner} {stage} v{revision}</html>" if status != "failed" else "",
            ))
    state.stage_review_reports.append(StageReviewReport(
        session_id="other-session", lineage_id=lineage, stage="stage1", revision=9,
        status="ready", html="FOREIGN SESSION",
    ))
    state.round_results.append(RoundResult(round_number=3, lineage_id=lineage))
    # The export also works after persistence/reload, without live workspace figures.
    from backend.utils.blob_store import externalize_blobs, inline_blobs
    saved = externalize_blobs(state.model_dump(mode="json"), tmp_path / "blobs")
    state = AnalysisState.model_validate(inline_blobs(saved, tmp_path / "blobs"))
    before = state.model_dump_json()
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", tmp_path / "staging")
    monkeypatch.setattr(downloads, "resolve_published_report_dir", lambda _id, branch: tmp_path / (branch or "main"))
    response = await downloads.download_full_report(
        state.id, state=state, branch_id=None if lineage == "main" else lineage,
    )
    with zipfile.ZipFile(response.path) as archive:
        assert archive.read("report.html").decode() == f"Published {lineage}"
        assert archive.read("figures/retained.png") == b"image"
        assert "previous_report_v1.zip" not in archive.namelist()
        index = json.loads(archive.read("history/index.json"))
        assert index["session_id"] == state.id and index["lineage_id"] == lineage
        rounds = [entry for entry in index["reports"] if entry["kind"] == "round"]
        assert [entry["round"] for entry in rounds] == [1, 2, 3]
        assert rounds[-1]["path"] is None and rounds[-1]["status"] == "unavailable"
        for entry in rounds[:2]:
            content = archive.read(entry["path"]).decode()
            assert f"Review {lineage} {entry['round']}" in content
            assert f"Supplement {lineage} {entry['round']}" in content
            assert "data:image/png;base64,eA==" in content
            assert f"Final {lineage}" not in content
            assert ("Review child" if lineage == "main" else "Review main") not in content
        stages = [entry for entry in index["reports"] if entry["kind"] == "stage"]
        assert len(stages) == 4 and stages[-1]["status"] == "failed" and stages[-1]["path"] is None
        for entry in stages[:3]:
            original = next(r for r in state.stage_review_reports if r.id == entry["report_id"])
            assert archive.read(entry["path"]) == original.html.encode()
    assert state.model_dump_json() == before
    await _consume_response(response)
    assert not Path(response.path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["../escape.html", "report.html"])
async def test_extra_report_member_rejection_cleans_up_archive(tmp_path, monkeypatch, name):
    from fastapi import HTTPException

    source = tmp_path / "source"
    source.mkdir()
    (source / "report.html").write_text("Published report")
    staging = tmp_path / "staging"
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", staging)
    with pytest.raises(HTTPException) as error:
        await downloads._disk_zip_response(
            source, filename="bundle.zip", recursive=True,
            extra_members=[(Path(name), b"retained report")],
        )
    assert error.value.status_code == 409
    assert not list(staging.iterdir())
