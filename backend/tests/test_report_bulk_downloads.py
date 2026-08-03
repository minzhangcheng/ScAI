from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from backend.routes.reports_parts import downloads
from backend.services.report.paths import resolve_report_output_dir


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
    archive.unlink()


@pytest.mark.asyncio
async def test_bulk_download_rejects_oversized_source_and_removes_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "large.bin").write_bytes(b"1234")
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", staging)
    monkeypatch.setattr(downloads, "_BULK_ZIP_MAX_SOURCE_BYTES", 3)

    with pytest.raises(HTTPException) as exc_info:
        await downloads._disk_zip_response(
            source,
            filename="bundle.zip",
            recursive=False,
        )

    assert exc_info.value.status_code == 413
    assert list(staging.iterdir()) == []


@pytest.mark.asyncio
async def test_bulk_download_rejects_when_staging_disk_cannot_fit_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "figure.png").write_bytes(b"image")
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", staging)
    monkeypatch.setattr(
        downloads.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(HTTPException) as exc_info:
        await downloads._disk_zip_response(
            source,
            filename="bundle.zip",
            recursive=False,
        )

    assert exc_info.value.status_code == 507
    assert list(staging.iterdir()) == []


def test_disk_zip_csv_filter_skips_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "table.csv").write_text("a\n1\n", encoding="utf-8")
    (source / "notes.txt").write_text("skip", encoding="utf-8")
    try:
        (source / "linked.csv").symlink_to(source / "table.csv")
    except OSError:
        pytest.skip("symlinks are not available on this platform")
    destination = tmp_path / "tables.zip"

    downloads._build_disk_zip(
        source,
        destination,
        recursive=False,
        csv_only=True,
    )

    with zipfile.ZipFile(destination) as zf:
        assert zf.namelist() == ["table.csv"]


@pytest.mark.asyncio
async def test_branch_archive_list_and_download_use_private_validated_copy(
    tmp_path, monkeypatch,
):
    from backend.models.schemas import AnalysisState

    reports_root = tmp_path / "reports"
    monkeypatch.setattr(
        "backend.services.report.paths.get_session_reports",
        lambda _session_id: reports_root,
    )
    branch_root = resolve_report_output_dir("sid", "branch/A")
    branch_root.mkdir(parents=True)
    archive = branch_root / "previous_report_v1.zip"
    archive.write_bytes(b"branch archive")
    try:
        (branch_root / "previous_report_v2.zip").symlink_to(archive)
    except OSError:
        pass
    monkeypatch.setattr(
        downloads,
        "require_session_ownership",
        AsyncMock(return_value=AnalysisState(id="sid", user_id="u1")),
    )
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", tmp_path / "staging")
    request = SimpleNamespace(state=SimpleNamespace(user_id="u1"))

    listed = await downloads.list_report_archives(
        "sid", request, branch_id="branch/A"
    )
    assert [item["filename"] for item in listed] == [archive.name]
    assert listed[0]["download_url"].endswith("?branch_id=branch%2FA")

    response = await downloads.download_report_archive(
        "sid", archive.name, request, branch_id="branch/A"
    )
    staged = Path(response.path)
    assert staged != archive
    assert staged.read_bytes() == b"branch archive"
    staged.unlink()


def test_archive_copy_rejects_inode_replacement(tmp_path):
    source = tmp_path / "archive.zip"
    source.write_bytes(b"old")
    metadata = source.stat()
    replacement = tmp_path / "replacement.zip"
    replacement.write_bytes(b"new")
    replacement.replace(source)

    with pytest.raises(downloads._ArchiveChanged):
        downloads._copy_validated_archive(
            source,
            tmp_path / "staged.zip",
            (metadata.st_dev, metadata.st_ino),
        )
