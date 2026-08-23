from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from backend.routes.reports_parts import downloads
from backend.services.report.paths import resolve_report_output_dir


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
async def test_bulk_download_removes_stage_after_body_send_failure(
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
    with pytest.raises(RuntimeError, match="stream disconnected"):
        await _consume_response(response, fail_body=True)
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
@pytest.mark.parametrize("method, expected_body", [("GET", True), ("HEAD", False)])
async def test_bulk_download_response_preserves_get_and_head_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_body: bool,
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
        {"type": "http", "method": method, "headers": [], "extensions": {}},
        receive,
        send,
    )
    assert messages[0]["status"] == 200
    body = b"".join(message.get("body", b"") for message in messages)
    assert bool(body) is expected_body
    if expected_body:
        assert body == payload
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


@pytest.mark.asyncio
async def test_bulk_download_unsatisfiable_range_returns_416_and_cleans_stage(
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
    payload_size = archive.stat().st_size
    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    await response(
        {
            "type": "http",
            "method": "GET",
            "headers": [(b"range", f"bytes={payload_size}-".encode())],
            "extensions": {},
        },
        receive,
        send,
    )
    headers = {
        key.decode().lower(): value.decode()
        for key, value in messages[0]["headers"]
    }
    assert messages[0]["status"] == 416
    assert headers["content-range"] == f"bytes */{payload_size}"
    assert not archive.exists()


@pytest.mark.asyncio
async def test_bulk_download_head_range_has_empty_body_and_cleans_stage(
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
    payload_size = archive.stat().st_size
    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    await response(
        {
            "type": "http",
            "method": "HEAD",
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
    assert headers["content-range"] == f"bytes 1-3/{payload_size}"
    assert headers["content-length"] == "3"
    assert body == b""
    assert not archive.exists()


@pytest.mark.asyncio
async def test_bulk_download_over_many_ranges_falls_back_to_full_200_and_cleans_stage(
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
    ranges = ",".join("0-0" for _ in range(response.max_ranges + 1))
    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    await response(
        {
            "type": "http",
            "method": "GET",
            "headers": [(b"range", f"bytes={ranges}".encode())],
            "extensions": {},
        },
        receive,
        send,
    )
    body = b"".join(message.get("body", b"") for message in messages[1:])
    assert messages[0]["status"] == 200
    assert body == payload
    assert not archive.exists()


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
    fd, destination = _allocated_stage(tmp_path)
    try:
        downloads._build_disk_zip(
            source,
            destination,
            fd,
            recursive=False,
            csv_only=True,
        )

        with zipfile.ZipFile(destination) as zf:
            assert zf.namelist() == ["table.csv"]
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


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
    assert await _consume_response(response) == b"branch archive"
    assert not staged.exists()


def test_archive_copy_rejects_inode_replacement(tmp_path):
    source = tmp_path / "archive.zip"
    source.write_bytes(b"old")
    metadata = source.stat()
    replacement = tmp_path / "replacement.zip"
    replacement.write_bytes(b"new")
    replacement.replace(source)

    fd, staged = _allocated_stage(tmp_path)
    try:
        with pytest.raises(downloads._ArchiveChanged):
            downloads._copy_validated_archive(
                source,
                staged,
                fd,
                (metadata.st_dev, metadata.st_ino),
            )
    finally:
        os.close(fd)
        staged.unlink(missing_ok=True)


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


def test_report_zip_rejects_selected_leaf_swap(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "leaf.bin"
    target.write_bytes(b"old")
    fd, destination = _allocated_stage(tmp_path)
    expected = _six_identity(target)
    original_fstat = os.fstat
    swapped = False

    def fstat(fd):
        nonlocal swapped
        metadata = original_fstat(fd)
        if not swapped and stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == expected[:2]:
            replacement = source / "replacement.bin"
            replacement.write_bytes(b"replacement")
            replacement.replace(target)
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


def test_report_zip_rejects_midstream_identity_change(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "stream.bin"
    target.write_bytes(b"x" * (2 * 1024 * 1024))
    fd, destination = _allocated_stage(tmp_path)
    expected = _six_identity(target)
    original_fstat = os.fstat
    original_read = os.read
    target_fd = None
    mutated = False

    def fstat(fd):
        nonlocal target_fd
        metadata = original_fstat(fd)
        if stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == expected[:2]:
            target_fd = fd
        return metadata

    def read(fd, size):
        nonlocal mutated
        block = original_read(fd, size)
        if fd == target_fd and block and not mutated:
            with target.open("ab") as handle:
                handle.write(b"mutation")
            mutated = True
        return block

    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "read", read)
    try:
        with pytest.raises(downloads._BulkZipRejected) as exc_info:
            downloads._build_disk_zip(source, destination, fd, recursive=True)

        assert exc_info.value.status_code == 409
        assert mutated
        assert not destination.exists()
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


def test_direct_report_copy_rejects_midstream_identity_change(monkeypatch, tmp_path):
    source = tmp_path / "archive.zip"
    source.write_bytes(b"x" * (2 * 1024 * 1024))
    fd, destination = _allocated_stage(tmp_path)
    metadata = source.stat()
    original_fstat = os.fstat
    original_read = os.read
    target_fd = None
    mutated = False

    def fstat(fd):
        nonlocal target_fd
        info = original_fstat(fd)
        if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == (metadata.st_dev, metadata.st_ino):
            target_fd = fd
        return info

    def read(fd, size):
        nonlocal mutated
        block = original_read(fd, size)
        if fd == target_fd and block and not mutated:
            with source.open("ab") as handle:
                handle.write(b"mutation")
            mutated = True
        return block

    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "read", read)
    try:
        with pytest.raises(downloads._ArchiveChanged):
            downloads._copy_validated_archive(
                source,
                destination,
                fd,
                (metadata.st_dev, metadata.st_ino),
            )

        assert mutated
        assert not destination.exists()
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)
