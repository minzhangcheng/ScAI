from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.responses import FileResponse

from backend.routes.reports_parts import downloads


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
