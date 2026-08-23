"""Security contracts for the V1 session export route."""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import stat
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from backend.models.schemas import AnalysisState
from backend.routes import v1


@pytest.fixture
def v1_export_env(tmp_path, monkeypatch):
    session_id = "v1-export-security"
    session_dir = tmp_path / "session"
    workspace = session_dir / "workspace"
    states = session_dir / "states"
    workspace.mkdir(parents=True)
    states.mkdir(parents=True)
    (workspace / "results.zip").write_bytes(b"workspace-only-zip")
    (states / "checkpoint.bin").write_bytes(b"ordinary-session-state")
    staging = tmp_path / "staging"
    staging.mkdir()
    state = AnalysisState(id=session_id, user_id="owner")
    monkeypatch.setattr(v1, "get_session_dir", lambda _session_id: session_dir)
    monkeypatch.setattr(v1, "UPLOAD_TEMP_DIR", staging, raising=False)
    monkeypatch.setattr(
        v1,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    return SimpleNamespace(
        session_id=session_id,
        session_dir=session_dir,
        workspace=workspace,
        states=states,
        staging=staging,
        state=state,
        request=SimpleNamespace(state=SimpleNamespace(user_id="owner")),
    )


async def _run_export(env):
    return await v1.export_analysis(env.session_id, env.request)


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


def _archive_paths(staging: Path) -> set[Path]:
    return {path for path in staging.iterdir() if path.is_file()}


def _cleanup_response(response) -> None:
    path = Path(response.path)
    marker = path.with_name(f"{path.name}.scai-export-owner")
    path.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)


def _patch_leaf_swap(monkeypatch, target: Path) -> None:
    expected = target.stat()
    original_fstat = os.fstat
    swapped = False

    def fstat(fd):
        nonlocal swapped
        metadata = original_fstat(fd)
        if (
            not swapped
            and stat.S_ISREG(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == (expected.st_dev, expected.st_ino)
        ):
            replacement = target.with_name("replacement-leaf.bin")
            replacement.write_bytes(b"replacement-leaf")
            replacement.replace(target)
            swapped = True
        return metadata

    monkeypatch.setattr(os, "fstat", fstat)


def _patch_parent_swap(monkeypatch, target: Path) -> None:
    expected = target.stat()
    original_fstat = os.fstat
    swapped = False

    def fstat(fd):
        nonlocal swapped
        metadata = original_fstat(fd)
        if (
            not swapped
            and stat.S_ISDIR(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == (expected.st_dev, expected.st_ino)
        ):
            old_parent = target.with_name(f"{target.name}-old")
            target.rename(old_parent)
            target.mkdir()
            (target / "results.zip").write_bytes(b"replacement-parent")
            swapped = True
        return metadata

    monkeypatch.setattr(os, "fstat", fstat)


def _patch_midstream_mutation(monkeypatch, target: Path) -> None:
    expected = target.stat()
    original_fstat = os.fstat
    original_read = os.read
    target_fd = None
    mutated = False

    def fstat(fd):
        nonlocal target_fd
        metadata = original_fstat(fd)
        if stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == (expected.st_dev, expected.st_ino):
            target_fd = fd
        return metadata

    def read(fd, size):
        nonlocal mutated
        block = original_read(fd, size)
        if fd == target_fd and block and not mutated:
            with target.open("ab") as handle:
                handle.write(b"midstream-mutation")
            mutated = True
        return block

    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "read", read)


@pytest.mark.asyncio
async def test_v1_export_builds_actual_session_not_workspace_zip(v1_export_env):
    response = await _run_export(v1_export_env)
    assert isinstance(response, FileResponse)
    archive_path = Path(response.path)
    assert archive_path.parent == v1_export_env.staging
    assert archive_path != v1_export_env.workspace / "results.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert {"workspace/results.zip", "states/checkpoint.bin"}.issubset(archive.namelist())
        assert archive.read("workspace/results.zip") == b"workspace-only-zip"
        assert archive.read("states/checkpoint.bin") == b"ordinary-session-state"
    assert await _consume_response(response)
    assert _archive_paths(v1_export_env.staging) == set()


@pytest.mark.asyncio
async def test_v1_export_marker_failure_deletes_archive_before_response(
    v1_export_env, monkeypatch,
):
    def fail_marker(_path, **_kwargs):
        raise OSError("marker unavailable")

    monkeypatch.setattr(v1, "create_export_ownership_marker", fail_marker, raising=False)
    with pytest.raises(OSError, match="marker unavailable"):
        await _run_export(v1_export_env)

    assert _archive_paths(v1_export_env.staging) == set()


@pytest.mark.asyncio
async def test_v1_export_cleans_archive_and_marker_after_stream_failure(v1_export_env):
    response = await _run_export(v1_export_env)
    archive_path = Path(response.path)
    assert archive_path.parent == v1_export_env.staging
    marker = archive_path.with_name(f"{archive_path.name}.scai-export-owner")
    assert archive_path.exists() and marker.exists()

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        if message["type"] == "http.response.body":
            raise RuntimeError("stream disconnected")

    with pytest.raises(RuntimeError, match="stream disconnected"):
        await response(
            {"type": "http", "method": "GET", "headers": [], "extensions": {}},
            receive,
            send,
        )

    assert not archive_path.exists()
    assert not marker.exists()


@pytest.mark.asyncio
async def test_v1_export_build_cancellation_drains_worker_before_cleanup(
    v1_export_env, monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    original_build = v1.build_secure_zip

    def blocking_build(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(v1, "build_secure_zip", blocking_build)
    task = asyncio.create_task(_run_export(v1_export_env))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1)
    task.cancel()
    await asyncio.wait_for(asyncio.to_thread(lambda: None), timeout=1)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)
    assert _archive_paths(v1_export_env.staging) == set()


@pytest.mark.asyncio
async def test_v1_export_passes_no_whole_session_file_or_byte_caps(
    v1_export_env, monkeypatch,
):
    calls = []
    original_build = v1.build_secure_zip

    def capture_build(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_build(*args, **kwargs)

    monkeypatch.setattr(v1, "build_secure_zip", capture_build)
    response = await _run_export(v1_export_env)
    assert calls
    assert calls[0]["max_files"] is None
    assert calls[0]["max_source_bytes"] is None
    body = await _consume_response(response)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        assert {"workspace/results.zip", "states/checkpoint.bin"}.issubset(archive.namelist())


@pytest.mark.asyncio
async def test_v1_marker_collision_preserves_the_unrelated_marker(
    v1_export_env, monkeypatch,
):
    original_mkstemp = v1.tempfile.mkstemp
    collision_marker = {}

    def mkstemp(*args, **kwargs):
        fd, raw = original_mkstemp(*args, **kwargs)
        archive = Path(raw)
        marker = archive.with_name(f"{archive.name}.scai-export-owner")
        marker.write_bytes(b"unrelated marker")
        collision_marker["path"] = marker
        return fd, raw

    monkeypatch.setattr(v1.tempfile, "mkstemp", mkstemp)
    with pytest.raises(FileExistsError):
        await _run_export(v1_export_env)
    marker = collision_marker["path"]
    assert marker.read_bytes() == b"unrelated marker"
    assert _archive_paths(v1_export_env.staging) == {marker}
    marker.unlink()


@pytest.mark.asyncio
async def test_v1_response_streams_the_owned_inode_after_path_rebinding(
    v1_export_env,
):
    response = await _run_export(v1_export_env)
    archive_path = Path(response.path)
    marker = archive_path.with_name(f"{archive_path.name}.scai-export-owner")
    foreign = v1_export_env.staging / "foreign-replacement"
    foreign.write_bytes(b"foreign bytes")
    archive_path.unlink()
    os.link(foreign, archive_path)

    body = await _consume_response(response)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        assert "workspace/results.zip" in archive.namelist()
    assert archive_path.read_bytes() == b"foreign bytes"
    assert foreign.read_bytes() == b"foreign bytes"
    assert not marker.exists()
    archive_path.unlink()
    foreign.unlink()


@pytest.mark.asyncio
async def test_v1_response_cleans_archive_and_marker_after_normal_stream(
    v1_export_env,
):
    response = await _run_export(v1_export_env)
    archive_path = Path(response.path)
    marker = archive_path.with_name(f"{archive_path.name}.scai-export-owner")
    assert await _consume_response(response)
    assert not archive_path.exists()
    assert not marker.exists()


@pytest.mark.asyncio
async def test_v1_response_cleanup_survives_repeated_cancellation(
    v1_export_env, monkeypatch,
):
    from backend.services import export_lifecycle as lifecycle

    response = await _run_export(v1_export_env)
    archive_path = Path(response.path)
    marker = archive_path.with_name(f"{archive_path.name}.scai-export-owner")
    body_started = asyncio.Event()
    hold_send = asyncio.Event()
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    original_unlink = lifecycle.os.unlink

    def unlink(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(archive_path):
            cleanup_started.set()
            assert release_cleanup.wait(timeout=3)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(lifecycle.os, "unlink", unlink)

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
    await asyncio.wait_for(asyncio.to_thread(cleanup_started.wait), timeout=1)
    task.cancel()
    await asyncio.wait_for(asyncio.to_thread(lambda: None), timeout=1)
    assert not task.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)
    assert not archive_path.exists()
    assert not marker.exists()


@pytest.mark.asyncio
async def test_v1_export_rejects_leaf_swap(v1_export_env, monkeypatch):
    _patch_leaf_swap(monkeypatch, v1_export_env.workspace / "results.zip")

    with pytest.raises(HTTPException) as exc_info:
        await _run_export(v1_export_env)

    assert exc_info.value.status_code == 409
    assert _archive_paths(v1_export_env.staging) == set()


@pytest.mark.asyncio
async def test_v1_export_rejects_parent_swap(v1_export_env, monkeypatch):
    _patch_parent_swap(monkeypatch, v1_export_env.workspace)

    with pytest.raises(HTTPException) as exc_info:
        await _run_export(v1_export_env)

    assert exc_info.value.status_code == 409
    assert _archive_paths(v1_export_env.staging) == set()


@pytest.mark.asyncio
async def test_v1_export_rejects_midstream_mutation(v1_export_env, monkeypatch):
    target = v1_export_env.states / "checkpoint.bin"
    target.write_bytes(b"x" * (2 * 1024 * 1024))
    _patch_midstream_mutation(monkeypatch, target)

    with pytest.raises(HTTPException) as exc_info:
        await _run_export(v1_export_env)

    assert exc_info.value.status_code == 409
    assert _archive_paths(v1_export_env.staging) == set()


@pytest.mark.asyncio
async def test_v1_export_fails_closed_without_descriptor_primitives(
    v1_export_env, monkeypatch,
):
    monkeypatch.setattr(os, "supports_dir_fd", set())

    with pytest.raises(HTTPException) as exc_info:
        await _run_export(v1_export_env)

    assert exc_info.value.status_code == 501
    assert _archive_paths(v1_export_env.staging) == set()


@pytest.mark.asyncio
async def test_v1_export_rejects_insufficient_disk(v1_export_env, monkeypatch):
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0, total=0, used=0),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _run_export(v1_export_env)

    assert exc_info.value.status_code == 507
    assert _archive_paths(v1_export_env.staging) == set()
