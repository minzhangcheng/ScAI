"""Chunked file upload service: init, chunk, complete, and cleanup.

No imports from backend.main — uses backend.state for sessions and
callback injection for session persistence.
"""
import hashlib
import asyncio
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Callable, Awaitable, Optional

from fastapi import HTTPException, Request

from backend.config import UPLOAD_CHUNK_SIZE, UPLOAD_TEMP_DIR, MAX_UPLOAD_SIZE, get_session_uploads
from backend.utils.auth import get_db
from backend.utils.files import safe_filename, validate_path_within
from backend.utils.upload import ChunkedUploadManager
from backend.services.session_lifecycle import (
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.services.session_store import _async_save_session_under_lifecycle
from backend.state import session_access_times, sessions, state_lock

# Singleton upload manager (supports chunked uploads up to 10GB)
upload_manager = ChunkedUploadManager(UPLOAD_TEMP_DIR, UPLOAD_CHUNK_SIZE)

logger = logging.getLogger(__name__)

# Per-upload locks to prevent concurrent complete calls
_complete_locks: Dict[str, asyncio.Lock] = {}
_complete_lock_refs: Dict[str, int] = {}
_registry_lock = asyncio.Lock()


async def read_chunk_with_limit(
    upload_file,
    *,
    max_bytes: int = UPLOAD_CHUNK_SIZE,
    read_size: int = 1024 * 1024,
) -> bytes:
    """Read one multipart chunk with a sentinel byte and a hard memory bound."""
    payload = bytearray()
    while True:
        remaining_with_sentinel = max_bytes + 1 - len(payload)
        chunk = await upload_file.read(min(read_size, remaining_with_sentinel))
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Chunk exceeds maximum size ({max_bytes} bytes)",
            )


async def _acquire_complete_lock(upload_id: str):
    async with _registry_lock:
        lock = _complete_locks.setdefault(upload_id, asyncio.Lock())
        _complete_lock_refs[upload_id] = _complete_lock_refs.get(upload_id, 0) + 1
    return lock, upload_id


def _release_complete_lock(upload_id: str) -> None:
    n = _complete_lock_refs.get(upload_id, 0) - 1
    if n <= 0:
        _complete_lock_refs.pop(upload_id, None)
        _complete_locks.pop(upload_id, None)
    else:
        _complete_lock_refs[upload_id] = n

# Callbacks injected by the route layer at startup
_on_session_update: Optional[Callable[[str, object], Awaitable[None]]] = None
_on_session_save: Optional[Callable[[str, object], Awaitable[None]]] = None


def set_session_callbacks(
    on_update: Callable[[str, object], Awaitable[None]],
    on_save: Callable[[str, object], Awaitable[None]],
) -> None:
    """Register session persistence callbacks (called once at app startup)."""
    global _on_session_update, _on_session_save
    _on_session_update = on_update
    _on_session_save = on_save


async def init_upload(session_id: str, request: Request) -> dict:
    """Initialize a chunked file upload for a session's placeholder file.

    Expects JSON body with: filename, expected_sha256, total_size, total_chunks, chunk_size?
    Returns: {"upload_id": str, "chunk_size": int}
    """
    body = await request.json()
    raw_filename = body.get("filename", "")
    expected_sha256 = str(body.get("expected_sha256", "") or "").strip().lower()
    try:
        total_chunks = int(body.get("total_chunks", 0) or 0)
        chunk_size = int(body.get("chunk_size", UPLOAD_CHUNK_SIZE))
        total_size = int(body.get("total_size", 0) or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid upload dimensions")

    if total_size <= 0 or total_size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="total_size out of allowed range")
    if chunk_size <= 0 or chunk_size > UPLOAD_CHUNK_SIZE:
        raise HTTPException(status_code=400, detail="chunk_size out of allowed range")
    expected_chunks = math.ceil(total_size / chunk_size)
    if total_chunks <= 0 or total_chunks != expected_chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                "total_chunks does not match total_size and chunk_size: "
                f"expected {expected_chunks}"
            ),
        )

    filename = safe_filename(raw_filename)
    if not filename or not expected_sha256:
        raise HTTPException(status_code=400, detail="filename and expected_sha256 required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise HTTPException(
            status_code=400,
            detail="expected_sha256 must be a 64-character hex digest",
        )

    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    user_id = getattr(request.state, "user_id", "")
    if getattr(state, "user_id", "") != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    target = next(
        (f for f in state.data_files if f.filename == filename and not f.file_path),
        None,
    )
    if target is None:
        raise HTTPException(status_code=400, detail=f"No placeholder for filename {filename}")

    async with get_lifecycle_lock(session_id):
        assert_generation_writable(session_id, state)
        db = await get_db()
        expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=24)
        ).isoformat()

        upload_id = await db.create_upload_session(
            session_id=session_id,
            filename=filename,
            expected_sha256=expected_sha256,
            total_size=total_size,
            total_chunks=total_chunks,
            chunk_size=chunk_size,
            expires_at_iso=expires_at,
            placeholder_file_id=target.id,
        )

        UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return {"upload_id": upload_id, "chunk_size": chunk_size}


async def upload_chunk(
    session_id: str,
    upload_id: str,
    chunk_index: int,
    chunk_data: bytes,
) -> dict:
    """Write a single chunk to disk and mark it received.

    Returns: {"received_chunks": int, "total_chunks": int}
    Raises: HTTPException on not found, already completed, or chunk too large.
    """
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async with get_lifecycle_lock(session_id):
        assert_generation_writable(session_id, state)
        db = await get_db()
        upload = await db.get_upload_session(upload_id)

        if upload is None or upload["session_id"] != session_id:
            raise HTTPException(status_code=404, detail="Upload session not found")

        if upload["completed"]:
            raise HTTPException(status_code=400, detail="Upload already completed")

        total_chunks = upload.get("total_chunks", 0) or 0
        if chunk_index < 0 or chunk_index >= total_chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"chunk_index {chunk_index} out of range "
                    f"[0,{total_chunks})"
                ),
            )
        total_size = upload.get("total_size")
        if not isinstance(total_size, int) or total_size <= 0:
            raise HTTPException(
                status_code=409,
                detail="Upload session predates size-bound uploads; restart upload",
            )
        chunk_size = int(upload.get("chunk_size", 0) or 0)
        expected_length = (
            chunk_size
            if chunk_index < total_chunks - 1
            else total_size - chunk_size * (total_chunks - 1)
        )
        if expected_length <= 0 or len(chunk_data) != expected_length:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Chunk {chunk_index} has {len(chunk_data)} bytes; "
                    f"expected {expected_length}"
                ),
            )

        chunk_path = UPLOAD_TEMP_DIR / f"{upload_id}.chunk.{chunk_index}"
        tmp_path = chunk_path.with_suffix(".tmp")
        tmp_path.write_bytes(chunk_data)
        tmp_path.rename(chunk_path)

        result = await db.mark_chunk_received(upload_id, chunk_index)
        if result is None:
            raise HTTPException(status_code=404, detail="Upload session not found")
        return {
            "received_chunks": result["received_chunks"],
            "total_chunks": result["total_chunks"],
        }


async def complete_upload(session_id: str, upload_id: str) -> dict:
    """Assemble chunks, verify SHA256, move to session dir, update state.

    Returns: {"success": True, "filename": str, "path": str}
    Raises: HTTPException on missing chunks, integrity failure, or assembly error.
    """
    lock, _ = await _acquire_complete_lock(upload_id)
    async with lock:
        generation_lock = get_lifecycle_lock(session_id)
        await generation_lock.acquire()
        try:
            state = sessions.get(session_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Session not found")
            assert_generation_writable(session_id, state)
            db = await get_db()
            upload = await db.get_upload_session(upload_id)

            if upload is None or upload["session_id"] != session_id:
                raise HTTPException(status_code=404, detail="Upload session not found")

            # Idempotent: already completed — return the actual stored path
            if upload["completed"]:
                stored_path = ""
                if state:
                    placeholder_id = upload.get("placeholder_file_id")
                    target = next(
                        (df for df in state.data_files if df.id == placeholder_id), None
                    ) if placeholder_id else None
                    if target and target.file_path:
                        stored_path = target.file_path
                return {"success": True, "filename": upload["filename"], "path": stored_path}

            if upload["received_chunks"] != upload["total_chunks"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing chunks: {upload['received_chunks']}/{upload['total_chunks']}",
                )
            total_size = upload.get("total_size")
            if not isinstance(total_size, int) or total_size <= 0:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Upload session predates size-bound uploads; restart upload"
                    ),
                )
            if total_size > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="Upload exceeds the configured size limit",
                )

            upload_dir = get_session_uploads(session_id)
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = safe_filename(upload["filename"])
            if not safe_name:
                raise HTTPException(status_code=400, detail="Invalid filename")
            final_path = upload_dir / safe_name
            final_path = validate_path_within(final_path, upload_dir)
            sha256 = hashlib.sha256()
            assembled_size = 0

            # Assembly with rollback on failure
            try:
                with open(final_path, "wb") as f:
                    for i in range(upload["total_chunks"]):
                        chunk_path = UPLOAD_TEMP_DIR / f"{upload_id}.chunk.{i}"
                        try:
                            data = chunk_path.read_bytes()
                        except FileNotFoundError:
                            raise HTTPException(status_code=400, detail=f"Chunk {i} missing")
                        expected_length = (
                            upload["chunk_size"]
                            if i < upload["total_chunks"] - 1
                            else total_size - upload["chunk_size"] * (
                                upload["total_chunks"] - 1
                            )
                        )
                        if expected_length <= 0 or len(data) != expected_length:
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    f"Chunk {i} has {len(data)} bytes; "
                                    f"expected {expected_length}"
                                ),
                            )
                        assembled_size += len(data)
                        if assembled_size > total_size:
                            raise HTTPException(
                                status_code=413,
                                detail="Upload exceeds initialized total_size",
                            )
                        f.write(data)
                        sha256.update(data)
            except HTTPException:
                try:
                    final_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            except Exception as e:
                try:
                    final_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HTTPException(status_code=500, detail=f"Assembly failed: {e}")

            if assembled_size != total_size:
                try:
                    final_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File size check failed for {upload['filename']}: "
                        f"expected {total_size}, got {assembled_size}"
                    ),
                )

            # SHA256 integrity check with consistent rollback
            actual_sha256 = sha256.hexdigest()
            if actual_sha256 != upload["expected_sha256"]:
                try:
                    final_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to unlink %s on integrity failure", final_path)
                for i in range(upload["total_chunks"]):
                    chunk_path = UPLOAD_TEMP_DIR / f"{upload_id}.chunk.{i}"
                    try:
                        chunk_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise HTTPException(
                    status_code=400,
                    detail=f"File integrity check failed for {upload['filename']}: "
                           f"expected {upload['expected_sha256']}, got {actual_sha256}",
                )

            # Cleanup chunks after successful assembly
            for i in range(upload["total_chunks"]):
                chunk_path = UPLOAD_TEMP_DIR / f"{upload_id}.chunk.{i}"
                try:
                    chunk_path.unlink(missing_ok=True)
                except OSError:
                    pass

            await db.complete_upload_session(upload_id)

            placeholder_id = upload.get("placeholder_file_id")
            target = next(
                (df for df in state.data_files if df.id == placeholder_id), None
            ) if placeholder_id else None
            if target:
                target.file_path = str(final_path)
            async with state_lock:
                if sessions.get(session_id) is not state:
                    raise HTTPException(
                        status_code=409,
                        detail="Session generation changed during upload",
                    )
                session_access_times[session_id] = time.time()
            await _async_save_session_under_lifecycle(
                session_id, state, force=True,
            )

            return {"success": True, "filename": upload["filename"], "path": str(final_path)}
        finally:
            generation_lock.release()
            _release_complete_lock(upload_id)


async def cleanup_session_uploads(session_id: str) -> None:
    """Delete all upload session records and chunk files for a session."""
    # The web upload API predates the DB-backed V1 uploader and stores JSON
    # metadata beside its chunks. Both stores are generation-agnostic, so both
    # must be empty before a tombstone can permit same-ID reuse.
    await upload_manager.cleanup_session_artifacts(session_id)

    db = await get_db()
    uploads = await db.list_uploads_for_session(session_id)
    failures: list[tuple[Path, OSError]] = []
    for u in uploads:
        for i in range(u.get("total_chunks", 0) or 0):
            p = UPLOAD_TEMP_DIR / f"{u['id']}.chunk.{i}"
            try:
                p.unlink(missing_ok=True)
            except OSError as exc:
                failures.append((p, exc))
    if failures:
        first_path, first_error = failures[0]
        raise OSError(
            f"failed to remove {len(failures)} upload chunk(s); "
            f"first path: {first_path}"
        ) from first_error
    await db.delete_uploads_for_session(session_id)
