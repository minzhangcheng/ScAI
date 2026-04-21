"""Chunked file upload service: init, chunk, complete, and cleanup.

No imports from backend.main — uses backend.state for sessions and
callback injection for session persistence.
"""
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Callable, Awaitable, Optional

from fastapi import HTTPException, Request

from backend.config import UPLOAD_CHUNK_SIZE, UPLOAD_TEMP_DIR, get_session_uploads
from backend.utils.auth import get_db
from backend.utils.files import safe_filename, validate_path_within
from backend.state import sessions

logger = logging.getLogger(__name__)

# Per-upload locks to prevent concurrent complete calls
_complete_locks: Dict[str, asyncio.Lock] = {}

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
    expected_sha256 = body.get("expected_sha256", "")
    total_chunks = body.get("total_chunks", 0)
    chunk_size = body.get("chunk_size", UPLOAD_CHUNK_SIZE)

    filename = safe_filename(raw_filename)
    if not filename or not expected_sha256:
        raise HTTPException(status_code=400, detail="filename and expected_sha256 required")

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

    db = await get_db()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

    upload_id = await db.create_upload_session(
        session_id=session_id,
        filename=filename,
        expected_sha256=expected_sha256,
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
    db = await get_db()
    upload = await db.get_upload_session(upload_id)

    if upload is None or upload["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Upload session not found")

    if upload["completed"]:
        raise HTTPException(status_code=400, detail="Upload already completed")

    if len(chunk_data) > upload["chunk_size"] * 2:
        raise HTTPException(status_code=400, detail="Chunk too large")

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
    lock = _complete_locks.setdefault(upload_id, asyncio.Lock())
    async with lock:
        db = await get_db()
        upload = await db.get_upload_session(upload_id)

        if upload is None or upload["session_id"] != session_id:
            raise HTTPException(status_code=404, detail="Upload session not found")

        # Idempotent: already completed — return the actual stored path
        if upload["completed"]:
            stored_path = ""
            state = sessions.get(session_id)
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

        upload_dir = get_session_uploads(session_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = safe_filename(upload["filename"])
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        final_path = upload_dir / safe_name
        final_path = validate_path_within(final_path, upload_dir)
        sha256 = hashlib.sha256()

        # Assembly with rollback on failure
        try:
            with open(final_path, "wb") as f:
                for i in range(upload["total_chunks"]):
                    chunk_path = UPLOAD_TEMP_DIR / f"{upload_id}.chunk.{i}"
                    try:
                        data = chunk_path.read_bytes()
                    except FileNotFoundError:
                        raise HTTPException(status_code=400, detail=f"Chunk {i} missing")
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

        # Update session state via callbacks
        state = sessions.get(session_id)
        if state:
            placeholder_id = upload.get("placeholder_file_id")
            target = next(
                (df for df in state.data_files if df.id == placeholder_id), None
            ) if placeholder_id else None
            if target:
                target.file_path = str(final_path)
            if _on_session_update:
                await _on_session_update(session_id, state)
            if _on_session_save:
                await _on_session_save(session_id, state)

        return {"success": True, "filename": upload["filename"], "path": str(final_path)}


async def cleanup_session_uploads(session_id: str) -> None:
    """Delete all upload session records and chunk files for a session."""
    db = await get_db()
    uploads = await db.list_uploads_for_session(session_id)
    for u in uploads:
        for i in range(u.get("total_chunks", 0) or 0):
            p = UPLOAD_TEMP_DIR / f"{u['id']}.chunk.{i}"
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    await db.delete_uploads_for_session(session_id)
