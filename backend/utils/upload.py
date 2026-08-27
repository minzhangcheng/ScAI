"""
Chunked file upload utilities for handling large files (up to 10GB)
Supports resume on interruption via chunk-level tracking.
"""
import asyncio
import hashlib
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import aiofiles

from backend.utils.upload_parts import maintenance, transactions
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    run_in_owned_thread,
)

logger = logging.getLogger(__name__)


class ChunkedUploadManager:
    """
    Manages chunked file uploads with resume support.

    Features:
    - Split large files into configurable chunks (default 10MB)
    - Track received chunks for resume capability
    - Verify chunk integrity via MD5 hash
    - Atomic file assembly
    - Automatic cleanup of temporary files
    """

    # Maximum concurrent uploads per session
    MAX_CONCURRENT_UPLOADS_PER_SESSION = 5
    ALLOWED_FILE_TYPES = {
        "snrna_seq",
        "loom",
        "bulk_rna_seq",
        "pathology",
        "biochemistry",
        "file",
    }
    ALLOWED_GROUPS = {None, "control", "treatment"}
    _LEGACY_UPLOAD_ID_RE = re.compile(
        r"^(?P<session>.+)_(?P<timestamp>[0-9]{14})_(?P<hash>[0-9a-f]{8})$"
    )
    _UPLOAD_ID_RE = re.compile(
        r"^(?P<session>.+)_(?P<nonce>[0-9a-f]{32})$"
    )

    def __init__(self, temp_dir: Path, chunk_size: int = 10 * 1024 * 1024):
        """
        Initialize the upload manager.

        Args:
            temp_dir: Directory to store temporary chunks
            chunk_size: Size of each chunk in bytes (default 10MB)
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        # Per-upload locks to prevent concurrent metadata read/write race conditions
        self._upload_locks: Dict[str, asyncio.Lock] = {}
        self._upload_lock_users: Dict[str, int] = {}
        self._upload_locks_guard = asyncio.Lock()
        # Track active uploads per session for rate limiting
        self._active_uploads: Dict[str, int] = {}  # session_id -> count
        self._active_uploads_lock = asyncio.Lock()

    async def _get_upload_lock(self, upload_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific upload (thread-safe)"""
        async with self._upload_locks_guard:
            if upload_id not in self._upload_locks:
                self._upload_locks[upload_id] = asyncio.Lock()
            return self._upload_locks[upload_id]

    @asynccontextmanager
    async def _hold_upload_lock(self, upload_id: str):
        """Reference-count a per-upload lock and retire idle unknown IDs."""
        async with self._upload_locks_guard:
            lock = self._upload_locks.get(upload_id)
            if lock is None:
                lock = asyncio.Lock()
                self._upload_locks[upload_id] = lock
            self._upload_lock_users[upload_id] = (
                self._upload_lock_users.get(upload_id, 0) + 1
            )
        try:
            async with lock:
                yield
        finally:
            async with self._upload_locks_guard:
                remaining = self._upload_lock_users.get(upload_id, 1) - 1
                if remaining <= 0:
                    self._upload_lock_users.pop(upload_id, None)
                    if self._upload_locks.get(upload_id) is lock:
                        self._upload_locks.pop(upload_id, None)
                else:
                    self._upload_lock_users[upload_id] = remaining

    def _get_upload_meta_path(self, upload_id: str) -> Path:
        """Get path to upload metadata file"""
        return self.temp_dir / f"{upload_id}.meta.json"

    def _get_chunk_path(self, upload_id: str, chunk_index: int) -> Path:
        """Get path to a specific chunk file"""
        return self.temp_dir / f"{upload_id}.chunk.{chunk_index}"

    @classmethod
    def _session_from_upload_id(cls, upload_id: str) -> Optional[str]:
        """Recover the exact session component from a generated upload ID."""
        match = cls._UPLOAD_ID_RE.fullmatch(upload_id)
        if match is None:
            match = cls._LEGACY_UPLOAD_ID_RE.fullmatch(upload_id)
        return match.group("session") if match else None

    @staticmethod
    def _artifact_upload_id(name: str) -> Optional[str]:
        if name.endswith(".assembled.tmp"):
            return name[:-len(".assembled.tmp")]
        if name.endswith(".tmp"):
            name = name[:-len(".tmp")]
        if name.endswith(".meta.json"):
            return name[:-len(".meta.json")]
        if ".chunk." in name:
            upload_id, index = name.rsplit(".chunk.", 1)
            if upload_id and index.isdigit():
                return upload_id
        return None

    async def _decrement_active_uploads(self, session_id: str, default: int = 0) -> None:
        """Decrement active upload counter for a session, removing entry when it reaches zero."""
        async with self._active_uploads_lock:
            count = self._active_uploads.get(session_id, default)
            if count <= 1:
                self._active_uploads.pop(session_id, None)
            else:
                self._active_uploads[session_id] = count - 1

    async def _load_upload_meta_locked(
        self,
        upload_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read and bind metadata while the upload lock is held."""
        inferred_session = self._session_from_upload_id(upload_id)
        if inferred_session is None:
            return None
        meta_path = self._get_upload_meta_path(upload_id)
        if not meta_path.is_file():
            return None
        try:
            payload = await run_in_owned_thread(
                meta_path.read_text,
                encoding="utf-8",
            )
            meta = json.loads(payload)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(meta, dict):
            return None
        declared_upload = str(meta.get("upload_id") or "")
        declared_session = str(meta.get("session_id") or "")
        if declared_upload != upload_id:
            return None
        if not declared_session or inferred_session != declared_session:
            return None
        if session_id is not None and declared_session != session_id:
            return None
        return meta

    async def _cleanup_loaded_upload_locked(
        self,
        upload_id: str,
        meta: Dict[str, Any],
    ) -> None:
        return await maintenance._cleanup_loaded_upload_locked(self, upload_id, meta)

    async def _write_upload_meta_locked(
        self,
        upload_id: str,
        meta: Dict[str, Any],
    ) -> None:
        """Atomically replace metadata while the upload lock is held."""
        meta_path = self._get_upload_meta_path(upload_id)
        meta_tmp_path = meta_path.with_name(f"{meta_path.name}.tmp")

        def _write() -> None:
            try:
                meta_tmp_path.write_text(
                    json.dumps(meta, indent=2),
                    encoding="utf-8",
                )
                meta_tmp_path.replace(meta_path)
            except BaseException:
                meta_tmp_path.unlink(missing_ok=True)
                raise

        await run_in_owned_thread(_write)

    @staticmethod
    async def _sha256_file(path: Path) -> str:
        def _digest() -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        return digest.hexdigest()
                    digest.update(chunk)

        return await run_in_owned_thread(_digest)

    def _assemble_chunks_sync(
        self,
        upload_id: str,
        total_chunks: int,
        final_path: Path,
    ) -> tuple[int, str]:
        return transactions._assemble_chunks_sync(
            self, upload_id, total_chunks, final_path,
        )

    @staticmethod
    def _publish_staged_sync(staging_path: Path, final_path: Path) -> None:
        return transactions._publish_staged_sync(staging_path, final_path)

    async def init_upload(
        self,
        session_id: str,
        filename: str,
        total_size: int,
        total_chunks: int,
        file_type: str,
        group: Optional[str] = None,
        expected_sha256: Optional[str] = None,
    ) -> str:
        """
        Initialize a chunked upload and return upload_id.

        Args:
            session_id: Session ID for this upload
            filename: Original filename
            total_size: Total file size in bytes
            total_chunks: Expected number of chunks
            file_type: Type of file ('snrna_seq', 'loom', 'pathology', 'biochemistry')
            group: Optional group ('control' or 'treatment' for seq data)

        Returns:
            upload_id: Unique identifier for this upload
        """
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        if file_type not in self.ALLOWED_FILE_TYPES:
            raise ValueError("Unsupported file_type")
        if group not in self.ALLOWED_GROUPS:
            raise ValueError("Unsupported upload group")
        if expected_sha256 is not None:
            expected_sha256 = str(expected_sha256).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                raise ValueError("expected_sha256 must be a 64-character hex digest")
        expected_chunks = (total_size + self.chunk_size - 1) // self.chunk_size
        if total_chunks != expected_chunks:
            raise ValueError(
                f"total_chunks must equal {expected_chunks} for the declared size"
            )

        # Enforce per-session concurrent upload limit and atomically increment counter
        async with self._active_uploads_lock:
            active = self._active_uploads.get(session_id, 0)
            if active >= self.MAX_CONCURRENT_UPLOADS_PER_SESSION:
                raise ValueError(
                    f"Too many concurrent uploads for session {session_id} "
                    f"(limit: {self.MAX_CONCURRENT_UPLOADS_PER_SESSION})"
                )
            self._active_uploads[session_id] = active + 1

        # A random suffix prevents same-second collisions and makes a leaked
        # session identifier insufficient to guess another upload credential.
        upload_id = f"{session_id}_{uuid.uuid4().hex}"

        try:
            meta = {
                "upload_id": upload_id,
                "session_id": session_id,
                "filename": filename,
                "total_size": total_size,
                "total_chunks": total_chunks,
                "file_type": file_type,
                "group": group,
                "expected_sha256": expected_sha256,
                "received_chunks": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completed": False
            }

            await self._write_upload_meta_locked(upload_id, meta)
        except BaseException:
            meta_path = self._get_upload_meta_path(upload_id)
            meta_tmp_path = meta_path.with_name(f"{meta_path.name}.tmp")

            def _remove_unpublished_init() -> None:
                meta_tmp_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)

            try:
                await run_in_owned_thread(_remove_unpublished_init)
            except BaseException as cleanup_exc:
                logger.warning(
                    "Could not reclaim cancelled upload initialization %s: %s",
                    upload_id,
                    cleanup_exc,
                )
            # Cancellation is a failed initialization too. Roll the quota back
            # before propagating it so repeated cancelled requests cannot
            # exhaust the per-session upload slots.
            await self._decrement_active_uploads(session_id, default=1)
            raise

        logger.info(
            f"Initialized chunked upload: {upload_id} for {filename} "
            f"({total_size} bytes, {total_chunks} chunks)"
        )
        return upload_id

    async def save_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        chunk_data: bytes,
        chunk_hash: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a single chunk.

        Args:
            upload_id: Upload identifier
            chunk_index: Index of this chunk (0-based)
            chunk_data: Raw chunk bytes
            chunk_hash: Optional MD5 hash for verification

        Returns:
            Dict with success status and progress info
        """
        if len(chunk_data) > self.chunk_size:
            return {
                "success": False,
                "error": f"Chunk exceeds maximum size ({self.chunk_size} bytes)",
                "too_large": True,
            }

        # Verify hash before taking the disk mutation lock.
        if chunk_hash:
            actual_hash = hashlib.md5(chunk_data).hexdigest()
            if actual_hash != chunk_hash:
                logger.warning(
                    f"Chunk hash mismatch for {upload_id} chunk {chunk_index}: "
                    f"expected {chunk_hash}, got {actual_hash}"
                )
                return {
                    "success": False,
                    "error": f"Chunk hash mismatch: expected {chunk_hash}, got {actual_hash}"
                }

        async with self._hold_upload_lock(upload_id):
            meta = await self._load_upload_meta_locked(upload_id, session_id)
            if meta is None:
                return {
                    "success": False,
                    "error": "Upload not found",
                    "not_found": True,
                }
            if meta.get("completed"):
                return {"success": False, "error": "Upload already completed"}
            if not isinstance(meta.get("received_chunks"), list):
                return {"success": False, "error": "Corrupted upload metadata: received_chunks is not a list"}
            total_chunks = int(meta.get("total_chunks") or 0)
            if chunk_index < 0 or chunk_index >= total_chunks:
                return {
                    "success": False,
                    "error": (
                        f"chunk_index {chunk_index} out of range "
                        f"[0,{total_chunks})"
                    ),
                }

            chunk_path = self._get_chunk_path(upload_id, chunk_index)
            tmp_path = chunk_path.with_name(f"{chunk_path.name}.tmp")
            already_received = chunk_index in meta["received_chunks"]
            if not already_received:
                meta["received_chunks"].append(chunk_index)
                meta["received_chunks"].sort()
            meta["updated_at"] = datetime.now().isoformat()
            meta_path = self._get_upload_meta_path(upload_id)
            meta_tmp_path = meta_path.with_name(f"{meta_path.name}.tmp")

            def _write_chunk_and_meta() -> None:
                try:
                    tmp_path.write_bytes(chunk_data)
                    tmp_path.replace(chunk_path)
                    meta_tmp_path.write_text(
                        json.dumps(meta, indent=2),
                        encoding="utf-8",
                    )
                    meta_tmp_path.replace(meta_path)
                except BaseException:
                    tmp_path.unlink(missing_ok=True)
                    meta_tmp_path.unlink(missing_ok=True)
                    if not already_received:
                        chunk_path.unlink(missing_ok=True)
                    raise

            try:
                await run_in_owned_thread(_write_chunk_and_meta)
            except BaseException:
                if not already_received:
                    meta["received_chunks"].remove(chunk_index)
                raise

        logger.debug(f"Saved chunk {chunk_index}/{meta['total_chunks']} for upload {upload_id}")

        return {
            "success": True,
            "chunk_index": chunk_index,
            "received_count": len(meta["received_chunks"]),
            "total_chunks": meta["total_chunks"],
            "progress": (len(meta["received_chunks"]) / meta["total_chunks"] * 100) if meta["total_chunks"] > 0 else 0
        }

    async def complete_upload(
        self,
        upload_id: str,
        final_path: Path,
        *,
        session_id: Optional[str] = None,
        defer_cleanup: bool = False,
    ) -> Dict[str, Any]:
        return await transactions.complete_upload(
            self,
            upload_id,
            final_path,
            session_id=session_id,
            defer_cleanup=defer_cleanup,
        )

    async def rollback_assembled_upload(
        self,
        upload_id: str,
        final_path: Path,
        *,
        session_id: Optional[str] = None,
    ) -> bool:
        return await transactions.rollback_assembled_upload(
            self, upload_id, final_path, session_id=session_id,
        )

    async def commit_completed_upload(
        self,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
        data_file_id: Optional[str] = None,
    ) -> bool:
        return await transactions.commit_completed_upload(
            self,
            upload_id,
            session_id=session_id,
            data_file_id=data_file_id,
        )

    async def cleanup_upload(
        self,
        upload_id: str,
        *,
        strict: bool = False,
        session_id: Optional[str] = None,
    ):
        return await maintenance.cleanup_upload(
            self, upload_id, strict=strict, session_id=session_id,
        )

    async def cleanup_session_artifacts(self, session_id: str) -> None:
        return await maintenance.cleanup_session_artifacts(self, session_id)

    async def get_upload_status(
        self,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await maintenance.get_upload_status(
            self, upload_id, session_id=session_id,
        )

    async def list_pending_uploads(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await maintenance.list_pending_uploads(self, session_id)

    async def cancel_upload(
        self,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await maintenance.cancel_upload(
            self, upload_id, session_id=session_id,
        )

async def stream_save_file(file, dest_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """
    Stream file to disk without loading entirely into memory.
    Use this for regular (non-chunked) uploads to avoid memory issues.

    Args:
        file: FastAPI UploadFile object
        dest_path: Destination path
        chunk_size: Read chunk size (default 1MB)

    Returns:
        Path to saved file
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    async def _stream() -> None:
        async with aiofiles.open(dest_path, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)

    stream_task = asyncio.create_task(_stream())
    await await_with_deferred_cancellation(stream_task)

    return dest_path
