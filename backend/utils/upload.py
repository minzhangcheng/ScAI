"""
Chunked file upload utilities for handling large files (up to 10GB)
Supports resume on interruption via chunk-level tracking.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import aiofiles

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
        """Delete one validated upload without re-entering its lock."""
        failures: list[tuple[Path, OSError]] = []
        artifacts = [
            path for path in self.temp_dir.iterdir()
            if self._artifact_upload_id(path.name) == upload_id
        ]
        meta_path = self._get_upload_meta_path(upload_id)
        # A completed metadata record is the idempotency marker.  Reclaim it
        # only after every chunk/tmp artifact has gone so partial cleanup can
        # never turn a committed upload back into an apparently retryable one.
        artifacts.sort(key=lambda path: path == meta_path)
        for path in artifacts:
            if path == meta_path and failures:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append((path, exc))
        if failures:
            first_path, first_error = failures[0]
            raise OSError(
                f"failed to remove {len(failures)} upload artifact(s); "
                f"first path: {first_path}"
            ) from first_error
        await self._decrement_active_uploads(
            str(meta.get("session_id") or ""), default=0,
        )

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
        """Assemble one upload in a worker that its caller always joins."""
        digest = hashlib.sha256()
        with final_path.open("wb") as out_f:
            for index in range(total_chunks):
                chunk_path = self._get_chunk_path(upload_id, index)
                with chunk_path.open("rb") as chunk_f:
                    while True:
                        data = chunk_f.read(1024 * 1024)
                        if not data:
                            break
                        digest.update(data)
                        out_f.write(data)
        return final_path.stat().st_size, digest.hexdigest()

    @staticmethod
    def _publish_staged_sync(staging_path: Path, final_path: Path) -> None:
        """Copy one owned staging file to a destination without clobbering."""
        created = False
        try:
            with staging_path.open("rb") as source, final_path.open("xb") as target:
                created = True
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            if created:
                final_path.unlink(missing_ok=True)
            raise

    async def init_upload(
        self,
        session_id: str,
        filename: str,
        total_size: int,
        total_chunks: int,
        file_type: str,
        group: Optional[str] = None
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
        """
        Assemble chunks into final file.

        Args:
            upload_id: Upload identifier
            final_path: Destination path for assembled file

        Returns:
            Dict with success status and file info
        """
        async with self._hold_upload_lock(upload_id):
            meta = await self._load_upload_meta_locked(upload_id, session_id)
            if meta is None:
                return {
                    "success": False,
                    "error": "Upload not found",
                    "not_found": True,
                }
            if meta.get("completed"):
                return {
                    "success": False,
                    "error": "Upload already completed",
                    "completed": True,
                }
            total_chunks = int(meta.get("total_chunks") or 0)
            total_size = int(meta.get("total_size") or 0)
            max_chunks = (
                (total_size + self.chunk_size - 1) // self.chunk_size
                if total_size > 0 else 0
            )
            received = meta.get("received_chunks")
            if (
                total_chunks <= 0
                or total_chunks != max_chunks
                or not isinstance(received, list)
                or any(
                    not isinstance(index, int)
                    or index < 0
                    or index >= total_chunks
                    for index in received
                )
            ):
                return {
                    "success": False,
                    "error": "Corrupted upload metadata",
                }

            expected_chunks = set(range(total_chunks))
            received_chunks = set(received)
            if expected_chunks != received_chunks:
                missing = sorted(expected_chunks - received_chunks)
                await self._cleanup_loaded_upload_locked(upload_id, meta)
                return {
                    "success": False,
                    "error": f"Missing chunks: {missing}",
                    "missing_chunks": missing,
                }

            final_path = Path(final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                assembled_path = str(meta.get("assembled_path") or "")
                assembled_size = int(meta.get("assembled_size") or -1)
                assembled_sha256 = str(meta.get("assembled_sha256") or "")
                staged_size = int(meta.get("staged_size") or -1)
                staged_sha256 = str(meta.get("staged_sha256") or "")
                expected_sha256 = ""
                if (
                    defer_cleanup
                    and assembled_path == str(final_path.resolve())
                    and assembled_size == final_path.stat().st_size
                    and assembled_size == total_size
                    and assembled_sha256
                ):
                    expected_sha256 = assembled_sha256
                elif (
                    defer_cleanup
                    and staged_size == final_path.stat().st_size == total_size
                    and staged_sha256
                ):
                    # Publication can finish while caller cancellation is being
                    # deferred. The already-durable staging digest proves the
                    # final content even if the assembled marker was not yet
                    # written.
                    expected_sha256 = staged_sha256
                if expected_sha256:
                    actual_sha256 = await self._sha256_file(final_path)
                    if actual_sha256 == expected_sha256:
                        return {
                            "success": True,
                            "path": str(final_path),
                            "size": total_size,
                            "filename": meta["filename"],
                            "file_type": meta["file_type"],
                            "group": meta.get("group"),
                            "cleanup_pending": True,
                            "reused_assembly": True,
                        }
                return {
                    "success": False,
                    "error": "Destination file already exists",
                    "conflict": True,
                }

            staging_path = self.temp_dir / f"{upload_id}.assembled.tmp"
            staged_size = int(meta.get("staged_size") or -1)
            staged_sha256 = str(meta.get("staged_sha256") or "")
            reusable_staging = (
                defer_cleanup
                and str(meta.get("staged_path") or "")
                == str(staging_path.resolve())
                and staging_path.is_file()
                and staging_path.stat().st_size == staged_size == total_size
                and bool(staged_sha256)
            )
            if reusable_staging:
                reusable_staging = (
                    await self._sha256_file(staging_path) == staged_sha256
                )
            if not reusable_staging:
                # This name is derived from the unguessable, session-bound
                # upload ID and never points at user content.
                staging_path.unlink(missing_ok=True)
                logger.info(
                    "Assembling %d chunks into owned staging %s",
                    total_chunks,
                    staging_path,
                )
                try:
                    staged_size, staged_sha256 = await run_in_owned_thread(
                        self._assemble_chunks_sync,
                        upload_id,
                        total_chunks,
                        staging_path,
                    )
                except asyncio.CancelledError:
                    try:
                        staging_path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning(
                            "Cancelled upload %s retained owned staging %s: %s",
                            upload_id,
                            staging_path,
                            exc,
                        )
                    raise
                except Exception as exc:
                    logger.exception("Failed to assemble file")
                    staging_path.unlink(missing_ok=True)
                    await self._cleanup_loaded_upload_locked(upload_id, meta)
                    return {
                        "success": False,
                        "error": f"Assembly failed: {exc}",
                    }

                if staged_size != total_size:
                    logger.error(
                        "Size mismatch: expected %d, got %d",
                        total_size,
                        staged_size,
                    )
                    staging_path.unlink(missing_ok=True)
                    await self._cleanup_loaded_upload_locked(upload_id, meta)
                    return {
                        "success": False,
                        "error": (
                            f"Size mismatch: expected {total_size}, "
                            f"got {staged_size}"
                        ),
                    }

                if defer_cleanup:
                    meta["staged_path"] = str(staging_path.resolve())
                    meta["staged_size"] = staged_size
                    meta["staged_sha256"] = staged_sha256
                    meta["staged_at"] = datetime.now().isoformat()
                    await self._write_upload_meta_locked(upload_id, meta)

            try:
                await run_in_owned_thread(
                    self._publish_staged_sync,
                    staging_path,
                    final_path,
                )
            except FileExistsError:
                return {
                    "success": False,
                    "error": "Destination file already exists",
                    "conflict": True,
                }

            actual_size = staged_size
            assembled_sha256 = staged_sha256

            if defer_cleanup:
                # Bind the unpublished assembly to this upload before handing
                # it to the route-level state transaction.  If rollback cannot
                # unlink the file, a later retry can safely reclaim this exact
                # assembly instead of becoming permanently wedged on a path
                # conflict.
                meta["assembled_path"] = str(final_path.resolve())
                meta["assembled_size"] = actual_size
                meta["assembled_sha256"] = assembled_sha256
                meta["assembled_at"] = datetime.now().isoformat()
                try:
                    await self._write_upload_meta_locked(upload_id, meta)
                except BaseException:
                    final_path.unlink(missing_ok=True)
                    raise
            else:
                await self._cleanup_loaded_upload_locked(upload_id, meta)
            logger.info(
                "Completed chunked upload: %s -> %s (%d bytes)",
                upload_id, final_path, actual_size,
            )
            return {
                "success": True,
                "path": str(final_path),
                "size": actual_size,
                "filename": meta["filename"],
                "file_type": meta["file_type"],
                "group": meta.get("group"),
                "cleanup_pending": bool(defer_cleanup),
            }

    async def rollback_assembled_upload(
        self,
        upload_id: str,
        final_path: Path,
        *,
        session_id: Optional[str] = None,
    ) -> bool:
        """Remove an unpublished assembly while retaining resumable chunks."""
        async with self._hold_upload_lock(upload_id):
            inferred_session = self._session_from_upload_id(upload_id)
            if inferred_session is None or (
                session_id is not None and inferred_session != session_id
            ):
                return False
            meta = await self._load_upload_meta_locked(upload_id, session_id)
            if meta is None or meta.get("completed"):
                return False
            staging_path = self.temp_dir / f"{upload_id}.assembled.tmp"
            recorded_staging = str(meta.get("staged_path") or "")
            if recorded_staging and recorded_staging != str(staging_path.resolve()):
                return False

            def _rollback_files() -> None:
                Path(final_path).unlink(missing_ok=True)
                staging_path.unlink(missing_ok=True)

            try:
                await run_in_owned_thread(_rollback_files)
            except OSError:
                # Preserve the assembly/staging markers so the exact same
                # upload can retry when either owned file is temporarily busy.
                return False
            for key in (
                "assembled_path",
                "assembled_size",
                "assembled_sha256",
                "assembled_at",
                "staged_path",
                "staged_size",
                "staged_sha256",
                "staged_at",
            ):
                meta.pop(key, None)
            await self._write_upload_meta_locked(upload_id, meta)
            return True

    async def commit_completed_upload(
        self,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
        data_file_id: Optional[str] = None,
    ) -> bool:
        """Mark a route-level commit, then reclaim resumability artifacts.

        A durable ``completed`` marker is written before cleanup.  If unlinking
        one artifact fails, a surviving metadata file remains non-retryable and
        session cleanup can reclaim it later without registering the file twice.
        The committed data file itself is never removed by this method.
        """
        async with self._hold_upload_lock(upload_id):
            meta = await self._load_upload_meta_locked(upload_id, session_id)
            if meta is None:
                return False
            meta["completed"] = True
            meta["completed_at"] = datetime.now().isoformat()
            if data_file_id:
                meta["data_file_id"] = data_file_id
            try:
                await self._write_upload_meta_locked(upload_id, meta)
            except OSError as exc:
                # The session record (or generic final file) is already
                # durable.  A housekeeping marker failure must not turn that
                # successful commit into a client-visible failed retry.
                logger.warning(
                    "Committed upload %s could not persist cleanup marker: %s",
                    upload_id,
                    exc,
                )
                await self._decrement_active_uploads(
                    str(meta.get("session_id") or ""), default=0,
                )
                return False
            try:
                await self._cleanup_loaded_upload_locked(upload_id, meta)
            except OSError as exc:
                # The data file and session record are already committed. Keep
                # this a housekeeping warning rather than turning a valid
                # upload into a retry that could duplicate its DataFile record.
                await self._decrement_active_uploads(
                    str(meta.get("session_id") or ""), default=0,
                )
                logger.warning(
                    "Committed upload %s retained cleanup artifacts: %s",
                    upload_id,
                    exc,
                )
                return False
            return True

    async def cleanup_upload(
        self,
        upload_id: str,
        *,
        strict: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Clean up temporary files for an upload.

        Args:
            upload_id: Upload identifier
        """
        try:
            async with self._hold_upload_lock(upload_id):
                meta = await self._load_upload_meta_locked(
                    upload_id, session_id,
                )
                if meta is None:
                    return False
                await self._cleanup_loaded_upload_locked(upload_id, meta)
                logger.info("Cleaned up upload %s", upload_id)
                return True
        except Exception as exc:
            logger.warning("Error cleaning up upload %s: %s", upload_id, exc)
            if strict:
                raise
            return False

    async def cleanup_session_artifacts(self, session_id: str) -> None:
        """Remove every provably session-owned legacy upload artifact.

        Metadata may be truncated or unreadable after a crash, and chunks may
        outlive their metadata. Generated upload IDs have an unambiguous UUID
        suffix (with a timestamp/hash parser retained for legacy IDs), so both
        cases can still be attributed without a prefix match that would confuse
        IDs such as ``sample`` and ``sample_extra``. Any ownership
        contradiction fails closed.
        """
        def _clean() -> set[str]:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            paths = list(self.temp_dir.iterdir())
            candidates: set[str] = set()

            for path in paths:
                upload_id = self._artifact_upload_id(path.name)
                if upload_id is None:
                    continue
                inferred_session = self._session_from_upload_id(upload_id)
                if inferred_session == session_id:
                    candidates.add(upload_id)

                if not path.name.endswith(".meta.json"):
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    declared_session = str(payload.get("session_id") or "")
                    declared_upload = str(payload.get("upload_id") or "")
                except Exception:
                    if inferred_session == session_id:
                        candidates.add(upload_id)
                    continue

                if declared_session == session_id:
                    if declared_upload and declared_upload != upload_id:
                        raise ValueError(
                            f"upload metadata filename/id mismatch: {path.name}"
                        )
                    if inferred_session not in (None, session_id):
                        raise ValueError(
                            f"upload metadata ownership mismatch: {path.name}"
                        )
                    candidates.add(upload_id)
                elif inferred_session == session_id:
                    raise ValueError(
                        f"upload metadata session mismatch: {path.name}"
                    )

            failures: list[tuple[Path, OSError]] = []
            for path in paths:
                upload_id = self._artifact_upload_id(path.name)
                if upload_id not in candidates:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    failures.append((path, exc))
            if failures:
                first_path, first_error = failures[0]
                raise OSError(
                    f"failed to remove {len(failures)} legacy upload artifact(s); "
                    f"first path: {first_path}"
                ) from first_error
            return candidates

        candidates = await run_in_owned_thread(_clean)

        async with self._upload_locks_guard:
            for upload_id in candidates:
                self._upload_locks.pop(upload_id, None)
                self._upload_lock_users.pop(upload_id, None)
        async with self._active_uploads_lock:
            self._active_uploads.pop(session_id, None)

    async def get_upload_status(
        self,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get upload progress and status.

        Args:
            upload_id: Upload identifier

        Returns:
            Dict with status info or None if not found
        """
        async with self._hold_upload_lock(upload_id):
            meta = await self._load_upload_meta_locked(upload_id, session_id)
            if meta is None:
                return None
            received = meta.get("received_chunks")
            total_chunks = int(meta.get("total_chunks") or 0)
            if not isinstance(received, list) or total_chunks <= 0:
                return None
            return {
                "upload_id": upload_id,
                "filename": meta["filename"],
                "file_type": meta["file_type"],
                "group": meta.get("group"),
                "total_size": meta["total_size"],
                "total_chunks": total_chunks,
                "received_chunks": len(received),
                "progress": len(received) / total_chunks * 100,
                "missing_chunks": sorted(
                    set(range(total_chunks)) - set(received)
                ),
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
                "completed": meta.get("completed", False),
            }

    async def list_pending_uploads(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all pending (incomplete) uploads.

        Args:
            session_id: Optional filter by session ID

        Returns:
            List of upload status dicts
        """
        pending = []

        for meta_path in self.temp_dir.glob("*.meta.json"):
            try:
                payload = await run_in_owned_thread(
                    meta_path.read_text,
                    encoding="utf-8",
                )
                meta = json.loads(payload)

                if meta.get("completed", False):
                    continue

                if session_id and meta.get("session_id") != session_id:
                    continue

                status = await self.get_upload_status(meta["upload_id"])
                if status:
                    pending.append(status)
            except Exception as e:
                logger.warning(f"Error reading upload metadata {meta_path}: {e}")

        return pending

    async def cancel_upload(
        self,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cancel and cleanup an upload.

        Args:
            upload_id: Upload identifier

        Returns:
            Dict with success status and optional error
        """
        async with self._hold_upload_lock(upload_id):
            meta = await self._load_upload_meta_locked(upload_id, session_id)
            if meta is None:
                return {
                    "success": False,
                    "error": "Upload not found",
                    "not_found": True,
                }
            await self._cleanup_loaded_upload_locked(upload_id, meta)
            return {"success": True}


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
