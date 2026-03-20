"""
Chunked file upload utilities for handling large files (up to 10GB)
Supports resume on interruption via chunk-level tracking.
"""
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import aiofiles

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
        self._upload_locks_guard = asyncio.Lock()

    async def _get_upload_lock(self, upload_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific upload (thread-safe)"""
        async with self._upload_locks_guard:
            if upload_id not in self._upload_locks:
                self._upload_locks[upload_id] = asyncio.Lock()
            return self._upload_locks[upload_id]

    def _get_upload_meta_path(self, upload_id: str) -> Path:
        """Get path to upload metadata file"""
        return self.temp_dir / f"{upload_id}.meta.json"

    def _get_chunk_path(self, upload_id: str, chunk_index: int) -> Path:
        """Get path to a specific chunk file"""
        return self.temp_dir / f"{upload_id}.chunk.{chunk_index}"

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
        # Generate unique upload ID
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        upload_id = f"{session_id}_{timestamp}_{file_hash}"

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

        async with aiofiles.open(self._get_upload_meta_path(upload_id), 'w') as f:
            await f.write(json.dumps(meta, indent=2))

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
        chunk_hash: Optional[str] = None
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
        # Verify hash if provided
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

        # Save chunk to disk
        chunk_path = self._get_chunk_path(upload_id, chunk_index)
        async with aiofiles.open(chunk_path, 'wb') as f:
            await f.write(chunk_data)

        # Update metadata with lock to prevent concurrent read-modify-write race
        async with await self._get_upload_lock(upload_id):
            meta_path = self._get_upload_meta_path(upload_id)
            if not meta_path.exists():
                return {"success": False, "error": "Upload not found"}

            async with aiofiles.open(meta_path, 'r') as f:
                meta = json.loads(await f.read())

            if chunk_index not in meta["received_chunks"]:
                meta["received_chunks"].append(chunk_index)
                meta["received_chunks"].sort()

            meta["updated_at"] = datetime.now().isoformat()

            async with aiofiles.open(meta_path, 'w') as f:
                await f.write(json.dumps(meta, indent=2))

        logger.debug(f"Saved chunk {chunk_index}/{meta['total_chunks']} for upload {upload_id}")

        return {
            "success": True,
            "chunk_index": chunk_index,
            "received_count": len(meta["received_chunks"]),
            "total_chunks": meta["total_chunks"],
            "progress": (len(meta["received_chunks"]) / meta["total_chunks"] * 100) if meta["total_chunks"] > 0 else 0
        }

    async def complete_upload(self, upload_id: str, final_path: Path) -> Dict[str, Any]:
        """
        Assemble chunks into final file.

        Args:
            upload_id: Upload identifier
            final_path: Destination path for assembled file

        Returns:
            Dict with success status and file info
        """
        meta_path = self._get_upload_meta_path(upload_id)

        if not meta_path.exists():
            return {"success": False, "error": "Upload not found"}

        # Read metadata under lock to avoid race with concurrent save_chunk
        async with await self._get_upload_lock(upload_id):
            async with aiofiles.open(meta_path, 'r') as f:
                meta = json.loads(await f.read())

        # Verify all chunks received
        expected_chunks = set(range(meta["total_chunks"]))
        received_chunks = set(meta["received_chunks"])

        if expected_chunks != received_chunks:
            missing = sorted(expected_chunks - received_chunks)
            return {
                "success": False,
                "error": f"Missing chunks: {missing}",
                "missing_chunks": missing
            }

        # Ensure destination directory exists
        final_path = Path(final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # Assemble file using streaming to avoid memory issues
        logger.info(f"Assembling {meta['total_chunks']} chunks into {final_path}")

        try:
            async with aiofiles.open(final_path, 'wb') as out_f:
                for i in range(meta["total_chunks"]):
                    chunk_path = self._get_chunk_path(upload_id, i)
                    async with aiofiles.open(chunk_path, 'rb') as chunk_f:
                        # Read and write in smaller sub-chunks to manage memory
                        while True:
                            data = await chunk_f.read(1024 * 1024)  # 1MB at a time
                            if not data:
                                break
                            await out_f.write(data)
        except Exception as e:
            logger.exception("Failed to assemble file")
            if final_path.exists():
                final_path.unlink()
            return {"success": False, "error": f"Assembly failed: {str(e)}"}

        # Verify final file size
        actual_size = final_path.stat().st_size
        if actual_size != meta["total_size"]:
            logger.error(
                f"Size mismatch: expected {meta['total_size']}, got {actual_size}"
            )
            final_path.unlink()
            return {
                "success": False,
                "error": f"Size mismatch: expected {meta['total_size']}, got {actual_size}"
            }

        # Cleanup temp files
        await self.cleanup_upload(upload_id)

        logger.info(f"Completed chunked upload: {upload_id} -> {final_path} ({actual_size} bytes)")

        return {
            "success": True,
            "path": str(final_path),
            "size": actual_size,
            "filename": meta["filename"],
            "file_type": meta["file_type"],
            "group": meta.get("group")
        }

    async def cleanup_upload(self, upload_id: str):
        """
        Clean up temporary files for an upload.

        Args:
            upload_id: Upload identifier
        """
        meta_path = self._get_upload_meta_path(upload_id)

        if meta_path.exists():
            try:
                async with aiofiles.open(meta_path, 'r') as f:
                    meta = json.loads(await f.read())

                # Delete all chunk files
                for i in meta.get("received_chunks", []):
                    chunk_path = self._get_chunk_path(upload_id, i)
                    if chunk_path.exists():
                        chunk_path.unlink()

                # Delete metadata file
                meta_path.unlink()
                # Clean up per-upload lock
                self._upload_locks.pop(upload_id, None)
                logger.info(f"Cleaned up upload {upload_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up upload {upload_id}: {e}")

    async def get_upload_status(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """
        Get upload progress and status.

        Args:
            upload_id: Upload identifier

        Returns:
            Dict with status info or None if not found
        """
        meta_path = self._get_upload_meta_path(upload_id)

        if not meta_path.exists():
            return None

        # Read metadata under lock to avoid race with concurrent save_chunk
        async with await self._get_upload_lock(upload_id):
            async with aiofiles.open(meta_path, 'r') as f:
                meta = json.loads(await f.read())

        received_count = len(meta["received_chunks"])
        total_chunks = meta["total_chunks"]

        return {
            "upload_id": upload_id,
            "filename": meta["filename"],
            "file_type": meta["file_type"],
            "group": meta.get("group"),
            "total_size": meta["total_size"],
            "total_chunks": total_chunks,
            "received_chunks": received_count,
            "progress": received_count / total_chunks * 100 if total_chunks > 0 else 0,
            "missing_chunks": sorted(
                set(range(total_chunks)) - set(meta["received_chunks"])
            ),
            "created_at": meta["created_at"],
            "updated_at": meta["updated_at"],
            "completed": meta.get("completed", False)
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
                async with aiofiles.open(meta_path, 'r') as f:
                    meta = json.loads(await f.read())

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

    async def cancel_upload(self, upload_id: str) -> bool:
        """
        Cancel and cleanup an upload.

        Args:
            upload_id: Upload identifier

        Returns:
            True if cancelled successfully
        """
        meta_path = self._get_upload_meta_path(upload_id)

        if not meta_path.exists():
            return False

        await self.cleanup_upload(upload_id)
        return True


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

    async with aiofiles.open(dest_path, 'wb') as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            await f.write(chunk)

    return dest_path
