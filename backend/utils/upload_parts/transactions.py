"""Transaction phases for the legacy chunked upload manager."""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from backend.utils.asyncio_tasks import run_in_owned_thread

logger = logging.getLogger("backend.utils.upload")


def _assemble_chunks_sync(
    self,
    upload_id: str,
    total_chunks: int,
    final_path: Path,
) -> tuple[int, str]:
    """Assemble one upload in a worker that its caller always joins."""
    meta = json.loads(
        self._get_upload_meta_path(upload_id).read_text(encoding="utf-8")
    )
    total_size = int(meta.get("total_size") or 0)
    digest = hashlib.sha256()
    with final_path.open("wb") as out_f:
        for index in range(total_chunks):
            chunk_path = self._get_chunk_path(upload_id, index)
            expected_size = (
                self.chunk_size
                if index < total_chunks - 1
                else total_size - self.chunk_size * (total_chunks - 1)
            )
            chunk_bytes = 0
            with chunk_path.open("rb") as chunk_f:
                while True:
                    data = chunk_f.read(1024 * 1024)
                    if not data:
                        break
                    chunk_bytes += len(data)
                    digest.update(data)
                    out_f.write(data)
            if chunk_bytes != expected_size:
                raise ValueError(
                    f"Chunk size mismatch at index {index}: expected "
                    f"{expected_size}, got {chunk_bytes}"
                )
    return final_path.stat().st_size, digest.hexdigest()

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

        expected_sha256 = str(
            meta.get("expected_sha256") or ""
        ).strip().lower()
        if expected_sha256 and staged_sha256 != expected_sha256:
            logger.error(
                "SHA-256 mismatch for upload %s: expected %s, got %s",
                upload_id,
                expected_sha256,
                staged_sha256,
            )
            staging_path.unlink(missing_ok=True)
            await self._cleanup_loaded_upload_locked(upload_id, meta)
            return {
                "success": False,
                "error": "Whole-file SHA-256 mismatch",
                "integrity_mismatch": True,
            }

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
