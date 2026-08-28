"""Maintenance operations for the legacy chunked upload manager."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.utils.asyncio_tasks import run_in_owned_thread

logger = logging.getLogger("backend.utils.upload")


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
