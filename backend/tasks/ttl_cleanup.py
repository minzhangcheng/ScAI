"""Background task to clean up expired upload sessions and their temp files."""
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.utils.auth import get_db
from backend.config import UPLOAD_TEMP_DIR

logger = logging.getLogger(__name__)

TTL_CLEANUP_INTERVAL = 3600  # Run every hour


def _unlink_chunks(upload_id: str, total: int, base: Path) -> int:
    cleaned = 0
    for i in range(total):
        chunk_path = base / f"{upload_id}.chunk.{i}"
        try:
            chunk_path.unlink(missing_ok=True)
            cleaned += 1
        except OSError:
            pass
    return cleaned


async def ttl_cleanup_task():
    """Periodically scan for expired upload sessions and clean them up."""
    while True:
        try:
            db = await get_db()
            now_iso = datetime.now(timezone.utc).isoformat()
            expired = await db.list_expired_uploads(now_iso)

            for upload in expired:
                upload_id = upload["id"]
                total_chunks = upload["total_chunks"]

                # Clean up chunk files (offloaded to thread)
                cleaned = await asyncio.to_thread(
                    _unlink_chunks, upload_id, total_chunks, UPLOAD_TEMP_DIR
                )

                # Remove DB record
                await db.delete_upload_session(upload_id)
                logger.info(
                    f"TTL cleanup: removed expired upload {upload_id}, "
                    f"cleaned {cleaned} chunk files"
                )

        except Exception as e:
            logger.error(f"TTL cleanup error: {e}")

        await asyncio.sleep(TTL_CLEANUP_INTERVAL)
