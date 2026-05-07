"""User data cleanup service: cascade delete sessions and files for a user."""
import shutil
import logging
from pathlib import Path

from backend.config import STATE_DIR, SESSION_DIR
from backend.services.session_store import list_saved_sessions
from backend.utils.auth import get_db
from backend.state import sessions, state_lock

logger = logging.getLogger(__name__)


async def delete_user_data(user_id: str) -> dict:
    """
    Delete all data associated with a user:
    - Remove session directories under SESSION_DIR matching user_id
    - Delete {sid}* state files under STATE_DIR
    - Remove from in-memory sessions dict
    - Delete DB records (api_keys via delete_api_keys_by_user, then the user)

    Returns a dict with counts of deleted items.
    """
    deleted_sessions = 0
    deleted_state_files = 0

    # 1. Remove from in-memory sessions cache
    async with state_lock:
        to_evict = [sid for sid, state in sessions.items() if getattr(state, "user_id", None) == user_id]
        for sid in to_evict:
            sessions.pop(sid, None)

    # 2. Iterate saved sessions and delete those belonging to this user
    saved = list_saved_sessions()
    for sess in saved:
        if sess.get("user_id") != user_id:
            continue
        sid = sess.get("id", "")
        # Session directory
        session_dir = SESSION_DIR / sid
        if session_dir.exists():
            shutil.rmtree(session_dir)
            deleted_sessions += 1
            logger.info(f"Deleted session directory: {session_dir}")
        # Legacy state files: {sid}*.pkl, {sid}*.json under STATE_DIR
        if STATE_DIR.exists():
            for pattern in [f"{sid}.pkl", f"{sid}.json"]:
                for f in STATE_DIR.glob(pattern):
                    if f.exists():
                        f.unlink()
                        deleted_state_files += 1
                        logger.info(f"Deleted state file: {f}")

    # 3. Delete from database
    db = await get_db()
    await db.delete_api_keys_by_user(user_id)
    await db.delete_user_llm_config(user_id)
    await db.delete_user(user_id)

    logger.info(f"User {user_id} deleted: {deleted_sessions} sessions, {deleted_state_files} state files")
    return {
        "user_id": user_id,
        "deleted_sessions": deleted_sessions,
        "deleted_state_files": deleted_state_files,
    }
