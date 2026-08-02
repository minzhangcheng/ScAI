"""User data cleanup service: cascade delete sessions and files for a user."""
import asyncio
import logging

from backend.services.session_store import list_saved_sessions, load_session_from_disk
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

    # Build the complete owner set before deleting anything. Memory-only
    # pending sessions matter just as much as persisted ones.
    async with state_lock:
        owned_states = {
            sid: state
            for sid, state in sessions.items()
            if getattr(state, "user_id", None) == user_id
        }

    saved = await asyncio.to_thread(list_saved_sessions)
    for sess in saved:
        if sess.get("user_id") != user_id:
            continue
        sid = sess.get("id", "")
        if sid and sid not in owned_states:
            loaded = await asyncio.to_thread(load_session_from_disk, sid)
            if loaded is not None and loaded.user_id == user_id:
                owned_states[sid] = loaded

    # Every cascade uses the same tombstone-first generation boundary as both
    # public delete APIs. ``force_stop`` cancels active workflow/report/chat
    # producers only after the tombstone makes any late save fail closed.
    from backend.services.session_deletion import delete_session_generation

    for sid, state in owned_states.items():
        stats = await delete_session_generation(
            sid,
            state,
            user_id,
            force_stop=True,
        )
        deleted_sessions += max(1, int(stats.get("deleted_sessions", 0)))
        deleted_state_files += int(stats.get("deleted_state_files", 0))

    # Delete credentials/user only after upload cleanup has used the DB.
    db = await get_db()
    await db.delete_api_keys_by_user(user_id)
    await db.delete_user_llm_config(user_id)
    await db.delete_user(user_id)

    logger.info(
        f"User {user_id} deleted: {deleted_sessions} sessions, "
        f"{deleted_state_files} state files"
    )
    return {
        "user_id": user_id,
        "deleted_sessions": deleted_sessions,
        "deleted_state_files": deleted_state_files,
    }
