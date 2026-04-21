"""Session persistence: memory cache + atomic disk writes.

Extracted from backend/main.py. Public API:
  - save_session_to_disk(session_id, state)   # sync write, event-loop thread
  - async_save_session(session_id, state, force=False)  # throttled async write
  - load_session_from_disk(session_id) -> AnalysisState | None
  - ensure_session_loaded(session_id) -> AnalysisState | None
  - list_saved_sessions() -> list[dict]
  - update_session_cache(session_id, state)
  - touch_session(session_id)
  - pause_and_restart_ws(session_id, state, websockets)
"""
import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import (
    SESSION_DIR,
    get_session_metadata_path,
    resolve_session_metadata_path,
)
from backend.models.schemas import AnalysisState, TaskStatus
from backend.utils.files import validate_session_id
from backend.state import (
    sessions, state_lock, session_access_times, last_save_times,
)

logger = logging.getLogger(__name__)

_MIN_SAVE_INTERVAL = 5.0
_FORCE_SAVE_STATUSES = frozenset({
    TaskStatus.COMPLETED, TaskStatus.FAILED,
    TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION,
    TaskStatus.GENERATING_REPORT,
})


def _write_session_json(session_id: str, json_str: str) -> None:
    try:
        session_file = get_session_metadata_path(session_id)
        session_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp", prefix=f"{session_id}_", dir=session_file.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_str)
            os.replace(tmp_path, session_file)
            logger.info(f"Session {session_id} saved to disk")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except Exception:
        logger.exception(f"Failed to save session {session_id}")


def save_session_to_disk(session_id: str, state: AnalysisState) -> None:
    _write_session_json(session_id, state.model_dump_json())


async def async_save_session(session_id: str, state: AnalysisState, force: bool = False) -> None:
    if not force and hasattr(state, "status") and state.status in _FORCE_SAVE_STATUSES:
        force = True
    now = time.monotonic()
    if not force:
        last = last_save_times.get(session_id, 0)
        if now - last < _MIN_SAVE_INTERVAL:
            logger.debug("Throttled save for %s", session_id)
            return
    last_save_times[session_id] = now
    json_str = state.model_dump_json()
    await asyncio.to_thread(_write_session_json, session_id, json_str)


def load_session_from_disk(session_id: str) -> Optional[AnalysisState]:
    try:
        session_file = resolve_session_metadata_path(session_id)
        if session_file and session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("user_id"):
                logger.warning(f"Session {session_id} has empty user_id, refusing to load")
                return None
            state = AnalysisState(**data)
            logger.info(f"Session {session_id} loaded from disk")
            return state
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        logger.exception(f"Failed to load session {session_id}")
    return None


async def update_session_cache(session_id: str, state: AnalysisState) -> None:
    async with state_lock:
        sessions[session_id] = state
        session_access_times[session_id] = time.time()


async def touch_session(session_id: str) -> None:
    async with state_lock:
        session_access_times[session_id] = time.time()


async def ensure_session_loaded(session_id: str) -> Optional[AnalysisState]:
    validate_session_id(session_id)
    if session_id in sessions:
        async with state_lock:
            session_access_times[session_id] = time.time()
        return sessions.get(session_id)
    async with state_lock:
        if session_id in sessions:
            session_access_times[session_id] = time.time()
            return sessions[session_id]
        state = await asyncio.to_thread(load_session_from_disk, session_id)
        if state:
            sessions[session_id] = state
            session_access_times[session_id] = time.time()
            return state
    return None


def list_saved_sessions() -> List[Dict[str, Any]]:
    sessions_list: List[Dict[str, Any]] = []
    seen_ids: set = set()

    def _parse(session_file: Path, fallback_id: str = "") -> None:
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            sid = data.get("id", fallback_id)
            if sid in seen_ids:
                return
            seen_ids.add(sid)
            data_files = data.get("data_files", [])
            has_path = has_seq = has_bio = False
            for f in data_files:
                s = f.get("section")
                if s == "pathology": has_path = True
                elif s == "sequencing": has_seq = True
                elif s == "biochemistry": has_bio = True
            if data.get("biochemistry_indicators"):
                has_bio = True
            sessions_list.append({
                "id": sid,
                "user_id": data.get("user_id", ""),
                "name": data.get("name", "Unknown"),
                "status": data.get("status", "unknown"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "current_round": data.get("current_round", 0),
                "has_pathology": has_path,
                "has_sequencing": has_seq,
                "has_biochemistry": has_bio,
            })
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to read session file {session_file}: {e}")

    try:
        for entry in SESSION_DIR.iterdir():
            if entry.is_dir():
                sf = entry / "session.json"
                if sf.exists():
                    _parse(sf, fallback_id=entry.name)
        for session_file in SESSION_DIR.glob("*.json"):
            if session_file.is_file():
                _parse(session_file, fallback_id=session_file.stem)
    except Exception:
        logger.exception("Failed to list sessions")

    sessions_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions_list


async def pause_and_restart_ws(session_id: str, state: AnalysisState, websockets: Dict[str, Any]) -> None:
    state.status = TaskStatus.PAUSED
    await update_session_cache(session_id, state)
    await async_save_session(session_id, state)
    ws = websockets.get(session_id)
    if ws:
        try:
            await ws.close(4000, "workflow_restart")
        except Exception:
            logger.debug("WebSocket close failed during pause-and-restart", exc_info=True)
