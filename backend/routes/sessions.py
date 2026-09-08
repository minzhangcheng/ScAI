"""Session and group management routes."""
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from cryptography.fernet import InvalidToken

from backend.config import (
    SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR, REPORTS_DIR,
    SystemConfig, get_session_dir, resolve_agent_context_window_default,
)
from backend.models.api_requests import (
    BiochemistryInput, GroupCreateRequest, GroupUpdateRequest,
    SessionUpdateRequest, StartRequest,
)
from backend.models.schemas import APIResponse, TaskStatus
from backend.services.session_store import async_save_session, list_saved_sessions
from backend.state import (
    sessions, workflows, state_lock, initializing_sessions,
    session_access_times, last_save_times,
)
from backend.utils.auth import require_session_ownership, require_session_read_ownership
from backend.utils.auth import get_db
from backend.utils import crypto
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import validate_path_within, validate_session_id
from backend.utils.url_validation import validated_http_url
from backend.utils.websocket_delivery import serialize_realtime_message

router = APIRouter(prefix="/api", tags=["sessions"])

logger = logging.getLogger(__name__)

_AGENT_NAMES = (
    "pi", "pathology", "bioinformatics", "biochemistry",
    "literature", "painter", "supervisor", "technique",
)


def _new_session_agent_settings(
    state,
    settings,
    secret_defaults,
    config: SystemConfig | None = None,
) -> None:
    """Copy one browser default snapshot into a new authoritative session."""
    if settings is not None:
        state.custom_prompts = {
            agent: settings.prompts[agent]
            for agent in _AGENT_NAMES
            if agent in settings.prompts
        }
        state.custom_models = {
            agent: settings.models[agent]
            for agent in _AGENT_NAMES
            if agent in settings.models
        }
        state.custom_temperatures = {
            agent: settings.temperatures[agent]
            for agent in _AGENT_NAMES
            if agent in settings.temperatures
        }
        state.agent_vision_settings = {
            agent: settings.vision_settings[agent]
            for agent in _AGENT_NAMES
            if agent in settings.vision_settings
        }

    saved_keys = secret_defaults.get("agent_api_keys") or {}
    for agent in _AGENT_NAMES:
        incoming = settings.agent_server_configs.get(agent) if settings is not None else None
        api_key = saved_keys.get(agent, "")
        if incoming is None and not api_key and config is None:
            continue
        raw_url = incoming.llm_base_url if incoming is not None else ""
        try:
            llm_base_url = validated_http_url(
                raw_url, f"agent_server_configs.{agent}.llm_base_url", allow_empty=True
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        server = {"llm_base_url": llm_base_url, "api_key": api_key}
        if config is not None:
            server["default_context_window"] = resolve_agent_context_window_default(
                config, agent
            )
        if incoming is not None and incoming.context_window is not None:
            server["context_window"] = incoming.context_window
        state.agent_server_configs[agent] = server

    literature = dict(state.literature_config or {})
    if settings is not None and settings.literature_config is not None:
        incoming_literature = settings.literature_config.model_dump(exclude_unset=True)
        if "deep_research_llm_base_url" in incoming_literature:
            try:
                incoming_literature["deep_research_llm_base_url"] = validated_http_url(
                    incoming_literature["deep_research_llm_base_url"] or "",
                    "literature_config.deep_research_llm_base_url",
                    allow_empty=True,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        literature.update(incoming_literature)
    saved_literature = secret_defaults.get("literature") or {}
    literature["deep_research_api_key"] = saved_literature.get("deep_research_api_key", "")
    literature["google_api_key"] = saved_literature.get("google_api_key", "")
    state.literature_config = literature


def _serialize_realtime_messages(messages: list, limit: int = 500) -> list:
    """Serialize WebSocketMessage list for WebSocket transmission."""
    return [
        {
            "id": msg.id,
            "type": msg.type,
            "agent": msg.agent.value if hasattr(msg.agent, 'value') else (str(msg.agent) if msg.agent else None),
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else datetime.now().isoformat(),
        }
        for msg in messages[-limit:]
    ]


def _strip_message_base64(msg_type: str, content) -> Any:
    """Strip large base64 image data from message content to reduce response size."""
    if msg_type == "figure" and isinstance(content, str) and len(content) > 1000:
        return "[image]"
    if msg_type == "graphic_abstract" and isinstance(content, dict):
        return {k: v for k, v in content.items() if k not in ("image_base64", "imageBase64")}
    if msg_type in ("figure_review", "figure_approved", "figure_revision") and isinstance(content, dict):
        return {k: v for k, v in content.items() if k not in ("png_base64", "pngBase64")}
    if isinstance(content, dict) and "figures" in content and isinstance(content.get("figures"), list):
        stripped_figures = []
        for f in content["figures"]:
            if isinstance(f, dict):
                stripped_figures.append({k: v for k, v in f.items() if k not in ("data", "pngBase64", "png_base64")})
            else:
                stripped_figures.append(f)
        return {**content, "figures": stripped_figures}
    return content


def _strip_legacy_blob_references(value: Any) -> Any:
    """Remove legacy blob handles from durable history after JSON conversion."""
    if isinstance(value, str):
        return "[image]" if value.startswith("@scai-blob:") else value
    if isinstance(value, list):
        return [_strip_legacy_blob_references(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_legacy_blob_references(item) for key, item in value.items()}
    return value


def _serialize_realtime_history_message(message: Any) -> dict:
    return _strip_legacy_blob_references(serialize_realtime_message(message))


@router.post("/session/start")
async def start_session(request: Request, body: StartRequest):
    """Start a new analysis session"""
    user_id = getattr(request.state, "user_id", "")

    import uuid
    session_id = str(uuid.uuid4())
    from backend.models.schemas import AnalysisState
    from backend.services.report.artifacts import ARTIFACT_REGISTRY_VERSION
    state = AnalysisState(
        id=session_id,
        name=body.session_name or f"Analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        user_id=user_id,
        initial_instruction=body.instruction,
        status=TaskStatus.PENDING,
        round_wait_seconds=body.round_wait_seconds,
        artifact_registry_version=ARTIFACT_REGISTRY_VERSION,
    )
    from backend.services.agent_default_secrets import load_agent_default_secrets

    try:
        secret_defaults = await load_agent_default_secrets(await get_db(), user_id)
    except (crypto.ConfigEncryptionUnavailable, InvalidToken) as exc:
        raise HTTPException(
            503, "Encryption service unavailable; cannot load Agent defaults"
        ) from exc
    config = await resolve_config_for_user(user_id) or SystemConfig()
    _new_session_agent_settings(state, body.agent_settings, secret_defaults, config)

    from backend.services.session_lifecycle import register_new_session_generation
    from backend.services.session_deletion import delete_session_generation
    from backend.services.session_store import update_session_cache as _update_session_cache
    await register_new_session_generation(state)
    try:
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    except BaseException:
        # A generated identifier is never recycled after partial creation.
        # Tombstoning also removes any cache/disk fragments already produced.
        try:
            await asyncio.shield(
                delete_session_generation(session_id, state, user_id)
            )
        except Exception:
            logger.exception(
                "Failed to clean up partially-created session %s", session_id
            )
        raise

    return APIResponse(
        success=True,
        message="Session created",
        session_id=session_id,
        data={"name": state.name}
    )


@router.get("/sessions")
async def list_sessions(request: Request):
    """List all available sessions (from disk and memory), filtered by user ownership"""
    user_id = getattr(request.state, "user_id", None)
    disk_sessions = await asyncio.to_thread(list_saved_sessions)

    if user_id:
        disk_sessions = [s for s in disk_sessions if s.get("user_id") == user_id]

    # P3-8: dedup-by-id when merging disk + memory sessions.
    # Disk entries take precedence (they reflect persisted state); memory-only
    # sessions fill in any not-yet-flushed sessions.
    sessions_by_id: dict[str, dict] = {s["id"]: s for s in disk_sessions}

    for session_id, state in list(sessions.items()):
        if session_id in sessions_by_id:
            continue
        if state is None or (user_id and state.user_id != user_id):
            continue
        sessions_by_id[session_id] = {
            "id": session_id,
            "name": state.name,
            "status": state.status.value if hasattr(state.status, 'value') else str(state.status),
            "created_at": state.created_at.isoformat() if state.created_at else "",
            "updated_at": state.updated_at.isoformat() if state.updated_at else "",
            "current_round": state.current_round,
            "waiting_input_started_at": state.waiting_input_started_at.isoformat() if state.waiting_input_started_at else "",
            "round_wait_seconds": state.round_wait_seconds,
            "has_pathology": any(f.section == "pathology" for f in state.data_files),
            "has_sequencing": state.has_sequencing_data(),
            "has_biochemistry": any(f.section == "biochemistry" for f in state.data_files) or len(state.biochemistry_indicators) > 0,
        }

    merged = list(sessions_by_id.values())
    merged.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"sessions": merged}


@router.get("/session/{session_id}/load")
async def load_session(session_id: str, request: Request):
    """Load a session by ID (from memory or disk)"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_read_ownership(session_id=session_id, user_id=user_id)
    return state.to_api_dict(
        strip_message_base64=_strip_message_base64,
        include_messages=False,
    )


@router.get("/session/{session_id}/messages")
async def load_realtime_messages(
    session_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    before: Optional[str] = Query(default=None, min_length=1, max_length=128),
):
    """Load realtime history in chronological pages using stable cursors."""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_read_ownership(session_id=session_id, user_id=user_id)
    snapshot = list(state.realtime_messages)
    end = len(snapshot)
    if before is not None:
        cursor_index = next(
            (index for index, message in enumerate(snapshot) if message.id == before),
            None,
        )
        if cursor_index is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "MESSAGE_CURSOR_EXPIRED",
                    "message": "Realtime message cursor expired; restart from the newest page.",
                },
            )
        end = cursor_index
    start = max(0, end - limit)
    selected = snapshot[start:end]
    has_more = start > 0
    return {
        "messages": [_serialize_realtime_history_message(message) for message in selected],
        "count": len(selected),
        "total": len(snapshot),
        "page_size": limit,
        "has_more": has_more,
        "next_before": selected[0].id if has_more and selected else None,
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a session completely"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    from backend.services.session_lifecycle import tombstone_visible_to_owner
    from backend.services.session_deletion import (
        SessionProducerStopError,
        cleanup_tombstoned_generation,
        delete_session_generation,
    )

    if tombstone_visible_to_owner(session_id, user_id):
        stats = await cleanup_tombstoned_generation(session_id, user_id)
        pending = int(stats.get("pending_producers") or 0)
        return APIResponse(
            success=pending == 0,
            message=(
                "Session deletion cleanup is still waiting for producers"
                if pending
                else "Session deletion cleanup completed"
            ),
            data=stats,
        )
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    try:
        await delete_session_generation(session_id, state, user_id)
    except SessionProducerStopError:
        return APIResponse(
            success=False,
            message=(
                "Session is deleted and remains inaccessible; physical cleanup "
                "will retry after background producers stop"
            ),
            data={"pending_cleanup": True},
        )
    return APIResponse(success=True, message="Session deleted")


@router.post("/session/{session_id}/ensure")
async def ensure_session(session_id: str, request: Request, body: StartRequest):
    """Ensure a persisted session exists; never recreate a client-supplied ID."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    existing = await require_session_ownership(
        session_id=session_id,
        user_id=user_id,
    )
    return APIResponse(
        success=True,
        message="Session exists",
        session_id=session_id,
        data={"name": existing.name, "created": False},
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str, request: Request):
    """Get session status"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    return {
        "session_id": session_id,
        "name": state.name,
        "status": state.status.value,
        "current_round": state.current_round,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "has_pathology": any(f.section == "pathology" for f in state.data_files),
        "has_sequencing": state.has_sequencing_data(),
        "has_biochemistry": any(f.section == "biochemistry" for f in state.data_files) or len(state.biochemistry_indicators) > 0,
        "treatment_groups": len(state.treatment_groups),
        "data_files_count": len(state.data_files),
        "round_count": len(state.round_results)
    }


@router.post("/session/{session_id}/groups")
async def create_group(session_id: str, request: Request, body: GroupCreateRequest):
    """Create a new treatment group"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    from backend.models.schemas import TreatmentGroup
    group = TreatmentGroup(
        name=body.name,
        description=body.description,
        sample_count=body.sample_count,
        order=len(state.treatment_groups)
    )
    state.treatment_groups.append(group)
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=f"Treatment group created: {body.name}",
        data={"id": group.id, "name": group.name, "description": group.description,
              "sampleCount": group.sample_count, "order": group.order}
    )


@router.put("/session/{session_id}/info")
async def update_session_info(session_id: str, request: Request, body: SessionUpdateRequest):
    """Update session name and/or instruction"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if body.name is not None:
        state.name = body.name
    if body.instruction is not None:
        state.initial_instruction = body.instruction

    await async_save_session(session_id, state)
    return APIResponse(
        success=True,
        message="Session info updated",
        data={"name": state.name, "instruction": state.initial_instruction}
    )


@router.put("/session/{session_id}/groups/{group_id}")
async def update_group(session_id: str, group_id: str, request: Request, body: GroupUpdateRequest):
    """Update a treatment group"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    group = state.get_group_by_id(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")

    if body.name is not None:
        group.name = body.name
    if body.description is not None:
        group.description = body.description
    if body.sample_count is not None:
        if body.sample_count < group.sample_count:
            sample_numbers_in_use = (
                [f.sample_number for f in state.data_files
                 if f.group_id == group_id and f.sample_number is not None] +
                [i.sample_number for i in state.biochemistry_indicators
                 if i.group_id == group_id and i.sample_number is not None]
            )
            if sample_numbers_in_use:
                max_sample_in_use = max(sample_numbers_in_use)
                if body.sample_count < max_sample_in_use:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot reduce sample count below {max_sample_in_use}: data exists for sample {max_sample_in_use}"
                    )
        group.sample_count = body.sample_count

    await async_save_session(session_id, state)
    return APIResponse(
        success=True,
        message=f"Group updated: {group.name}",
        data={"id": group.id, "name": group.name, "description": group.description,
              "sampleCount": group.sample_count, "order": group.order}
    )


def _safe_remove_session_file(file_path: str, session_dir: Path) -> None:
    try:
        candidate = Path(file_path)
        validate_path_within(candidate, session_dir)
        if candidate.exists():
            candidate.unlink()
    except Exception as e:
        logger.warning(f"Skipped removing out-of-session or missing file {file_path}: {e}")


@router.delete("/session/{session_id}/groups/{group_id}")
async def delete_group(session_id: str, group_id: str, request: Request):
    """Delete a treatment group and all associated data files"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    group = state.get_group_by_id(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")

    files_to_remove = [f for f in state.data_files if f.group_id == group_id]
    session_dir = get_session_dir(session_id)
    for df in files_to_remove:
        _safe_remove_session_file(df.file_path, session_dir)

    state.data_files = [f for f in state.data_files if f.group_id != group_id]
    state.biochemistry_indicators = [i for i in state.biochemistry_indicators if i.group_id != group_id]
    state.treatment_groups = [g for g in state.treatment_groups if g.id != group_id]

    await async_save_session(session_id, state)
    return APIResponse(
        success=True,
        message=f"Group deleted: {group.name} (and {len(files_to_remove)} associated files)"
    )


@router.post("/session/{session_id}/save")
async def save_session(session_id: str, request: Request):
    """Manually save session to disk"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    # Forced: an explicit "save now" that the throttle can swallow is not a
    # save. The endpoint reports success either way, so the user is told the
    # session is on disk when it may not be.
    await async_save_session(session_id, state, force=True)

    return APIResponse(success=True, message="Session saved")
