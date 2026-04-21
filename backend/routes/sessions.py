"""Session and group management routes."""
import asyncio
import os
import shutil
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.config import (
    SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR, REPORTS_DIR,
    get_session_dir,
)
from backend.models.api_requests import (
    BiochemistryInput, GroupCreateRequest, GroupUpdateRequest,
    SessionUpdateRequest, StartRequest,
)
from backend.models.schemas import APIResponse, TaskStatus
from backend.services.session_store import (
    async_save_session, ensure_session_loaded, list_saved_sessions,
)
from backend.state import (
    sessions, workflows, state_lock, initializing_sessions,
    session_access_times, last_save_times,
    system_config,
)
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id

router = APIRouter(prefix="/api", tags=["sessions"])


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


@router.post("/session/start")
async def start_session(request: Request, body: StartRequest):
    """Start a new analysis session"""
    cfg = system_config["current"]
    if cfg is None:
        raise HTTPException(status_code=400, detail="API configuration not set")

    user_id = getattr(request.state, "user_id", "")

    import uuid
    session_id = str(uuid.uuid4())
    from backend.models.schemas import AnalysisState
    state = AnalysisState(
        id=session_id,
        name=body.session_name or f"Analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        user_id=user_id,
        initial_instruction=body.instruction,
        status=TaskStatus.PENDING
    )

    from backend.services.session_store import update_session_cache as _update_session_cache
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

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

    memory_session_ids = set(list(sessions.keys()))
    disk_session_ids = {s["id"] for s in disk_sessions}

    for session_id in memory_session_ids - disk_session_ids:
        state = sessions.get(session_id)
        if state is None or (user_id and state.user_id != user_id):
            continue
        disk_sessions.append({
            "id": session_id,
            "name": state.name,
            "status": state.status.value if hasattr(state.status, 'value') else str(state.status),
            "created_at": state.created_at.isoformat() if state.created_at else "",
            "updated_at": state.updated_at.isoformat() if state.updated_at else "",
            "current_round": state.current_round,
            "has_pathology": any(f.section == "pathology" for f in state.data_files),
            "has_sequencing": state.has_sequencing_data(),
            "has_biochemistry": any(f.section == "biochemistry" for f in state.data_files) or len(state.biochemistry_indicators) > 0,
        })

    disk_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"sessions": disk_sessions}


@router.get("/session/{session_id}/load")
async def load_session(session_id: str, request: Request):
    """Load a session by ID (from memory or disk)"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return state.to_api_dict(strip_message_base64=_strip_message_base64)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a session completely"""
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    async with state_lock:
        if session_id in workflows or session_id in initializing_sessions:
            raise HTTPException(status_code=409, detail="Cannot delete a running session. Stop the analysis first.")
        sessions.pop(session_id, None)
        session_access_times.pop(session_id, None)
        last_save_times.pop(session_id, None)

    def _delete_from_disk():
        session_dir = get_session_dir(session_id)
        if session_dir.exists() and session_dir.is_dir():
            shutil.rmtree(session_dir)
        legacy_file = SESSION_DIR / f"{session_id}.json"
        if legacy_file.is_file():
            legacy_file.unlink()
        for legacy_dir in [UPLOAD_DIR / session_id, WORKSPACE_DIR / session_id, REPORTS_DIR / session_id]:
            if legacy_dir.exists():
                shutil.rmtree(legacy_dir)
        for state_file in STATE_DIR.glob(f"{session_id}*"):
            state_file.unlink()

    await asyncio.to_thread(_delete_from_disk)
    return APIResponse(success=True, message="Session deleted")


@router.post("/session/{session_id}/ensure")
async def ensure_session(session_id: str, request: StartRequest):
    """Ensure session exists, create if not found"""
    cfg = system_config["current"]
    validate_session_id(session_id)

    if cfg is None:
        raise HTTPException(status_code=400, detail="API configuration not set")

    state = sessions.get(session_id)
    if state is not None:
        return APIResponse(
            success=True, message="Session exists",
            session_id=session_id, data={"name": state.name, "created": False}
        )

    existing = await ensure_session_loaded(session_id)
    if existing:
        return APIResponse(
            success=True, message="Session loaded from disk",
            session_id=session_id, data={"name": existing.name, "created": False}
        )

    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    from backend.models.schemas import AnalysisState
    state = AnalysisState(
        id=session_id,
        name=request.session_name or f"Analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        initial_instruction=request.instruction,
        status=TaskStatus.PENDING,
        user_id=user_id,
    )

    from backend.services.session_store import update_session_cache as _update_session_cache
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Recreated session {session_id} after server restart")

    return APIResponse(
        success=True, message="Session recreated",
        session_id=session_id, data={"name": state.name, "created": True}
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
async def create_group(session_id: str, request: GroupCreateRequest):
    """Create a new treatment group"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from backend.models.schemas import TreatmentGroup
    group = TreatmentGroup(
        name=request.name,
        description=request.description,
        sample_count=request.sample_count,
        order=len(state.treatment_groups)
    )
    state.treatment_groups.append(group)
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=f"Treatment group created: {request.name}",
        data={"id": group.id, "name": group.name, "description": group.description,
              "sampleCount": group.sample_count, "order": group.order}
    )


@router.put("/session/{session_id}/info")
async def update_session_info(session_id: str, request: SessionUpdateRequest):
    """Update session name and/or instruction"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.name is not None:
        state.name = request.name
    if request.instruction is not None:
        state.initial_instruction = request.instruction

    await async_save_session(session_id, state)
    return APIResponse(
        success=True,
        message="Session info updated",
        data={"name": state.name, "instruction": state.initial_instruction}
    )


@router.put("/session/{session_id}/groups/{group_id}")
async def update_group(session_id: str, group_id: str, request: GroupUpdateRequest):
    """Update a treatment group"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    group = state.get_group_by_id(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")

    if request.name is not None:
        group.name = request.name
    if request.description is not None:
        group.description = request.description
    if request.sample_count is not None:
        if request.sample_count < group.sample_count:
            sample_numbers_in_use = (
                [f.sample_number for f in state.data_files
                 if f.group_id == group_id and f.sample_number is not None] +
                [i.sample_number for i in state.biochemistry_indicators
                 if i.group_id == group_id and i.sample_number is not None]
            )
            if sample_numbers_in_use:
                max_sample_in_use = max(sample_numbers_in_use)
                if request.sample_count < max_sample_in_use:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot reduce sample count below {max_sample_in_use}: data exists for sample {max_sample_in_use}"
                    )
        group.sample_count = request.sample_count

    await async_save_session(session_id, state)
    return APIResponse(
        success=True,
        message=f"Group updated: {group.name}",
        data={"id": group.id, "name": group.name, "description": group.description,
              "sampleCount": group.sample_count, "order": group.order}
    )


@router.delete("/session/{session_id}/groups/{group_id}")
async def delete_group(session_id: str, group_id: str):
    """Delete a treatment group and all associated data files"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    group = state.get_group_by_id(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")

    files_to_remove = [f for f in state.data_files if f.group_id == group_id]
    for df in files_to_remove:
        try:
            if os.path.exists(df.file_path):
                os.remove(df.file_path)
        except OSError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to delete file {df.file_path}: {e}")

    state.data_files = [f for f in state.data_files if f.group_id != group_id]
    state.biochemistry_indicators = [i for i in state.biochemistry_indicators if i.group_id != group_id]
    state.treatment_groups = [g for g in state.treatment_groups if g.id != group_id]

    await async_save_session(session_id, state)
    return APIResponse(
        success=True,
        message=f"Group deleted: {group.name} (and {len(files_to_remove)} associated files)"
    )


@router.post("/session/{session_id}/save")
async def save_session(session_id: str):
    """Manually save session to disk"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await async_save_session(session_id, state)

    return APIResponse(success=True, message="Session saved")
