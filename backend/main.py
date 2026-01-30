"""
FastAPI backend server for the Multi-Agent Liver Analysis System
"""
import os
import json
import asyncio
import logging
import traceback
import zipfile
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import (
    SystemConfig,
    DATA_BASE_DIR, UPLOAD_DIR, STATE_DIR, WORKSPACE_DIR, SESSION_DIR, REPORTS_DIR, LOG_DIR,
    setup_logging
)
from .models.schemas import (
    AnalysisState, APIRequest, APIResponse, TaskStatus,
    WebSocketMessage, AgentType, PathologyImage, BiochemistryResult, BiochemistryFile
)
from .workflow import LiverAnalysisWorkflow
from .utils.llm_client import OpenAICompatibleClient

# Configure logging - must be called before any logging
setup_logging()
logger = logging.getLogger(__name__)


# FastAPI app
app = FastAPI(
    title="InteSc - Integrated and Intelligent Single Cell Sequencing Analysis",
    description="Integrated & Intelligent Single Cell Analysis System with pathology, bioinformatics, and biochemistry analysis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state with lock for thread-safe access
sessions: Dict[str, AnalysisState] = {}
workflows: Dict[str, LiverAnalysisWorkflow] = {}
websockets: Dict[str, WebSocket] = {}
config: Optional[SystemConfig] = None

# Lock for protecting concurrent access to global dicts
_state_lock = asyncio.Lock()

# Log the data directory location (directories are created in config.py)
logger.info(f"Data directory: {DATA_BASE_DIR.resolve()}")


# Session persistence helpers
def save_session_to_disk(session_id: str, state: AnalysisState):
    """Save session state to disk for persistence"""
    try:
        session_file = SESSION_DIR / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        logger.info(f"Session {session_id} saved to disk")
    except Exception as e:
        logger.error(f"Failed to save session {session_id}: {e}")


def load_session_from_disk(session_id: str) -> Optional[AnalysisState]:
    """Load session state from disk"""
    try:
        session_file = SESSION_DIR / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = AnalysisState(**data)
            logger.info(f"Session {session_id} loaded from disk")
            return state
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
    return None


def ensure_session_loaded(session_id: str) -> Optional[AnalysisState]:
    """Ensure session is loaded into memory (from disk if necessary)"""
    global sessions
    
    # Check if already in memory
    if session_id in sessions:
        return sessions[session_id]
    
    # Try to load from disk
    state = load_session_from_disk(session_id)
    if state:
        sessions[session_id] = state
        return state
    
    return None


def list_saved_sessions() -> List[Dict[str, Any]]:
    """List all saved sessions on disk"""
    sessions_list = []
    try:
        for session_file in SESSION_DIR.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions_list.append({
                    "id": data.get("id", session_file.stem),
                    "name": data.get("name", "Unknown"),
                    "status": data.get("status", "unknown"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "current_round": data.get("current_round", 0),
                    "has_pathology": len(data.get("pathology_images", [])) > 0,
                    "has_sequencing": bool(data.get("control_seq_path") or data.get("treatment_seq_path")),
                    "has_biochemistry": len(data.get("biochemistry_results", [])) > 0 or len(data.get("biochemistry_files", [])) > 0,
                })
            except Exception as e:
                logger.warning(f"Failed to read session file {session_file}: {e}")
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
    
    # Sort by updated_at descending
    sessions_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions_list


# Request/Response models
class ConfigRequest(BaseModel):
    api_key: str  # Main LLM API key
    google_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    proxy_url: Optional[str] = None


class StartRequest(BaseModel):
    instruction: str
    session_name: Optional[str] = None


class PromptUpdateRequest(BaseModel):
    session_id: str
    agent: str
    prompt: str


class ModelUpdateRequest(BaseModel):
    session_id: str
    agent: str
    model: str


class TemperatureUpdateRequest(BaseModel):
    session_id: str
    agent: str
    temperature: float


class BiochemistryInput(BaseModel):
    session_id: str
    indicator: str
    value: float
    unit: str


# Endpoints
@app.get("/")
async def root():
    """API root"""
    return {"message": "InteSc - Integrated and Intelligent Single Cell Sequencing Analysis", "version": "1.0.0", "docs": "/docs"}


@app.get("/api")
async def api_root():
    return {"message": "InteSc API", "version": "1.0.0"}


@app.post("/api/config")
async def set_config(request: ConfigRequest):
    """Set API configuration"""
    global config
    
    config = SystemConfig()
    config.api.api_key = request.api_key
    config.api.google_api_key = request.google_api_key
    config.api.llm_base_url = request.llm_base_url
    config.api.proxy_url = request.proxy_url
    
    # Set proxy environment variables if provided
    if request.proxy_url:
        os.environ['HTTP_PROXY'] = request.proxy_url
        os.environ['HTTPS_PROXY'] = request.proxy_url
        os.environ['http_proxy'] = request.proxy_url
        os.environ['https_proxy'] = request.proxy_url
    else:
        # Clear proxy environment variables if not set
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(var, None)
    
    return APIResponse(
        success=True,
        message="Configuration saved successfully"
    )


@app.get("/api/config/status")
async def config_status():
    """Check if configuration is set"""
    global config
    
    if config is None:
        return {"configured": False}
    
    return {
        "configured": True,
        "has_api_key": bool(config.api.api_key),
        "has_google": bool(config.api.google_api_key)
    }


@app.post("/api/session/start")
async def start_session(request: StartRequest):
    """Start a new analysis session"""
    global config
    
    if config is None:
        raise HTTPException(status_code=400, detail="API configuration not set")
    
    # Create new session
    session_id = str(uuid.uuid4())
    state = AnalysisState(
        id=session_id,
        name=request.session_name or f"Analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        initial_instruction=request.instruction,
        status=TaskStatus.PENDING
    )
    
    sessions[session_id] = state
    
    # Save to disk for persistence
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message="Session created",
        session_id=session_id,
        data={"name": state.name}
    )


@app.get("/api/sessions")
async def list_sessions():
    """List all available sessions (from disk and memory)"""
    # Get sessions from disk
    disk_sessions = list_saved_sessions()
    
    # Also include any in-memory sessions not yet saved
    memory_session_ids = set(sessions.keys())
    disk_session_ids = {s["id"] for s in disk_sessions}
    
    for session_id in memory_session_ids - disk_session_ids:
        state = sessions[session_id]
        disk_sessions.append({
            "id": session_id,
            "name": state.name,
            "status": state.status.value if hasattr(state.status, 'value') else str(state.status),
            "created_at": state.created_at.isoformat() if state.created_at else "",
            "updated_at": state.updated_at.isoformat() if state.updated_at else "",
            "current_round": state.current_round,
            "has_pathology": len(state.pathology_images) > 0,
            "has_sequencing": bool(state.control_seq_path or state.treatment_seq_path),
            "has_biochemistry": len(state.biochemistry_results) > 0 or len(state.biochemistry_files) > 0,
        })
    
    # Sort by updated_at descending
    disk_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    
    return {"sessions": disk_sessions}


@app.get("/api/session/{session_id}/load")
async def load_session(session_id: str):
    """Load a session by ID (from memory or disk)"""
    # First check memory
    if session_id in sessions:
        state = sessions[session_id]
    else:
        # Try to load from disk
        state = load_session_from_disk(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        # Put in memory
        sessions[session_id] = state
    
    # Return complete session data including analysis results
    return {
        "session_id": state.id,
        "name": state.name,
        "status": state.status.value if hasattr(state.status, 'value') else str(state.status),
        "current_round": state.current_round,
        "max_rounds": state.max_rounds,
        "created_at": state.created_at.isoformat() if state.created_at else "",
        "updated_at": state.updated_at.isoformat() if state.updated_at else "",
        "initial_instruction": state.initial_instruction,
        # Data uploads
        "pathology_images": [{"id": img.id, "filename": img.filename, "imageType": img.image_type} for img in state.pathology_images],
        "biochemistry_results": [{"id": r.id, "indicator": r.indicator, "value": r.value, "unit": r.unit} for r in state.biochemistry_results],
        "biochemistry_files": [{"id": f.id, "filename": f.filename, "fileType": f.file_type} for f in state.biochemistry_files],
        "control_seq_path": state.control_seq_path,
        "treatment_seq_path": state.treatment_seq_path,
        "control_loom_path": state.control_loom_path,
        "treatment_loom_path": state.treatment_loom_path,
        # Analysis results - complete round results
        "round_results": [
            {
                "round": r.round_number,
                "pi_summary": r.pi_summary,
                "pathology_findings": r.pathology_findings,
                "bioinformatics_findings": r.bioinformatics_findings,
                "biochemistry_findings": r.biochemistry_findings,
                "literature_findings": r.literature_findings,
                "next_steps": r.next_steps,
                "conflicts": r.conflicts,
                "timestamp": r.timestamp.isoformat() if r.timestamp else ""
            } 
            for r in state.round_results
        ],
        "final_summary": state.final_summary,
        # Messages history (realtime messages)
        "messages": [
            {
                "agent": msg.agent.value if hasattr(msg.agent, 'value') else str(msg.agent),
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else "",
                "metadata": msg.metadata
            }
            for msg in state.messages
        ],
        # Custom settings
        "custom_prompts": state.custom_prompts,
        "custom_models": state.custom_models,
        "custom_temperatures": state.custom_temperatures,
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session completely"""
    # Remove from memory
    if session_id in sessions:
        del sessions[session_id]
    
    # Remove from disk
    session_file = SESSION_DIR / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()
    
    # Optionally remove uploaded files
    upload_dir = UPLOAD_DIR / session_id
    if upload_dir.exists():
        import shutil
        shutil.rmtree(upload_dir)
    
    return APIResponse(
        success=True,
        message="Session deleted"
    )


@app.post("/api/session/{session_id}/ensure")
async def ensure_session(session_id: str, request: StartRequest):
    """Ensure session exists, create if not found (for recovering after server restart)"""
    global config
    
    if config is None:
        raise HTTPException(status_code=400, detail="API configuration not set")
    
    # If session already exists, just return it
    if session_id in sessions:
        state = sessions[session_id]
        return APIResponse(
            success=True,
            message="Session exists",
            session_id=session_id,
            data={"name": state.name, "created": False}
        )
    
    # Create new session with the provided ID
    state = AnalysisState(
        id=session_id,
        name=request.session_name or f"Analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        initial_instruction=request.instruction,
        status=TaskStatus.PENDING
    )
    
    sessions[session_id] = state
    logger.info(f"Recreated session {session_id} after server restart")
    
    return APIResponse(
        success=True,
        message="Session recreated",
        session_id=session_id,
        data={"name": state.name, "created": True}
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session status"""
    # Ensure session is loaded (from disk if necessary)
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "name": state.name,
        "status": state.status.value,
        "current_round": state.current_round,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "has_pathology": len(state.pathology_images) > 0,
        "has_control_seq": state.control_seq_path is not None,
        "has_treatment_seq": state.treatment_seq_path is not None,
        "has_control_loom": state.control_loom_path is not None,
        "has_treatment_loom": state.treatment_loom_path is not None,
        "has_biochemistry": len(state.biochemistry_results) > 0,
        "round_count": len(state.round_results)
    }


@app.post("/api/session/{session_id}/upload/pathology")
async def upload_pathology(
    session_id: str,
    file: UploadFile = File(...),
    image_type: str = Form("H&E")
):
    """Upload pathology image"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save file with absolute path
    file_path = (UPLOAD_DIR / session_id / "pathology" / file.filename).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create PathologyImage record with absolute path
    img = PathologyImage(
        filename=file.filename,
        image_type=image_type,
        original_path=str(file_path)
    )
    
    state.pathology_images.append(img)
    
    # Auto-save session
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message=f"Pathology image uploaded: {file.filename}",
        data={"image_id": img.id, "image_type": image_type}
    )


@app.post("/api/session/{session_id}/upload/file")
async def upload_generic_file(
    session_id: str,
    file: UploadFile = File(...)
):
    """Upload a generic file (for supplementary data)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save file with absolute path
    file_path = (UPLOAD_DIR / session_id / "files" / file.filename).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return APIResponse(
        success=True,
        message=f"File uploaded: {file.filename}",
        data={"filename": file.filename, "path": str(file_path)}
    )


@app.post("/api/session/{session_id}/upload/snrna_seq")
async def upload_snrna_seq(
    session_id: str,
    file: UploadFile = File(...),
    group: str = Form("control")  # 'control' or 'treatment'
):
    """Upload snRNA-seq data file (h5ad or 10x format)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate group
    if group not in ["control", "treatment"]:
        raise HTTPException(status_code=400, detail="Invalid group. Must be 'control' or 'treatment'")
    
    # Save file with absolute path
    file_path = (UPLOAD_DIR / session_id / "snrna_seq" / group / file.filename).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Update the appropriate path based on group (use absolute path)
    if group == "control":
        state.control_seq_path = str(file_path)
    else:
        state.treatment_seq_path = str(file_path)
    
    # Auto-save session
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message=f"{group.capitalize()} snRNA-seq data uploaded: {file.filename}",
        data={"path": str(file_path), "group": group}
    )


@app.post("/api/session/{session_id}/upload/loom")
async def upload_loom(
    session_id: str,
    file: UploadFile = File(...),
    group: str = Form("control")  # 'control' or 'treatment'
):
    """Upload loom file for RNA velocity analysis"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate group
    if group not in ["control", "treatment"]:
        raise HTTPException(status_code=400, detail="Invalid group. Must be 'control' or 'treatment'")
    
    # Save file with absolute path
    file_path = (UPLOAD_DIR / session_id / "loom" / group / file.filename).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Update the appropriate path based on group (use absolute path)
    if group == "control":
        state.control_loom_path = str(file_path)
    else:
        state.treatment_loom_path = str(file_path)
    
    # Auto-save session
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message=f"{group.capitalize()} loom file uploaded: {file.filename}",
        data={"path": str(file_path), "group": group}
    )


@app.post("/api/session/{session_id}/save")
async def save_session(session_id: str):
    """Manually save session to disk"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message="Session saved"
    )


@app.post("/api/session/{session_id}/biochemistry")
async def add_biochemistry(request: BiochemistryInput):
    """Add biochemistry result"""
    state = ensure_session_loaded(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = BiochemistryResult(
        indicator=request.indicator,
        value=request.value,
        unit=request.unit
    )
    
    state.biochemistry_results.append(result)
    
    # Auto-save session
    save_session_to_disk(request.session_id, state)
    
    return APIResponse(
        success=True,
        message=f"Biochemistry result added: {request.indicator}",
        data={"result_id": result.id}
    )


@app.post("/api/session/{session_id}/upload/biochemistry")
async def upload_biochemistry_file(
    session_id: str,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """Upload biochemistry file (image or data file like csv, xlsx, txt)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Determine file type
    filename_lower = file.filename.lower()
    if filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
        file_type = 'image'
    else:
        file_type = 'file'
    
    # Save file
    file_path = (UPLOAD_DIR / session_id / "biochemistry" / file.filename).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Create biochemistry file record
    bio_file = BiochemistryFile(
        filename=file.filename,
        file_type=file_type,
        file_path=str(file_path),
        description=description
    )
    
    state.biochemistry_files.append(bio_file)
    
    # Notify via WebSocket if connected
    if session_id in websockets:
        try:
            await websockets[session_id].send_json({
                "type": "data_uploaded",
                "data_type": "biochemistry_file",
                "message": f"Biochemistry {file_type} uploaded: {file.filename}",
                "file_id": bio_file.id,
                "file_type": file_type
            })
        except Exception:
            pass  # WebSocket may not be connected
    
    # Auto-save session
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message=f"Biochemistry {file_type} uploaded: {file.filename}",
        data={
            "file_id": bio_file.id,
            "file_type": file_type,
            "path": str(file_path)
        }
    )


# ============ DELETE ENDPOINTS ============

@app.delete("/api/session/{session_id}/pathology/{image_id}")
async def delete_pathology(session_id: str, image_id: str):
    """Delete a pathology image"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find and remove the image
    for i, img in enumerate(state.pathology_images):
        if img.id == image_id:
            # Delete file from disk
            try:
                if os.path.exists(img.original_path):
                    os.remove(img.original_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {img.original_path}: {e}")
            
            # Remove from state
            state.pathology_images.pop(i)
            
            # Auto-save session
            save_session_to_disk(session_id, state)
            
            return APIResponse(
                success=True,
                message=f"Pathology image deleted: {img.filename}"
            )
    
    raise HTTPException(status_code=404, detail="Image not found")


@app.delete("/api/session/{session_id}/biochemistry/file/{file_id}")
async def delete_biochemistry_file(session_id: str, file_id: str):
    """Delete a biochemistry file"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find and remove the file
    for i, bio_file in enumerate(state.biochemistry_files):
        if bio_file.id == file_id:
            # Delete file from disk
            try:
                if os.path.exists(bio_file.file_path):
                    os.remove(bio_file.file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {bio_file.file_path}: {e}")
            
            # Remove from state
            state.biochemistry_files.pop(i)
            
            # Auto-save session
            save_session_to_disk(session_id, state)
            
            return APIResponse(
                success=True,
                message=f"Biochemistry file deleted: {bio_file.filename}"
            )
    
    raise HTTPException(status_code=404, detail="File not found")


@app.delete("/api/session/{session_id}/biochemistry/result/{result_id}")
async def delete_biochemistry_result(session_id: str, result_id: str):
    """Delete a biochemistry result (manual entry)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find and remove the result
    for i, result in enumerate(state.biochemistry_results):
        if result.id == result_id:
            state.biochemistry_results.pop(i)
            
            # Auto-save session
            save_session_to_disk(session_id, state)
            
            return APIResponse(
                success=True,
                message=f"Biochemistry result deleted: {result.indicator}"
            )
    
    raise HTTPException(status_code=404, detail="Result not found")


@app.delete("/api/session/{session_id}/sequencing/{group}")
async def delete_sequencing(session_id: str, group: str):
    """Delete sequencing data (control or treatment)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if group not in ["control", "treatment"]:
        raise HTTPException(status_code=400, detail="Group must be 'control' or 'treatment'")
    
    # Get the path and clear it
    if group == "control":
        file_path = state.control_seq_path
        if file_path:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
            state.control_seq_path = None
    else:
        file_path = state.treatment_seq_path
        if file_path:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
            state.treatment_seq_path = None
    
    # Auto-save session
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message=f"{group.capitalize()} sequencing data deleted"
    )


@app.delete("/api/session/{session_id}/loom/{group}")
async def delete_loom(session_id: str, group: str):
    """Delete loom file (control or treatment)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if group not in ["control", "treatment"]:
        raise HTTPException(status_code=400, detail="Group must be 'control' or 'treatment'")
    
    # Get the path and clear it
    if group == "control":
        file_path = state.control_loom_path
        if file_path:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
            state.control_loom_path = None
    else:
        file_path = state.treatment_loom_path
        if file_path:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
            state.treatment_loom_path = None
    
    # Auto-save session
    save_session_to_disk(session_id, state)
    
    return APIResponse(
        success=True,
        message=f"{group.capitalize()} loom file deleted"
    )


@app.post("/api/session/{session_id}/prompts")
async def update_prompts(request: PromptUpdateRequest):
    """Update agent prompts"""
    state = ensure_session_loaded(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_prompts[request.agent] = request.prompt

    # Save to disk
    save_session_to_disk(request.session_id, state)

    return APIResponse(
        success=True,
        message=f"Prompt updated for {request.agent}"
    )


@app.get("/api/session/{session_id}/prompts")
async def get_prompts(session_id: str):
    """Get current prompts"""
    global config

    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Merge default and custom prompts
    prompts = {
        "pi": state.custom_prompts.get("pi", config.prompts.pi_prompt if config else ""),
        "pathology": state.custom_prompts.get("pathology", config.prompts.pathology_prompt if config else ""),
        "bioinformatics": state.custom_prompts.get("bioinformatics", config.prompts.bioinformatics_prompt if config else ""),
        "biochemistry": state.custom_prompts.get("biochemistry", config.prompts.biochemistry_prompt if config else ""),
        "literature": state.custom_prompts.get("literature", config.prompts.literature_prompt if config else "")
    }
    
    return prompts


@app.post("/api/session/{session_id}/models")
async def update_models(request: ModelUpdateRequest):
    """Update agent models"""
    state = ensure_session_loaded(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_models[request.agent] = request.model

    # Save to disk
    save_session_to_disk(request.session_id, state)

    return APIResponse(
        success=True,
        message=f"Model updated for {request.agent}"
    )


@app.get("/api/session/{session_id}/models")
async def get_models(session_id: str):
    """Get current models"""
    global config

    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Merge default and custom models
    models = {
        "pi": state.custom_models.get("pi", config.models.pi_model if config else ""),
        "pathology": state.custom_models.get("pathology", config.models.pathology_model if config else ""),
        "bioinformatics": state.custom_models.get("bioinformatics", config.models.bioinformatics_model if config else ""),
        "biochemistry": state.custom_models.get("biochemistry", config.models.biochemistry_model if config else ""),
        "literature": state.custom_models.get("literature", "google-deep-research")
    }
    
    return models


# Default temperatures
DEFAULT_TEMPERATURES = {
    "pi": 0.5,           # PI can be more creative in synthesis and planning
    "pathology": 0.2,    # Pathology needs objective, consistent analysis
    "bioinformatics": 0.1, # Bioinformatics must be highly deterministic
    "biochemistry": 0.2,  # Biochemistry needs objective interpretation
    "literature": 0.6    # Literature search can explore more broadly
}


@app.post("/api/session/{session_id}/temperatures")
async def update_temperatures(request: TemperatureUpdateRequest):
    """Update agent temperature"""
    state = ensure_session_loaded(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_temperatures[request.agent] = request.temperature

    # Save to disk
    save_session_to_disk(request.session_id, state)

    return APIResponse(
        success=True,
        message=f"Temperature updated for {request.agent}: {request.temperature}"
    )


@app.get("/api/session/{session_id}/temperatures")
async def get_temperatures(session_id: str):
    """Get current temperatures"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Merge default and custom temperatures
    temperatures = {
        "pi": state.custom_temperatures.get("pi", DEFAULT_TEMPERATURES["pi"]),
        "pathology": state.custom_temperatures.get("pathology", DEFAULT_TEMPERATURES["pathology"]),
        "bioinformatics": state.custom_temperatures.get("bioinformatics", DEFAULT_TEMPERATURES["bioinformatics"]),
        "biochemistry": state.custom_temperatures.get("biochemistry", DEFAULT_TEMPERATURES["biochemistry"]),
        "literature": state.custom_temperatures.get("literature", DEFAULT_TEMPERATURES["literature"])
    }
    
    return temperatures


class MaxRoundsRequest(BaseModel):
    session_id: str
    max_rounds: int = Field(ge=1, le=50, default=3)


@app.post("/api/session/{session_id}/max-rounds")
async def update_max_rounds(request: MaxRoundsRequest):
    """Update maximum analysis rounds"""
    state = ensure_session_loaded(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    state.max_rounds = request.max_rounds
    
    # Auto-save session
    save_session_to_disk(request.session_id, state)
    
    return APIResponse(
        success=True,
        message=f"Max rounds updated to {request.max_rounds}"
    )


@app.get("/api/session/{session_id}/max-rounds")
async def get_max_rounds(session_id: str):
    """Get current max rounds setting"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"max_rounds": state.max_rounds}


@app.post("/api/session/{session_id}/run")
async def run_analysis(session_id: str):
    """Start running analysis (use WebSocket for real-time updates)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis already running")
    
    state.status = TaskStatus.RUNNING
    
    return APIResponse(
        success=True,
        message="Analysis started. Connect to WebSocket for updates.",
        data={"websocket_url": f"/ws/{session_id}"}
    )


@app.post("/api/session/{session_id}/pause")
async def pause_analysis(session_id: str):
    """Pause running analysis"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.status != TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis not running")
    
    state.status = TaskStatus.PAUSED
    
    if session_id in workflows:
        await workflows[session_id].pause(session_id)
    
    return APIResponse(
        success=True,
        message="Analysis paused"
    )


@app.post("/api/session/{session_id}/resume")
async def resume_analysis(session_id: str):
    """Resume paused analysis"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.status != TaskStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Analysis not paused")
    
    state.status = TaskStatus.RUNNING
    
    return APIResponse(
        success=True,
        message="Analysis resumed"
    )


class UploadedFileInfo(BaseModel):
    """Info about an uploaded file"""
    type: str  # 'image' or 'file'
    filename: str
    id: Optional[str] = None


class HumanInputRequest(BaseModel):
    """Request for human intervention"""
    message: str
    target_agent: str = "pi"  # Which agent to send the message to
    uploaded_files: List[UploadedFileInfo] = Field(default_factory=list)


@app.post("/api/session/{session_id}/human-input")
async def send_human_input(session_id: str, request: HumanInputRequest):
    """Send human input/intervention to the analysis"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Store the human input with file info
    state.pending_human_input = request.message
    state.pending_uploaded_files = [f.model_dump() for f in request.uploaded_files]
    state.human_input_requested = False  # Clear request flag
    
    # Record in history
    state.human_messages.append({
        "timestamp": datetime.now().isoformat(),
        "message": request.message,
        "target_agent": request.target_agent,
        "uploaded_files": [f.model_dump() for f in request.uploaded_files]
    })
    
    # If there's an active WebSocket, send notification
    if session_id in websockets:
        try:
            await websockets[session_id].send_json({
                "type": "human_input_received",
                "content": {
                    "message": request.message,
                    "uploaded_files": [f.model_dump() for f in request.uploaded_files]
                },
                "target_agent": request.target_agent,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass
    
    return APIResponse(
        success=True,
        message="Human input received",
        data={
            "target_agent": request.target_agent,
            "files_count": len(request.uploaded_files)
        }
    )


@app.get("/api/session/{session_id}/human-input")
async def get_human_input_status(session_id: str):
    """Check if system is waiting for human input"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "waiting_for_input": state.human_input_requested,
        "pending_input": state.pending_human_input,
        "history": state.human_messages[-10:]  # Last 10 messages
    }


@app.post("/api/session/{session_id}/request-input")
async def request_human_input(session_id: str):
    """Request human input (called by workflow when needing guidance)"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    state.human_input_requested = True
    
    # Notify via WebSocket
    if session_id in websockets:
        try:
            await websockets[session_id].send_json({
                "type": "human_input_requested",
                "content": "System is requesting human guidance",
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass
    
    return APIResponse(
        success=True,
        message="Human input requested"
    )


@app.post("/api/session/{session_id}/export")
async def export_state(session_id: str):
    """Export session state including notebook and session dump"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add session state JSON
        state_json = state.model_dump_json(indent=2)
        zf.writestr(f"{session_id}/state.json", state_json)
        
        # Add notebook file if exists
        notebook_path = STATE_DIR / f"{session_id}.ipynb"
        if notebook_path.exists():
            zf.write(notebook_path, f"{session_id}/notebook.ipynb")
        
        # Add session dump (dill) if exists
        session_dump_path = STATE_DIR / f"{session_id}_session.pkl"
        if session_dump_path.exists():
            zf.write(session_dump_path, f"{session_id}/session.pkl")
        
        # Add metadata JSON if exists
        metadata_path = STATE_DIR / f"{session_id}_state.json"
        if metadata_path.exists():
            zf.write(metadata_path, f"{session_id}/metadata.json")
        
        # Add workspace files if exist
        workspace_dir = WORKSPACE_DIR / session_id
        if workspace_dir.exists():
            for file_path in workspace_dir.rglob("*"):
                if file_path.is_file():
                    arcname = f"{session_id}/workspace/{file_path.relative_to(workspace_dir)}"
                    zf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={session_id}_export.zip"}
    )


@app.post("/api/session/import")
async def import_state(file: UploadFile = File(...)):
    """Import session state including notebook and session dump"""
    import zipfile
    import io
    
    content = await file.read()
    
    # Check if it's a zip file or JSON
    if file.filename.endswith('.zip'):
        # Handle zip file
        zip_buffer = io.BytesIO(content)
        
        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            # Find the session ID from the first directory
            namelist = zf.namelist()
            if not namelist:
                raise HTTPException(status_code=400, detail="Empty zip file")
            
            session_id = namelist[0].split('/')[0]
            
            # Extract state.json
            state_json_path = f"{session_id}/state.json"
            if state_json_path not in namelist:
                raise HTTPException(status_code=400, detail="state.json not found in archive")
            
            state_data = json.loads(zf.read(state_json_path).decode('utf-8'))
            state = AnalysisState(**state_data)
            
            # Extract notebook if exists
            notebook_path = f"{session_id}/notebook.ipynb"
            if notebook_path in namelist:
                notebook_content = zf.read(notebook_path)
                dest_notebook = STATE_DIR / f"{session_id}.ipynb"
                dest_notebook.write_bytes(notebook_content)
            
            # Extract session dump if exists
            session_pkl_path = f"{session_id}/session.pkl"
            if session_pkl_path in namelist:
                pkl_content = zf.read(session_pkl_path)
                dest_pkl = STATE_DIR / f"{session_id}_session.pkl"
                dest_pkl.write_bytes(pkl_content)
            
            # Extract metadata if exists
            metadata_path = f"{session_id}/metadata.json"
            if metadata_path in namelist:
                metadata_content = zf.read(metadata_path)
                dest_metadata = STATE_DIR / f"{session_id}_state.json"
                dest_metadata.write_bytes(metadata_content)
            
            # Extract workspace files if exist
            workspace_files = [n for n in namelist if n.startswith(f"{session_id}/workspace/")]
            for wf in workspace_files:
                if not wf.endswith('/'):  # Skip directories
                    relative_path = wf.replace(f"{session_id}/workspace/", "")
                    dest_path = WORKSPACE_DIR / session_id / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_bytes(zf.read(wf))
    else:
        # Handle plain JSON (backward compatibility)
        state_data = json.loads(content.decode('utf-8'))
        state = AnalysisState(**state_data)
    
    sessions[state.id] = state
    
    return APIResponse(
        success=True,
        message="State imported",
        session_id=state.id,
        data={
            "name": state.name,
            "has_session_dump": (STATE_DIR / f"{state.id}_session.pkl").exists()
        }
    )


@app.get("/api/session/{session_id}/code")
async def get_code_history(session_id: str):
    """Get all executed code for reproducibility"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    code_blocks = []
    for i, ce in enumerate(state.code_history):
        code_blocks.append({
            "step": i + 1,
            "code": ce.code,
            "output": ce.output[:500] if ce.output else None,
            "error": ce.error,
            "execution_time": ce.execution_time
        })
    
    return {"code_history": code_blocks}


@app.get("/api/session/{session_id}/report")
async def get_report(session_id: str):
    """Get final analysis report"""
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "name": state.name,
        "status": state.status.value,
        "round_results": [
            {
                "round": r.round_number,
                "summary": r.pi_summary,
                "pathology_findings": r.pathology_findings,
                "bioinformatics_findings": r.bioinformatics_findings,
                "biochemistry_findings": r.biochemistry_findings,
                "literature_findings": r.literature_findings
            }
            for r in state.round_results
        ],
        "final_summary": state.final_summary
    }


# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates during analysis"""
    global config
    
    await websocket.accept()
    
    # Ensure session is loaded (from disk if necessary)
    state = ensure_session_loaded(session_id)
    if state is None:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    if config is None:
        await websocket.send_json({"error": "API configuration not set"})
        await websocket.close()
        return
    
    logger.info(f"WebSocket connected for session {session_id}, state={state.status.value}")

    # Check if workflow is already running BEFORE updating websocket reference
    # Use lock to prevent race condition between check and set
    async with _state_lock:
        workflow_running = session_id in workflows
        if not workflow_running:
            # Only set websocket reference if we're the main connection (not an observer)
            websockets[session_id] = websocket

    if workflow_running:
        logger.info(f"Workflow already running for session {session_id}, observing...")

        # Send current status (but don't take over the websocket reference)
        try:
            await websocket.send_json({
                "type": "status",
                "agent": None,
                "content": f"Analysis in progress (Round {state.current_round})...",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to send status to observer: {e}")

        # Wait for existing workflow to complete or for disconnect
        # Note: We don't update websockets[session_id] - let the main workflow keep control
        try:
            while session_id in workflows:
                # Keep connection alive by waiting for messages
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    break
        except WebSocketDisconnect:
            logger.info(f"WebSocket observer disconnected for session {session_id}")
        # Don't delete websockets[session_id] - we never owned it
        return
    
    # Create message callback that uses the global websockets dict
    # This way, if the websocket reconnects, messages go to the new connection
    # Track WebSocket closed state
    ws_closed = False
    
    async def message_callback(msg: WebSocketMessage):
        nonlocal ws_closed
        if ws_closed:
            return  # Skip if WebSocket is closed
        
        current_ws = websockets.get(session_id)
        if not current_ws:
            logger.warning(f"[{session_id}] No WebSocket connection, skipping message: {msg.type}")
            return
        try:
            msg_data = {
                "type": msg.type,
                "agent": msg.agent.value if msg.agent else None,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            await current_ws.send_json(msg_data)
        except Exception as e:
            # Mark as closed to prevent further attempts
            ws_closed = True
            logger.error(f"[{session_id}] Failed to send WS message, marking as closed: {e}")
    
    # Helper to safely send JSON
    async def safe_send(data: dict):
        nonlocal ws_closed
        if ws_closed:
            return False
        
        current_ws = websockets.get(session_id)
        if not current_ws:
            return False
        try:
            await current_ws.send_json(data)
            return True
        except Exception:
            ws_closed = True
            return False
    
    # Send initial connection confirmation message
    try:
        await websocket.send_json({
            "type": "status",
            "agent": None,
            "content": "WebSocket connected, starting analysis...",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[{session_id}] Failed to send initial message: {e}")
    
    # State callback to save after each round
    async def state_callback(state: AnalysisState):
        sessions[session_id] = state
        save_session_to_disk(session_id, state)
    
    # Create workflow with session_id for state persistence
    workflow = LiverAnalysisWorkflow(
        config=config,
        session_id=session_id,
        message_callback=message_callback,
        state_callback=state_callback
    )
    async with _state_lock:
        workflows[session_id] = workflow
    
    try:
        # Run workflow
        logger.info(f"Starting workflow for session {session_id}")
        final_state = await workflow.run(state, session_id)
        sessions[session_id] = final_state
        logger.info(f"Workflow completed for session {session_id}")
        
        # Save completed state to disk
        save_session_to_disk(session_id, final_state)
        
        # Send completion message (safely)
        await safe_send({
            "type": "completed",
            "content": {
                "status": final_state.status.value,
                "rounds": len(final_state.round_results),
                "final_summary": final_state.final_summary
            }
        })
        
    except WebSocketDisconnect:
        # Client disconnected - pause workflow
        logger.info(f"WebSocket disconnected for session {session_id}")
        state.status = TaskStatus.PAUSED
        sessions[session_id] = state
        save_session_to_disk(session_id, state)
        
    except Exception as e:
        # Log the full error traceback
        logger.error(f"Workflow error for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Try to send error, but don't fail if WebSocket is closed
        await safe_send({
            "type": "error",
            "content": str(e)
        })
        state.status = TaskStatus.FAILED
        sessions[session_id] = state
        save_session_to_disk(session_id, state)
        
    finally:
        # Clean up with lock protection
        logger.info(f"Cleaning up session {session_id}")
        try:
            await workflow.close()
        except Exception as e:
            logger.error(f"Error closing workflow: {e}")
        async with _state_lock:
            if session_id in workflows:
                del workflows[session_id]
            if session_id in websockets:
                del websockets[session_id]


# Run with: uvicorn backend.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============== Detailed Report Generation ==============

from .report_generator import DetailedReportGenerator

# Store for detailed reports
detailed_reports: Dict[str, Any] = {}
report_websockets: Dict[str, WebSocket] = {}


@app.get("/api/session/{session_id}/detailed-report")
async def get_detailed_report(session_id: str):
    """Get existing detailed report for a session"""
    # Ensure session is loaded (from disk if necessary)
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_id in detailed_reports:
        return {"report": detailed_reports[session_id]}
    
    # Try to load from disk
    report_path = REPORTS_DIR / session_id / "report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
            detailed_reports[session_id] = report
            return {"report": report}
    
    return {"report": None}


@app.post("/api/session/{session_id}/generate-detailed-report")
async def generate_detailed_report(session_id: str, background_tasks: BackgroundTasks):
    """Start generating a detailed report"""
    # Ensure session is loaded (from disk if necessary)
    state = ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis must be completed before generating report")
    
    # Start generation in background
    background_tasks.add_task(
        _generate_report_task,
        session_id,
        state
    )
    
    return {"status": "started", "message": "Report generation started"}


async def _generate_report_task(session_id: str, state: AnalysisState):
    """Background task for report generation"""
    global config
    
    # Debug: log state data paths
    logger.info(f"Report generation for session {session_id}")
    logger.info(f"  control_seq_path: {state.control_seq_path}")
    logger.info(f"  treatment_seq_path: {state.treatment_seq_path}")
    logger.info(f"  state.status: {state.status}")
    logger.info(f"  round_results count: {len(state.round_results)}")
    
    output_dir = str(REPORTS_DIR / session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Progress callback
    async def progress_callback(data: dict):
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "progress",
                    **data
                })
            except:
                pass
    
    # Create LLM client using global config
    if config is None:
        logger.error("Config not set, cannot generate report")
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "error",
                    "message": "API configuration not set"
                })
            except:
                pass
        return
    
    llm_client = OpenAICompatibleClient(
        api_key=config.api.api_key,
        base_url=config.api.llm_base_url
    )
    
    # Get models from state or use defaults
    pi_model = state.custom_models.get("pi", "google/gemini-2.5-pro-preview-05-06")
    bio_model = state.custom_models.get("bioinformatics", "anthropic/claude-sonnet-4")
    
    try:
        generator = DetailedReportGenerator(
            llm_client=llm_client,
            pi_model=pi_model,
            bio_model=bio_model,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        
        # Generate report (sandbox is created internally)
        report = await generator.generate_report(state)
        detailed_reports[session_id] = report
        
        # Notify completion via WebSocket
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "complete",
                    "report": report
                })
            except:
                pass
                
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        logger.error(traceback.format_exc())
        
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass


@app.websocket("/ws/report/{session_id}")
async def report_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for report generation progress"""
    await websocket.accept()
    report_websockets[session_id] = websocket
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if session_id in report_websockets:
            del report_websockets[session_id]


@app.get("/api/file")
async def get_file(path: str):
    """Serve a file from the filesystem"""
    # Security: only allow files in the data directory
    allowed_prefixes = [
        str(REPORTS_DIR) + os.sep,
        str(UPLOAD_DIR) + os.sep,
        str(STATE_DIR) + os.sep,
    ]

    normalized_path = os.path.normpath(path)
    if not any(normalized_path.startswith(prefix) for prefix in allowed_prefixes):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(path)


@app.get("/api/download")
async def download_file(path: str):
    """Download a file"""
    # Security check - only allow files in the data directory
    allowed_prefixes = [
        str(REPORTS_DIR) + os.sep,
        str(UPLOAD_DIR) + os.sep,
        str(STATE_DIR) + os.sep,
    ]
    normalized_path = os.path.normpath(path)
    
    if not any(normalized_path.startswith(prefix) for prefix in allowed_prefixes):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = os.path.basename(path)
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


@app.post("/api/session/{session_id}/report/download-all-figures")
async def download_all_figures(session_id: str):
    """Download all figures as a ZIP file"""
    figures_dir = REPORTS_DIR / session_id / "figures"
    
    if not figures_dir.exists():
        raise HTTPException(status_code=404, detail="No figures found")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in figures_dir.glob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.name)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={session_id}_figures.zip"}
    )


@app.post("/api/session/{session_id}/report/download-all-tables")
async def download_all_tables(session_id: str):
    """Download all tables as a ZIP file"""
    tables_dir = REPORTS_DIR / session_id / "tables"
    
    if not tables_dir.exists():
        raise HTTPException(status_code=404, detail="No tables found")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in tables_dir.glob("*.csv"):
            if file_path.is_file():
                zf.write(file_path, file_path.name)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={session_id}_tables.zip"}
    )


@app.post("/api/session/{session_id}/report/download-full")
async def download_full_report(session_id: str):
    """Download full report including figures, tables, and report JSON"""
    report_dir = REPORTS_DIR / session_id
    
    if not report_dir.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in report_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(report_dir)
                zf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    
    session_name = sessions.get(session_id, {})
    if hasattr(session_name, 'name'):
        filename = f"{session_name.name}_full_report.zip"
    else:
        filename = f"{session_id}_full_report.zip"
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/session/{session_id}/report/html")
async def get_report_html(session_id: str):
    """Get the HTML report for preview"""
    report_dir = REPORTS_DIR / session_id
    html_path = report_dir / "report.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found")

    return FileResponse(
        html_path,
        media_type="text/html",
        filename=f"{session_id}_report.html"
    )


@app.get("/api/session/{session_id}/report/preview")
async def preview_report(session_id: str):
    """Get report preview data including HTML content and metadata"""
    report_dir = REPORTS_DIR / session_id

    if not report_dir.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    # Load report metadata
    report_json_path = report_dir / "report.json"
    html_path = report_dir / "report.html"

    report_data = None
    if report_json_path.exists():
        with open(report_json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

    html_content = None
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

    # Get list of figures and tables
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    figures = []
    if figures_dir.exists():
        for f in figures_dir.glob("*"):
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                pdf_path = f.with_suffix('.pdf')
                figures.append({
                    "name": f.stem,
                    "png_url": f"/api/session/{session_id}/report/figure/{f.name}",
                    "pdf_url": f"/api/session/{session_id}/report/figure/{pdf_path.name}" if pdf_path.exists() else None
                })

    tables = []
    if tables_dir.exists():
        for f in tables_dir.glob("*.csv"):
            tables.append({
                "name": f.stem,
                "csv_url": f"/api/session/{session_id}/report/table/{f.name}"
            })

    return {
        "session_id": session_id,
        "html_url": f"/api/session/{session_id}/report/html",
        "download_url": f"/api/session/{session_id}/report/download-full",
        "report": report_data,
        "html_content": html_content,
        "figures": figures,
        "tables": tables,
        "has_html": html_path.exists()
    }


@app.get("/api/session/{session_id}/report/figure/{filename}")
async def get_report_figure(session_id: str, filename: str):
    """Get a specific figure from the report"""
    figure_path = REPORTS_DIR / session_id / "figures" / filename

    if not figure_path.exists():
        raise HTTPException(status_code=404, detail="Figure not found")

    # Determine media type based on extension
    ext = figure_path.suffix.lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.pdf': 'application/pdf'
    }
    media_type = media_types.get(ext, 'application/octet-stream')

    return FileResponse(
        figure_path,
        media_type=media_type,
        filename=filename
    )


@app.get("/api/session/{session_id}/report/table/{filename}")
async def get_report_table(session_id: str, filename: str):
    """Get a specific table from the report"""
    table_path = REPORTS_DIR / session_id / "tables" / filename

    if not table_path.exists():
        raise HTTPException(status_code=404, detail="Table not found")

    return FileResponse(
        table_path,
        media_type="text/csv",
        filename=filename
    )
