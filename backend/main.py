"""
FastAPI backend server for the Multi-Agent Single-Cell Analysis System
"""
import os
import json
import asyncio
import logging
import time
import traceback
import zipfile
import tarfile
import tempfile
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
import uuid
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

from .config import (
    SystemConfig, AgentTemperatures,
    DATA_BASE_DIR, UPLOAD_DIR, STATE_DIR, WORKSPACE_DIR, SESSION_DIR, REPORTS_DIR, LOG_DIR,
    UPLOAD_CHUNK_SIZE, MAX_UPLOAD_SIZE, UPLOAD_TEMP_DIR,
    setup_logging
)
from .models.schemas import (
    AnalysisState, APIRequest, APIResponse, TaskStatus,
    WebSocketMessage, AgentType, TreatmentGroup, DataFile, BiochemistryIndicator
)
from .workflow import SingleCellAnalysisWorkflow
from .utils.llm_client import OpenAICompatibleClient, LLMManager
from .utils.upload import ChunkedUploadManager, stream_save_file

# Initialize upload manager for chunked uploads (supports up to 10GB files)
upload_manager = ChunkedUploadManager(UPLOAD_TEMP_DIR, UPLOAD_CHUNK_SIZE)


def safe_filename(filename: str) -> str:
    """Extract safe base filename, preventing path traversal. Returns empty string for invalid input."""
    name = Path(filename).name
    # Reject hidden files and names that are just dots
    if not name or name.startswith('.'):
        return ""
    return name


def validate_session_id(session_id: str) -> None:
    """Validate session ID to prevent path traversal"""
    if not session_id or '/' in session_id or '\\' in session_id or '..' in session_id:
        raise HTTPException(status_code=400, detail="Invalid session ID")


def validate_path_within(path: Path, parent: Path) -> Path:
    """Validate that resolved path stays within expected parent directory"""
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if not str(resolved).startswith(str(parent_resolved) + os.sep) and resolved != parent_resolved:
        raise HTTPException(status_code=400, detail="Invalid path: path traversal detected")
    return resolved

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
workflows: Dict[str, SingleCellAnalysisWorkflow] = {}
websockets: Dict[str, WebSocket] = {}
config: Optional[SystemConfig] = None
# Track sessions that are being initialized (prevents concurrent workflow creation)
_initializing_sessions: set = set()

# Lock for protecting concurrent access to global dicts
_state_lock = asyncio.Lock()

# Chat delegation tracking
_chat_tasks: Dict[str, asyncio.Task] = {}  # session_id → background delegation task
_chat_task_lock = asyncio.Lock()
_chat_sandboxes: Dict[str, Any] = {}  # session_id → sandbox for active chat (delegation/figure gen)

# Memory management settings
MAX_CACHED_SESSIONS = 50  # Maximum number of sessions to keep in memory
SESSION_CACHE_TTL = 3600  # Seconds before an inactive session is evicted (1 hour)
_session_access_times: Dict[str, float] = {}  # Track last access time for each session

# Message buffer for WebSocket reconnection recovery
MAX_MESSAGE_BUFFER_SIZE = 100  # Keep last N messages per session
_message_buffers: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> list of messages
_message_buffer_lock = asyncio.Lock()

# Background import jobs: import_id -> {"status": "pending"|"completed"|"failed", ...}
_import_jobs: Dict[str, dict] = {}

# Log the data directory location (directories are created in config.py)
logger.info(f"Data directory: {DATA_BASE_DIR.resolve()}")


async def cleanup_stale_sessions():
    """Background task to clean up stale sessions from memory"""
    import time
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes

            current_time = time.time()
            sessions_to_evict = []

            async with _state_lock:
                # Find sessions that are not running and have been inactive
                for session_id, last_access in list(_session_access_times.items()):
                    if session_id in workflows:
                        # Don't evict running sessions
                        continue
                    if current_time - last_access > SESSION_CACHE_TTL:
                        sessions_to_evict.append(session_id)

                # Also evict if we're over the limit (evict oldest first)
                if len(sessions) > MAX_CACHED_SESSIONS:
                    # Sort by access time, oldest first
                    sorted_sessions = sorted(
                        [(sid, _session_access_times.get(sid, 0))
                         for sid in sessions.keys() if sid not in workflows],
                        key=lambda x: x[1]
                    )
                    # Evict oldest sessions until we're under the limit
                    evict_count = len(sessions) - MAX_CACHED_SESSIONS
                    for sid, _ in sorted_sessions[:evict_count]:
                        if sid not in sessions_to_evict:
                            sessions_to_evict.append(sid)

            # Evict sessions - release lock during I/O operations to prevent blocking
            for session_id in sessions_to_evict:
                async with _state_lock:
                    if session_id in sessions:
                        state_to_save = sessions[session_id]
                        del sessions[session_id]
                    else:
                        state_to_save = None
                    if session_id in _session_access_times:
                        del _session_access_times[session_id]

                # Also clean up message buffer for evicted session
                async with _message_buffer_lock:
                    if session_id in _message_buffers:
                        del _message_buffers[session_id]

                # Save to disk outside the lock using thread pool to avoid blocking
                if state_to_save:
                    try:
                        await asyncio.to_thread(save_session_to_disk, session_id, state_to_save)
                    except Exception as save_error:
                        logger.error(f"Failed to save session {session_id} during eviction: {save_error}")

            if sessions_to_evict:
                logger.info(f"Evicted {len(sessions_to_evict)} stale sessions from memory")

        except Exception as e:
            logger.error(f"Error in session cleanup task: {e}")


async def touch_session(session_id: str):
    """Update the last access time for a session (thread-safe)"""
    import time
    async with _state_lock:
        _session_access_times[session_id] = time.time()


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup"""
    asyncio.create_task(cleanup_stale_sessions())
    logger.info("Started background session cleanup task")


# Session persistence helpers
def save_session_to_disk(session_id: str, state: AnalysisState):
    """Save session state to disk for persistence using atomic write"""
    import tempfile
    try:
        session_file = SESSION_DIR / f"{session_id}.json"
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        # Use atomic write: write to temp file first, then rename
        # This prevents data corruption if the process crashes during write
        fd, tmp_path = tempfile.mkstemp(
            suffix='.json.tmp',
            prefix=f'{session_id}_',
            dir=SESSION_DIR
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(state.model_dump_json(indent=2))
            # Atomic rename (on POSIX systems)
            os.replace(tmp_path, session_file)
            logger.info(f"Session {session_id} saved to disk")
        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
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


async def ensure_session_loaded(session_id: str) -> Optional[AnalysisState]:
    """Ensure session is loaded into memory (from disk if necessary)

    Uses lock to prevent race conditions when multiple requests try to load
    the same session simultaneously.
    """
    global sessions

    # Validate session_id to prevent path traversal
    validate_session_id(session_id)

    import time

    # Quick check without lock (optimization for common case)
    if session_id in sessions:
        # Update access time directly instead of calling touch_session() to avoid deadlock
        async with _state_lock:
            _session_access_times[session_id] = time.time()
        return sessions.get(session_id)  # use .get() for safety against eviction race

    # Use lock for the load operation to prevent race conditions
    async with _state_lock:
        # Double-check after acquiring lock (another coroutine might have loaded it)
        if session_id in sessions:
            _session_access_times[session_id] = time.time()
            return sessions[session_id]

        # Load from disk using thread pool to avoid blocking
        state = await asyncio.to_thread(load_session_from_disk, session_id)
        if state:
            sessions[session_id] = state
            _session_access_times[session_id] = time.time()
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
                data_files = data.get("data_files", [])
                sessions_list.append({
                    "id": data.get("id", session_file.stem),
                    "name": data.get("name", "Unknown"),
                    "status": data.get("status", "unknown"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "current_round": data.get("current_round", 0),
                    "has_pathology": any(f.get("section") == "pathology" for f in data_files),
                    "has_sequencing": any(f.get("section") == "sequencing" for f in data_files),
                    "has_biochemistry": any(f.get("section") == "biochemistry" for f in data_files) or len(data.get("biochemistry_indicators", [])) > 0,
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
    llm_base_url: str = "https://openrouter.ai/api/v1"
    proxy_url: Optional[str] = None
    google_api_key: str = ""  # Deprecated: kept for backward compat, now in literature_config


class ValidateKeyRequest(BaseModel):
    api_key: str
    llm_base_url: str = "https://openrouter.ai/api/v1"
    proxy_url: Optional[str] = None


class StartRequest(BaseModel):
    instruction: str
    session_name: Optional[str] = None


class PromptUpdateRequest(BaseModel):
    agent: str
    prompt: str


class ModelUpdateRequest(BaseModel):
    agent: str
    model: str


class TemperatureUpdateRequest(BaseModel):
    agent: str
    temperature: float


class LiteratureConfigRequest(BaseModel):
    mode: Literal["deep_research", "gemini_deep_research", "pubmed_llm"] = Field(default="deep_research")
    deep_research_model: str = ""
    google_api_key: str = ""
    pubmed_email: str = ""
    max_search_rounds: int = Field(default=5, ge=1, le=20)


class BiochemistryInput(BaseModel):
    indicator: str = Field(..., min_length=1, max_length=200, description="Biochemistry indicator name")
    value: float = Field(..., description="Indicator value")
    unit: str = Field(..., max_length=50, description="Measurement unit")
    group_id: Optional[str] = None
    sample_number: Optional[int] = None


class SessionUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    instruction: Optional[str] = None


class GroupCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    sample_count: int = Field(1, ge=1, le=100)


class GroupUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    sample_count: Optional[int] = Field(None, ge=1, le=100)


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
    }


@app.post("/api/config/validate")
async def validate_api_key(request: ValidateKeyRequest):
    """Validate API key by attempting to list models from the LLM endpoint"""
    base_url = request.llm_base_url.rstrip('/')
    models_url = f"{base_url}/models"

    headers = {
        "Authorization": f"Bearer {request.api_key}",
        "Content-Type": "application/json",
    }

    proxy = request.proxy_url if request.proxy_url else None

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            proxy=proxy,
            follow_redirects=True,
        ) as client:
            response = await client.get(models_url, headers=headers)

            if response.status_code == 200:
                return {"valid": True, "message": "API key is valid"}
            elif response.status_code in (401, 403):
                return {
                    "valid": False,
                    "message": f"Invalid API key or insufficient permissions (HTTP {response.status_code})",
                }
            else:
                return {
                    "valid": False,
                    "message": f"Unexpected response from LLM server (HTTP {response.status_code})",
                    "uncertain": True,
                }
    except httpx.TimeoutException:
        return {
            "valid": False,
            "message": "Connection timed out while validating API key",
            "uncertain": True,
        }
    except httpx.ConnectError as e:
        return {
            "valid": False,
            "message": f"Could not connect to LLM server: {str(e)}",
            "uncertain": True,
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"Validation error: {str(e)}",
            "uncertain": True,
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
    await asyncio.to_thread(save_session_to_disk, session_id, state)
    
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
    
    # Also include any in-memory sessions not yet saved (snapshot keys to avoid race)
    memory_session_ids = set(list(sessions.keys()))
    disk_session_ids = {s["id"] for s in disk_sessions}

    for session_id in memory_session_ids - disk_session_ids:
        state = sessions.get(session_id)
        if state is None:
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
    
    # Sort by updated_at descending
    disk_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    
    return {"sessions": disk_sessions}


def _strip_message_base64(msg_type: str, content) -> any:
    """Strip large base64 image data from message content to reduce response size."""
    # figure type: content is raw base64 string
    if msg_type == "figure" and isinstance(content, str) and len(content) > 1000:
        return "[image]"
    # graphic_abstract type: content has image_base64
    if msg_type == "graphic_abstract" and isinstance(content, dict):
        return {k: v for k, v in content.items() if k not in ("image_base64", "imageBase64")}
    # figure_review / figure_approved / figure_revision: content has png_base64
    if msg_type in ("figure_review", "figure_approved", "figure_revision") and isinstance(content, dict):
        return {k: v for k, v in content.items() if k not in ("png_base64", "pngBase64")}
    # Strip base64 from figures arrays in result/integrated_report messages
    if isinstance(content, dict) and "figures" in content and isinstance(content.get("figures"), list):
        stripped_figures = []
        for f in content["figures"]:
            if isinstance(f, dict):
                stripped_figures.append({k: v for k, v in f.items() if k not in ("data", "pngBase64", "png_base64")})
            else:
                stripped_figures.append(f)
        return {**content, "figures": stripped_figures}
    return content


@app.get("/api/session/{session_id}/load")
async def load_session(session_id: str):
    """Load a session by ID (from memory or disk)"""
    validate_session_id(session_id)
    # First check memory
    if session_id in sessions:
        state = sessions[session_id]
    else:
        # Try to load from disk
        state = await asyncio.to_thread(load_session_from_disk, session_id)
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
        "round_wait_seconds": state.round_wait_seconds,
        "created_at": state.created_at.isoformat() if state.created_at else "",
        "updated_at": state.updated_at.isoformat() if state.updated_at else "",
        "initial_instruction": state.initial_instruction,
        # Treatment groups and data uploads
        "treatment_groups": [
            {"id": g.id, "name": g.name, "description": g.description, "sampleCount": g.sample_count, "order": g.order}
            for g in state.treatment_groups
        ],
        "data_files": [
            {"id": f.id, "filename": f.filename, "filePath": f.file_path, "section": f.section, "dataType": f.data_type,
             "groupId": f.group_id, "sampleNumber": f.sample_number, "description": f.description, "timestamp": f.timestamp.isoformat() if f.timestamp else ""}
            for f in state.data_files
        ],
        "biochemistry_indicators": [
            {"id": r.id, "indicator": r.indicator, "value": r.value, "unit": r.unit,
             "referenceRange": r.reference_range, "interpretation": r.interpretation,
             "groupId": r.group_id, "sampleNumber": r.sample_number}
            for r in state.biochemistry_indicators
        ],
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
        # Messages history (realtime WebSocket messages for UI restoration)
        # Strip large base64 image data from messages to reduce response size
        "messages": [
            {
                "id": msg.id,
                "type": msg.type,
                "agent": msg.agent.value if hasattr(msg.agent, 'value') else (str(msg.agent) if msg.agent else None),
                "content": _strip_message_base64(msg.type, msg.content),
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else "",
            }
            for msg in state.realtime_messages
        ],
        # Custom settings
        "custom_prompts": state.custom_prompts,
        "custom_models": state.custom_models,
        "custom_temperatures": state.custom_temperatures,
        # State version control
        "state_version": state.state_version,
        "last_sync_timestamp": state.last_sync_timestamp.isoformat() if state.last_sync_timestamp else "",
        # Agent errors tracking
        "agent_errors": [
            {
                "id": err.id,
                "agent": err.agent.value if hasattr(err.agent, 'value') else str(err.agent),
                "error_type": err.error_type,
                "message": err.message,
                "details": err.details,
                "round_number": err.round_number,
                "timestamp": err.timestamp.isoformat() if err.timestamp else "",
                "resolved": err.resolved,
                "resolution": err.resolution
            }
            for err in state.agent_errors
        ],
        # Publication outputs
        "publication_figures": [
            {
                "id": fig.id,
                "name": fig.name,
                "description": fig.description,
                "figure_type": fig.figure_type,
                "png_path": fig.png_path,
                "pdf_path": fig.pdf_path,
                "png_base64": fig.png_base64,
                "review_status": fig.review_status.value if hasattr(fig.review_status, 'value') else str(fig.review_status),
                "pi_feedback": fig.pi_feedback,
                "revision_count": fig.revision_count,
                "created_at": fig.created_at.isoformat() if fig.created_at else ""
            }
            for fig in state.publication_figures
        ],
        "publication_tables": [
            {
                "id": tbl.id,
                "name": tbl.name,
                "description": tbl.description,
                "csv_path": tbl.csv_path,
                "columns": tbl.columns,
                "preview_data": tbl.preview_data,
                "review_status": tbl.review_status.value if hasattr(tbl.review_status, 'value') else str(tbl.review_status),
                "pi_feedback": tbl.pi_feedback,
                "created_at": tbl.created_at.isoformat() if tbl.created_at else ""
            }
            for tbl in state.publication_tables
        ],
        "graphic_abstract": {
            "id": state.graphic_abstract.id,
            "prompt": state.graphic_abstract.prompt,
            "image_base64": state.graphic_abstract.image_base64,
            "created_at": state.graphic_abstract.created_at.isoformat() if state.graphic_abstract.created_at else ""
        } if state.graphic_abstract else None,
        "integrated_report": state.integrated_report,
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session completely"""
    validate_session_id(session_id)

    # Atomically check workflow status and remove session from memory
    async with _state_lock:
        if session_id in workflows or session_id in _initializing_sessions:
            raise HTTPException(status_code=409, detail="Cannot delete a running session. Stop the analysis first.")
        # Remove from memory while holding lock
        sessions.pop(session_id, None)
        _session_access_times.pop(session_id, None)

    # Remove message buffer
    async with _message_buffer_lock:
        _message_buffers.pop(session_id, None)

    # Remove from disk (use thread pool for blocking I/O)
    def _delete_from_disk():
        session_file = SESSION_DIR / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        upload_dir = UPLOAD_DIR / session_id
        if upload_dir.exists():
            import shutil
            shutil.rmtree(upload_dir)

    await asyncio.to_thread(_delete_from_disk)

    return APIResponse(
        success=True,
        message="Session deleted"
    )


@app.post("/api/session/{session_id}/ensure")
async def ensure_session(session_id: str, request: StartRequest):
    """Ensure session exists, create if not found (for recovering after server restart)"""
    global config
    validate_session_id(session_id)

    if config is None:
        raise HTTPException(status_code=400, detail="API configuration not set")

    # If session already exists in memory, just return it
    if session_id in sessions:
        state = sessions[session_id]
        return APIResponse(
            success=True,
            message="Session exists",
            session_id=session_id,
            data={"name": state.name, "created": False}
        )

    # Try loading from disk first (session may exist from previous server run)
    existing = await ensure_session_loaded(session_id)
    if existing:
        return APIResponse(
            success=True,
            message="Session loaded from disk",
            session_id=session_id,
            data={"name": existing.name, "created": False}
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
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
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


@app.post("/api/session/{session_id}/groups")
async def create_group(session_id: str, request: GroupCreateRequest):
    """Create a new treatment group"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    group = TreatmentGroup(
        name=request.name,
        description=request.description,
        sample_count=request.sample_count,
        order=len(state.treatment_groups)
    )
    state.treatment_groups.append(group)
    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"Treatment group created: {request.name}",
        data={"id": group.id, "name": group.name, "description": group.description,
              "sampleCount": group.sample_count, "order": group.order}
    )


@app.put("/api/session/{session_id}/info")
async def update_session_info(session_id: str, request: SessionUpdateRequest):
    """Update session name and/or instruction"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.name is not None:
        state.name = request.name
    if request.instruction is not None:
        state.initial_instruction = request.instruction

    await asyncio.to_thread(save_session_to_disk, session_id, state)
    return APIResponse(
        success=True,
        message="Session info updated",
        data={"name": state.name, "instruction": state.initial_instruction}
    )


@app.put("/api/session/{session_id}/groups/{group_id}")
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
        # Prevent reducing sample count below highest sample number with data
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

    await asyncio.to_thread(save_session_to_disk, session_id, state)
    return APIResponse(
        success=True,
        message=f"Group updated: {group.name}",
        data={"id": group.id, "name": group.name, "description": group.description,
              "sampleCount": group.sample_count, "order": group.order}
    )


@app.delete("/api/session/{session_id}/groups/{group_id}")
async def delete_group(session_id: str, group_id: str):
    """Delete a treatment group and all associated data files"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    group = state.get_group_by_id(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")

    # Delete associated data files from disk
    files_to_remove = [f for f in state.data_files if f.group_id == group_id]
    for df in files_to_remove:
        try:
            if os.path.exists(df.file_path):
                os.remove(df.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete file {df.file_path}: {e}")

    # Remove files and indicators from state
    state.data_files = [f for f in state.data_files if f.group_id != group_id]
    state.biochemistry_indicators = [i for i in state.biochemistry_indicators if i.group_id != group_id]
    state.treatment_groups = [g for g in state.treatment_groups if g.id != group_id]

    await asyncio.to_thread(save_session_to_disk, session_id, state)
    return APIResponse(
        success=True,
        message=f"Group deleted: {group.name} (and {len(files_to_remove)} associated files)"
    )


@app.post("/api/session/{session_id}/upload/data")
async def upload_data(
    session_id: str,
    file: UploadFile = File(...),
    section: str = Form(...),
    data_type: str = Form(...),
    group_id: Optional[str] = Form(None),
    sample_number: Optional[int] = Form(None),
    description: Optional[str] = Form(None)
):
    """Unified data file upload endpoint"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate section
    if section not in ("pathology", "sequencing", "biochemistry"):
        raise HTTPException(status_code=400, detail="Section must be 'pathology', 'sequencing', or 'biochemistry'")

    # Validate group_id if provided
    if group_id and group_id != "none":
        group = state.get_group_by_id(group_id)
        if group is None:
            raise HTTPException(status_code=400, detail="Group not found")
        # Validate sample_number
        if sample_number is not None and sample_number > group.sample_count:
            raise HTTPException(status_code=400, detail=f"Sample number exceeds group sample count ({group.sample_count})")
    else:
        group_id = None
        sample_number = None  # No sample distinction when no group

    # Build storage path
    if group_id:
        sample_dir = str(sample_number) if sample_number else "_unsorted"
        upload_parent = UPLOAD_DIR / session_id / section / group_id / sample_dir
    else:
        upload_parent = UPLOAD_DIR / session_id / section / "_ungrouped"

    sanitized_name = safe_filename(file.filename)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = upload_parent / sanitized_name
    upload_parent.mkdir(parents=True, exist_ok=True)
    validate_path_within(file_path, upload_parent)

    # For large sequencing files, use stream save
    if section == "sequencing":
        await stream_save_file(file, file_path)
    else:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

    # Create DataFile record
    data_file = DataFile(
        filename=sanitized_name,
        file_path=str(file_path),
        section=section,
        data_type=data_type,
        group_id=group_id,
        sample_number=sample_number,
        description=description
    )
    state.data_files.append(data_file)

    # Register as pending if workflow is waiting
    if state.status in (TaskStatus.WAITING_INPUT, TaskStatus.COMPLETED):
        state.pending_uploaded_files.append({
            "type": section,
            "filename": sanitized_name,
            "id": data_file.id
        })

    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"{section} {data_type} uploaded: {sanitized_name}",
        data={"file_id": data_file.id, "section": section, "data_type": data_type,
              "file_path": str(file_path), "group_id": group_id, "sample_number": sample_number}
    )


@app.post("/api/session/{session_id}/upload/file")
async def upload_generic_file(
    session_id: str,
    file: UploadFile = File(...)
):
    """Upload a generic file (for supplementary data)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save file with absolute path (sanitize filename to prevent path traversal)
    sanitized_name = safe_filename(file.filename)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    upload_parent = UPLOAD_DIR / session_id / "files"
    file_path = upload_parent / sanitized_name
    upload_parent.mkdir(parents=True, exist_ok=True)
    validate_path_within(file_path, upload_parent)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return APIResponse(
        success=True,
        message=f"File uploaded: {sanitized_name}",
        data={"filename": sanitized_name}
    )




@app.post("/api/session/{session_id}/save")
async def save_session(session_id: str):
    """Manually save session to disk"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await asyncio.to_thread(save_session_to_disk, session_id, state)
    
    return APIResponse(
        success=True,
        message="Session saved"
    )


@app.post("/api/session/{session_id}/indicators")
async def add_indicator(session_id: str, request: BiochemistryInput):
    """Add a manual biochemistry indicator"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    import math
    if math.isnan(request.value) or math.isinf(request.value):
        raise HTTPException(status_code=400, detail="Value must be a finite number (not NaN or Infinity)")
    if not request.indicator.strip():
        raise HTTPException(status_code=400, detail="Indicator name cannot be empty or whitespace")
    if not request.unit.strip():
        raise HTTPException(status_code=400, detail="Unit cannot be empty or whitespace")

    # Validate group_id if provided
    group_id = request.group_id
    sample_number = request.sample_number
    if group_id and group_id != "none":
        group = state.get_group_by_id(group_id)
        if group is None:
            raise HTTPException(status_code=400, detail="Group not found")
        if sample_number is not None and sample_number > group.sample_count:
            raise HTTPException(status_code=400, detail=f"Sample number exceeds group sample count ({group.sample_count})")
    else:
        group_id = None
        sample_number = None

    indicator = BiochemistryIndicator(
        indicator=request.indicator.strip(),
        value=request.value,
        unit=request.unit.strip(),
        group_id=group_id,
        sample_number=sample_number
    )
    state.biochemistry_indicators.append(indicator)
    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"Biochemistry indicator added: {request.indicator}",
        data={"indicator_id": indicator.id}
    )


@app.delete("/api/session/{session_id}/indicators/{indicator_id}")
async def delete_indicator(session_id: str, indicator_id: str):
    """Delete a biochemistry indicator"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the indicator name before deletion (for response message)
    deleted_name = next((ind.indicator for ind in state.biochemistry_indicators if ind.id == indicator_id), None)
    if deleted_name is None:
        raise HTTPException(status_code=404, detail="Indicator not found")

    state.biochemistry_indicators = [ind for ind in state.biochemistry_indicators if ind.id != indicator_id]
    await asyncio.to_thread(save_session_to_disk, session_id, state)
    return APIResponse(success=True, message=f"Indicator deleted: {deleted_name}")


# ============ DELETE ENDPOINTS ============

@app.delete("/api/session/{session_id}/data/{file_id}")
async def delete_data_file(session_id: str, file_id: str):
    """Delete any data file (pathology, sequencing, or biochemistry)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    for i, df in enumerate(state.data_files):
        if df.id == file_id:
            # Delete file from disk
            try:
                if os.path.exists(df.file_path):
                    os.remove(df.file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {df.file_path}: {e}")

            state.data_files.pop(i)
            await asyncio.to_thread(save_session_to_disk, session_id, state)

            return APIResponse(
                success=True,
                message=f"{df.section} file deleted: {df.filename}"
            )

    raise HTTPException(status_code=404, detail="Data file not found")


@app.post("/api/session/{session_id}/prompts")
async def update_prompts(session_id: str, request: PromptUpdateRequest):
    """Update agent prompts"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_prompts[request.agent] = request.prompt

    # Save to disk
    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"Prompt updated for {request.agent}"
    )


@app.get("/api/session/{session_id}/prompts")
async def get_prompts(session_id: str):
    """Get current prompts"""
    global config

    state = await ensure_session_loaded(session_id)
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
async def update_models(session_id: str, request: ModelUpdateRequest):
    """Update agent models"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_models[request.agent] = request.model

    # Save to disk
    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"Model updated for {request.agent}"
    )


@app.get("/api/session/{session_id}/models")
async def get_models(session_id: str):
    """Get current models"""
    global config

    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Merge default and custom models
    models = {
        "pi": state.custom_models.get("pi", config.models.pi_model if config else ""),
        "pathology": state.custom_models.get("pathology", config.models.pathology_model if config else ""),
        "bioinformatics": state.custom_models.get("bioinformatics", config.models.bioinformatics_model if config else ""),
        "biochemistry": state.custom_models.get("biochemistry", config.models.biochemistry_model if config else ""),
        "literature": state.custom_models.get("literature", config.models.pi_model if config else "google/gemini-3-flash-preview"),
        "painter": state.custom_models.get("painter", config.models.painter_model if config else "gemini-2.0-flash-exp-image-generation")
    }

    return models


@app.post("/api/session/{session_id}/temperatures")
async def update_temperatures(session_id: str, request: TemperatureUpdateRequest):
    """Update agent temperature"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_temperatures[request.agent] = request.temperature

    # Save to disk
    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"Temperature updated for {request.agent}: {request.temperature}"
    )


@app.get("/api/session/{session_id}/temperatures")
async def get_temperatures(session_id: str):
    """Get current temperatures - uses centralized config with custom overrides"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Use centralized config as base, merge with custom temperatures
    base_temps = config.temperatures if config else AgentTemperatures()
    merged = base_temps.merge_custom(state.custom_temperatures)

    return merged.model_dump()


class MaxRoundsRequest(BaseModel):
    max_rounds: int = Field(ge=1, le=50, default=3)


@app.post("/api/session/{session_id}/max-rounds")
async def update_max_rounds(session_id: str, request: MaxRoundsRequest):
    """Update maximum analysis rounds"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    state.max_rounds = request.max_rounds

    # Auto-save session
    await asyncio.to_thread(save_session_to_disk, session_id, state)
    
    return APIResponse(
        success=True,
        message=f"Max rounds updated to {request.max_rounds}"
    )


@app.get("/api/session/{session_id}/max-rounds")
async def get_max_rounds(session_id: str):
    """Get current max rounds setting"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"max_rounds": state.max_rounds}


class RoundWaitRequest(BaseModel):
    round_wait_seconds: int = Field(ge=-1, le=3600, default=300)


@app.post("/api/session/{session_id}/round-wait")
async def update_round_wait(session_id: str, request: RoundWaitRequest):
    """Update round wait countdown seconds (-1=manual, 0=no wait, positive=countdown)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    state.round_wait_seconds = request.round_wait_seconds

    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"Round wait updated to {request.round_wait_seconds}s"
    )


@app.post("/api/session/{session_id}/literature-config")
async def update_literature_config(session_id: str, request: LiteratureConfigRequest):
    """Update literature agent configuration"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.literature_config = {
        "mode": request.mode,
        "deep_research_model": request.deep_research_model,
        "google_api_key": request.google_api_key,
        "pubmed_email": request.pubmed_email,
        "max_search_rounds": request.max_search_rounds,
    }

    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"Literature config updated: mode={request.mode}"
    )


@app.get("/api/session/{session_id}/literature-config")
async def get_literature_config(session_id: str):
    """Get current literature agent configuration"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return state.literature_config


@app.post("/api/session/{session_id}/run")
async def run_analysis(session_id: str):
    """Start running analysis (use WebSocket for real-time updates)"""
    state = await ensure_session_loaded(session_id)
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
    state = await ensure_session_loaded(session_id)
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
    """Resume paused or completed analysis (for re-analysis)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status == TaskStatus.COMPLETED:
        # Re-analysis: reset for a new round while keeping existing data
        state.status = TaskStatus.RUNNING
        # Increment max_rounds to allow additional analysis
        state.max_rounds = state.current_round + 1
        return APIResponse(
            success=True,
            message="Re-analysis started"
        )
    elif state.status == TaskStatus.WAITING_INPUT:
        # Continue from round-end wait (user sent input or skipped)
        state.status = TaskStatus.RUNNING
        # Signal the workflow to continue from _wait_for_user_input
        if session_id in workflows:
            workflows[session_id].signal_continue()
        return APIResponse(
            success=True,
            message="Continuing analysis"
        )
    elif state.status == TaskStatus.PAUSED:
        state.status = TaskStatus.RUNNING
        return APIResponse(
            success=True,
            message="Analysis resumed"
        )
    elif state.status in (TaskStatus.GENERATING_REPORT, TaskStatus.FAILED):
        # Interrupted during report generation or failed — allow re-run
        state.status = TaskStatus.RUNNING
        return APIResponse(
            success=True,
            message="Resuming analysis"
        )
    else:
        raise HTTPException(status_code=400, detail="Analysis not paused, waiting, or completed")


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
    state = await ensure_session_loaded(session_id)
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
    state = await ensure_session_loaded(session_id)
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
    state = await ensure_session_loaded(session_id)
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


@app.api_route("/api/session/{session_id}/export", methods=["GET", "HEAD"])
async def export_state(session_id: str):
    """Export complete session as tar.gz including all uploads, workspace, reports, and state files.

    Uses a temporary file on disk to avoid loading the entire archive into memory.
    """
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save latest state to disk first
    await asyncio.to_thread(save_session_to_disk, session_id, state)

    # Write tar.gz to a temp file on disk (not in memory)
    tmp = tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False, dir=str(UPLOAD_TEMP_DIR))
    tmp_path = tmp.name
    tmp.close()

    def _build_tar():
        with tarfile.open(tmp_path, mode='w:gz') as tf:
            def _add_bytes(arcname: str, data: bytes):
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            # Manifest
            manifest = json.dumps({
                "session_id": session_id,
                "version": "1.0",
                "data_base_dir": str(DATA_BASE_DIR)
            }, indent=2).encode("utf-8")
            _add_bytes(f"{session_id}/manifest.json", manifest)

            # 1. sessions/{session_id}.json
            session_file = SESSION_DIR / f"{session_id}.json"
            if session_file.exists():
                tf.add(str(session_file), arcname=f"{session_id}/sessions/{session_id}.json")

            # 2. states/{session_id}* (including .pkl for sandbox restore)
            for state_file in STATE_DIR.glob(f"{session_id}*"):
                if state_file.is_file():
                    tf.add(str(state_file), arcname=f"{session_id}/states/{state_file.name}")

            # 3. uploads/{session_id}/ (recursive)
            upload_dir = UPLOAD_DIR / session_id
            if upload_dir.exists():
                for file_path in upload_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(DATA_BASE_DIR)
                        tf.add(str(file_path), arcname=f"{session_id}/{rel}")

            # 4. workspace/{session_id}/ (recursive)
            workspace_dir = WORKSPACE_DIR / session_id
            if workspace_dir.exists():
                for file_path in workspace_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(DATA_BASE_DIR)
                        tf.add(str(file_path), arcname=f"{session_id}/{rel}")

            # 5. reports/{session_id}/ (recursive)
            reports_dir = REPORTS_DIR / session_id
            if reports_dir.exists():
                for file_path in reports_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(DATA_BASE_DIR)
                        tf.add(str(file_path), arcname=f"{session_id}/{rel}")

    await asyncio.to_thread(_build_tar)

    # Build safe filename
    safe_name = "".join(c if c.isalnum() or c in '-_ ' else '' for c in state.name).strip().replace(' ', '_')
    if not safe_name:
        safe_name = "session"
    filename = f"{safe_name}_{session_id[:8]}.tar.gz"

    # FileResponse streams from disk; BackgroundTask deletes the temp file after response is sent
    return FileResponse(
        tmp_path,
        media_type="application/gzip",
        filename=filename,
        background=BackgroundTask(os.unlink, tmp_path)
    )


# --- Export: two-phase (prepare + download) for large files ---

# In-memory store for prepared export file paths: session_id -> {"path": str, "filename": str, "size": int}
_prepared_exports: Dict[str, Dict[str, Any]] = {}


@app.post("/api/session/{session_id}/export/prepare")
async def export_prepare(session_id: str):
    """Phase 1: Build the tar.gz archive on disk and return metadata.

    The frontend calls this first (showing a spinner), then triggers
    the streaming download via GET /export/download once ready.
    """
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save latest state to disk first
    await asyncio.to_thread(save_session_to_disk, session_id, state)

    # Build tar to a stable temp file
    export_dir = REPORTS_DIR / session_id
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in '-_ ' else '' for c in state.name).strip().replace(' ', '_')
    if not safe_name:
        safe_name = "session"
    filename = f"{safe_name}_{session_id[:8]}.tar.gz"
    tar_path = export_dir / filename

    def _build_tar():
        with tarfile.open(str(tar_path), mode='w:gz') as tf:
            def _add_bytes(arcname: str, data: bytes):
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            manifest = json.dumps({
                "session_id": session_id,
                "version": "1.0",
                "data_base_dir": str(DATA_BASE_DIR)
            }, indent=2).encode("utf-8")
            _add_bytes(f"{session_id}/manifest.json", manifest)

            session_file = SESSION_DIR / f"{session_id}.json"
            if session_file.exists():
                tf.add(str(session_file), arcname=f"{session_id}/sessions/{session_id}.json")

            for state_file in STATE_DIR.glob(f"{session_id}*"):
                if state_file.is_file():
                    tf.add(str(state_file), arcname=f"{session_id}/states/{state_file.name}")

            upload_dir = UPLOAD_DIR / session_id
            if upload_dir.exists():
                for file_path in upload_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(DATA_BASE_DIR)
                        tf.add(str(file_path), arcname=f"{session_id}/{rel}")

            workspace_dir = WORKSPACE_DIR / session_id
            if workspace_dir.exists():
                for file_path in workspace_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(DATA_BASE_DIR)
                        tf.add(str(file_path), arcname=f"{session_id}/{rel}")

            reports_dir = REPORTS_DIR / session_id
            if reports_dir.exists():
                for file_path in reports_dir.rglob("*"):
                    if file_path.is_file():
                        # Skip the export tar itself
                        if file_path.name == filename:
                            continue
                        rel = file_path.relative_to(DATA_BASE_DIR)
                        tf.add(str(file_path), arcname=f"{session_id}/{rel}")

    await asyncio.to_thread(_build_tar)

    size_bytes = tar_path.stat().st_size

    # Store reference for the download endpoint
    _prepared_exports[session_id] = {
        "path": str(tar_path),
        "filename": filename,
        "size": size_bytes,
    }

    return {
        "status": "ready",
        "size_bytes": size_bytes,
        "filename": filename,
    }


@app.get("/api/session/{session_id}/export/download")
async def export_download(session_id: str):
    """Phase 2: Stream the pre-built tar.gz to the browser.

    The file is deleted after the response is fully sent.
    """
    validate_session_id(session_id)

    info = _prepared_exports.pop(session_id, None)
    if not info or not os.path.exists(info["path"]):
        raise HTTPException(status_code=404, detail="Export not prepared. Call POST /export/prepare first.")

    return FileResponse(
        info["path"],
        media_type="application/gzip",
        filename=info["filename"],
        background=BackgroundTask(os.unlink, info["path"])
    )


@app.post("/api/session/import")
async def import_state(background_tasks: BackgroundTasks, file: UploadFile = File(...), import_id: Optional[str] = Form(None)):
    """Import session from tar.gz/zip/json archive, restoring all files to their directories.

    For tar.gz: streams upload to temp file, then extracts in a background task and returns
    an import_id immediately to avoid proxy timeouts on large archives.
    The frontend can pre-generate import_id so that even if a proxy timeout prevents the
    response from reaching the client, it can still poll for status.
    For zip/json: processed inline (small files).
    """
    filename = file.filename or ""

    if filename.endswith(('.tar.gz', '.tgz')):
        # ---- tar.gz: save to disk, then extract in background to avoid proxy timeouts ----
        # Register import job BEFORE saving file so frontend can poll even if proxy times out
        import_job_id = import_id or str(uuid.uuid4())
        _import_jobs[import_job_id] = {"status": "uploading"}

        tmp_path = UPLOAD_TEMP_DIR / f"import_{uuid.uuid4().hex}.tar.gz"
        try:
            await stream_save_file(file, tmp_path)
        except Exception as e:
            _import_jobs[import_job_id] = {"status": "failed", "error": f"Upload failed: {str(e)}"}
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

        _import_jobs[import_job_id] = {"status": "pending"}
        background_tasks.add_task(_process_import_job, str(tmp_path), import_job_id)

        return APIResponse(
            success=True,
            message="File uploaded, extraction in progress",
            data={"import_id": import_job_id, "status": "pending"}
        )

    elif filename.endswith('.zip'):
        # ---- Legacy zip import (typically small) ----
        content = await file.read()
        state = _import_zip(content)

    else:
        # ---- Plain JSON (backward compatibility) ----
        content = await file.read()
        state_data = json.loads(content.decode('utf-8'))
        state = AnalysisState(**state_data)

    # Validate imported session ID (zip/json path)
    if not state.id or '/' in state.id or '\\' in state.id or '..' in state.id:
        raise HTTPException(status_code=400, detail="Invalid session ID in imported data")

    # Save state to disk and load into memory
    await asyncio.to_thread(save_session_to_disk, state.id, state)
    sessions[state.id] = state

    return APIResponse(
        success=True,
        message="Session imported successfully",
        session_id=state.id,
        data={"name": state.name}
    )


async def _process_import_job(tmp_path_str: str, import_job_id: str):
    """Background task: extract tar.gz archive and load session."""
    try:
        state = await asyncio.to_thread(_import_tar_gz, tmp_path_str)

        if not state.id or '/' in state.id or '\\' in state.id or '..' in state.id:
            _import_jobs[import_job_id] = {"status": "failed", "error": "Invalid session ID in imported data"}
            return

        await asyncio.to_thread(save_session_to_disk, state.id, state)
        sessions[state.id] = state
        _import_jobs[import_job_id] = {"status": "completed", "session_id": state.id, "name": state.name}
    except HTTPException as e:
        _import_jobs[import_job_id] = {"status": "failed", "error": e.detail}
    except Exception as e:
        _import_jobs[import_job_id] = {"status": "failed", "error": str(e)}
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)


@app.get("/api/session/import/{import_id}")
async def get_import_status(import_id: str):
    """Poll the status of a background tar.gz import job."""
    if import_id not in _import_jobs:
        raise HTTPException(status_code=404, detail="Import job not found")
    job = _import_jobs[import_id]
    return {"import_id": import_id, **job}


def _import_tar_gz(file_path: str) -> AnalysisState:
    """Import from tar.gz archive file on disk with path remapping"""
    data_base_str = str(DATA_BASE_DIR)

    with tarfile.open(file_path, mode='r:gz') as tf:
        # Read manifest first
        manifest_data = None
        session_id = None
        for member in tf.getmembers():
            if member.name.endswith('/manifest.json'):
                f = tf.extractfile(member)
                if f:
                    manifest_data = json.loads(f.read().decode('utf-8'))
                    session_id = manifest_data.get("session_id")
                    break

        if not session_id:
            # Fallback: infer session_id from first path component
            members = tf.getmembers()
            if not members:
                raise HTTPException(status_code=400, detail="Empty archive")
            session_id = members[0].name.split('/')[0]

        # Validate session_id
        if not session_id or '/' in session_id or '\\' in session_id or '..' in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID in archive")

        old_base_dir = manifest_data.get("data_base_dir", "") if manifest_data else ""

        # Extract files with security checks
        state_json_data = None
        for member in tf.getmembers():
            if not member.isfile():
                continue

            # Security: skip paths with .. or absolute paths
            if '..' in member.name or member.name.startswith('/'):
                logger.warning(f"Skipping archive entry with suspicious path: {member.name}")
                continue

            # Skip manifest (already processed)
            if member.name.endswith('/manifest.json'):
                continue

            # Determine destination: strip session_id prefix, rest maps to DATA_BASE_DIR
            parts = member.name.split('/', 1)
            if len(parts) < 2:
                continue
            relative_to_base = parts[1]  # e.g. "sessions/{id}.json", "uploads/{id}/pathology/img.png"

            # Read state JSON for later processing
            if relative_to_base == f"sessions/{session_id}.json":
                f = tf.extractfile(member)
                if f:
                    state_json_data = f.read()

            dest_path = (DATA_BASE_DIR / relative_to_base).resolve()

            # Validate path stays within DATA_BASE_DIR
            if not str(dest_path).startswith(data_base_str + os.sep) and str(dest_path) != data_base_str:
                logger.warning(f"Skipping entry outside data dir: {member.name}")
                continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(member)
            if f:
                dest_path.write_bytes(f.read())

    if state_json_data is None:
        raise HTTPException(status_code=400, detail="Session state JSON not found in archive")

    # Path remapping: replace old data_base_dir with current one
    state_json_str = state_json_data.decode('utf-8')
    if old_base_dir and old_base_dir != data_base_str:
        state_json_str = state_json_str.replace(old_base_dir, data_base_str)

    state_data = json.loads(state_json_str)
    state = AnalysisState(**state_data)

    # Overwrite the session file with remapped paths
    session_file = SESSION_DIR / f"{state.id}.json"
    session_file.write_text(state.model_dump_json(indent=2), encoding='utf-8')

    return state


def _import_zip(content: bytes) -> AnalysisState:
    """Legacy zip import (backward compatibility)"""
    zip_buffer = io.BytesIO(content)

    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        namelist = zf.namelist()
        if not namelist:
            raise HTTPException(status_code=400, detail="Empty zip file")

        session_id = namelist[0].split('/')[0]
        if not session_id or '/' in session_id or '\\' in session_id or '..' in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID in archive")

        # Extract state.json
        state_json_path = f"{session_id}/state.json"
        if state_json_path not in namelist:
            raise HTTPException(status_code=400, detail="state.json not found in archive")

        state_data = json.loads(zf.read(state_json_path).decode('utf-8'))
        state = AnalysisState(**state_data)

        # Extract notebook if exists
        notebook_path = f"{session_id}/notebook.ipynb"
        if notebook_path in namelist:
            (STATE_DIR / f"{session_id}.ipynb").write_bytes(zf.read(notebook_path))

        # Extract session.pkl if exists (for sandbox restore)
        pkl_path = f"{session_id}/session.pkl"
        if pkl_path in namelist:
            (STATE_DIR / f"{session_id}_session.pkl").write_bytes(zf.read(pkl_path))

        # Extract metadata if exists
        metadata_path = f"{session_id}/metadata.json"
        if metadata_path in namelist:
            (STATE_DIR / f"{session_id}_state.json").write_bytes(zf.read(metadata_path))

        # Extract workspace files with path traversal protection
        workspace_parent = (WORKSPACE_DIR / session_id).resolve()
        for wf in namelist:
            if wf.startswith(f"{session_id}/workspace/") and not wf.endswith('/'):
                relative_path = wf.replace(f"{session_id}/workspace/", "")
                dest_path = (WORKSPACE_DIR / session_id / relative_path).resolve()
                if not str(dest_path).startswith(str(workspace_parent) + os.sep):
                    logger.warning(f"Skipping zip entry with path traversal: {wf}")
                    continue
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(zf.read(wf))

    return state


@app.get("/api/session/{session_id}/code")
async def get_code_history(session_id: str):
    """Get all executed code for reproducibility"""
    state = await ensure_session_loaded(session_id)
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
    state = await ensure_session_loaded(session_id)
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
    state = await ensure_session_loaded(session_id)
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
        workflow_initializing = session_id in _initializing_sessions
        if not workflow_running and not workflow_initializing:
            # Only set websocket reference if we're the main connection (not an observer)
            websockets[session_id] = websocket
            # Mark as initializing to prevent concurrent workflow creation
            _initializing_sessions.add(session_id)

    if workflow_running or workflow_initializing:
        logger.info(f"Workflow already running for session {session_id}, observing...")

        # Send current status (but don't take over the websocket reference)
        try:
            await websocket.send_json({
                "type": "status",
                "agent": None,
                "content": f"Analysis in progress (Round {state.current_round})...",
                "timestamp": datetime.now().isoformat()
            })

            # Send buffered messages to allow observer to catch up
            async with _message_buffer_lock:
                buffered_messages = _message_buffers.get(session_id, [])
                if buffered_messages:
                    logger.info(f"Sending {len(buffered_messages)} buffered messages to observer")
                    await websocket.send_json({
                        "type": "message_history",
                        "content": {
                            "messages": buffered_messages,
                            "count": len(buffered_messages)
                        },
                        "timestamp": datetime.now().isoformat()
                    })
        except Exception as e:
            logger.error(f"Failed to send status/history to observer: {e}")

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

    # Check if session is already completed - don't restart workflow
    if state.status == TaskStatus.COMPLETED:
        logger.info(f"Session {session_id} already completed, not restarting workflow")
        # Update websocket reference for this completed session
        websockets[session_id] = websocket
        # Remove from initializing set since we're not starting a workflow
        async with _state_lock:
            _initializing_sessions.discard(session_id)

        try:
            # Send completed message with full session data for UI restoration
            await websocket.send_json({
                "type": "completed",
                "agent": None,
                "content": {
                    "status": "completed",
                    "message": "Analysis already completed. View results in the Reports tab.",
                    "final_summary": state.final_summary,
                    "round_results": [
                        {
                            "round": r.round_number,
                            "summary": r.pi_summary,
                            "pathology_findings": r.pathology_findings,
                            "bioinformatics_findings": r.bioinformatics_findings,
                            "biochemistry_findings": r.biochemistry_findings,
                            "literature_findings": r.literature_findings,
                        }
                        for r in state.round_results
                    ],
                },
                "timestamp": datetime.now().isoformat()
            })

            # Send message_history for UI message stream restoration
            if state.realtime_messages:
                msg_data_list = [
                    {
                        "id": msg.id,
                        "type": msg.type,
                        "agent": msg.agent.value if hasattr(msg.agent, 'value') else (str(msg.agent) if msg.agent else None),
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else datetime.now().isoformat(),
                    }
                    for msg in state.realtime_messages[-500:]  # Limit to prevent oversized payload
                ]
                await websocket.send_json({
                    "type": "message_history",
                    "content": {
                        "messages": msg_data_list,
                        "count": len(msg_data_list)
                    },
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"[{session_id}] Sent {len(msg_data_list)} messages from history to reconnecting client")
        except Exception as e:
            logger.error(f"[{session_id}] Failed to send completed status: {e}")

        # Keep connection open for potential client requests
        try:
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for completed session {session_id}")
        finally:
            if websockets.get(session_id) == websocket:
                del websockets[session_id]
        return
    
    # Create message callback that uses the global websockets dict
    # This way, if the websocket reconnects, messages go to the new connection
    # Track WebSocket closed state with lock for thread-safe access
    ws_closed = False
    ws_lock = asyncio.Lock()

    async def message_callback(msg: WebSocketMessage):
        nonlocal ws_closed

        # Save message to state.realtime_messages for session restoration
        state.realtime_messages.append(msg)

        msg_data = {
            "type": msg.type,
            "agent": msg.agent.value if msg.agent else None,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else datetime.now().isoformat(),
            "id": msg.id  # Include message ID for deduplication on client
        }

        # Add to message buffer for reconnection recovery
        async with _message_buffer_lock:
            if session_id not in _message_buffers:
                _message_buffers[session_id] = []
            _message_buffers[session_id].append(msg_data)
            # Trim buffer to max size
            if len(_message_buffers[session_id]) > MAX_MESSAGE_BUFFER_SIZE:
                _message_buffers[session_id] = _message_buffers[session_id][-MAX_MESSAGE_BUFFER_SIZE:]

        # Get websocket reference atomically with ws_closed check
        async with ws_lock:
            if ws_closed:
                # Check if a new websocket has connected (reset ws_closed if so)
                current_ws = websockets.get(session_id)
                if current_ws and current_ws != websocket:
                    # New connection detected, reset closed state
                    ws_closed = False
                else:
                    return  # Still closed, skip
            else:
                current_ws = websockets.get(session_id)

        if not current_ws:
            logger.warning(f"[{session_id}] No WebSocket connection, message buffered: {msg.type}")
            return
        try:
            await current_ws.send_json(msg_data)
        except Exception as e:
            # Mark as closed to prevent further attempts
            async with ws_lock:
                ws_closed = True
            logger.error(f"[{session_id}] Failed to send WS message, marking as closed: {e}")

    # Helper to safely send JSON
    async def safe_send(data: dict):
        nonlocal ws_closed
        # Get websocket reference atomically with ws_closed check
        async with ws_lock:
            if ws_closed:
                # Check if a new websocket has connected
                current_ws = websockets.get(session_id)
                if current_ws and current_ws != websocket:
                    ws_closed = False
                else:
                    return False
            else:
                current_ws = websockets.get(session_id)

        if not current_ws:
            return False
        try:
            await current_ws.send_json(data)
            return True
        except Exception:
            async with ws_lock:
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
        # Use thread pool for disk I/O to avoid blocking event loop
        await asyncio.to_thread(save_session_to_disk, session_id, state)

    # Create workflow with session_id for state persistence
    workflow = SingleCellAnalysisWorkflow(
        config=config,
        session_id=session_id,
        message_callback=message_callback,
        state_callback=state_callback
    )
    # Add workflow to tracking dict (protected by lock to prevent race conditions)
    async with _state_lock:
        workflows[session_id] = workflow

    # Background task to monitor WebSocket for client disconnect
    async def monitor_websocket():
        """Listen for WebSocket disconnect while workflow is running"""
        nonlocal ws_closed
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, asyncio.CancelledError):
            logger.info(f"[{session_id}] Client disconnected during workflow execution")
            async with ws_lock:
                ws_closed = True
        except Exception as e:
            logger.error(f"[{session_id}] Unexpected WebSocket monitor error: {e}")
            async with ws_lock:
                ws_closed = True

    monitor_task = asyncio.create_task(monitor_websocket())

    # Detect sessions that completed analysis but were interrupted during report generation
    effective_max_rounds = state.max_rounds if state.max_rounds > 0 else config.max_rounds
    needs_report_only = (
        state.status in (TaskStatus.RUNNING, TaskStatus.WAITING_INPUT,
                         TaskStatus.FAILED, TaskStatus.GENERATING_REPORT)
        and state.current_round >= effective_max_rounds
        and len(state.round_results) >= effective_max_rounds
    )

    try:
        # Run workflow (or report-only if analysis was already complete)
        if needs_report_only:
            logger.info(f"Session {session_id} needs report generation only "
                        f"(round {state.current_round}/{effective_max_rounds}, "
                        f"{len(state.round_results)} round results)")
            final_state = await workflow.generate_report(state, session_id)
        else:
            logger.info(f"Starting workflow for session {session_id}")
            final_state = await workflow.run(state, session_id)
        sessions[session_id] = final_state
        logger.info(f"Workflow completed for session {session_id}")

        # Save completed state to disk (use thread pool for I/O)
        await asyncio.to_thread(save_session_to_disk, session_id, final_state)

        # Send completion message (safely)
        await safe_send({
            "type": "completed",
            "content": {
                "status": final_state.status.value,
                "rounds": len(final_state.round_results),
                "final_summary": final_state.final_summary
            }
        })

    except Exception as e:
        # Log the full error traceback
        logger.error(f"Workflow error for session {session_id}: {e}")
        logger.error(traceback.format_exc())

        # Try to send error, but don't fail if WebSocket is closed
        await safe_send({
            "type": "error",
            "content": str(e)
        })
        # Get the latest state to avoid race condition
        current_state = sessions.get(session_id, state)
        current_state.status = TaskStatus.FAILED
        sessions[session_id] = current_state
        await asyncio.to_thread(save_session_to_disk, session_id, current_state)

    finally:
        # Cancel the WebSocket monitor task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

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
            # Remove from initializing set
            _initializing_sessions.discard(session_id)


# Run with: uvicorn backend.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============== Detailed Report Generation ==============

from .report_generator import DetailedReportGenerator

# Store for detailed reports
detailed_reports: Dict[str, Any] = {}
report_websockets: Dict[str, WebSocket] = {}


def _sanitize_nan(obj):
    """Recursively replace NaN/Infinity float values with None for JSON compliance."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


@app.get("/api/session/{session_id}/detailed-report")
async def get_detailed_report(session_id: str):
    """Get existing detailed report for a session"""
    # First check memory cache
    if session_id in detailed_reports:
        return {"report": _sanitize_nan(detailed_reports[session_id])}

    # Try to load from disk first (avoid session loading overhead)
    report_path = REPORTS_DIR / session_id / "report.json"
    if report_path.exists():
        try:
            # Use parse_float to handle NaN/Infinity in JSON text
            import re as _re
            with open(report_path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Replace bare NaN/Infinity tokens with null before parsing
            raw = _re.sub(r'\bNaN\b', 'null', raw)
            raw = _re.sub(r'\b-?Infinity\b', 'null', raw)
            report = json.loads(raw)
            report = _sanitize_nan(report)
            detailed_reports[session_id] = report
            return {"report": report}
        except Exception as e:
            logger.error(f"Failed to load report for {session_id}: {e}")

    # Only check session exists if no report found
    session_file = SESSION_DIR / f"{session_id}.json"
    if not session_file.exists() and session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"report": None}


@app.post("/api/session/{session_id}/generate-detailed-report")
async def generate_detailed_report(session_id: str, background_tasks: BackgroundTasks):
    """Start generating a detailed report"""
    # Ensure session is loaded (from disk if necessary)
    state = await ensure_session_loaded(session_id)
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
    seq_files = state.get_files_by_section("sequencing")
    logger.info(f"  sequencing files: {len(seq_files)}")
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
            except Exception as e:
                logger.debug(f"Failed to send progress update for session {session_id}: {e}")
    
    # Create LLM client using global config
    if config is None:
        logger.error("Config not set, cannot generate report")
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "error",
                    "message": "API configuration not set"
                })
            except Exception as e:
                logger.debug(f"Failed to send error to WebSocket for session {session_id}: {e}")
        return
    
    llm_client = OpenAICompatibleClient(
        api_key=config.api.api_key,
        base_url=config.api.llm_base_url,
        proxy_url=config.api.proxy_url
    )

    # Get models from state or use defaults
    pi_model = state.custom_models.get("pi", "google/gemini-2.5-pro-preview-05-06")
    bio_model = state.custom_models.get("bioinformatics", "anthropic/claude-opus-4.6")
    
    try:
        workspace_dir = str(WORKSPACE_DIR / session_id)
        generator = DetailedReportGenerator(
            llm_client=llm_client,
            pi_model=pi_model,
            bio_model=bio_model,
            output_dir=output_dir,
            workspace_dir=workspace_dir,
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
            except Exception as e:
                logger.debug(f"Failed to send completion to WebSocket for session {session_id}: {e}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        logger.error(traceback.format_exc())

        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception as ws_error:
                logger.debug(f"Failed to send error to WebSocket for session {session_id}: {ws_error}")


@app.websocket("/ws/report/{session_id}")
async def report_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for report generation progress"""
    if not session_id or '/' in session_id or '\\' in session_id or '..' in session_id:
        await websocket.close(code=1008, reason="Invalid session ID")
        return
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
    # Use Path.relative_to() for proper path containment check (handles symlinks correctly)
    allowed_dirs = [
        Path(REPORTS_DIR).resolve(),
        Path(UPLOAD_DIR).resolve(),
        Path(STATE_DIR).resolve(),
        Path(WORKSPACE_DIR).resolve(),
    ]

    # Resolve the real path to prevent traversal attacks like /../../../etc/passwd
    try:
        real_path = Path(path).resolve()
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid path")

    # Check if the resolved path is within any allowed directory using relative_to
    # This properly handles symlinks that might point outside allowed directories
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            real_path.relative_to(allowed_dir)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied")

    if not real_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not real_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    return FileResponse(str(real_path))


@app.get("/api/download")
async def download_file(path: str):
    """Download a file"""
    # Security: only allow files in the data directory
    # Use Path.relative_to() for proper path containment check (handles symlinks correctly)
    allowed_dirs = [
        Path(REPORTS_DIR).resolve(),
        Path(UPLOAD_DIR).resolve(),
        Path(STATE_DIR).resolve(),
        Path(WORKSPACE_DIR).resolve(),
    ]

    # Resolve the real path to prevent traversal attacks
    try:
        real_path = Path(path).resolve()
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid path")

    # Check if the resolved path is within any allowed directory using relative_to
    # This properly handles symlinks that might point outside allowed directories
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            real_path.relative_to(allowed_dir)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied")

    if not real_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not real_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    filename = real_path.name
    return FileResponse(str(real_path), filename=filename, media_type="application/octet-stream")


@app.post("/api/session/{session_id}/report/download-all-figures")
async def download_all_figures(session_id: str):
    """Download all figures as a ZIP file"""
    validate_session_id(session_id)
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
    validate_session_id(session_id)
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
    validate_session_id(session_id)
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
    validate_session_id(session_id)
    report_dir = REPORTS_DIR / session_id
    html_path = report_dir / "report.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found")

    # Don't set filename to avoid triggering download
    return FileResponse(
        html_path,
        media_type="text/html"
    )


@app.get("/api/session/{session_id}/report/preview")
async def preview_report(session_id: str):
    """Get report preview data including HTML content and metadata"""
    validate_session_id(session_id)
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
    safe_name = safe_filename(filename)
    figure_parent = REPORTS_DIR / session_id / "figures"
    figure_path = figure_parent / safe_name
    validate_path_within(figure_path, figure_parent)

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


# Also handle "figures/" path for HTML relative URLs
@app.get("/api/session/{session_id}/report/figures/{filename}")
async def get_report_figure_alt(session_id: str, filename: str):
    """Get a specific figure from the report (alternative path for HTML relative URLs)"""
    return await get_report_figure(session_id, filename)


@app.get("/api/session/{session_id}/report/table/{filename}")
async def get_report_table(session_id: str, filename: str):
    """Get a specific table from the report"""
    safe_name = safe_filename(filename)
    table_parent = REPORTS_DIR / session_id / "tables"
    table_path = table_parent / safe_name
    validate_path_within(table_path, table_parent)

    if not table_path.exists():
        raise HTTPException(status_code=404, detail="Table not found")

    return FileResponse(
        table_path,
        media_type="text/csv",
        filename=filename
    )


# Also handle "tables/" path for HTML relative URLs
@app.get("/api/session/{session_id}/report/tables/{filename}")
async def get_report_table_alt(session_id: str, filename: str):
    """Get a specific table from the report (alternative path for HTML relative URLs)"""
    return await get_report_table(session_id, filename)


# ============================================
# Publication Figures and Graphic Abstract Endpoints
# ============================================

@app.get("/api/session/{session_id}/graphic-abstract")
async def get_graphic_abstract(session_id: str):
    """Get the graphic abstract for a session"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.graphic_abstract:
        return {
            "success": True,
            "graphic_abstract": {
                "id": state.graphic_abstract.id,
                "image_base64": state.graphic_abstract.image_base64,
                "prompt": state.graphic_abstract.prompt,
                "created_at": state.graphic_abstract.created_at.isoformat()
            }
        }

    return {
        "success": False,
        "error": "Graphic abstract not generated yet",
        "graphic_abstract": None
    }


@app.get("/api/session/{session_id}/publication-figures")
async def get_publication_figures(session_id: str):
    """Get all publication figures for a session"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    figures = [
        {
            "id": fig.id,
            "name": fig.name,
            "description": fig.description,
            "figure_type": fig.figure_type,
            "png_path": fig.png_path,
            "pdf_path": fig.pdf_path,
            "png_base64": fig.png_base64,
            "review_status": fig.review_status.value,
            "pi_feedback": fig.pi_feedback,
            "revision_count": fig.revision_count,
            "created_at": fig.created_at.isoformat()
        }
        for fig in state.publication_figures
    ]

    return {
        "success": True,
        "figures": figures,
        "total": len(figures),
        "approved": sum(1 for f in state.publication_figures if f.review_status.value == "approved")
    }


@app.get("/api/session/{session_id}/publication-tables")
async def get_publication_tables(session_id: str):
    """Get all publication tables for a session"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    tables = [
        {
            "id": tbl.id,
            "name": tbl.name,
            "description": tbl.description,
            "csv_path": tbl.csv_path,
            "columns": tbl.columns,
            "preview_data": tbl.preview_data,
            "review_status": tbl.review_status.value,
            "pi_feedback": tbl.pi_feedback,
            "created_at": tbl.created_at.isoformat()
        }
        for tbl in state.publication_tables
    ]

    return {
        "success": True,
        "tables": tables,
        "total": len(tables)
    }


def _find_workspace_figure(state: AnalysisState, session_id: str, figure_id: str) -> Optional[Dict[str, Any]]:
    """Search integrated_report message for a workspace figure by ID.

    Returns a dict with 'name' and optional 'pdf_path' if found, else None.
    """
    for msg in state.realtime_messages:
        msg_type = msg.type if hasattr(msg, 'type') else (msg.get('type') if isinstance(msg, dict) else None)
        msg_content = msg.content if hasattr(msg, 'content') else (msg.get('content') if isinstance(msg, dict) else None)
        if msg_type == 'integrated_report' and isinstance(msg_content, dict):
            for fig_data in msg_content.get('figures', []):
                if isinstance(fig_data, dict) and fig_data.get('id') == figure_id:
                    return fig_data
    return None


@app.get("/api/session/{session_id}/figure/{figure_id}/pdf")
async def download_publication_figure_pdf(session_id: str, figure_id: str):
    """Download a specific publication figure as PDF"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    figure = next((f for f in state.publication_figures if f.id == figure_id), None)

    if figure is not None:
        if not figure.pdf_path:
            raise HTTPException(status_code=404, detail="PDF not available for this figure")
        pdf_path = Path(figure.pdf_path)
        fig_name = figure.name
    else:
        # Fallback: search workspace figures from integrated_report message
        fig_data = _find_workspace_figure(state, session_id, figure_id)
        if fig_data is None:
            raise HTTPException(status_code=404, detail="Figure not found")
        fig_name = fig_data.get('name', '')
        # Try explicit pdf_path first, then derive from workspace directory
        explicit_pdf = fig_data.get('pdf_path', '')
        if explicit_pdf and Path(explicit_pdf).exists():
            pdf_path = Path(explicit_pdf)
        else:
            pdf_path = WORKSPACE_DIR / session_id / "figures" / f"{fig_name}.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    # Validate path is within allowed directories
    resolved = pdf_path.resolve()
    allowed_parents = [REPORTS_DIR.resolve(), WORKSPACE_DIR.resolve(), UPLOAD_DIR.resolve()]
    if not any(str(resolved).startswith(str(p) + os.sep) or resolved == p for p in allowed_parents):
        raise HTTPException(status_code=403, detail="Access denied: file outside allowed directories")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{fig_name}.pdf"
    )


@app.get("/api/session/{session_id}/figure/{figure_id}/png")
async def download_publication_figure_png(session_id: str, figure_id: str):
    """Download a specific publication figure as PNG"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    figure = next((f for f in state.publication_figures if f.id == figure_id), None)

    if figure is not None:
        if not figure.png_path:
            raise HTTPException(status_code=404, detail="PNG not available for this figure")
        png_path = Path(figure.png_path)
        fig_name = figure.name
    else:
        # Fallback: search workspace figures from integrated_report message
        fig_data = _find_workspace_figure(state, session_id, figure_id)
        if fig_data is None:
            raise HTTPException(status_code=404, detail="Figure not found")
        fig_name = fig_data.get('name', '')
        png_path = WORKSPACE_DIR / session_id / "figures" / f"{fig_name}.png"

    if not png_path.exists():
        raise HTTPException(status_code=404, detail="PNG file not found on disk")

    # Validate path is within allowed directories
    resolved = png_path.resolve()
    allowed_parents = [REPORTS_DIR.resolve(), WORKSPACE_DIR.resolve(), UPLOAD_DIR.resolve()]
    if not any(str(resolved).startswith(str(p) + os.sep) or resolved == p for p in allowed_parents):
        raise HTTPException(status_code=403, detail="Access denied: file outside allowed directories")

    return FileResponse(
        png_path,
        media_type="image/png",
        filename=f"{fig_name}.png"
    )


@app.get("/api/session/{session_id}/integrated-report")
async def get_integrated_report(session_id: str):
    """Get the integrated report for a session"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.integrated_report:
        return {
            "success": True,
            "report": state.integrated_report,
            "figures_count": len(state.publication_figures),
            "tables_count": len(state.publication_tables)
        }

    return {
        "success": False,
        "error": "Integrated report not generated yet",
        "report": None
    }


# ============== Chunked Upload Endpoints (for large files up to 10GB) ==============

class InitChunkedUploadRequest(BaseModel):
    """Request to initialize a chunked upload"""
    filename: str
    total_size: int
    total_chunks: int
    file_type: str  # 'snrna_seq', 'loom', 'pathology', 'biochemistry'
    group: Optional[str] = None  # 'control' or 'treatment' for seq data


@app.post("/api/session/{session_id}/upload/init-chunked")
async def init_chunked_upload(session_id: str, request: InitChunkedUploadRequest):
    """Initialize a chunked upload for large files"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.total_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed ({MAX_UPLOAD_SIZE / (1024**3):.1f}GB)"
        )

    upload_id = await upload_manager.init_upload(
        session_id=session_id,
        filename=request.filename,
        total_size=request.total_size,
        total_chunks=request.total_chunks,
        file_type=request.file_type,
        group=request.group
    )

    return {
        "upload_id": upload_id,
        "chunk_size": UPLOAD_CHUNK_SIZE,
        "total_chunks": request.total_chunks
    }


@app.post("/api/session/{session_id}/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(
    session_id: str,
    upload_id: str,
    chunk_index: int,
    file: UploadFile = File(...),
    chunk_hash: Optional[str] = Form(None)
):
    """Upload a single chunk"""
    chunk_data = await file.read()

    result = await upload_manager.save_chunk(
        upload_id=upload_id,
        chunk_index=chunk_index,
        chunk_data=chunk_data,
        chunk_hash=chunk_hash
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/api/session/{session_id}/upload/complete-chunked/{upload_id}")
async def complete_chunked_upload(
    session_id: str,
    upload_id: str,
    file_type: str = Form(None),
    group: Optional[str] = Form(None)
):
    """Complete chunked upload and assemble file"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get upload metadata
    status = await upload_manager.get_upload_status(upload_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Use metadata values (set at init), not client-overridable form params
    file_type = status.get("file_type") or file_type
    group = status.get("group") or group

    # Determine final path based on file type
    filename = safe_filename(status["filename"])
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if file_type == "snrna_seq":
        final_path = UPLOAD_DIR / session_id / "snrna_seq" / (group or "control") / filename
    elif file_type == "loom":
        final_path = UPLOAD_DIR / session_id / "loom" / (group or "control") / filename
    elif file_type == "pathology":
        final_path = UPLOAD_DIR / session_id / "pathology" / filename
    elif file_type == "biochemistry":
        final_path = UPLOAD_DIR / session_id / "biochemistry" / filename
    else:
        final_path = UPLOAD_DIR / session_id / "files" / filename

    # Complete upload
    result = await upload_manager.complete_upload(upload_id, final_path.resolve())

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    # Update session state - create DataFile for all chunked uploads
    section_map = {"snrna_seq": "sequencing", "loom": "sequencing", "pathology": "pathology", "biochemistry": "biochemistry"}
    type_map = {"snrna_seq": "scRNA-seq", "loom": "loom", "pathology": "uploaded", "biochemistry": "file"}
    section = section_map.get(file_type, file_type)
    data_type = type_map.get(file_type, file_type)

    data_file = DataFile(
        filename=filename,
        file_path=str(final_path.resolve()),
        section=section,
        data_type=data_type,
        group_id=status.get("group_id"),
        sample_number=status.get("sample_number"),
    )
    state.data_files.append(data_file)

    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=f"File uploaded successfully: {filename}",
        data=result
    )


@app.get("/api/session/{session_id}/upload/status/{upload_id}")
async def get_chunked_upload_status(session_id: str, upload_id: str):
    """Get chunked upload progress"""
    status = await upload_manager.get_upload_status(upload_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return status


@app.delete("/api/session/{session_id}/upload/cancel/{upload_id}")
async def cancel_chunked_upload(session_id: str, upload_id: str):
    """Cancel and cleanup a chunked upload"""
    success = await upload_manager.cancel_upload(upload_id)
    if not success:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {"success": True, "message": "Upload cancelled"}


# ============== Node Retry Endpoints ==============

class RetryNodeRequest(BaseModel):
    """Request to retry or skip a failed node"""
    node_name: str
    skip: bool = False  # If True, skip instead of retry


@app.post("/api/session/{session_id}/retry-node")
async def retry_node(session_id: str, request: RetryNodeRequest):
    """Retry or skip a failed node"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status != TaskStatus.FAILED:
        raise HTTPException(status_code=400, detail="Session is not in failed state")

    # Note: The actual retry/skip logic is handled by the workflow's with_retry decorator
    # This endpoint just updates the state to allow resumption

    if request.skip:
        # Store skip_nodes in sandbox_state (used by workflow on resume)
        if state.sandbox_state is None:
            state.sandbox_state = {}
        if "skip_nodes" not in state.sandbox_state:
            state.sandbox_state["skip_nodes"] = []
        if request.node_name not in state.sandbox_state["skip_nodes"]:
            state.sandbox_state["skip_nodes"].append(request.node_name)
        message = f"Node {request.node_name} will be skipped on resume"
    else:
        # Clear failed status for retry
        state.status = TaskStatus.PAUSED  # Allow resumption
        message = f"Session ready to retry from node {request.node_name}"

    await asyncio.to_thread(save_session_to_disk, session_id, state)

    return APIResponse(
        success=True,
        message=message,
        data={"node": request.node_name, "action": "skip" if request.skip else "retry"}
    )


@app.get("/api/session/{session_id}/failed-node")
async def get_failed_node(session_id: str):
    """Get information about the failed node"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if we have workflow state with error info
    if session_id in workflows:
        workflow = workflows[session_id]
        config = {"configurable": {"thread_id": session_id}}
        try:
            graph_state = await workflow.compiled_graph.aget_state(config)
            if graph_state and graph_state.values:
                return {
                    "failed_node": graph_state.values.get("failed_node"),
                    "node_errors": graph_state.values.get("node_errors", {}),
                    "retry_counts": graph_state.values.get("node_retry_count", {})
                }
        except Exception as e:
            logger.error(f"Error getting workflow state: {e}")

    # Fallback: check sandbox_state
    if state.sandbox_state:
        return {
            "failed_node": state.sandbox_state.get("failed_node"),
            "node_errors": state.sandbox_state.get("node_errors", {}),
            "retry_counts": state.sandbox_state.get("node_retry_count", {})
        }

    return {"failed_node": None, "node_errors": {}, "retry_counts": {}}


# ============== Bioinformatics Resume Endpoint ==============

@app.post("/api/session/{session_id}/resume-bioinformatics")
async def resume_bioinformatics(session_id: str):
    """Resume bioinformatics analysis from last checkpoint"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check for checkpoint
    checkpoint_path = STATE_DIR / f"{session_id}_bioinfo_checkpoint.json"
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="No bioinformatics checkpoint found")

    # Read checkpoint info
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return APIResponse(
            success=True,
            message=f"Checkpoint found at step {checkpoint.get('step_number', 0)}. Connect to WebSocket to resume.",
            data={
                "has_checkpoint": True,
                "step_number": checkpoint.get("step_number", 0),
                "total_steps": checkpoint.get("total_steps", 80),
                "findings_count": len(checkpoint.get("findings", [])),
                "figures_count": len(checkpoint.get("figures", []))
            }
        )
    except Exception as e:
        logger.error(f"Error reading checkpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading checkpoint: {str(e)}")


# ============== Chat Endpoints (PI Agent free-form chat) ==============

def _build_chat_context(state: AnalysisState) -> tuple:
    """Build analysis context summary for PI Agent chat.

    Returns:
        (context_str, figure_inventory) where figure_inventory is a list of
        {"name": str, "path": str} dicts for PNG files found on disk.
    """
    parts = []

    if state.initial_instruction:
        parts.append(f"Research Goal: {state.initial_instruction}")

    parts.append(f"Current Round: {state.current_round}")
    parts.append(f"Status: {state.status.value}")

    # Experimental design summary
    design_summary = state.build_experimental_design_summary()
    if design_summary:
        parts.append(design_summary)

    # Biochemistry indicators summary
    if state.biochemistry_indicators:
        indicators = [f"{r.indicator}={r.value}{r.unit}" for r in state.biochemistry_indicators]
        parts.append(f"Biochemistry: {', '.join(indicators)}")

    # Round results (last 3)
    for rr in state.round_results[-3:]:
        round_parts = [f"\n--- Round {rr.round_number} Summary ---"]
        if rr.pathology_findings:
            round_parts.append(f"Pathology: {'; '.join(rr.pathology_findings[:3])}")
        if rr.bioinformatics_findings:
            round_parts.append(f"Bioinformatics: {'; '.join(rr.bioinformatics_findings[:3])}")
        if rr.biochemistry_findings:
            round_parts.append(f"Biochemistry: {'; '.join(rr.biochemistry_findings[:3])}")
        if rr.literature_findings:
            round_parts.append(f"Literature: {'; '.join(rr.literature_findings[:3])}")
        if rr.pi_summary:
            round_parts.append(f"PI Summary: {rr.pi_summary[:500]}")
        parts.extend(round_parts)

    # Final summary
    if state.final_summary:
        parts.append(f"\nFinal Summary: {state.final_summary[:1000]}")

    # Literature reports
    if state.literature_reports:
        parts.append(f"\nLiterature Reports: {len(state.literature_reports)} available")
        for lr in state.literature_reports[-2:]:
            parts.append(f"- Topic: {lr.topic}")

    # Scan disk for actual figure files
    figure_inventory: List[Dict[str, str]] = []
    seen_names: set = set()

    def _add_pngs(directory: Path):
        if not directory.exists():
            return
        for png in sorted(directory.glob("*.png")):
            if png.stem not in seen_names:
                seen_names.add(png.stem)
                figure_inventory.append({"name": png.stem, "path": str(png)})

    _add_pngs(WORKSPACE_DIR / state.id)
    _add_pngs(WORKSPACE_DIR / state.id / "figures")
    _add_pngs(REPORTS_DIR / state.id / "figures")

    if figure_inventory:
        parts.append(f"\nAvailable Figures ({len(figure_inventory)} on disk):")
        for fig in figure_inventory:
            parts.append(f"  - {fig['name']}")

    return "\n".join(parts), figure_inventory


async def _generate_figures_in_chat(
    session_id: str, state: AnalysisState, descriptions: List[str]
) -> List[Dict[str, str]]:
    """Generate figures via bioinformatics sandbox for chat responses."""
    from backend.utils.sandbox import LocalJupyterSandbox
    import time as _time

    generated_figures: List[Dict[str, str]] = []

    sandbox = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(STATE_DIR),
        working_dir=str(WORKSPACE_DIR),
    )
    _chat_sandboxes[session_id] = sandbox
    try:
        start_info = await sandbox.start()
        if not start_info.get("session_restored"):
            logger.warning("Chat sandbox: session not restored, figure generation may fail")

        for desc_idx, desc in enumerate(descriptions):
            # Ask LLM to write figure code
            code_prompt = (
                "You are a bioinformatics expert. Write Python code using scanpy and matplotlib "
                "to generate the following figure. The variable `adata` is already loaded in the "
                "environment with all prior analysis results.\n"
                "IMPORTANT: End your code with plt.show() to display the figure. "
                "Do NOT call plt.close(). Do NOT use plt.savefig().\n"
                "Output ONLY a single ```python code block, no explanations.\n\n"
                f"Figure request: {desc}"
            )

            llm_client = OpenAICompatibleClient(
                api_key=config.api.api_key,
                base_url=config.api.llm_base_url,
                proxy_url=config.api.proxy_url,
            )
            bio_model = state.custom_models.get("bioinformatics", "google/gemini-3.1-pro-preview")
            code_resp = await llm_client.chat(
                model=bio_model,
                messages=[{"role": "user", "content": code_prompt}],
                temperature=0.1,
                max_tokens=32768,
            )

            code_text = code_resp.content or ""
            code_match = re.search(r'```python\s*\n(.*?)```', code_text, re.DOTALL)
            code = code_match.group(1) if code_match else code_text
            if not code.strip():
                logger.warning(f"Chat sandbox: empty code for description '{desc[:80]}'")
                continue

            ts = _time.strftime("%Y%m%d_%H%M%S")
            fig_dir = WORKSPACE_DIR / session_id / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)

            # Wrap code: neutralize plt.show/plt.close, capture figures as PNG via IPython.display
            # Also save high-res PDF for download
            escaped_dir = str(fig_dir).replace("\\", "\\\\")
            wrapped = (
                "import matplotlib\n"
                "import matplotlib.pyplot as plt\n"
                "import io as _io\n"
                "from IPython.display import display as _display, Image as _Image\n"
                "# Neutralize plt.show() and plt.close() to keep figures alive\n"
                "_orig_show = plt.show\n"
                "_orig_close_func = plt.close\n"
                "plt.show = lambda *a, **kw: None\n"
                "plt.close = lambda *a, **kw: None\n"
                "_code_error = None\n"
                "try:\n"
                f"    {code.replace(chr(10), chr(10) + '    ')}\n"
                "except Exception as _e:\n"
                "    _code_error = str(_e)\n"
                "    print(f'CODE_ERROR: {_e}')\n"
                "    import traceback; traceback.print_exc()\n"
                "finally:\n"
                "    plt.show = _orig_show\n"
                "    plt.close = _orig_close_func\n"
                "# Capture all open figures as PNG + save PDF for download\n"
                "_fignums = plt.get_fignums()\n"
                "for _fn in _fignums:\n"
                "    _f = plt.figure(_fn)\n"
                "    # Save high-res PDF\n"
                f"    _f.savefig(r\"{escaped_dir}/chat_{ts}_\" + str(_fn) + \".pdf\", "
                "format='pdf', dpi=300, bbox_inches='tight', facecolor='white')\n"
                "    # Capture PNG for display\n"
                "    _buf = _io.BytesIO()\n"
                "    _f.savefig(_buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')\n"
                "    _buf.seek(0)\n"
                "    _display(_Image(data=_buf.getvalue()))\n"
                "plt.close('all')\n"
            )

            result = await sandbox.execute(wrapped, save_to_history=False)
            logger.info(f"Chat sandbox execute: success={result.success}, "
                        f"kernel_figures={len(result.figures)}, error={result.error}")
            if result.output:
                logger.info(f"Chat sandbox output: {result.output[:2000]}")

            # Primary: use kernel-captured base64 figures (most reliable)
            if result.figures:
                import base64 as _b64
                for i, b64_data in enumerate(result.figures):
                    fig_name = f"chat_{ts}_{i + 1}"
                    fig_path = fig_dir / f"{fig_name}.png"
                    pdf_path = fig_dir / f"{fig_name}.pdf"
                    try:
                        fig_path.write_bytes(_b64.b64decode(b64_data))
                        logger.info(f"Chat sandbox: saved figure to {fig_path}")
                        entry: Dict[str, str] = {
                            "name": fig_name,
                            "path": str(fig_path),
                            "base64": b64_data,
                        }
                        # PDF was saved by the kernel code directly
                        if pdf_path.exists():
                            entry["pdfPath"] = str(pdf_path)
                        generated_figures.append(entry)
                    except Exception as save_err:
                        logger.warning(f"Chat sandbox: disk save failed ({save_err}), using base64 only")
                        generated_figures.append({
                            "name": fig_name,
                            "base64": b64_data,
                        })
            else:
                logger.warning(f"Chat sandbox: no figures captured from kernel output "
                               f"(success={result.success})")

    except Exception as e:
        logger.error(f"Chat figure generation error: {e}")
        logger.error(traceback.format_exc())
    finally:
        _chat_sandboxes.pop(session_id, None)
        try:
            await sandbox.stop()
        except Exception:
            pass

    return generated_figures


# ============== Chat Delegation Helpers ==============

async def _send_chat_ws(session_id: str, msg_type: str, content: Any):
    """Send a WebSocket message for chat delegation events."""
    ws = websockets.get(session_id)
    if ws:
        try:
            await ws.send_json({
                "type": msg_type,
                "agent": None,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4()),
            })
        except Exception as e:
            logger.warning(f"Chat WS send failed for {session_id} ({msg_type}): {e}")
    else:
        logger.warning(f"Chat WS: no connection for {session_id}, message {msg_type} not delivered")


def _try_parse_delegation(content: str) -> Optional[Dict[str, Any]]:
    """Try to parse PI response as a delegation plan. Returns None if not delegation."""
    content = content.strip()
    # Direct JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("delegation") is True:
            delegates = parsed.get("delegates", [])
            if isinstance(delegates, list) and len(delegates) > 0:
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # JSON inside markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict) and parsed.get("delegation") is True:
                delegates = parsed.get("delegates", [])
                if isinstance(delegates, list) and len(delegates) > 0:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _build_synthesis_prompt(
    user_message: str,
    delegation: Dict[str, Any],
    agent_results: Dict[str, Dict[str, Any]],
) -> str:
    """Build PI synthesis prompt from delegation results."""
    parts = [
        "You are the PI Agent. Synthesize the following agent results into a clear, "
        "helpful response to the user's original request.",
        f"\n## User's Request\n{user_message}",
        f"\n## Delegation Plan\n{delegation.get('plan', '')}",
        "\n## Agent Results",
    ]
    for agent_name, result in agent_results.items():
        if "error" in result and not result.get("findings"):
            parts.append(f"\n### {agent_name.upper()} Agent: ERROR\n{result['error']}")
        else:
            # Extract the most useful content from agent result
            summary = result.get("synthesis", result.get("report", result.get("summary", "")))
            if isinstance(summary, dict):
                summary = json.dumps(summary, ensure_ascii=False, indent=2)
            findings = result.get("findings", [])
            if isinstance(findings, list) and findings:
                findings_text = "\n".join(f"- {f}" for f in findings[:20])
            else:
                findings_text = str(findings) if findings else ""
            agent_text = ""
            if summary:
                agent_text += f"Summary:\n{str(summary)[:6000]}\n"
            if findings_text:
                agent_text += f"Key Findings:\n{findings_text}\n"
            if not agent_text:
                agent_text = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            parts.append(f"\n### {agent_name.upper()} Agent\n{agent_text}")

    parts.append(
        "\n## Instructions\n"
        "- Synthesize these results into a comprehensive response\n"
        "- Reference specific findings from each agent\n"
        "- If figures were generated during the analysis, you can reference them with [[FIGURE:figure_name]]\n"
        "- Respond in the same language as the user's message\n"
    )
    if delegation.get("synthesis_instruction"):
        parts.append(f"Additional guidance: {delegation['synthesis_instruction']}")

    return "\n".join(parts)


def _collect_session_figures(
    session_id: str,
    since_time: Optional[float] = None
) -> List[Dict[str, str]]:
    """Collect figures from workspace, optionally filtering by creation time.

    Scans both the figures/ subdirectory (generate_publication_figure output)
    and the workspace root (execute_code explicit savefig output, e.g. UMAP plots).
    """
    figures: List[Dict[str, str]] = []
    seen_stems: set = set()

    def _add_png(png: Path) -> None:
        if since_time and png.stat().st_mtime < since_time:
            return
        if png.stem in seen_stems:
            return
        entry: Dict[str, str] = {"name": png.stem, "path": str(png)}
        pdf_path = png.with_suffix(".pdf")
        if pdf_path.exists():
            entry["pdfPath"] = str(pdf_path)
        figures.append(entry)
        seen_stems.add(png.stem)

    # 1. figures/ subdirectory (highest priority — publication figures)
    fig_dir = WORKSPACE_DIR / session_id / "figures"
    if fig_dir.exists():
        for png in sorted(fig_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
            _add_png(png)

    # 2. Workspace root — picks up UMAP and other plots saved via explicit plt.savefig()
    #    when LLM code calls plt.close() before the auto-save wrapper can capture them.
    root_dir = WORKSPACE_DIR / session_id
    if root_dir.exists():
        for png in sorted(root_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
            _add_png(png)

    figures.sort(key=lambda e: Path(e["path"]).stat().st_mtime, reverse=True)
    return figures[:30]


async def _execute_chat_agent(
    session_id: str,
    state: AnalysisState,
    agent_name: str,
    instruction: str,
    llm_client: OpenAICompatibleClient,
    message_callback,
    saved_files: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Execute a single agent in chat delegation context."""
    from .agents.base import AgentConfig
    from .utils.sandbox import LocalJupyterSandbox

    async def _send_agent_msg(msg_type: str, content: Any, agent_type: AgentType):
        msg = WebSocketMessage(type=msg_type, agent=agent_type, content=content)
        await message_callback(msg)

    agent_type = AgentType(agent_name)
    await _send_agent_msg("status", f"Starting {agent_name} analysis...", agent_type)

    if agent_name == "bioinformatics":
        from .agents.bioinformatics_agent import BioinformaticsAgent

        # Create sandbox with session restore
        sandbox = LocalJupyterSandbox(
            session_id=session_id,
            state_dir=str(STATE_DIR),
            working_dir=str(WORKSPACE_DIR),
            timeout=config.sandbox_timeout if hasattr(config, 'sandbox_timeout') else 120
        )
        _chat_sandboxes[session_id] = sandbox

        # Create literature agent for callbacks
        lit_model = state.custom_models.get("literature", config.models.literature_model)
        lit_prompt = state.custom_prompts.get("literature", config.prompts.literature_prompt)
        lit_config = state.literature_config or {}
        pi_model = state.custom_models.get("pi", config.models.pi_model)

        from .agents.literature_agent import LiteratureAgent
        literature_agent = LiteratureAgent(
            llm_client=llm_client,
            model=lit_model,
            system_prompt=lit_prompt,
            config=AgentConfig(temperature=state.custom_temperatures.get("literature", 0.6)),
            literature_mode=lit_config.get("mode", "deep_research"),
            deep_research_model=lit_config.get("deep_research_model", ""),
            google_api_key=lit_config.get("google_api_key", ""),
            pubmed_email=lit_config.get("pubmed_email", ""),
            max_search_rounds=lit_config.get("max_search_rounds", 5),
            proxy_url=config.api.proxy_url if config.api else None,
            fallback_model=pi_model,
        )

        async def lit_callback(topic: str) -> str:
            result = await literature_agent.process(state, topic, use_deep_research=True)
            return result.get("report", "")

        async def pi_callback(question: str, context: str) -> str:
            await _send_agent_msg("pi_question", f"Question: {question}\nContext: {context}", AgentType.BIOINFORMATICS)
            pi_model = state.custom_models.get("pi", config.models.pi_model)
            prompt = (
                f"The bioinformatics analyst is asking for your guidance:\n\n"
                f"QUESTION: {question}\n\nCONTEXT: {context}\n\n"
                f"Please provide clear, actionable guidance."
            )
            resp = await llm_client.chat(
                model=pi_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=state.custom_temperatures.get("pi", 0.5),
                max_tokens=2048
            )
            answer = resp.content or "Proceed with standard bioinformatics practices."
            await _send_agent_msg("pi_response", answer, AgentType.PI)
            return answer

        async def annotate_callback(cluster_markers, tissue, organism):
            await _send_agent_msg("status", f"Annotating {len(cluster_markers)} clusters via literature...", AgentType.LITERATURE)
            return await literature_agent.annotate_clusters(cluster_markers, tissue, organism, state)

        async def agent_msg_callback(msg_type: str, content: str):
            await _send_agent_msg(msg_type, content, AgentType.BIOINFORMATICS)

        bio_agent = BioinformaticsAgent(
            llm_client=llm_client,
            model=state.custom_models.get("bioinformatics", config.models.bioinformatics_model),
            system_prompt=state.custom_prompts.get("bioinformatics", config.prompts.bioinformatics_prompt),
            sandbox=sandbox,
            literature_callback=lit_callback,
            pi_callback=pi_callback,
            annotate_clusters_callback=annotate_callback,
            message_callback=agent_msg_callback,
            config=AgentConfig(temperature=state.custom_temperatures.get("bioinformatics", 0.1)),
        )

        try:
            analysis_start_time = time.time()
            await sandbox.start()
            result = await bio_agent.run_autonomous_analysis(
                state, instruction, max_steps=30
            )
            # Collect only figures generated during THIS analysis (not historical ones)
            result["figures"] = _collect_session_figures(session_id, since_time=analysis_start_time)
            return result
        finally:
            _chat_sandboxes.pop(session_id, None)
            try:
                await sandbox.stop()
            except Exception:
                pass

    elif agent_name == "pathology":
        from .agents.pathology_agent import PathologyAgent

        pathology_agent = PathologyAgent(
            llm_client=llm_client,
            model=state.custom_models.get("pathology", config.models.pathology_model),
            system_prompt=state.custom_prompts.get("pathology", config.prompts.pathology_prompt),
            tile_size=config.image_tile_size if hasattr(config, 'image_tile_size') else 1024,
            max_tiles=50,
            config=AgentConfig(temperature=state.custom_temperatures.get("pathology", 0.2)),
        )
        result = await pathology_agent.process(state, instruction)
        return result

    elif agent_name == "biochemistry":
        from .agents.biochemistry_agent import BiochemistryAgent

        biochem_agent = BiochemistryAgent(
            llm_client=llm_client,
            model=state.custom_models.get("biochemistry", config.models.biochemistry_model),
            system_prompt=state.custom_prompts.get("biochemistry", config.prompts.biochemistry_prompt),
            config=AgentConfig(temperature=state.custom_temperatures.get("biochemistry", 0.2)),
        )
        result = await biochem_agent.process(state, instruction)
        return result

    elif agent_name == "literature":
        from .agents.literature_agent import LiteratureAgent
        lit_config = state.literature_config or {}
        pi_model = state.custom_models.get("pi", config.models.pi_model)

        lit_agent = LiteratureAgent(
            llm_client=llm_client,
            model=state.custom_models.get("literature", config.models.literature_model),
            system_prompt=state.custom_prompts.get("literature", config.prompts.literature_prompt),
            config=AgentConfig(temperature=state.custom_temperatures.get("literature", 0.6)),
            literature_mode=lit_config.get("mode", "deep_research"),
            deep_research_model=lit_config.get("deep_research_model", ""),
            google_api_key=lit_config.get("google_api_key", ""),
            pubmed_email=lit_config.get("pubmed_email", ""),
            max_search_rounds=lit_config.get("max_search_rounds", 5),
            proxy_url=config.api.proxy_url if config.api else None,
            fallback_model=pi_model,
        )
        result = await lit_agent.process(state, instruction)
        return result

    else:
        raise ValueError(f"Unknown agent for chat delegation: {agent_name}")


async def _run_chat_delegation(
    session_id: str,
    state: AnalysisState,
    delegation: Dict[str, Any],
    user_message: str,
    image_b64_list: List[str],
    saved_files: List[Dict[str, str]],
):
    """Background task: orchestrate delegated agent work, then PI synthesizes results."""
    llm_manager = None
    start_time = time.time()
    try:
        # 1. Send started notification
        delegates = delegation.get("delegates", [])
        agent_names = [d.get("agent", "unknown") for d in delegates]
        await _send_chat_ws(session_id, "chat_delegation", {
            "status": "started",
            "plan": delegation.get("plan", ""),
            "agents": agent_names,
        })

        # 2. Create shared LLM client
        llm_manager = LLMManager(
            api_key=config.api.api_key,
            base_url=config.api.llm_base_url,
            proxy_url=config.api.proxy_url,
            session_id=session_id,
            agent_name="chat_delegation"
        )

        # 3. Create message callback that sends to WebSocket + persists
        async def chat_msg_callback(msg: WebSocketMessage):
            ws = websockets.get(session_id)
            if ws:
                try:
                    await ws.send_json({
                        "type": msg.type,
                        "agent": msg.agent.value if msg.agent else None,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "id": msg.id,
                    })
                except Exception:
                    pass
            # Persist to realtime_messages
            if hasattr(state, 'realtime_messages'):
                state.realtime_messages.append(msg)

        # 4. Check for sandbox conflicts with running workflow
        workflow_running = session_id in workflows
        if workflow_running:
            # Filter out bioinformatics to avoid sandbox conflict
            delegates = [d for d in delegates if d.get("agent") != "bioinformatics"]
            if not delegates:
                await _send_chat_ws(session_id, "chat_error", {
                    "error": "Cannot run bioinformatics analysis while main workflow is active. "
                             "Please wait for the workflow to complete."
                })
                return

        # 5. Group by priority and execute
        priority_groups: Dict[int, List[Dict]] = {}
        for d in delegates:
            p = d.get("priority", 1)
            priority_groups.setdefault(p, []).append(d)

        agent_results: Dict[str, Dict[str, Any]] = {}
        all_figures: List[Dict[str, str]] = []
        figure_timestamp = time.time()  # Track when delegation started for figure collection

        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]

            if len(group) == 1:
                # Single agent — execute directly
                d = group[0]
                agent_name = d.get("agent", "unknown")
                try:
                    result = await _execute_chat_agent(
                        session_id, state, agent_name, d.get("instruction", ""),
                        llm_manager.openrouter, chat_msg_callback, saved_files
                    )
                    agent_results[agent_name] = result
                    if "figures" in result:
                        all_figures.extend(result["figures"])
                    await _send_chat_ws(session_id, "chat_delegation", {
                        "status": "agent_complete", "agent": agent_name,
                    })
                except Exception as e:
                    logger.error(f"Chat delegation agent {agent_name} failed: {traceback.format_exc()}")
                    agent_results[agent_name] = {"error": str(e), "findings": []}
                    await _send_chat_ws(session_id, "chat_delegation", {
                        "status": "agent_error", "agent": agent_name, "error": str(e),
                    })
            else:
                # Multiple agents at same priority — execute in parallel
                tasks = []
                for d in group:
                    tasks.append(_execute_chat_agent(
                        session_id, state, d.get("agent", "unknown"), d.get("instruction", ""),
                        llm_manager.openrouter, chat_msg_callback, saved_files
                    ))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for d, result in zip(group, results):
                    agent_name = d.get("agent", "unknown")
                    if isinstance(result, Exception):
                        logger.error(f"Chat delegation agent {agent_name} failed: {result}")
                        agent_results[agent_name] = {"error": str(result), "findings": []}
                        await _send_chat_ws(session_id, "chat_delegation", {
                            "status": "agent_error", "agent": agent_name, "error": str(result),
                        })
                    else:
                        agent_results[agent_name] = result
                        if "figures" in result:
                            all_figures.extend(result["figures"])
                        await _send_chat_ws(session_id, "chat_delegation", {
                            "status": "agent_complete", "agent": agent_name,
                        })

        # 6. PI Synthesis
        await _send_chat_ws(session_id, "chat_delegation", {
            "status": "synthesizing", "message": "PI synthesizing results...",
        })

        synthesis_prompt = _build_synthesis_prompt(user_message, delegation, agent_results)
        pi_model = state.custom_models.get("pi", config.models.pi_model)
        pi_temp = state.custom_temperatures.get("pi", 0.5)

        pi_response = await llm_manager.openrouter.chat(
            model=pi_model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=pi_temp,
            max_tokens=32768,
        )
        final_content = pi_response.content or "Analysis completed but synthesis failed."

        # 7. Post-process: FIGURE and GENERATE_FIGURE markers (reuse existing patterns)
        context, figure_inventory = _build_chat_context(state)
        figure_pattern = re.compile(r'\[\[FIGURE:([\w.-]+)\]\]')
        referenced_names = figure_pattern.findall(final_content)
        synthesis_figures: List[Dict[str, str]] = []

        if referenced_names and figure_inventory:
            fig_lookup = {f["name"]: f for f in figure_inventory}
            for name in dict.fromkeys(referenced_names):
                fig = fig_lookup.get(name)
                if fig:
                    entry = {"name": fig["name"], "path": fig["path"]}
                    pdf_path = Path(fig["path"]).with_suffix(".pdf")
                    if pdf_path.exists():
                        entry["pdfPath"] = str(pdf_path)
                    synthesis_figures.append(entry)

        generate_pattern = re.compile(r'\[\[GENERATE_FIGURE:(.*?)\]\]', re.DOTALL)
        generate_requests = generate_pattern.findall(final_content)
        if generate_requests:
            try:
                generated = await _generate_figures_in_chat(session_id, state, generate_requests)
                synthesis_figures.extend(generated)
            except Exception as e:
                logger.error(f"Chat delegation figure generation failed: {e}")

        # Merge: all_figures from agents + synthesis_figures
        # Deduplicate by name
        seen_names = set()
        merged_figures = []
        for fig in all_figures + synthesis_figures:
            if fig["name"] not in seen_names:
                seen_names.add(fig["name"])
                merged_figures.append(fig)

        # Clean markers from text
        cleaned_content = figure_pattern.sub('', final_content)
        cleaned_content = generate_pattern.sub('', cleaned_content).strip()
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

        # 8. Build final chat message
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": cleaned_content,
            "timestamp": datetime.now().isoformat(),
        }
        if merged_figures:
            assistant_msg["figures"] = merged_figures

        # Replace placeholder message in chat_messages
        replaced = False
        for i in range(len(state.chat_messages) - 1, -1, -1):
            if state.chat_messages[i].get("delegation_active"):
                state.chat_messages[i] = assistant_msg
                replaced = True
                break
        if not replaced:
            state.chat_messages.append(assistant_msg)

        # 9. Send chat_result via WebSocket
        await _send_chat_ws(session_id, "chat_result", {
            "content": cleaned_content,
            "figures": merged_figures,
            "timestamp": datetime.now().isoformat(),
        })

        # 10. Persist state
        sessions[session_id] = state
        await asyncio.to_thread(save_session_to_disk, session_id, state)

        duration = time.time() - start_time
        logger.info(f"Chat delegation completed for {session_id} in {duration:.1f}s, "
                     f"{len(agent_results)} agents, {len(merged_figures)} figures")

    except asyncio.CancelledError:
        logger.info(f"Chat delegation cancelled by user for {session_id}")
        for i in range(len(state.chat_messages) - 1, -1, -1):
            if state.chat_messages[i].get("delegation_active"):
                state.chat_messages[i] = {
                    "role": "assistant",
                    "content": "已取消",
                    "timestamp": datetime.now().isoformat(),
                }
                break
        try:
            await _send_chat_ws(session_id, "chat_cancelled", {"message": "Cancelled by user"})
            sessions[session_id] = state
            await asyncio.to_thread(save_session_to_disk, session_id, state)
        except Exception:
            pass
        raise

    except Exception as e:
        logger.error(f"Chat delegation error for {session_id}: {traceback.format_exc()}")
        error_msg = f"Analysis failed: {str(e)}"
        # Update placeholder message
        for i in range(len(state.chat_messages) - 1, -1, -1):
            if state.chat_messages[i].get("delegation_active"):
                state.chat_messages[i] = {
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat(),
                }
                break
        await _send_chat_ws(session_id, "chat_error", {"error": error_msg})
        sessions[session_id] = state
        await asyncio.to_thread(save_session_to_disk, session_id, state)

    finally:
        async with _chat_task_lock:
            _chat_tasks.pop(session_id, None)
        _chat_sandboxes.pop(session_id, None)
        if llm_manager:
            try:
                await llm_manager.close()
            except Exception:
                pass


@app.post("/api/session/{session_id}/chat")
async def chat_with_pi(
    session_id: str,
    message: str = Form(...),
    target_agents: str = Form(default="[]"),
    file_groups: str = Form(default="[]"),
    file_samples: str = Form(default="[]"),
    file_types: str = Form(default="[]"),
    files: List[UploadFile] = File(default=[]),
):
    """Chat with PI Agent using analysis context. Supports file uploads with per-file agent targeting."""
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if config is None:
        raise HTTPException(status_code=400, detail="API configuration not set")

    # Parse per-file metadata arrays
    def _parse_json_list(raw: str) -> list:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    per_file_agents = _parse_json_list(target_agents)
    per_file_groups = _parse_json_list(file_groups)
    per_file_samples = _parse_json_list(file_samples)
    per_file_types = _parse_json_list(file_types)

    # --- Save uploaded files ---
    import base64
    saved_files: List[Dict[str, str]] = []
    image_b64_list: List[str] = []
    data_file_contents: List[str] = []
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}

    chat_upload_dir = UPLOAD_DIR / session_id / "chat"
    chat_upload_dir.mkdir(parents=True, exist_ok=True)

    for idx, upload_file in enumerate(files):
        fname = safe_filename(upload_file.filename or "file")
        if not fname:
            continue
        dest = chat_upload_dir / fname
        # Deduplicate filenames
        counter = 1
        while dest.exists():
            stem = Path(fname).stem
            suffix = Path(fname).suffix
            dest = chat_upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        await stream_save_file(upload_file, dest)
        ext = dest.suffix.lower()
        is_image = ext in IMAGE_EXTS
        file_type = "image" if is_image else "file"
        file_agent = per_file_agents[idx] if idx < len(per_file_agents) else "pi"

        file_group = per_file_groups[idx] if idx < len(per_file_groups) else None
        file_sample = per_file_samples[idx] if idx < len(per_file_samples) else None
        file_dtype = per_file_types[idx] if idx < len(per_file_types) else None

        saved_files.append({
            "filename": dest.name,
            "type": file_dtype or file_type,
            "path": str(dest),
            "target_agent": file_agent,
            "group_id": file_group,
            "sample_number": file_sample,
        })

        # Load image as base64 for vision API
        if is_image:
            with open(dest, "rb") as f:
                image_b64_list.append(base64.b64encode(f.read()).decode("utf-8"))

        # Read data file content (first 5000 chars) for text inclusion
        if not is_image and ext in {'.csv', '.tsv', '.txt', '.json'}:
            try:
                with open(dest, "r", encoding="utf-8", errors="replace") as f:
                    content_preview = f.read(5000)
                data_file_contents.append(f"--- {dest.name} ---\n{content_preview}")
            except Exception:
                pass

    # --- Register files to target agent state (per-file) ---
    _agent_section_map = {"pathology": "pathology", "bioinformatics": "sequencing", "biochemistry": "biochemistry"}
    for sf in saved_files:
        agent = sf["target_agent"]
        section = _agent_section_map.get(agent)
        if section:
            state.data_files.append(DataFile(
                filename=sf["filename"],
                file_path=sf["path"],
                section=section,
                data_type=sf.get("type") or "uploaded_via_chat",
                group_id=sf.get("group_id"),
                sample_number=sf.get("sample_number"),
                description="Uploaded via chat",
            ))

    # --- Build context & prompt ---
    context, figure_inventory = _build_chat_context(state)
    pi_model = state.custom_models.get("pi", "google/gemini-2.5-pro-preview-05-06")
    pi_temp = state.custom_temperatures.get("pi", 0.5)

    system_prompt = (
        "You are the PI (Principal Investigator) Agent of a biomedical research analysis system. "
        "You have access to the current analysis state and can help the user understand results, "
        "answer questions about the data, and provide scientific insights.\n\n"
        "Current Analysis Context:\n" + context + "\n\n"
        "Instructions:\n"
        "- Answer the user's questions based on available analysis results\n"
        "- If relevant data exists in the analysis, reference it specifically\n"
        "- Be concise but thorough in scientific explanations\n"
        "- If you don't have enough information, say so clearly\n"
        "- You can suggest next steps or additional analyses if appropriate\n"
        "\n"
        "FIGURE DISPLAY RULES (CRITICAL — you MUST follow these):\n"
        "You have a built-in figure rendering system. You MUST use it. "
        "NEVER provide Python/R code for the user to run. NEVER tell the user to 'run this code'. "
        "The user CANNOT execute code — only you can display figures through these commands:\n\n"
        "1. To show an EXISTING figure from the analysis, write on its own line:\n"
        "   [[FIGURE:figure_name]]\n"
        "   Use exact names from the 'Available Figures' list in the context above.\n"
        "   Example: [[FIGURE:umap_cell_types]]\n\n"
        "2. To CREATE a NEW figure that doesn't exist, write on its own line:\n"
        "   [[GENERATE_FIGURE:detailed description]]\n"
        "   The system will automatically generate and display the figure.\n"
        "   Include specific details: what plot type, which genes/clusters, colors, labels, etc.\n"
        "   Example: [[GENERATE_FIGURE:UMAP plot colored by cell_type annotation with legend on the right margin, "
        "using tab20 palette, title 'Cell Type Clusters', frameon=False]]\n\n"
        "When the user asks to see, show, display, generate, or create any figure/plot/chart/image, "
        "you MUST use [[FIGURE:...]] or [[GENERATE_FIGURE:...]] — NEVER give code.\n"
        "\n"
        "- Respond in the same language as the user's message\n"
        "\n"
        "DELEGATION CAPABILITY:\n"
        "When the user's request requires running new computational analysis, processing new data, "
        "or using specialized agent capabilities beyond your own knowledge, you can delegate tasks "
        "to other agents. Available agents:\n"
        "- bioinformatics: Code execution, new plots, scRNA-seq analysis, clustering, DE analysis, trajectory\n"
        "- pathology: Pathology/histology image analysis or re-examination\n"
        "- biochemistry: Biochemistry data interpretation (Western blot, ELISA, etc.)\n"
        "- literature: Literature search, evidence gathering, Deep Research\n\n"
        "When delegation is needed, respond with ONLY a JSON block (no other text):\n"
        '{"delegation": true, "plan": "Brief description of what you will coordinate", "delegates": [\n'
        '  {"agent": "bioinformatics", "instruction": "Detailed task instruction with data paths and specifics", "priority": 1}\n'
        '], "synthesis_instruction": "How to summarize results for user"}\n\n'
        "Delegation rules:\n"
        "- Same priority number = parallel execution; lower number executes first\n"
        "- For simple questions answerable from existing results, respond normally (NO delegation)\n"
        "- For showing existing figures, use [[FIGURE:...]] (NO delegation)\n"
        "- Only delegate when NEW analysis, computation, or specialized processing is genuinely needed\n"
        "- Include specific data file paths and analysis requirements in the instruction field\n"
        "- The bioinformatics agent's notebook ALREADY has all previous analysis results loaded (adata with "
        "cell_type, clustering, PCA, UMAP, DE results, etc.). Do NOT include prerequisite steps like "
        "'first ensure cell annotations are available' or 'run clustering first' in the instruction. "
        "Focus ONLY on the NEW analysis the user is asking for. The agent will check existing results automatically.\n"
    )

    # Build messages: system + history + new message
    chat_messages_list = [{"role": "system", "content": system_prompt}]
    for msg in state.chat_messages[-20:]:
        chat_messages_list.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Build user content with file info
    user_content = message
    if saved_files:
        # Group files by target agent for readable description
        agent_files: Dict[str, List[str]] = {}
        for sf in saved_files:
            agent_files.setdefault(sf["target_agent"], []).append(sf["filename"])
        file_parts = []
        for agent, fnames in agent_files.items():
            label = agent.upper() if agent != "pi" else "PI"
            file_parts.append(f"{label} Agent: {', '.join(fnames)}")
        user_content += f"\n\n[Attached {len(saved_files)} file(s) — {'; '.join(file_parts)}]"
    if data_file_contents:
        user_content += "\n\n" + "\n\n".join(data_file_contents)

    chat_messages_list.append({"role": "user", "content": user_content})

    # Store user message
    state.chat_messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat(),
        "files": saved_files,
    })

    # --- Call LLM ---
    try:
        llm_client = OpenAICompatibleClient(
            api_key=config.api.api_key,
            base_url=config.api.llm_base_url,
            proxy_url=config.api.proxy_url
        )

        if image_b64_list:
            # Vision API for images
            full_prompt = system_prompt + "\n\nChat History:\n"
            for msg in state.chat_messages[-21:-1]:  # exclude the just-added user msg
                full_prompt += f"\n{msg.get('role', 'user')}: {msg.get('content', '')}"
            full_prompt += f"\n\nuser: {user_content}"

            response = await llm_client.chat_with_vision(
                model=pi_model,
                text_prompt=full_prompt,
                images=image_b64_list,
                temperature=pi_temp,
                max_tokens=32768
            )
        else:
            response = await llm_client.chat(
                model=pi_model,
                messages=chat_messages_list,
                temperature=pi_temp,
                max_tokens=32768
            )

        assistant_content = response.content if response.content else "I'm sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        logger.error(f"Chat LLM error: {e}")
        logger.error(traceback.format_exc())
        assistant_content = f"Error communicating with LLM: {str(e)}"

    # --- Check for delegation ---
    delegation = _try_parse_delegation(assistant_content)
    if delegation:
        logger.info(f"Chat delegation detected for {session_id}: {delegation.get('plan', '')}")

        # Concurrency guard
        async with _chat_task_lock:
            if session_id in _chat_tasks and not _chat_tasks[session_id].done():
                raise HTTPException(status_code=409, detail="Chat analysis already in progress for this session")

        # Add placeholder assistant message
        state.chat_messages.append({
            "role": "assistant",
            "content": f"Working on it: {delegation.get('plan', 'Coordinating agents...')}",
            "timestamp": datetime.now().isoformat(),
            "delegation_active": True,
        })
        await asyncio.to_thread(save_session_to_disk, session_id, state)

        # Launch background task
        task = asyncio.create_task(
            _run_chat_delegation(
                session_id, state, delegation, user_content,
                image_b64_list, saved_files
            )
        )
        async with _chat_task_lock:
            _chat_tasks[session_id] = task

        return {"status": "processing", "plan": delegation.get("plan", "")}

    # --- Post-process: extract figure references ---
    figure_pattern = re.compile(r'\[\[FIGURE:([\w.-]+)\]\]')
    referenced_names = figure_pattern.findall(assistant_content)
    logger.info(f"Chat post-process: found {len(referenced_names)} FIGURE refs, {len(figure_inventory)} in inventory")

    response_figures: List[Dict[str, str]] = []
    if referenced_names and figure_inventory:
        fig_lookup = {f["name"]: f for f in figure_inventory}
        for name in dict.fromkeys(referenced_names):  # deduplicate, preserve order
            fig = fig_lookup.get(name)
            if fig:
                entry = {"name": fig["name"], "path": fig["path"]}
                # Check for corresponding PDF
                pdf_path = Path(fig["path"]).with_suffix(".pdf")
                if pdf_path.exists():
                    entry["pdfPath"] = str(pdf_path)
                response_figures.append(entry)
        logger.info(f"Chat post-process: matched {len(response_figures)} existing figures")

    # --- Post-process: extract GENERATE_FIGURE requests ---
    generate_pattern = re.compile(r'\[\[GENERATE_FIGURE:(.*?)\]\]', re.DOTALL)
    generate_requests = generate_pattern.findall(assistant_content)
    logger.info(f"Chat post-process: found {len(generate_requests)} GENERATE_FIGURE requests")

    # Clean markers from text
    cleaned_content = figure_pattern.sub('', assistant_content)
    cleaned_content = generate_pattern.sub('', cleaned_content).strip()
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

    # Save text response IMMEDIATELY (before figure generation which can take minutes)
    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": cleaned_content,
        "timestamp": datetime.now().isoformat(),
    }
    if response_figures:
        assistant_msg["figures"] = list(response_figures)
    state.chat_messages.append(assistant_msg)
    await asyncio.to_thread(save_session_to_disk, session_id, state)
    logger.info(f"Chat response: text saved ({len(cleaned_content)} chars, {len(response_figures)} existing figures)")

    # Now try to generate new figures (may take long, but text is already saved)
    if generate_requests:
        try:
            generated = await _generate_figures_in_chat(session_id, state, generate_requests)
            logger.info(f"Chat post-process: generated {len(generated)} figures")
            if generated:
                existing = assistant_msg.get("figures", [])
                assistant_msg["figures"] = existing + generated
                response_figures.extend(generated)
                await asyncio.to_thread(save_session_to_disk, session_id, state)
        except Exception as e:
            logger.error(f"Chat figure generation failed: {e}")
            logger.error(traceback.format_exc())

    result: Dict[str, Any] = {"role": "assistant", "content": cleaned_content}
    if assistant_msg.get("figures"):
        result["figures"] = assistant_msg["figures"]
    return result


@app.post("/api/session/{session_id}/chat/cancel")
async def cancel_chat(session_id: str):
    """Cancel an in-progress chat request (delegation or simple)."""
    validate_session_id(session_id)

    cancelled = False

    # 1. Cancel delegation task if running
    async with _chat_task_lock:
        task = _chat_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            cancelled = True

    # 2. Interrupt sandbox kernel if running
    sandbox = _chat_sandboxes.get(session_id)
    if sandbox:
        try:
            if sandbox.km:
                sandbox.km.interrupt_kernel()
                logger.info(f"Chat cancel: interrupted sandbox kernel for {session_id}")
        except Exception as e:
            logger.warning(f"Chat cancel: sandbox interrupt failed for {session_id}: {e}")

    return {"status": "cancelled", "had_delegation": cancelled}


@app.get("/api/session/{session_id}/chat")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"messages": state.chat_messages}
