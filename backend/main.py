"""
FastAPI backend server for the Multi-Agent Single-Cell Analysis System
"""
import os
import json
import asyncio
import logging
import time
from contextlib import asynccontextmanager
import traceback
import zipfile
import tarfile
import tempfile
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, Tuple
import uuid
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

from .config import (
    SystemConfig, AgentTemperatures,
    DATA_BASE_DIR, UPLOAD_DIR, STATE_DIR, WORKSPACE_DIR, SESSION_DIR, REPORTS_DIR, LOG_DIR,
    UPLOAD_CHUNK_SIZE, MAX_UPLOAD_SIZE, UPLOAD_TEMP_DIR,
    MAX_CACHED_SESSIONS, MAX_REALTIME_MESSAGES, SESSION_CACHE_TTL,
    IMPORT_JOB_TTL, CLEANUP_INTERVAL,
    WS_RECEIVE_TIMEOUT, CHAT_DELEGATION_TIMEOUT,
    setup_logging,
    get_session_dir, get_session_metadata_path, resolve_session_metadata_path, get_session_workspace,
    get_session_states, get_session_uploads, get_session_reports, ensure_session_dirs,
)
from .models.schemas import (
    AnalysisState, AnalysisPhase, APIRequest, APIResponse, TaskStatus,
    WebSocketMessage, AgentType, TreatmentGroup, DataFile, BiochemistryIndicator,
    DATA_TYPE_SCRNASEQ, DATA_TYPE_BULK_RNASEQ,
)
from .workflow import SingleCellAnalysisWorkflow
from .utils.llm_client import OpenAICompatibleClient, LLMManager
from .utils.upload import ChunkedUploadManager, stream_save_file
from .routes.auth import router as auth_router
from .routes.setup import router as setup_router
from .routes.v1 import router as v1_router
from .routes.root import router as root_router
from .routes.config import router as config_router
from .routes.sessions import router as sessions_router
from .routes.session_config import router as session_config_router
from .routes.indicators import router as indicators_router
from .routes.upload import router as upload_router
from .routes.lifecycle import router as lifecycle_router
from .routes.interaction import router as interaction_router
from .routes.snapshots import router as snapshots_router
from .routes.export_import import router as export_import_router
from .routes.files import router as files_router
from .routes.chat import router as chat_router
from .routes.websockets import router as websockets_router
from .routes.reports import router as reports_router
from .routes.admin import router as admin_router
from .routes.user_config import router as user_config_router
from .middleware.auth import AuthMiddleware
from .utils.auth import get_db, require_auth, require_session_ownership, ws_require_auth, ws_authenticate_and_load, WS_AUTH_FAILED_CODE
from .utils.files import safe_filename, validate_session_id, validate_path_within
from .state import (
    sessions, workflows, _run_tasks,
    state_lock,
    initializing_sessions,
    session_access_times,
    last_save_times,
    get_session_lock,
    websockets,
    import_jobs,
    import_jobs_lock,
    prepared_exports,
    prepared_exports_lock,
    chat_tasks, chat_task_lock, chat_sandboxes,
    _monitor_tasks, report_websockets, detailed_reports,
)
from .services.session_store import (
    save_session_to_disk, load_session_from_disk, list_saved_sessions,
    async_save_session, ensure_session_loaded, touch_session,
    update_session_cache as _update_session_cache,
    pause_and_restart_ws as _pause_and_restart_ws_impl,
)

from .services.upload import upload_manager

from .utils.agent_llm import (
    validate_file_path_ownership as _validate_file_path_ownership,
    get_agent_llm_client as _get_agent_llm_client,
)

# Configure logging - must be called before any logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    from backend.utils.auth import get_db
    db = await get_db()
    logger.info(f"Auth database initialized at {db.path}")

    # Restore persisted global system config (survives server restarts)
    try:
        saved_cfg_json = await db.get_meta("global_system_config")
        if saved_cfg_json:
            from backend.config._config import SystemConfig
            from backend.state import system_config
            system_config["current"] = SystemConfig.model_validate_json(saved_cfg_json)
            logger.info("Restored global system config from database")
    except Exception as e:
        logger.warning("Failed to restore global system config: %s; using defaults", e)

    from backend.services.upload import set_session_callbacks
    set_session_callbacks(_update_session_cache, async_save_session)

    from backend.tasks.session_cleanup import cleanup_stale_sessions
    cleanup_task = asyncio.create_task(
        cleanup_stale_sessions(
            chat_task_lock=chat_task_lock,
            chat_sandboxes=chat_sandboxes,
            import_jobs=import_jobs,
            import_jobs_lock=import_jobs_lock,
        )
    )
    from backend.tasks.ttl_cleanup import ttl_cleanup_task
    ttl_task = asyncio.create_task(ttl_cleanup_task())
    logger.info("Background tasks started")
    try:
        yield
    finally:
        cleanup_task.cancel()
        ttl_task.cancel()


# FastAPI app
app = FastAPI(
    title="ScAI - Single Cell Sequencing Analysis Driven by AI",
    description="Integrated & Intelligent Single Cell Analysis System with pathology, bioinformatics, and biochemistry analysis",
    version="0.24.10",
    lifespan=lifespan,
)

# Mount auth router
app.include_router(auth_router)
app.include_router(setup_router)
app.include_router(v1_router)
app.include_router(root_router)
app.include_router(config_router)
app.include_router(sessions_router)
app.include_router(session_config_router)
app.include_router(indicators_router)
app.include_router(upload_router)
app.include_router(lifecycle_router)
app.include_router(interaction_router)
app.include_router(snapshots_router)
app.include_router(export_import_router)
app.include_router(files_router)
app.include_router(chat_router)
app.include_router(websockets_router)
app.include_router(reports_router)
app.include_router(admin_router)
app.include_router(user_config_router)

# CORS
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state with lock for thread-safe access
# Note: websockets are in backend.state; config is accessed via backend.state.system_config


# Log the data directory location (directories are created in config.py)
logger.info(f"Data directory: {DATA_BASE_DIR.resolve()}")

from .models.api_requests import (
    ConfigRequest, ValidateKeyRequest, StartRequest, VALID_AGENTS,
    PromptUpdateRequest, ModelUpdateRequest, TemperatureUpdateRequest,
    VisionSettingUpdateRequest, LiteratureConfigRequest,
    AgentServerConfigRequest, BiochemistryInput, SessionUpdateRequest,
    GroupCreateRequest, GroupUpdateRequest, StopAnalysisRequest,
    HumanInputRequest, AnnotationConfirmation,
)





# Run with: uvicorn backend.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


