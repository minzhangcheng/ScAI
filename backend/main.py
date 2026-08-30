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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
from .constants import VERSION
from .workflow import SingleCellAnalysisWorkflow
from .utils.llm_client import OpenAICompatibleClient, LLMManager
from .utils.upload import ChunkedUploadManager, stream_save_file
from .routes.auth import router as auth_router
from .routes.setup import router as setup_router
from .routes.v1 import router as v1_router
from .routes.root import frontend_file_response, router as root_router
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
from .middleware.upload_size import (
    MULTIPART_ENVELOPE_ALLOWANCE,
    UploadSizeLimitMiddleware,
)
from .utils.auth import (
    WS_AUTH_FAILED_CODE,
    get_db,
    require_auth,
    require_session_ownership,
    session_not_found_detail,
    ws_authenticate_and_load,
    ws_require_auth,
)
from .utils.files import safe_filename, validate_session_id, validate_path_within
from .services.report.output_generation import ReportGenerationError
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
from .services.session_lifecycle import (
    SessionGenerationError,
    SessionGoneError,
    SessionOwnerMismatchError,
)

from .utils.agent_llm import (
    validate_file_path_ownership as _validate_file_path_ownership,
    get_agent_llm_client as _get_agent_llm_client,
)

logger = logging.getLogger(__name__)


def _warn_about_unrecovered_journals() -> None:
    """Say so when startup is stepping over restore journals it will not touch.

    A crash during the APPLYING stage of a restore leaves the session's files
    half-swapped and a journal describing how to finish or roll back. With the
    subsystem disabled that journal is never read, so the session loads and runs
    against a mix of old and new files without anything being said. Recovery
    genuinely cannot run here — it needs the object store — but the operator
    should learn about it from a log line rather than from the results.
    """
    from backend.config._config import SESSION_DIR
    from backend.services.snapshot_v2.feature_gate import (
        has_unfinished_restore_journal,
    )

    if not SESSION_DIR.exists():
        return

    # Only a restore that never reached a terminal state describes a half-swapped
    # tree. A COMMITTED or FAILED journal is just waiting for the next enabled
    # startup to reap it, and warning about those would point at every session
    # that has ever been restored.
    stranded = []
    try:
        for sess_dir in sorted(SESSION_DIR.iterdir()):
            if sess_dir.is_dir() and has_unfinished_restore_journal(sess_dir.name):
                stranded.append(sess_dir.name)
    except OSError as exc:
        logger.debug("Could not scan for stranded restore journals: %s", exc)
        return
    if stranded:
        logger.warning(
            "Snapshot V2 is disabled, so %d session(s) with unfinished restore "
            "journals were loaded without recovery: %s. If any of them crashed "
            "mid-restore its files may mix pre- and post-restore content; "
            "re-enable SCAI_ENABLE_SNAPSHOT_V2 to let recovery finish or roll "
            "the restore back.",
            len(stranded),
            ", ".join(stranded[:10]) + ("…" if len(stranded) > 10 else ""),
        )


async def _recover_snapshots_on_startup() -> None:
    """Recover Snapshot V2 journals only for explicitly enabled deployments."""
    from backend.config import is_snapshot_v2_enabled

    if not is_snapshot_v2_enabled():
        await asyncio.to_thread(_warn_about_unrecovered_journals)
        return

    try:
        from backend.services.snapshot_v2.recovery import recover_for_all_sessions

        recovery_result = await recover_for_all_sessions()
        if recovery_result.recovered:
            logger.info(
                "Lifespan: recovered %d restore transactions across all sessions",
                len(recovery_result.recovered_transactions),
            )
            if recovery_result.errors:
                logger.warning("Lifespan recovery errors: %s", recovery_result.errors)
    except Exception:
        logger.exception("Lifespan snapshot recovery failed; continuing")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    setup_logging()
    from backend.utils.auth import get_db
    db = await get_db()
    logger.info(f"Auth database initialized at {db.path}")

    # Restore persisted global system config (survives server restarts)
    try:
        saved_cfg_json = await db.get_meta("global_system_config")
        if saved_cfg_json:
            import json
            from backend.config._config import SystemConfig
            from backend.state import system_config
            from backend.utils.secret_redaction import (
                SEALED_SECRET_PREFIX,
                open_system_config_payload,
                seal_system_config_payload,
            )

            persisted_payload = json.loads(saved_cfg_json)
            config_payload = open_system_config_payload(persisted_payload)
            system_config["current"] = SystemConfig.model_validate(config_payload)
            api_payload = persisted_payload.get("api") or {}
            if any(
                api_payload.get(field)
                and not str(api_payload[field]).startswith(SEALED_SECRET_PREFIX)
                for field in ("api_key", "proxy_url")
            ):
                migrated_payload = seal_system_config_payload(config_payload)
                await db.set_meta(
                    "global_system_config",
                    json.dumps(migrated_payload, ensure_ascii=False),
                )
                logger.info("Migrated global system credentials to encrypted storage")
            logger.info("Restored global system config from database")
    except Exception as e:
        logger.warning("Failed to restore global system config: %s; using defaults", e)

    from backend.services.upload import set_session_callbacks
    set_session_callbacks(_update_session_cache, async_save_session)

    # Recover incomplete restore transactions only when Snapshot V2 is enabled.
    await _recover_snapshots_on_startup()

    # Reclaim export archives stranded by a previous process. Unconditional is
    # safe *here only*: startup runs before any request is served, so
    # `prepared_exports` is empty by construction and every matching file on disk
    # is already unreachable. Both shipped launchers run a single uvicorn worker;
    # with `--workers N` this would need the age guard the periodic sweep uses.
    from backend.tasks.session_cleanup import sweep_orphaned_export_archives
    await sweep_orphaned_export_archives()

    from backend.tasks.session_cleanup import cleanup_stale_sessions
    cleanup_task = asyncio.create_task(
        cleanup_stale_sessions(
            chat_task_lock=chat_task_lock,
            chat_sandboxes=chat_sandboxes,
            import_jobs=import_jobs,
            import_jobs_lock=import_jobs_lock,
            prepared_exports=prepared_exports,
            prepared_exports_lock=prepared_exports_lock,
        )
    )
    from backend.tasks.ttl_cleanup import ttl_cleanup_task
    ttl_task = asyncio.create_task(ttl_cleanup_task())
    logger.info("Background tasks started")
    try:
        yield
    finally:
        await _cancel_and_await(cleanup_task, ttl_task)
        from backend.utils.websocket_delivery import (
            drain_abandoned_transport_tasks,
        )

        await drain_abandoned_transport_tasks()


async def _cancel_and_await(*tasks: asyncio.Task) -> None:
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# FastAPI app
app = FastAPI(
    title="ScAI - Single Cell Sequencing Analysis Driven by AI",
    description="Integrated & Intelligent Single Cell Analysis System with pathology, bioinformatics, and biochemistry analysis",
    version=VERSION,
    lifespan=lifespan,
)


@app.exception_handler(SessionGoneError)
async def session_gone_exception_handler(
    request: Request,
    exc: SessionGoneError,
) -> JSONResponse:
    """Translate a lifecycle tombstone race into the public HTTP contract."""
    request_user_id = getattr(request.state, "user_id", None)
    if request_user_id is None or request_user_id != exc.owner_id:
        return JSONResponse(
            status_code=404,
            content={"detail": session_not_found_detail()},
        )
    return JSONResponse(
        status_code=410,
        content={
            "detail": {
                "code": "SESSION_GONE",
                "message": "This session was deleted and cannot be restored.",
                "generation": exc.generation,
            }
        },
    )


@app.exception_handler(SessionOwnerMismatchError)
async def session_owner_mismatch_exception_handler(
    request: Request,
    exc: SessionOwnerMismatchError,
) -> JSONResponse:
    """Return a stable non-disclosing conflict for an owner mismatch."""
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "code": "SESSION_OWNER_MISMATCH",
                "message": (
                    "This session's stored owner does not match the session "
                    "being operated on. Reloading will not resolve it."
                ),
            }
        },
    )


@app.exception_handler(SessionGenerationError)
async def session_generation_exception_handler(
    request: Request,
    exc: SessionGenerationError,
) -> JSONResponse:
    """Translate a stale in-flight generation commit into HTTP 409."""
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "code": "SESSION_GENERATION_CHANGED",
                "message": (
                    "The session generation changed while this request was in progress. "
                    "Reload the session before retrying."
                ),
            }
        },
    )


@app.exception_handler(ReportGenerationError)
async def report_generation_exception_handler(
    request: Request,
    exc: ReportGenerationError,
) -> JSONResponse:
    """Expose an inconsistent report store as a stable conflict, not a 500."""
    logger.warning(
        "Rejected inconsistent report generation path=%s: %s",
        request.scope.get("path", ""),
        exc,
    )
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "code": "REPORT_GENERATION_INVALID",
                "message": (
                    "The published report is incomplete or inconsistent. "
                    "Regenerate the report before retrying this request."
                ),
            }
        },
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
    UploadSizeLimitMiddleware,
    max_body_bytes=MAX_UPLOAD_SIZE + MULTIPART_ENVELOPE_ALLOWANCE,
    advertised_bytes=MAX_UPLOAD_SIZE,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Production serves the Vite build from the same Uvicorn process as the API and
# WebSockets.  Keep this catch-all after every backend router so concrete
# routes always win.  ``frontend_file_response`` refuses API/WS namespaces,
# missing assets, and paths escaping the build directory.
@app.api_route("/{spa_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def frontend_spa_fallback(spa_path: str):
    response = frontend_file_response(spa_path)
    if response is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return response

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





# Direct execution follows the single-port production default.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
