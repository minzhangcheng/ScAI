"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import asyncio
import io
import json
import logging
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from backend.config import (
    SESSION_DIR, REPORTS_DIR, WORKSPACE_DIR, UPLOAD_DIR, UPLOAD_TEMP_DIR,
    get_session_dir, get_session_reports, get_session_workspace, PI_AGENT_NAME,
)
from backend.models.schemas import AnalysisState, APIResponse, TaskStatus
from backend.workflow import _retry_backoff_delay

from backend.state import detailed_reports, report_websockets
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import safe_filename, validate_path_within, validate_session_id, slugify_session_name, temp_file_on_error
from backend.utils.auth import require_session_ownership
from .serialization import serialize_round_results
from backend.utils.settings_provider import AgentSettingsProvider
from backend.services.session_store import async_save_session, update_session_cache as _update_session_cache
from backend.services.report.paths import resolve_report_output_dir, resolve_report_target_lineage
from backend.services.report.lineage import build_lineage_report_state

logger = logging.getLogger(__name__)

REPORT_ARCHIVE_PREFIX = "previous_report_v"


def _should_skip_in_report_zip(file_path: Path) -> bool:
    """Exclude report archives and leftover partial archive files."""
    name = file_path.name
    if name.startswith(REPORT_ARCHIVE_PREFIX) and name.endswith(".zip"):
        return True
    if name.endswith(".zip.tmp"):
        return True
    return False


def _branch_query(branch_id: Optional[str]) -> str:
    if not branch_id:
        return ""
    return f"?branch_id={quote(branch_id, safe='')}"


def _report_cache_key(session_id: str, branch_id: Optional[str]) -> str:
    target_lineage_id = resolve_report_target_lineage(branch_id)
    if target_lineage_id == "main":
        return session_id
    return f"{session_id}:{target_lineage_id}"


def _report_cache_key_for_lineage(session_id: str, target_lineage_id: str) -> str:
    if target_lineage_id == "main":
        return session_id
    return f"{session_id}:{target_lineage_id}"


router = APIRouter(prefix="/api/session", tags=["reports"])

# Note: _report_generation_pending is process-local. In multi-worker deployments
# (e.g., uvicorn --workers N), cross-worker concurrency is not prevented. This is
# consistent with the existing chat_task_lock pattern in chat.py. For multi-worker
# production, a Redis-based lock would be needed — out of scope for this fix.
_report_generation_pending: set = set()
_pending_lock = asyncio.Lock()


async def _try_begin_generation(session_id: str) -> bool:
    """Atomically check-and-set _report_generation_pending.

    Returns True if generation was started (session_id was not already
    pending), False if generation is already in progress.
    """
    async with _pending_lock:
        if session_id in _report_generation_pending:
            return False
        _report_generation_pending.add(session_id)
        return True


def clear_generation_pending(session_id: str) -> None:
    """Remove session_id from the report-generation-pending set.

    Public helper so that external cleanup code (session deletion, periodic
    cleanup) does not need to import the private ``_report_generation_pending``
    set directly.
    """
    _report_generation_pending.discard(session_id)


def clear_detailed_report_cache(session_id: str) -> None:
    """Remove all in-memory detailed report cache entries for a session."""
    detailed_reports.pop(session_id, None)
    prefix = f"{session_id}:"
    for key in list(detailed_reports.keys()):
        if key.startswith(prefix):
            detailed_reports.pop(key, None)


def _sanitize_nan(obj):
    """Recursively replace NaN/Infinity float values with None for JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _load_report_json(path: Path) -> dict:
    """Load a JSON report, handling NaN/Infinity via parse_constant (not regex on raw text)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    data = json.loads(content, parse_constant=lambda c: None)
    return _sanitize_nan(data)


def _next_archive_version(reports_dir: Path) -> int:
    """Determine the next archive version number as max(existing) + 1."""
    if not reports_dir.exists():
        return 1
    existing_versions = []
    for p in reports_dir.glob(f"{REPORT_ARCHIVE_PREFIX}*.zip"):
        try:
            version = int(p.stem.split("_v")[-1])
            existing_versions.append(version)
        except (ValueError, IndexError):
            continue
    return max(existing_versions, default=0) + 1


def _create_report_archive(reports_dir: Path, version: int) -> Path:
    """Create a versioned zip archive of the reports directory, excluding existing archives.
    
    Uses atomic write (tmp file + os.replace) to avoid self-inclusion race.
    """
    archive_path = reports_dir / f"{REPORT_ARCHIVE_PREFIX}{version}.zip"
    tmp_path = archive_path.with_suffix(".zip.tmp")
    
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
        for file_path in reports_dir.rglob("*"):
            if file_path.is_file():
                if _should_skip_in_report_zip(file_path):
                    continue
                arcname = file_path.relative_to(reports_dir)
                zf.write(file_path, arcname)
    
    os.replace(tmp_path, archive_path)
    return archive_path


def _enforce_archive_retention(reports_dir: Path, max_archives: int = 5):
    """Keep only the most recent max_archives archives, delete the rest sorted by mtime."""
    archives = sorted(
        reports_dir.glob(f"{REPORT_ARCHIVE_PREFIX}*.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(archives) > max_archives:
        oldest = archives.pop(0)
        oldest.unlink(missing_ok=True)


# ============================================
# Exploration and Annotation Report HTML
# ============================================

@router.get("/{session_id}/exploration-report/html")
async def get_exploration_report_html(session_id: str, request: Request):
    """Serve the exploration report HTML file"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    html_path = state.exploration_report_html_path
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=404, detail="Exploration report HTML not found")

    # Validate path is within the session directory to prevent path traversal
    session_dir = get_session_dir(session_id)
    validate_path_within(Path(html_path), session_dir)

    return FileResponse(html_path, media_type="text/html", filename="exploration_report.html")


@router.get("/{session_id}/annotation-report/html")
async def get_annotation_report_html(session_id: str, request: Request):
    """Serve the annotation context report HTML file"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    html_path = state.annotation_report_html_path
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=404, detail="Annotation report not found")

    # Validate path is within the session directory to prevent path traversal
    session_dir = get_session_dir(session_id)
    validate_path_within(Path(html_path), session_dir)

    return FileResponse(html_path, media_type="text/html", filename="annotation_context_report.html")


@router.get("/{session_id}/integration-report/html")
async def get_integration_report_html(session_id: str, request: Request):
    """Serve the integration report HTML file"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    html_path = state.integration_report_html_path
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=404, detail="Integration report HTML not found")

    session_dir = get_session_dir(session_id)
    validate_path_within(Path(html_path), session_dir)

    return FileResponse(html_path, media_type="text/html", filename="integration_report.html")


@router.get("/{session_id}/detailed-report/html")
async def get_detailed_report_html(session_id: str, request: Request, branch_id: Optional[str] = None):
    """Serve the combined incremental detailed report HTML."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if not branch_id:
        if not state.detailed_report_html_path:
            raise HTTPException(status_code=404, detail="Detailed report not yet generated")
        html_path = Path(state.detailed_report_html_path)
    else:
        target_lineage_id = resolve_report_target_lineage(branch_id)
        branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
        report_dir = resolve_report_output_dir(session_id, branch_id_resolved)
        html_path = report_dir / "report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Detailed report file not found")

    session_dir = get_session_dir(session_id)
    validate_path_within(html_path, session_dir)

    return FileResponse(str(html_path), media_type="text/html", filename="detailed_report.html")


# ============================================
# Detailed Report Generation
# ============================================

@router.get("/{session_id}/detailed-report")
async def get_detailed_report(session_id: str, request: Request, branch_id: Optional[str] = None):
    """Get existing detailed report for a session"""
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    report_dir = resolve_report_output_dir(session_id, branch_id_resolved)
    cache_key = _report_cache_key_for_lineage(session_id, target_lineage_id)

    # First check memory cache
    if cache_key in detailed_reports:
        return {"report": _sanitize_nan(detailed_reports[cache_key])}

    # Try to load from disk first (avoid session loading overhead)
    report_path = report_dir / "report.json"
    if report_path.exists():
        try:
            report = _load_report_json(report_path)
            detailed_reports[cache_key] = report
            return {"report": report}
        except Exception as e:
            logger.exception(f"Failed to load report for {session_id}")

    # Only check session exists if no report found
    from backend.config import resolve_session_metadata_path
    session_meta = resolve_session_metadata_path(session_id)
    if not session_meta.exists() and cache_key not in detailed_reports:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"report": None}


@router.post("/{session_id}/generate-detailed-report")
async def generate_detailed_report(session_id: str, background_tasks: BackgroundTasks, request: Request):
    """Start generating a detailed report"""
    if not await _try_begin_generation(session_id):
        raise HTTPException(status_code=409, detail="Report generation already in progress for this session")
    
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    
    if state.status.value != "completed":
        _report_generation_pending.discard(session_id)
        raise HTTPException(status_code=400, detail="Analysis must be completed before generating report")
    
    try:
        background_tasks.add_task(
            _generate_report_task,
            session_id,
            state
        )
    except Exception:
        _report_generation_pending.discard(session_id)
        raise
    
    return {"status": "started", "message": "Report generation started"}


async def _generate_report_task(session_id: str, state: AnalysisState):
    """Background task for report generation"""
    from backend.utils.agent_llm import get_agent_llm_client as _get_agent_llm_client

    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)
    from backend.config import get_session_workspace as _get_session_workspace
    
    # Debug: log state data paths
    logger.info(f"Report generation for session {session_id}")
    seq_files = state.get_files_by_section("sequencing")
    logger.info(f"  sequencing files: {len(seq_files)}")
    logger.info(f"  state.status: {state.status}")
    logger.info(f"  round_results count: {len(state.round_results)}")

    target_lineage_id = resolve_report_target_lineage(branch_id=None)
    branch_id = None if target_lineage_id == "main" else target_lineage_id
    output_dir = str(resolve_report_output_dir(session_id, branch_id))
    os.makedirs(output_dir, exist_ok=True)

    report_state = build_lineage_report_state(state, target_lineage_id)
    
    # Progress callback
    async def progress_callback(data: dict):
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "progress",
                    **data
                })
            except Exception as e:
                logger.warning(f"Failed to send progress update for session {session_id}: {e}", exc_info=True)
    
    # Create LLM client using global config
    if config is None:
        logger.error("Config not set, cannot generate report")
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "error",
                    "message": "No API key configured. Please configure your LLM API key in Settings before generating a report."
                })
            except Exception as e:
                logger.warning(f"Failed to send error to WebSocket for session {session_id}: {e}", exc_info=True)
        return
    
    llm_client = _get_agent_llm_client(state, config, PI_AGENT_NAME, session_id)

    # Create provider for resolving per-agent settings
    provider = AgentSettingsProvider(state, config, global_client=llm_client, session_id=session_id)

    # Resolve pi_model inside the retry loop so report retries pick up changes.
    try:
        workspace_dir = str(_get_session_workspace(session_id))
        from backend.agents.supervisor_agent import SupervisorAgent
        from backend.utils.llm_client import UpstreamExhaustedError

        # Create SupervisorAgent ONCE (stateless after log restore)
        supervisor = SupervisorAgent(
            settings_provider=provider,
            session_id=session_id,
        )

        # Archive and restore ONCE before retry loop
        report_json_path = Path(output_dir) / "report.json"
        if report_json_path.exists():
            try:
                next_version = _next_archive_version(Path(output_dir))
                try:
                    archive_path = await asyncio.to_thread(
                        _create_report_archive, Path(output_dir), next_version
                    )
                    logger.info(f"Archived previous report as {archive_path.name}")
                    await asyncio.to_thread(_enforce_archive_retention, Path(output_dir), 5)
                except Exception as archive_err:
                    logger.error(f"Failed to archive previous report: {archive_err}", exc_info=True)
                    if session_id in report_websockets:
                        try:
                            await report_websockets[session_id].send_json({
                                "type": "archive_failed",
                                "message": f"Failed to archive previous report: {archive_err}",
                            })
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Error during archive check: {e}", exc_info=True)

            try:
                with open(report_json_path, encoding="utf-8") as f:
                    saved_report = json.load(f)
                saved_log = saved_report.get("analysis_log", [])
                if saved_log:
                    supervisor.restore_log_from_saved(saved_log)
                    logger.info(f"Restored {len(saved_log)} supervision records from report.json")
            except Exception as e:
                logger.warning(f"Failed to restore supervision log from report.json: {e}")

        # Retry loop — only wraps the LLM call
        MAX_REPORT_RETRIES = 3
        last_error = None

        for attempt in range(MAX_REPORT_RETRIES + 1):  # 4 total attempts
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying report generation for {session_id} "
                        f"(attempt {attempt + 1}/{MAX_REPORT_RETRIES + 1})"
                    )
                    if session_id in report_websockets:
                        try:
                            await report_websockets[session_id].send_json({
                                "type": "retrying",
                                "attempt": attempt + 1,
                                "max_attempts": MAX_REPORT_RETRIES + 1,
                                "message": (
                                    f"Retrying report generation "
                                    f"(attempt {attempt + 1}/{MAX_REPORT_RETRIES + 1})..."
                                ),
                            })
                        except Exception:
                            pass
                    provider._client_cache.clear()
                    await asyncio.sleep(_retry_backoff_delay(attempt))

                pi_model = provider.get_model(PI_AGENT_NAME)
                report = await supervisor.generate_detailed_report(
                    analysis_state=report_state,
                    report_model=pi_model,
                    output_dir=output_dir,
                    workspace_dir=workspace_dir,
                    progress_callback=progress_callback,
                    chat_messages=state.chat_messages,
                )
                detailed_reports[_report_cache_key_for_lineage(session_id, target_lineage_id)] = report

                if session_id in report_websockets:
                    try:
                        await report_websockets[session_id].send_json({
                            "type": "complete",
                            "report": report,
                        })
                    except Exception as e:
                        logger.warning(
                            f"Failed to send completion to WebSocket for {session_id}: {e}",
                            exc_info=True,
                        )
                return  # success

            except UpstreamExhaustedError as e:
                last_error = e
                logger.error(
                    f"[{session_id}] Report generation LLM upstream exhausted "
                    f"(attempt {attempt + 1}/{MAX_REPORT_RETRIES + 1}): "
                    f"agent={e.agent_name} model={e.model} server={e.llm_server}"
                )
                if attempt < MAX_REPORT_RETRIES:
                    continue
                # All retries exhausted — fall through to error handling

            except Exception as e:
                last_error = e
                logger.exception("Report generation failed")
                break  # non-LLM error, don't retry

        # All retries exhausted (or non-LLM error)
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "error",
                    "message": str(last_error) if last_error else "Report generation failed",
                    "retryable": isinstance(last_error, UpstreamExhaustedError),
                })
            except Exception as ws_error:
                logger.warning(
                    f"Failed to send error to WebSocket for {session_id}: {ws_error}",
                    exc_info=True,
                )

    finally:
        _report_generation_pending.discard(session_id)


# ============================================
# Report Download Endpoints
# ============================================

@router.post("/{session_id}/report/download-all-figures")
async def download_all_figures(
    session_id: str,
    state: AnalysisState = Depends(require_session_ownership),
    branch_id: Optional[str] = None,
):
    """Download all figures as a ZIP file"""
    validate_session_id(session_id)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    figures_dir = resolve_report_output_dir(session_id, branch_id_resolved) / "figures"

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


@router.post("/{session_id}/report/download-all-tables")
async def download_all_tables(
    session_id: str,
    state: AnalysisState = Depends(require_session_ownership),
    branch_id: Optional[str] = None,
):
    """Download all tables as a ZIP file"""
    validate_session_id(session_id)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    tables_dir = resolve_report_output_dir(session_id, branch_id_resolved) / "tables"

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


@router.post("/{session_id}/report/download-reproducibility")
async def download_reproducibility_bundle(
    session_id: str,
    state: AnalysisState = Depends(require_session_ownership),
):
    """Download reproducibility bundle as a ZIP file"""
    validate_session_id(session_id)
    from backend.config import default_config, get_session_workspace
    if not default_config.enable_reproducibility:
        raise HTTPException(status_code=403, detail="Reproducibility bundle generation is disabled")
    from backend.services.reproducibility import ReproducibilityBundle, REPRODUCIBILITY_DIR_NAME

    repro_dir = get_session_workspace(session_id) / REPRODUCIBILITY_DIR_NAME
    if not repro_dir.exists():
        raise HTTPException(status_code=404, detail="Reproducibility bundle not found")

    zip_buffer = ReproducibilityBundle.create_zip_bytes(str(repro_dir))

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename={session_id}_reproducibility.zip"
            ),
        },
    )


@router.post("/{session_id}/report/download-full")
async def download_full_report(
    session_id: str,
    state: AnalysisState = Depends(require_session_ownership),
    branch_id: Optional[str] = None,
):
    """Download full report including figures, tables, and report JSON"""
    validate_session_id(session_id)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    report_dir = resolve_report_output_dir(session_id, branch_id_resolved)

    if not report_dir.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=str(UPLOAD_TEMP_DIR))
    tmp_path = tmp.name
    tmp.close()

    def _build_zip():
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in report_dir.rglob("*"):
                if file_path.is_file() and not _should_skip_in_report_zip(file_path):
                    arcname = file_path.relative_to(report_dir)
                    zf.write(file_path, arcname)

    async with temp_file_on_error(tmp_path):
        await asyncio.to_thread(_build_zip)
        safe_name = slugify_session_name(state.name) if state.name else session_id
        filename = f"{safe_name}_full_report.zip"
        return FileResponse(
            tmp_path,
            media_type="application/zip",
            filename=filename,
            background=BackgroundTask(Path(tmp_path).unlink, missing_ok=True),
        )


@router.get("/{session_id}/report/archives")
async def list_report_archives(session_id: str, request: Request):
    """List available report archive versions with metadata."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    
    reports_dir = get_session_reports(session_id)
    if not reports_dir.exists():
        return []
    
    archives = []
    for archive_path in reports_dir.glob(f"{REPORT_ARCHIVE_PREFIX}*.zip"):
        try:
            version = int(archive_path.stem.split("_v")[-1])
            stat = archive_path.stat()
            archives.append({
                "version": version,
                "filename": archive_path.name,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_bytes": stat.st_size,
                "download_url": f"session/{session_id}/report/download-archive/{archive_path.name}",
            })
        except (ValueError, IndexError):
            continue
    
    return sorted(archives, key=lambda a: a["version"])


@router.get("/{session_id}/report/download-archive/{filename}")
async def download_report_archive(session_id: str, filename: str, request: Request):
    """Download a specific report archive zip file."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    
    reports_dir = get_session_reports(session_id)
    archive_path = reports_dir / filename
    validate_path_within(archive_path, reports_dir)
    
    if not archive_path.exists() or not filename.startswith(REPORT_ARCHIVE_PREFIX) or not filename.endswith(".zip"):
        raise HTTPException(status_code=404, detail="Archive not found")
    
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=filename,
    )


# ============================================
# Report HTML and Preview
# ============================================

@router.get("/{session_id}/report/html")
async def get_report_html(session_id: str, request: Request, branch_id: Optional[str] = None):
    """Get the HTML report for preview"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    report_dir = resolve_report_output_dir(session_id, branch_id_resolved)
    html_path = report_dir / "report.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found")

    # Don't set filename to avoid triggering download
    return FileResponse(
        html_path,
        media_type="text/html"
    )


@router.get("/{session_id}/report/preview")
async def preview_report(session_id: str, request: Request, branch_id: Optional[str] = None):
    """Get report preview data including HTML content and metadata"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    report_dir = resolve_report_output_dir(session_id, branch_id_resolved)
    branch_query = _branch_query(branch_id_resolved)

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
                    "png_url": f"/api/session/{session_id}/report/figure/{f.name}{branch_query}",
                    "pdf_url": (
                        f"/api/session/{session_id}/report/figure/{pdf_path.name}{branch_query}"
                        if pdf_path.exists() else None
                    ),
                })

    tables = []
    if tables_dir.exists():
        for f in tables_dir.glob("*.csv"):
            tables.append({
                "name": f.stem,
                "csv_url": f"/api/session/{session_id}/report/table/{f.name}{branch_query}"
            })

    return {
        "session_id": session_id,
        "html_url": f"/api/session/{session_id}/report/html{branch_query}",
        "download_url": f"/api/session/{session_id}/report/download-full{branch_query}",
        "report": report_data,
        "html_content": html_content,
        "figures": figures,
        "tables": tables,
        "has_html": html_path.exists()
    }


@router.get("/{session_id}/report/figure/{filename}")
async def get_report_figure(
    session_id: str,
    filename: str,
    branch_id: Optional[str] = None,
    state: AnalysisState = Depends(require_session_ownership),
):
    """Get a specific figure from the report"""
    safe_name = safe_filename(filename)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    figure_parent = resolve_report_output_dir(session_id, branch_id_resolved) / "figures"
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


@router.get("/{session_id}/report/figures/{filename}")
async def get_report_figure_alt(
    session_id: str,
    filename: str,
    branch_id: Optional[str] = None,
    state: AnalysisState = Depends(require_session_ownership),
):
    """Get a specific figure from the report (alternative path for HTML relative URLs)"""
    return await get_report_figure(session_id, filename, branch_id=branch_id, state=state)


@router.get("/{session_id}/report/table/{filename}")
async def get_report_table(
    session_id: str,
    filename: str,
    branch_id: Optional[str] = None,
    state: AnalysisState = Depends(require_session_ownership),
):
    """Get a specific table from the report"""
    safe_name = safe_filename(filename)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    table_parent = resolve_report_output_dir(session_id, branch_id_resolved) / "tables"
    table_path = table_parent / safe_name
    validate_path_within(table_path, table_parent)

    if not table_path.exists():
        raise HTTPException(status_code=404, detail="Table not found")

    return FileResponse(
        table_path,
        media_type="text/csv",
        filename=filename
    )


@router.get("/{session_id}/report/tables/{filename}")
async def get_report_table_alt(
    session_id: str,
    filename: str,
    branch_id: Optional[str] = None,
    state: AnalysisState = Depends(require_session_ownership),
):
    """Get a specific table from the report (alternative path for HTML relative URLs)"""
    return await get_report_table(session_id, filename, branch_id=branch_id, state=state)


def _resolve_report_file(report_dir: Path, filename: str) -> Path:
    """Resolve a report file path, preserving subdirectories while blocking traversal."""
    file_path = report_dir / filename
    return validate_path_within(file_path, report_dir)


@router.get("/{session_id}/report/{filename:path}")
async def get_report_file(
    session_id: str,
    filename: str,
    branch_id: Optional[str] = None,
    state: AnalysisState = Depends(require_session_ownership),
):
    """Catch-all: serve any file from the report directory (notebook, sub-reports, etc.)"""
    validate_session_id(session_id)
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    report_dir = resolve_report_output_dir(session_id, branch_id_resolved)
    file_path = _resolve_report_file(report_dir, filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    ext = file_path.suffix.lower()
    media_types = {
        '.html': 'text/html', '.ipynb': 'application/json',
        '.json': 'application/json', '.pdf': 'application/pdf',
        '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.csv': 'text/csv',
    }
    return FileResponse(
        file_path,
        media_type=media_types.get(ext, 'application/octet-stream'),
        filename=filename,
    )


# ============================================
# Publication Figures and Graphic Abstract Endpoints
# ============================================

@router.get("/{session_id}/graphic-abstract")
async def get_graphic_abstract(session_id: str, request: Request):
    """Get the graphic abstract for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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


@router.get("/{session_id}/publication-figures")
async def get_publication_figures(session_id: str, request: Request):
    """Get all publication figures for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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


@router.get("/{session_id}/publication-tables")
async def get_publication_tables(session_id: str, request: Request):
    """Get all publication tables for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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


@router.get("/{session_id}/figure/{figure_id}/pdf")
async def download_publication_figure_pdf(session_id: str, figure_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download a specific publication figure as PDF"""
    from backend.config import SESSION_DIR as _SESSION_DIR, REPORTS_DIR as _REPORTS_DIR

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
            pdf_path = get_session_workspace(session_id) / "figures" / f"{fig_name}.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    # Validate path is within allowed directories
    resolved = pdf_path.resolve()
    allowed_parents = [_SESSION_DIR.resolve(), _REPORTS_DIR.resolve(), WORKSPACE_DIR.resolve(), UPLOAD_DIR.resolve()]
    if not any(str(resolved).startswith(str(p) + os.sep) or resolved == p for p in allowed_parents):
        raise HTTPException(status_code=403, detail="Access denied: file outside allowed directories")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{fig_name}.pdf"
    )


@router.get("/{session_id}/figure/{figure_id}/png")
async def download_publication_figure_png(session_id: str, figure_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download a specific publication figure as PNG/JPG"""
    from backend.config import SESSION_DIR as _SESSION_DIR, REPORTS_DIR as _REPORTS_DIR

    figure = next((f for f in state.publication_figures if f.id == figure_id), None)

    if figure is not None:
        if not figure.png_path:
            raise HTTPException(status_code=404, detail="Image not available for this figure")
        img_path = Path(figure.png_path)
        fig_name = figure.name
    else:
        # Fallback: search workspace figures from integrated_report message
        fig_data = _find_workspace_figure(state, session_id, figure_id)
        if fig_data is None:
            raise HTTPException(status_code=404, detail="Figure not found")
        fig_name = fig_data.get('name', '')
        explicit_path = fig_data.get('png_path', '')
        if explicit_path and Path(explicit_path).exists():
            img_path = Path(explicit_path)
        else:
            # Try JPG first, then PNG for backward compatibility
            img_path = get_session_workspace(session_id) / "figures" / f"{fig_name}.jpg"
            if not img_path.exists():
                img_path = get_session_workspace(session_id) / "figures" / f"{fig_name}.png"
            if not img_path.exists():
                img_path = get_session_workspace(session_id) / f"{fig_name}.jpg"
            if not img_path.exists():
                img_path = get_session_workspace(session_id) / f"{fig_name}.png"

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Validate path is within allowed directories
    resolved = img_path.resolve()
    allowed_parents = [_SESSION_DIR.resolve(), _REPORTS_DIR.resolve(), WORKSPACE_DIR.resolve(), UPLOAD_DIR.resolve()]
    if not any(str(resolved).startswith(str(p) + os.sep) or resolved == p for p in allowed_parents):
        raise HTTPException(status_code=403, detail="Access denied: file outside allowed directories")

    # Determine media type from extension
    ext = img_path.suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    dl_ext = ext if ext in (".jpg", ".jpeg") else ".png"

    return FileResponse(
        img_path,
        media_type=media_type,
        filename=f"{fig_name}{dl_ext}"
    )


@router.get("/{session_id}/integrated-report")
async def get_integrated_report(session_id: str, request: Request):
    """Get the integrated report for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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


# ============================================
# Code History and Report Summary
# ============================================

@router.get("/{session_id}/code")
async def get_code_history(session_id: str, request: Request):
    """Get all executed code for reproducibility"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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


@router.get("/{session_id}/report")
async def get_report(session_id: str, request: Request):
    """Get final analysis report"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    return {
        "session_id": session_id,
        "name": state.name,
        "status": state.status.value,
        "round_results": serialize_round_results(state.round_results),
        "final_summary": state.final_summary
    }


@router.post("/{session_id}/report/partial/approve")
async def approve_partial_report(session_id: str, request: Request):
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    state.allow_partial_report = True
    state.partial_report_requested_at = datetime.now()
    state.status = TaskStatus.RUNNING
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state, force=True)
    from backend.services.workflow_runner import start_report_generation
    await start_report_generation(session_id)
    return APIResponse(
        success=True,
        message="Partial report approved",
        data={"needs_reconnect": True},
    )
