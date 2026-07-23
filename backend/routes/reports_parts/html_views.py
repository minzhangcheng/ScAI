# HTML report endpoints and detailed-report generation trigger.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse

from backend.config import get_session_dir, PI_AGENT_NAME
from backend.models.schemas import AnalysisState
from backend.workflow import _retry_backoff_delay

from backend.state import detailed_reports, report_websockets
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import validate_path_within, validate_session_id
from backend.utils.auth import require_session_ownership
from backend.utils.settings_provider import AgentSettingsProvider
from backend.services.report.paths import resolve_report_output_dir, resolve_report_target_lineage
from backend.services.report.lineage import build_lineage_report_state

logger = logging.getLogger(__name__)

REPORT_ARCHIVE_PREFIX = "previous_report_v"


from .core import _create_report_archive, _enforce_archive_retention, _load_report_json, _next_archive_version, _report_cache_key_for_lineage, _report_generation_pending, _sanitize_nan, _try_begin_generation, router


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
