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

from backend.state import (
    detailed_report_tasks,
    detailed_reports,
    sessions,
    state_lock,
)
from backend.services.session_lifecycle import (
    SessionGenerationError,
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.utils.websocket_delivery import send_report_json
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import validate_path_within, validate_session_id
from backend.utils.auth import require_session_ownership
from backend.utils.asyncio_tasks import (
    consume_current_cancellation_request,
    run_in_owned_thread,
)
from backend.utils.llm_client import OpenAICompatibleClient
from backend.utils.settings_provider import AgentSettingsProvider
from backend.services.report.paths import (
    resolve_published_report_dir,
    resolve_report_output_dir,
    resolve_report_target_lineage,
)
from backend.services.report.output_generation import report_matches_published_dir
from backend.services.report.lineage import build_lineage_report_state

logger = logging.getLogger(__name__)

_REPORT_RESOURCE_CLEANUP_TIMEOUT_SECONDS = 10.0


from .core import _create_report_archive, _enforce_archive_retention, _load_report_json, _next_archive_version, _report_cache_key_for_lineage, _report_generation_pending, _sanitize_nan, _try_begin_generation, clear_generation_pending, router


@router.get("/{session_id}/exploration-report/html")
async def get_exploration_report_html(
    session_id: str,
    request: Request,
    branch_id: Optional[str] = None,
):
    """Serve the exploration report HTML file"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    target_lineage = resolve_report_target_lineage(
        branch_id if branch_id is not None else state.restore_branch_id
    )
    active_lineage = state.restore_branch_id or "main"
    html_path = (
        state.exploration_report_html_path
        if target_lineage == active_lineage
        else None
    )
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=404, detail="Exploration report HTML not found")

    # Validate path is within the session directory to prevent path traversal
    session_dir = get_session_dir(session_id)
    validate_path_within(Path(html_path), session_dir)

    return FileResponse(html_path, media_type="text/html", filename="exploration_report.html")

@router.get("/{session_id}/annotation-report/html")
async def get_annotation_report_html(
    session_id: str,
    request: Request,
    branch_id: Optional[str] = None,
):
    """Serve the annotation context report HTML file"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    target_lineage = resolve_report_target_lineage(
        branch_id if branch_id is not None else state.restore_branch_id
    )
    active_lineage = state.restore_branch_id or "main"
    html_path = (
        state.annotation_report_html_path
        if target_lineage == active_lineage
        else None
    )
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=404, detail="Annotation report not found")

    # Validate path is within the session directory to prevent path traversal
    session_dir = get_session_dir(session_id)
    validate_path_within(Path(html_path), session_dir)

    return FileResponse(html_path, media_type="text/html", filename="annotation_context_report.html")

@router.get("/{session_id}/integration-report/html")
async def get_integration_report_html(
    session_id: str,
    request: Request,
    branch_id: Optional[str] = None,
):
    """Serve the integration report HTML file"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    target_lineage = resolve_report_target_lineage(
        branch_id if branch_id is not None else state.restore_branch_id
    )
    active_lineage = state.restore_branch_id or "main"
    html_path = (
        state.integration_report_html_path
        if target_lineage == active_lineage
        else None
    )
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
        report_dir = resolve_published_report_dir(session_id, branch_id_resolved)
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
    report_dir = resolve_published_report_dir(session_id, branch_id_resolved)
    cache_key = _report_cache_key_for_lineage(session_id, target_lineage_id)

    # First check memory cache
    cached_report = detailed_reports.get(cache_key)
    if (
        cached_report is not None
        and report_matches_published_dir(cached_report, report_dir)
    ):
        return {"report": _sanitize_nan(cached_report)}

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
    pending_token = await _try_begin_generation(session_id)
    if pending_token is None:
        raise HTTPException(status_code=409, detail="Report generation already in progress for this session")

    handed_off = False
    try:
        user_id = getattr(request.state, "user_id", "")
        state = await require_session_ownership(
            session_id=session_id,
            user_id=user_id,
        )

        if state.status.value != "completed":
            raise HTTPException(
                status_code=400,
                detail="Analysis must be completed before generating report",
            )

        # Register a real task handle under the lifecycle barrier. FastAPI's
        # BackgroundTasks handoff leaves only a boolean marker and gives delete
        # nothing it can cancel/await; a late callback could recreate the
        # removed report tree or write into a reused session ID.
        async with get_lifecycle_lock(session_id):
            assert_generation_writable(session_id, state)
            async with state_lock:
                from backend.services.workflow_runner import (
                    session_has_any_workflow_owner,
                )

                if session_has_any_workflow_owner(session_id):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Another analysis or report workflow is active for "
                            "this session"
                        ),
                    )
                existing = detailed_report_tasks.get(session_id)
                if existing is not None and not existing.done():
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Report generation already in progress for this "
                            "session"
                        ),
                    )
                if sessions.get(session_id) is not state:
                    raise SessionGenerationError(
                        "session state changed before report task publication"
                    )
                task = asyncio.create_task(
                    _generate_report_task(session_id, state, pending_token),
                    name=f"detailed-report:{session_id}",
                )
                detailed_report_tasks[session_id] = task

        def _retire_report_task(done: asyncio.Task) -> None:
            if detailed_report_tasks.get(session_id) is done:
                detailed_report_tasks.pop(session_id, None)
                clear_generation_pending(session_id, pending_token)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Detailed report task failed for %s", session_id,
                )

        task.add_done_callback(_retire_report_task)
        handed_off = True
        return {"status": "started", "message": "Report generation started"}
    finally:
        # Before task publication succeeds, this request still owns the marker.
        # Afterwards the registered task and its identity-checked callback own
        # that exact token.
        if not handed_off:
            clear_generation_pending(session_id, pending_token)

async def _cleanup_report_llm_resources(provider, llm_client) -> None:
    """Close report-owned clients independently, retrying transient failures."""
    cancellation = None

    def _consume_detached_cleanup(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                "Detached report resource cleanup failed",
                exc_info=True,
            )

    async def _close_with_retry(label: str, closer) -> None:
        nonlocal cancellation
        for attempt in range(2):
            cleanup_task = asyncio.create_task(
                closer(),
                name=f"detailed-report-cleanup:{label}",
            )
            try:
                done, _pending = await asyncio.wait(
                    {cleanup_task},
                    timeout=_REPORT_RESOURCE_CLEANUP_TIMEOUT_SECONDS,
                )
                if not done:
                    logger.error(
                        "Timed out after %.1fs while %s; detaching cleanup",
                        _REPORT_RESOURCE_CLEANUP_TIMEOUT_SECONDS,
                        label,
                    )
                    cleanup_task.cancel()
                    cleanup_task.add_done_callback(_consume_detached_cleanup)
                    return
                await cleanup_task
                return
            except asyncio.CancelledError as exc:
                cleanup_task.cancel()
                cleanup_task.add_done_callback(_consume_detached_cleanup)
                cancellation = cancellation or exc
                consume_current_cancellation_request()
                logger.warning(
                    "Cancellation while %s; retrying owned-resource cleanup",
                    label,
                )
            except Exception:
                if attempt == 0:
                    logger.warning(
                        "Failed %s; retrying once",
                        label,
                        exc_info=True,
                    )
                else:
                    logger.error(
                        "Failed %s after two attempts",
                        label,
                        exc_info=True,
                    )

    if provider is not None:
        await _close_with_retry(
            "closing report settings-provider clients",
            provider.cleanup,
        )
    if llm_client is not None:
        await _close_with_retry(
            "closing report global LLM client",
            llm_client.close,
        )
    if cancellation is not None:
        raise cancellation


async def _await_owned_thread_call(func, *args):
    """Finish report-owned filesystem work before propagating cancellation."""
    return await run_in_owned_thread(func, *args)


async def _generate_report_task(
    session_id: str,
    state: AnalysisState,
    pending_token: str | None = None,
):
    """Run a report task and always release its per-session pending marker."""
    try:
        await _run_generate_report_task(session_id, state)
    finally:
        owner = detailed_report_tasks.get(session_id)
        if pending_token is not None and (
            owner is asyncio.current_task() or owner is None
        ):
            clear_generation_pending(session_id, pending_token)


async def _run_generate_report_task(session_id: str, state: AnalysisState):
    """Background task for report generation"""
    async with get_lifecycle_lock(session_id):
        assert_generation_writable(session_id, state)
        async with state_lock:
            if sessions.get(session_id) is not state:
                raise SessionGenerationError(
                    "session state changed before detailed report generation"
                )
    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)
    from backend.config import get_session_workspace as _get_session_workspace
    
    # Debug: log state data paths
    logger.info(f"Report generation for session {session_id}")
    seq_files = state.get_files_by_section("sequencing")
    logger.info(f"  sequencing files: {len(seq_files)}")
    logger.info(f"  state.status: {state.status}")
    logger.info(f"  round_results count: {len(state.round_results)}")

    target_lineage_id = resolve_report_target_lineage(
        getattr(state, "restore_branch_id", None)
    )
    branch_id = None if target_lineage_id == "main" else target_lineage_id
    output_dir = str(resolve_report_output_dir(session_id, branch_id))
    os.makedirs(output_dir, exist_ok=True)

    report_state = build_lineage_report_state(state, target_lineage_id)
    
    # Progress callback
    async def progress_callback(data: dict):
        await send_report_json(session_id, {"type": "progress", **data})
    
    # Create LLM client using global config
    if config is None:
        logger.error("Config not set, cannot generate report")
        await send_report_json(session_id, {
            "type": "error",
            "message": "No API key configured. Please configure your LLM API key in Settings before generating a report.",
        })
        return
    
    llm_client = None
    provider = None
    try:
        # The provider's global client must represent the resolved global
        # endpoint, not a PI override. It can then select a matching PI client
        # from the freshest session state on every retry.
        llm_client = OpenAICompatibleClient(
            api_key=config.api.api_key,
            base_url=config.api.llm_base_url,
            proxy_url=config.api.proxy_url,
            session_id=session_id,
            agent_name=PI_AGENT_NAME,
        )

        # Create provider for resolving per-agent settings.
        provider = AgentSettingsProvider(
            state,
            config,
            global_client=llm_client,
            session_id=session_id,
        )

        # Resolve the PI model/client pair inside the retry loop so per-session
        # endpoint, credential, and model changes apply together on the next try.
        workspace_dir = str(_get_session_workspace(session_id))
        from backend.agents.supervisor_agent import SupervisorAgent
        from backend.utils.llm_client import UpstreamExhaustedError

        # Create SupervisorAgent ONCE (stateless after log restore)
        supervisor = SupervisorAgent(
            settings_provider=provider,
            session_id=session_id,
        )

        # Archive and restore ONCE before retry loop
        report_json_path = (
            resolve_published_report_dir(session_id, branch_id) / "report.json"
        )
        if report_json_path.exists():
            try:
                next_version = _next_archive_version(Path(output_dir))
                try:
                    archive_path = await _await_owned_thread_call(
                        _create_report_archive, Path(output_dir), next_version
                    )
                    logger.info(f"Archived previous report as {archive_path.name}")
                    await _await_owned_thread_call(
                        _enforce_archive_retention, Path(output_dir), 5,
                    )
                except Exception as archive_err:
                    logger.error(f"Failed to archive previous report: {archive_err}", exc_info=True)
                    await send_report_json(session_id, {
                        "type": "archive_failed",
                        "message": f"Failed to archive previous report: {archive_err}",
                    })
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
                    await send_report_json(session_id, {
                        "type": "retrying",
                        "attempt": attempt + 1,
                        "max_attempts": MAX_REPORT_RETRIES + 1,
                        "message": (
                            f"Retrying report generation "
                            f"(attempt {attempt + 1}/{MAX_REPORT_RETRIES + 1})..."
                        ),
                    })
                    provider.clear_client_cache()
                    await asyncio.sleep(_retry_backoff_delay(attempt))

                report_llm_client = provider.get_llm_client(PI_AGENT_NAME)
                pi_model = provider.get_model(PI_AGENT_NAME)
                report = await supervisor.generate_detailed_report(
                    analysis_state=report_state,
                    report_model=pi_model,
                    output_dir=output_dir,
                    workspace_dir=workspace_dir,
                    progress_callback=progress_callback,
                    report_llm_client=report_llm_client,
                    chat_messages=state.chat_messages,
                )
                async with get_lifecycle_lock(session_id):
                    assert_generation_writable(session_id, state)
                    async with state_lock:
                        if sessions.get(session_id) is not state:
                            raise SessionGenerationError(
                                "session state changed before report publication"
                            )
                        detailed_reports[
                            _report_cache_key_for_lineage(
                                session_id, target_lineage_id,
                            )
                        ] = report

                await send_report_json(session_id, {
                    "type": "complete",
                    "report": report,
                })
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
        await send_report_json(session_id, {
            "type": "error",
            "message": str(last_error) if last_error else "Report generation failed",
            "retryable": isinstance(last_error, UpstreamExhaustedError),
        })

    finally:
        await _cleanup_report_llm_resources(provider, llm_client)
