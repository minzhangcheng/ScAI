# Publication figures/tables, graphic abstract, integrated report, code history.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import logging
import os
import asyncio
from pathlib import Path

from datetime import datetime
from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from backend.config import WORKSPACE_DIR, UPLOAD_DIR, get_session_workspace
from backend.models.schemas import AnalysisState, APIResponse, TaskStatus

from backend.utils.files import validate_session_id
from backend.utils.auth import require_session_ownership
from ..serialization import serialize_round_results
from backend.services.session_store import _async_save_session_under_lifecycle
from backend.services.session_lifecycle import (
    SessionGenerationError,
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.state import sessions, state_lock, touch_access_time
from backend.utils.asyncio_tasks import await_with_deferred_cancellation
from backend.services.report.artifacts import (
    archival_render_available,
    eligible_artifact_by_id,
    eligible_artifacts,
    resolve_legacy_artifact_pdf,
    resolve_legacy_artifact_render,
    resolve_archival_pdf,
    resolve_archival_render,
    verified_artifact_bytes,
)
from backend.services.report.figure_discovery import ensure_legacy_artifact_inventory

logger = logging.getLogger(__name__)


from .core import router


async def _ensure_frozen_artifact_inventory(
    state: AnalysisState,
    workspace: Path,
) -> None:
    """Apply the one-time compatibility scan to the request's loaded state.

    ``ensure_session_loaded`` performs and durably persists this migration under
    the session lifecycle lock before an authenticated route receives the state.
    Keeping the idempotent call here covers direct/internal callers without
    opening a second, route-local read-modify-write race.
    """
    ensure_legacy_artifact_inventory(state, workspace)


def _legacy_artifact_by_id(
    state: AnalysisState,
    artifact_id: str,
    lineage_id: str,
):
    return next(
        (
            item for item in (state.legacy_artifacts or [])
            if str(getattr(item, "id", "")) == str(artifact_id)
            and str(getattr(item, "lineage_id", None) or "main") == lineage_id
        ),
        None,
    )


def _immutable_download_response(
    payload: bytes,
    source_path: Path,
    *,
    media_type: str,
    filename: str,
) -> Response:
    """Build a delayed-response-safe download from already verified bytes."""
    safe_filename = os.path.basename(str(filename)).replace('"', "_")
    response = Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"'
        },
    )
    # Preserve the introspection attribute that FileResponse exposed to direct
    # route callers while ensuring ASGI delivery never reopens this path.
    response.path = os.fspath(source_path)
    return response


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

    figures = []
    approved = 0
    target_lineage = getattr(state, "restore_branch_id", None) or "main"
    for fig in eligible_artifacts(state.publication_figures):
        if not getattr(fig, "is_current", True) or getattr(fig, "invalidated_at", None):
            continue
        if (getattr(fig, "lineage_id", None) or "main") != target_lineage:
            continue
        if not archival_render_available(fig):
            continue
        review_status = fig.review_status.value
        if review_status == "approved":
            approved += 1
        figures.append({
            "id": fig.id,
            "name": fig.name,
            "description": fig.description,
            "figure_type": fig.figure_type,
            "png_path": fig.png_path,
            "pdf_path": fig.pdf_path,
            "png_base64": fig.png_base64,
            "review_status": review_status,
            "pi_feedback": fig.pi_feedback,
            "revision_count": fig.revision_count,
            "plan_step_id": fig.plan_step_id,
            "plan_id": getattr(fig, "plan_id", None),
            "purpose": fig.purpose,
            "logical_figure_id": getattr(fig, "logical_figure_id", None),
            "render_attempt_id": getattr(fig, "render_attempt_id", None),
            "producer_status": getattr(fig, "producer_status", "published"),
            "revision_of_id": getattr(fig, "revision_of_id", None),
            "supersedes_id": getattr(fig, "supersedes_id", None),
            "is_published_head": bool(getattr(fig, "is_published_head", False)),
            "is_publication": True,
            "round": fig.round_number or None,
            "diagnostic": bool(fig.diagnostic),
            "created_at": fig.created_at.isoformat()
        })

    return {
        "success": True,
        "figures": figures,
        "total": len(figures),
        "approved": approved
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

@router.get("/{session_id}/figure/{figure_id}/pdf")
async def download_publication_figure_pdf(session_id: str, figure_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download a specific publication figure as PDF"""
    from backend.config import SESSION_DIR as _SESSION_DIR, REPORTS_DIR as _REPORTS_DIR

    workspace = get_session_workspace(session_id)
    await _ensure_frozen_artifact_inventory(state, workspace)
    target_lineage = getattr(state, "restore_branch_id", None) or "main"
    figure = eligible_artifact_by_id(
        state.publication_figures,
        figure_id,
        lineage_id=target_lineage,
    )
    registered_id = any(
        str(getattr(item, "id", "")) == str(figure_id)
        for item in state.publication_figures
    )

    if figure is not None:
        pdf_path = resolve_archival_pdf(figure, workspace)
        if pdf_path is None:
            raise HTTPException(status_code=404, detail="PDF not available for this figure")
        fig_name = figure.name
        artifact_record = figure
    else:
        if registered_id:
            # A registry ID is never allowed to fall through to an old
            # integrated-report message. Pending/failed/abandoned/rejected or
            # superseded records remain audit entries, not download aliases.
            raise HTTPException(status_code=404, detail="Figure is not the current eligible head")
        legacy = _legacy_artifact_by_id(state, figure_id, target_lineage)
        if legacy is None:
            raise HTTPException(status_code=404, detail="Figure not found")
        fig_name = legacy.name
        artifact_record = legacy
        pdf_path = resolve_legacy_artifact_pdf(legacy, workspace)
        if pdf_path is None:
            raise HTTPException(
                status_code=404,
                detail="PDF file is missing or failed frozen-inventory validation",
            )

    # Validate path is within allowed directories
    resolved = pdf_path.resolve()
    allowed_parents = [_SESSION_DIR.resolve(), _REPORTS_DIR.resolve(), WORKSPACE_DIR.resolve(), UPLOAD_DIR.resolve()]
    if not any(str(resolved).startswith(str(p) + os.sep) or resolved == p for p in allowed_parents):
        raise HTTPException(status_code=403, detail="Access denied: file outside allowed directories")

    payload = verified_artifact_bytes(
        pdf_path, artifact_record, kind="pdf"
    )
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="PDF file changed or failed record-bound validation",
        )
    return _immutable_download_response(
        payload,
        pdf_path,
        media_type="application/pdf",
        filename=f"{fig_name}.pdf",
    )

@router.get("/{session_id}/figure/{figure_id}/png")
async def download_publication_figure_png(session_id: str, figure_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download a specific publication figure as PNG/JPG"""
    from backend.config import SESSION_DIR as _SESSION_DIR, REPORTS_DIR as _REPORTS_DIR

    workspace = get_session_workspace(session_id)
    await _ensure_frozen_artifact_inventory(state, workspace)
    target_lineage = getattr(state, "restore_branch_id", None) or "main"
    figure = eligible_artifact_by_id(
        state.publication_figures,
        figure_id,
        lineage_id=target_lineage,
    )
    registered_id = any(
        str(getattr(item, "id", "")) == str(figure_id)
        for item in state.publication_figures
    )

    if figure is not None:
        img_path = resolve_archival_render(figure, workspace)
        if img_path is None:
            raise HTTPException(status_code=404, detail="Image not available for this figure")
        fig_name = figure.name
        artifact_record = figure
    else:
        if registered_id:
            raise HTTPException(status_code=404, detail="Figure is not the current eligible head")
        legacy = _legacy_artifact_by_id(state, figure_id, target_lineage)
        if legacy is None:
            raise HTTPException(status_code=404, detail="Figure not found")
        fig_name = legacy.name
        artifact_record = legacy
        img_path = resolve_legacy_artifact_render(legacy, workspace)
        if img_path is None:
            raise HTTPException(
                status_code=404,
                detail="Image file is missing or failed frozen-inventory validation",
            )

    # Validate path is within allowed directories
    resolved = img_path.resolve()
    allowed_parents = [_SESSION_DIR.resolve(), _REPORTS_DIR.resolve(), WORKSPACE_DIR.resolve(), UPLOAD_DIR.resolve()]
    if not any(str(resolved).startswith(str(p) + os.sep) or resolved == p for p in allowed_parents):
        raise HTTPException(status_code=403, detail="Access denied: file outside allowed directories")

    # Determine media type from extension
    ext = img_path.suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    dl_ext = ext if ext in (".jpg", ".jpeg") else ".png"

    payload = verified_artifact_bytes(
        img_path, artifact_record, kind="image"
    )
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="Image file changed or failed record-bound validation",
        )
    return _immutable_download_response(
        payload,
        img_path,
        media_type=media_type,
        filename=f"{fig_name}{dl_ext}",
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
    from backend.services.workflow_runner import (
        SessionWorkflowConflictError,
        start_report_generation,
    )

    async def _prepare_partial_report() -> None:
        async def _persist_candidate_and_swap() -> None:
            async with get_lifecycle_lock(session_id):
                assert_generation_writable(session_id, state)
                async with state_lock:
                    if sessions.get(session_id) is not state:
                        raise SessionGenerationError(
                            "session state changed before partial-report approval"
                        )
                    candidate = state.model_copy(deep=True)
                    candidate.allow_partial_report = True
                    candidate.partial_report_requested_at = datetime.now()
                    candidate.status = TaskStatus.RUNNING

                # Publish durable state before exposing the candidate in cache.
                # A failed save leaves the original object completely untouched.
                await _async_save_session_under_lifecycle(
                    session_id, candidate, force=True,
                )
                async with state_lock:
                    if sessions.get(session_id) is not state:
                        raise SessionGenerationError(
                            "session state changed during partial-report approval"
                        )
                    sessions[session_id] = candidate
                    touch_access_time(session_id)

        transaction = asyncio.create_task(_persist_candidate_and_swap())
        await await_with_deferred_cancellation(transaction)

    try:
        await start_report_generation(
            session_id,
            expected_state=state,
            prepare=_prepare_partial_report,
        )
    except SessionWorkflowConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Another workflow, snapshot, or restore operation is active "
                "for this session"
            ),
        ) from exc
    return APIResponse(
        success=True,
        message="Partial report approved",
        data={"needs_reconnect": True},
    )
