# Publication figures/tables, graphic abstract, integrated report, code history.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from datetime import datetime
from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse

from backend.config import WORKSPACE_DIR, UPLOAD_DIR, get_session_workspace
from backend.models.schemas import AnalysisState, APIResponse, TaskStatus

from backend.utils.files import validate_session_id
from backend.utils.auth import require_session_ownership
from ..serialization import serialize_round_results
from backend.services.session_store import async_save_session, update_session_cache as _update_session_cache

logger = logging.getLogger(__name__)

REPORT_ARCHIVE_PREFIX = "previous_report_v"


from .core import router


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
