"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import io
import json
import logging
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import (
    SESSION_DIR, REPORTS_DIR, WORKSPACE_DIR, UPLOAD_DIR,
    get_session_dir, get_session_reports, get_session_workspace, PI_AGENT_NAME,
)
from backend.models.schemas import AnalysisState
from backend.services.session_store import ensure_session_loaded
from backend.state import detailed_reports, report_websockets
from backend.utils.files import safe_filename, validate_path_within, validate_session_id
from backend.utils.auth import require_session_ownership
from backend.utils.settings_provider import AgentSettingsProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["reports"])



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


# ============================================
# Exploration and Annotation Report HTML
# ============================================

@router.get("/{session_id}/exploration-report/html")
async def get_exploration_report_html(session_id: str):
    """Serve the exploration report HTML file"""
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    html_path = state.exploration_report_html_path
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=404, detail="Exploration report HTML not found")

    # Validate path is within the session directory to prevent path traversal
    session_dir = get_session_dir(session_id)
    validate_path_within(Path(html_path), session_dir)

    return FileResponse(html_path, media_type="text/html", filename="exploration_report.html")


@router.get("/{session_id}/annotation-report/html")
async def get_annotation_report_html(session_id: str):
    """Serve the annotation context report HTML file"""
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    html_path = state.annotation_report_html_path
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=404, detail="Annotation report not found")

    # Validate path is within the session directory to prevent path traversal
    session_dir = get_session_dir(session_id)
    validate_path_within(Path(html_path), session_dir)

    return FileResponse(html_path, media_type="text/html", filename="annotation_context_report.html")


@router.get("/{session_id}/detailed-report/html")
async def get_detailed_report_html(session_id: str):
    """Serve the combined incremental detailed report HTML."""
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not state.detailed_report_html_path:
        raise HTTPException(status_code=404, detail="Detailed report not yet generated")

    html_path = Path(state.detailed_report_html_path)
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Detailed report file not found")

    session_dir = get_session_dir(session_id)
    validate_path_within(html_path, session_dir)

    return FileResponse(str(html_path), media_type="text/html", filename="detailed_report.html")


# ============================================
# Detailed Report Generation
# ============================================

@router.get("/{session_id}/detailed-report")
async def get_detailed_report(session_id: str):
    """Get existing detailed report for a session"""
    # First check memory cache
    if session_id in detailed_reports:
        return {"report": _sanitize_nan(detailed_reports[session_id])}

    # Try to load from disk first (avoid session loading overhead)
    report_path = get_session_reports(session_id) / "report.json"
    if report_path.exists():
        try:
            # Use parse_float to handle NaN/Infinity in JSON text
            with open(report_path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Replace bare NaN/Infinity tokens with null before parsing
            raw = re.sub(r'\bNaN\b', 'null', raw)
            raw = re.sub(r'\b-?Infinity\b', 'null', raw)
            report = json.loads(raw)
            report = _sanitize_nan(report)
            detailed_reports[session_id] = report
            return {"report": report}
        except Exception as e:
            logger.exception(f"Failed to load report for {session_id}")

    # Only check session exists if no report found
    from backend.config import resolve_session_metadata_path
    session_meta = resolve_session_metadata_path(session_id)
    if not session_meta.exists() and session_id not in detailed_reports:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"report": None}


@router.post("/{session_id}/generate-detailed-report")
async def generate_detailed_report(session_id: str, background_tasks: BackgroundTasks):
    """Start generating a detailed report"""
    # Ensure session is loaded (from disk if necessary)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.status.value != "completed":
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
    # Import config inside function to avoid circular import
    from backend.state import system_config
    config = system_config.get("current")
    from backend.utils.agent_llm import get_agent_llm_client as _get_agent_llm_client
    from backend.config import get_session_workspace as _get_session_workspace
    
    # Debug: log state data paths
    logger.info(f"Report generation for session {session_id}")
    seq_files = state.get_files_by_section("sequencing")
    logger.info(f"  sequencing files: {len(seq_files)}")
    logger.info(f"  state.status: {state.status}")
    logger.info(f"  round_results count: {len(state.round_results)}")
    
    output_dir = str(get_session_reports(session_id))
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
                logger.warning(f"Failed to send progress update for session {session_id}: {e}", exc_info=True)
    
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
                logger.warning(f"Failed to send error to WebSocket for session {session_id}: {e}", exc_info=True)
        return
    
    llm_client = _get_agent_llm_client(state, config, PI_AGENT_NAME, session_id)

    # Create provider for resolving per-agent settings
    provider = AgentSettingsProvider(state, config, global_client=llm_client, session_id=session_id)

    # Get models via provider (respects session-level overrides)
    pi_model = provider.get_model(PI_AGENT_NAME)
    bio_model = provider.get_model("bioinformatics")

    try:
        workspace_dir = str(_get_session_workspace(session_id))

        # Import SupervisorAgent inside function to avoid circular import
        from backend.agents.supervisor_agent import SupervisorAgent

        # Use SupervisorAgent path for report generation
        supervisor = SupervisorAgent(
            settings_provider=provider,
            session_id=session_id,
        )

        # Restore supervision log from previously saved report.json
        report_json_path = Path(output_dir) / "report.json"
        if report_json_path.exists():
            try:
                with open(report_json_path, encoding="utf-8") as f:
                    saved_report = json.load(f)
                saved_log = saved_report.get("analysis_log", [])
                if saved_log:
                    supervisor.restore_log_from_saved(saved_log)
                    logger.info(f"Restored {len(saved_log)} supervision records from report.json")
            except Exception as e:
                logger.warning(f"Failed to restore supervision log from report.json: {e}")

        report = await supervisor.generate_detailed_report(
            analysis_state=state,
            report_model=pi_model,
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            progress_callback=progress_callback,
        )
        detailed_reports[session_id] = report

        # Notify completion via WebSocket
        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "complete",
                    "report": report
                })
            except Exception as e:
                logger.warning(f"Failed to send completion to WebSocket for session {session_id}: {e}", exc_info=True)

    except Exception as e:
        logger.exception("Report generation failed")

        if session_id in report_websockets:
            try:
                await report_websockets[session_id].send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception as ws_error:
                logger.warning(f"Failed to send error to WebSocket for session {session_id}: {ws_error}", exc_info=True)


# ============================================
# Report Download Endpoints
# ============================================

@router.post("/{session_id}/report/download-all-figures")
async def download_all_figures(session_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download all figures as a ZIP file"""
    validate_session_id(session_id)
    figures_dir = get_session_reports(session_id) / "figures"

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
async def download_all_tables(session_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download all tables as a ZIP file"""
    validate_session_id(session_id)
    tables_dir = get_session_reports(session_id) / "tables"

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
    from backend.config import get_session_workspace
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
async def download_full_report(session_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download full report including figures, tables, and report JSON"""
    validate_session_id(session_id)
    report_dir = get_session_reports(session_id)
    
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
    
    state = await ensure_session_loaded(session_id)
    if state and hasattr(state, 'name'):
        filename = f"{state.name}_full_report.zip"
    else:
        filename = f"{session_id}_full_report.zip"
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================
# Report HTML and Preview
# ============================================

@router.get("/{session_id}/report/html")
async def get_report_html(session_id: str):
    """Get the HTML report for preview"""
    validate_session_id(session_id)
    report_dir = get_session_reports(session_id)
    html_path = report_dir / "report.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found")

    # Don't set filename to avoid triggering download
    return FileResponse(
        html_path,
        media_type="text/html"
    )


@router.get("/{session_id}/report/preview")
async def preview_report(session_id: str):
    """Get report preview data including HTML content and metadata"""
    validate_session_id(session_id)
    report_dir = get_session_reports(session_id)

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


@router.get("/{session_id}/report/figure/{filename}")
async def get_report_figure(session_id: str, filename: str, state: AnalysisState = Depends(require_session_ownership)):
    """Get a specific figure from the report"""
    safe_name = safe_filename(filename)
    figure_parent = get_session_reports(session_id) / "figures"
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
async def get_report_figure_alt(session_id: str, filename: str, state: AnalysisState = Depends(require_session_ownership)):
    """Get a specific figure from the report (alternative path for HTML relative URLs)"""
    return await get_report_figure(session_id, filename)


@router.get("/{session_id}/report/table/{filename}")
async def get_report_table(session_id: str, filename: str, state: AnalysisState = Depends(require_session_ownership)):
    """Get a specific table from the report"""
    safe_name = safe_filename(filename)
    table_parent = get_session_reports(session_id) / "tables"
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
async def get_report_table_alt(session_id: str, filename: str, state: AnalysisState = Depends(require_session_ownership)):
    """Get a specific table from the report (alternative path for HTML relative URLs)"""
    return await get_report_table(session_id, filename)


@router.get("/{session_id}/report/{filename:path}")
async def get_report_file(session_id: str, filename: str, state: AnalysisState = Depends(require_session_ownership)):
    """Catch-all: serve any file from the report directory (notebook, sub-reports, etc.)"""
    validate_session_id(session_id)
    safe_name = safe_filename(filename)
    report_dir = get_session_reports(session_id)
    file_path = report_dir / safe_name
    validate_path_within(file_path, report_dir)

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


@router.get("/{session_id}/publication-figures")
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


@router.get("/{session_id}/publication-tables")
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


# ============================================
# Code History and Report Summary
# ============================================

@router.get("/{session_id}/code")
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


@router.get("/{session_id}/report")
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
