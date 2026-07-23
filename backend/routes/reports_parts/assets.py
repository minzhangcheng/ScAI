# Per-file report assets: html, figures, tables, arbitrary files.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse

from backend.models.schemas import AnalysisState

from backend.utils.files import safe_filename, validate_path_within, validate_session_id
from backend.utils.auth import require_session_ownership
from backend.services.report.paths import resolve_report_output_dir, resolve_report_target_lineage

logger = logging.getLogger(__name__)

REPORT_ARCHIVE_PREFIX = "previous_report_v"


from .core import _branch_query, router


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
