# Bulk downloads: figures, tables, reproducibility bundle, archives.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import asyncio
import io
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from datetime import datetime
from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from backend.config import UPLOAD_TEMP_DIR, get_session_reports
from backend.models.schemas import AnalysisState

from backend.utils.files import (
    validate_path_within,
    validate_session_id,
    slugify_session_name,
    temp_file_on_error,
)
from backend.utils.auth import require_session_ownership
from backend.services.report.paths import resolve_report_output_dir, resolve_report_target_lineage

logger = logging.getLogger(__name__)

REPORT_ARCHIVE_PREFIX = "previous_report_v"


from .core import _should_skip_in_report_zip, router


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
