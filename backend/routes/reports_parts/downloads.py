# Bulk downloads: figures, tables, reproducibility bundle, archives.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import logging
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote as url_quote

from datetime import datetime
from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from backend.config import MAX_UPLOAD_SIZE, UPLOAD_TEMP_DIR
from backend.models.schemas import AnalysisState

from backend.utils.files import (
    validate_path_within,
    validate_session_id,
    slugify_session_name,
    temp_file_on_error,
)
from backend.utils.auth import require_session_ownership
from backend.utils.asyncio_tasks import run_in_owned_thread
from backend.services.report.paths import (
    resolve_published_report_dir,
    resolve_report_output_dir,
    resolve_report_target_lineage,
)

logger = logging.getLogger(__name__)


from .core import REPORT_ARCHIVE_PREFIX, _should_skip_in_report_zip, router


_BULK_ZIP_MAX_FILES = 10_000
_BULK_ZIP_MAX_SOURCE_BYTES = MAX_UPLOAD_SIZE
_BULK_ZIP_DISK_RESERVE_BYTES = 64 * 1024 * 1024


class _BulkZipRejected(Exception):
    """A bounded bulk archive cannot be built safely."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _ArchiveChanged(Exception):
    """An archive path no longer names the validated regular file."""


def _copy_validated_archive(
    source: Path,
    destination: Path,
    expected_identity: tuple[int, int],
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise _ArchiveChanged from exc
    try:
        metadata = os.fstat(source_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_identity
        ):
            raise _ArchiveChanged
        with os.fdopen(source_fd, "rb") as source_handle:
            source_fd = -1
            with open(destination, "wb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
    finally:
        if source_fd >= 0:
            os.close(source_fd)


def _build_disk_zip(
    source_dir: Path,
    destination: Path,
    *,
    recursive: bool,
    csv_only: bool = False,
    skip_report_artifacts: bool = False,
) -> None:
    """Build a bounded ZIP without retaining inputs or output in memory."""
    candidates = source_dir.rglob("*") if recursive else source_dir.glob("*")
    selected: list[tuple[Path, Path, int]] = []
    total_bytes = 0

    for file_path in sorted(candidates):
        if file_path.is_symlink() or not file_path.is_file():
            continue
        if csv_only and file_path.suffix.lower() != ".csv":
            continue
        if skip_report_artifacts and _should_skip_in_report_zip(file_path):
            continue

        size = file_path.stat().st_size
        selected.append((file_path, file_path.relative_to(source_dir), size))
        total_bytes += size
        if len(selected) > _BULK_ZIP_MAX_FILES:
            raise _BulkZipRejected(
                413,
                f"Archive contains more than {_BULK_ZIP_MAX_FILES} files",
            )
        if total_bytes > _BULK_ZIP_MAX_SOURCE_BYTES:
            raise _BulkZipRejected(
                413,
                f"Archive source exceeds {_BULK_ZIP_MAX_SOURCE_BYTES} bytes",
            )

    # Stored or poorly-compressible data can produce an archive slightly larger
    # than its inputs. Refuse before publication when the staging filesystem
    # cannot accommodate that conservative estimate.
    estimated_bytes = total_bytes + (len(selected) * 1024)
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < estimated_bytes + _BULK_ZIP_DISK_RESERVE_BYTES:
        raise _BulkZipRejected(507, "Insufficient disk space for report archive")

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname, _size in selected:
            zf.write(file_path, arcname)


async def _disk_zip_response(
    source_dir: Path,
    *,
    filename: str,
    recursive: bool,
    csv_only: bool = False,
    skip_report_artifacts: bool = False,
) -> FileResponse:
    """Stage a bounded ZIP on disk and retire it after the response completes."""
    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix="report_bundle_",
        suffix=".zip",
        delete=False,
        dir=str(UPLOAD_TEMP_DIR),
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    async with temp_file_on_error(str(tmp_path)):
        try:
            await run_in_owned_thread(
                _build_disk_zip,
                source_dir,
                tmp_path,
                recursive=recursive,
                csv_only=csv_only,
                skip_report_artifacts=skip_report_artifacts,
            )
        except _BulkZipRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        return FileResponse(
            tmp_path,
            media_type="application/zip",
            filename=filename,
            background=BackgroundTask(tmp_path.unlink, missing_ok=True),
        )


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
    figures_dir = (
        resolve_published_report_dir(session_id, branch_id_resolved) / "figures"
    )

    if not figures_dir.exists():
        raise HTTPException(status_code=404, detail="No figures found")

    return await _disk_zip_response(
        figures_dir,
        filename=f"{session_id}_figures.zip",
        recursive=False,
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
    tables_dir = (
        resolve_published_report_dir(session_id, branch_id_resolved) / "tables"
    )

    if not tables_dir.exists():
        raise HTTPException(status_code=404, detail="No tables found")

    return await _disk_zip_response(
        tables_dir,
        filename=f"{session_id}_tables.zip",
        recursive=False,
        csv_only=True,
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
    from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME

    repro_dir = get_session_workspace(session_id) / REPRODUCIBILITY_DIR_NAME
    if not repro_dir.exists():
        raise HTTPException(status_code=404, detail="Reproducibility bundle not found")

    return await _disk_zip_response(
        repro_dir,
        filename=f"{session_id}_reproducibility.zip",
        recursive=True,
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
    report_dir = resolve_published_report_dir(session_id, branch_id_resolved)

    if not report_dir.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    safe_name = slugify_session_name(state.name) if state.name else session_id
    return await _disk_zip_response(
        report_dir,
        filename=f"{safe_name}_full_report.zip",
        recursive=True,
        skip_report_artifacts=True,
    )

@router.get("/{session_id}/report/archives")
async def list_report_archives(
    session_id: str,
    request: Request,
    branch_id: Optional[str] = None,
):
    """List available report archive versions with metadata."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    reports_dir = resolve_report_output_dir(session_id, branch_id_resolved)
    if not reports_dir.exists():
        return []
    
    archives = []
    for archive_path in reports_dir.glob(f"{REPORT_ARCHIVE_PREFIX}*.zip"):
        try:
            metadata = archive_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                continue
            version = int(archive_path.stem.split("_v")[-1])
            archives.append({
                "version": version,
                "filename": archive_path.name,
                "created_at": datetime.fromtimestamp(metadata.st_mtime).isoformat(),
                "size_bytes": metadata.st_size,
                "download_url": (
                    f"session/{session_id}/report/download-archive/{archive_path.name}"
                    + (
                        f"?branch_id={url_quote(branch_id_resolved, safe='')}"
                        if branch_id_resolved else ""
                    )
                ),
            })
        except (OSError, ValueError, IndexError):
            continue
    
    return sorted(archives, key=lambda a: a["version"])

@router.get("/{session_id}/report/download-archive/{filename}")
async def download_report_archive(
    session_id: str,
    filename: str,
    request: Request,
    branch_id: Optional[str] = None,
):
    """Download a specific report archive zip file."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    
    target_lineage_id = resolve_report_target_lineage(branch_id)
    branch_id_resolved = None if target_lineage_id == "main" else target_lineage_id
    reports_dir = resolve_report_output_dir(session_id, branch_id_resolved)
    archive_path = reports_dir / filename
    if (
        not filename.startswith(REPORT_ARCHIVE_PREFIX)
        or not filename.endswith(".zip")
        or Path(filename).name != filename
    ):
        raise HTTPException(status_code=404, detail="Archive not found")
    try:
        metadata = archive_path.lstat()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archive not found")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise HTTPException(status_code=404, detail="Archive not found")
    resolved_archive = validate_path_within(archive_path, reports_dir)
    resolved_metadata = resolved_archive.stat()
    if (
        not resolved_archive.is_file()
        or (resolved_metadata.st_dev, resolved_metadata.st_ino)
        != (metadata.st_dev, metadata.st_ino)
    ):
        raise HTTPException(status_code=404, detail="Archive not found")
    
    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix="report_archive_",
        suffix=".zip",
        delete=False,
        dir=str(UPLOAD_TEMP_DIR),
    )
    staged_archive = Path(tmp.name)
    tmp.close()
    async with temp_file_on_error(str(staged_archive)):
        try:
            await run_in_owned_thread(
                _copy_validated_archive,
                resolved_archive,
                staged_archive,
                (metadata.st_dev, metadata.st_ino),
            )
        except _ArchiveChanged as exc:
            raise HTTPException(status_code=404, detail="Archive not found") from exc
        return FileResponse(
            staged_archive,
            media_type="application/zip",
            filename=filename,
            background=BackgroundTask(staged_archive.unlink, missing_ok=True),
        )
