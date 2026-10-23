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
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote as url_quote

from datetime import datetime
from fastapi import Depends, HTTPException, Request
from backend.config import UPLOAD_TEMP_DIR
from backend.models.schemas import AnalysisState

from backend.utils.files import (
    validate_path_within,
    validate_session_id,
    slugify_session_name,
)
from backend.utils.auth import require_session_ownership
from backend.utils.asyncio_tasks import run_in_owned_bulk_thread
from backend.services.export_lifecycle import (
    OwnedArchiveFileResponse,
    SecureArchiveError,
    build_secure_zip,
    copy_secure_file,
    unlink_export_archive_and_marker,
)
from backend.services.report.paths import (
    resolve_published_report_dir,
    resolve_report_output_dir,
    resolve_report_target_lineage,
)

logger = logging.getLogger(__name__)


from .core import REPORT_ARCHIVE_PREFIX, _should_skip_in_report_zip, router


_BULK_ZIP_DISK_RESERVE_BYTES = 64 * 1024 * 1024


def _inode_identity_from_fd(fd: int) -> tuple[int, int]:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("archive staging descriptor is not a regular file")
    return int(metadata.st_dev), int(metadata.st_ino)


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
    destination_fd: int,
    expected_identity: tuple[int, ...],
) -> None:
    try:
        copy_secure_file(
            source.parent,
            Path(source.name),
            destination,
            destination_fd=destination_fd,
            expected_identity=expected_identity,
            reserve_bytes=_BULK_ZIP_DISK_RESERVE_BYTES,
            include_source_space_estimate=False,
        )
    except SecureArchiveError as exc:
        if exc.status_code == 409:
            raise _ArchiveChanged from exc
        raise _BulkZipRejected(exc.status_code, exc.detail) from exc


def _build_disk_zip(
    source_dir: Path,
    destination: Path,
    destination_fd: int,
    *,
    recursive: bool,
    csv_only: bool = False,
    skip_report_artifacts: bool = False,
    extra_members: Iterable[tuple[Path, bytes]] = (),
) -> None:
    """Build a bounded ZIP without retaining inputs or output in memory."""
    def include(relative: Path) -> bool:
        file_path = source_dir / relative
        if csv_only and file_path.suffix.lower() != ".csv":
            return False
        if skip_report_artifacts and _should_skip_in_report_zip(file_path):
            return False
        return True

    try:
        build_secure_zip(
            source_dir,
            destination,
            destination_fd=destination_fd,
            recursive=recursive,
            include=include,
            max_files=None,
            max_source_bytes=None,
            reserve_bytes=_BULK_ZIP_DISK_RESERVE_BYTES,
            include_source_space_estimate=False,
            extra_members=extra_members,
        )
    except SecureArchiveError as exc:
        raise _BulkZipRejected(exc.status_code, exc.detail) from exc


async def _disk_zip_response(
    source_dir: Path,
    *,
    filename: str,
    recursive: bool,
    csv_only: bool = False,
    skip_report_artifacts: bool = False,
    extra_members: Iterable[tuple[Path, bytes]] = (),
) -> OwnedArchiveFileResponse:
    """Stage a bounded ZIP on disk and retire it after the response completes."""
    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix="report_bundle_",
        suffix=".zip",
        dir=str(UPLOAD_TEMP_DIR),
    )
    tmp_path = Path(raw_path)
    owned_identity = _inode_identity_from_fd(fd)
    try:
        try:
            await run_in_owned_bulk_thread(
                _build_disk_zip,
                source_dir,
                tmp_path,
                fd,
                recursive=recursive,
                csv_only=csv_only,
                skip_report_artifacts=skip_report_artifacts,
                extra_members=extra_members,
            )
        except _BulkZipRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        return OwnedArchiveFileResponse(
            tmp_path,
            file_fd=fd,
            owned_identity=owned_identity,
            media_type="application/zip",
            filename=filename,
        )
    except BaseException:
        try:
            unlink_export_archive_and_marker(
                tmp_path,
                expected_identity=owned_identity,
            )
        finally:
            os.close(fd)
        raise


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

# GET serves report HTML links; POST remains compatible with existing API callers.
@router.get("/{session_id}/report/download-reproducibility")
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
    """Download published files and retained round/stage reports for one lineage."""
    from backend.services.report.history_exports import (
        iter_report_history_members, snapshot_report_history,
    )

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
        extra_members=iter_report_history_members(
            snapshot_report_history(state), target_lineage_id,
        ),
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
        stat.S_ISLNK(resolved_metadata.st_mode)
        or not stat.S_ISREG(resolved_metadata.st_mode)
        or (resolved_metadata.st_dev, resolved_metadata.st_ino)
        != (metadata.st_dev, metadata.st_ino)
    ):
        raise HTTPException(status_code=404, detail="Archive not found")
    expected_identity = (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )
    
    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix="report_archive_",
        suffix=".zip",
        dir=str(UPLOAD_TEMP_DIR),
    )
    staged_archive = Path(raw_path)
    owned_identity = _inode_identity_from_fd(fd)
    try:
        try:
            await run_in_owned_bulk_thread(
                _copy_validated_archive,
                resolved_archive,
                staged_archive,
                fd,
                expected_identity,
            )
        except _ArchiveChanged as exc:
            raise HTTPException(
                status_code=409,
                detail="Archive changed during streaming",
            ) from exc
        except _BulkZipRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return OwnedArchiveFileResponse(
            staged_archive,
            file_fd=fd,
            owned_identity=owned_identity,
            media_type="application/zip",
            filename=filename,
        )
    except BaseException:
        try:
            unlink_export_archive_and_marker(
                staged_archive,
                expected_identity=owned_identity,
            )
        finally:
            os.close(fd)
        raise
