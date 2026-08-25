"""Upload routes for session files (data, generic files, and chunked uploads)."""
import asyncio
import logging
import os
import re
import stat
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from backend.config import (
    UPLOAD_CHUNK_SIZE, MAX_UPLOAD_SIZE, get_session_uploads,
)
from backend.models.schemas import APIResponse, DataFile, TaskStatus, DATA_TYPE_SCRNASEQ, DATA_TYPE_BULK_RNASEQ
from backend.services.session_store import (
    _async_save_session_under_lifecycle,
)
from backend.services.session_lifecycle import (
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.services.upload import read_chunk_with_limit
from backend.utils.auth import require_session_ownership
from backend.utils.files import fsync_directory, safe_filename, validate_path_within
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    run_in_owned_thread,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])
_LEGACY_CHUNK_FILE_TYPES = {
    "snrna_seq",
    "loom",
    "bulk_rna_seq",
    "pathology",
    "biochemistry",
    "file",
}
_LEGACY_CHUNK_GROUPS = {None, "control", "treatment"}
_GENERIC_UPLOAD_MAX_BYTES = MAX_UPLOAD_SIZE


def _open_upload_destination(dest: Path) -> int:
    """Create an operation-owned upload path without following a symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(dest, flags, 0o600)


def _write_upload_chunk(fd: int, chunk: bytes) -> None:
    """Write a complete chunk to an unbuffered upload descriptor."""
    remaining = memoryview(chunk)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("upload write made no progress")
        remaining = remaining[written:]


@asynccontextmanager
async def _upload_generation_barrier(session_id: str, state):
    """Serialize an upload mutation against delete and same-ID reuse."""
    async with get_lifecycle_lock(session_id):
        assert_generation_writable(session_id, state)
        yield


def _safe_upload_log_token(value: str, limit: int = 160) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or ""))[:limit]


def _upload_validation_error(
    *,
    request_id: str,
    session_id: str,
    error_code: str,
    detail: str,
    section: str,
    data_type: str,
    sanitized_filename: str = "",
) -> HTTPException:
    """Build a compatible 400 response with safe correlation metadata."""
    logger.warning(
        "Upload validation failed request_id=%s session_id=%s error_code=%s "
        "stage=validation status=400 section=%s data_type=%s filename=%s",
        request_id,
        _safe_upload_log_token(session_id),
        error_code,
        _safe_upload_log_token(section),
        _safe_upload_log_token(data_type),
        _safe_upload_log_token(sanitized_filename) or "<unavailable>",
    )
    return HTTPException(
        status_code=400,
        detail=detail,
        headers={
            "X-Request-ID": request_id,
            "X-ScAI-Error-Code": error_code,
        },
    )


async def _stream_generic_upload(
    upload_file,
    dest: Path,
    max_bytes: int | None = MAX_UPLOAD_SIZE,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Stream into one operation-owned path without clobbering a prior file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    created = False

    async def _stream() -> None:
        nonlocal created
        total = 0
        fd: int | None = None
        try:
            # Keep the descriptor inside this independently drained task.  In
            # particular, do not use aiofiles here: its shared default
            # executor can outlive a cancelled request on CPython 3.13.
            fd = _open_upload_destination(dest)
            created = True
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File size exceeds maximum allowed "
                            f"({max_bytes // (1024 * 1024)}MB)"
                        ),
                    )
                await run_in_owned_thread(_write_upload_chunk, fd, chunk)
            await run_in_owned_thread(os.fsync, fd)
        finally:
            if fd is not None:
                os.close(fd)

    stream_task = asyncio.create_task(_stream())
    try:
        await await_with_deferred_cancellation(stream_task)
    except BaseException:
        if created:
            try:
                await run_in_owned_thread(dest.unlink, True)
            except OSError:
                logger.exception("Failed to clean upload staging file %s", dest)
        raise
    return dest


def _upload_staging_path(final_path: Path) -> Path:
    return final_path.with_name(
        f".{final_path.name}.{uuid.uuid4().hex}.uploading"
    )


def _same_regular_file(left: Path, right: Path) -> bool:
    try:
        left_stat = left.lstat()
        right_stat = right.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(left_stat.st_mode)
        and stat.S_ISREG(right_stat.st_mode)
        and left_stat.st_dev == right_stat.st_dev
        and left_stat.st_ino == right_stat.st_ino
    )


def _link_upload_staging(
    staging_path: Path,
    final_path: Path,
    publication: dict[str, bool],
) -> None:
    """Publish by hard link so an existing destination is never replaced."""
    os.link(staging_path, final_path, follow_symlinks=False)
    publication["linked"] = True
    if not _same_regular_file(staging_path, final_path):
        raise OSError("published upload does not match its staging inode")
    fsync_directory(final_path.parent)


def _cleanup_upload_transaction(
    staging_path: Path,
    final_path: Path | None = None,
) -> None:
    """Remove only paths still proven to belong to this upload operation."""
    if final_path is not None and _same_regular_file(staging_path, final_path):
        final_path.unlink(missing_ok=True)
    staging_path.unlink(missing_ok=True)
    try:
        fsync_directory(staging_path.parent)
    except OSError:
        # The unlink is already complete; some filesystems cannot fsync dirs.
        pass


async def _cleanup_upload_transaction_safely(
    staging_path: Path,
    final_path: Path | None = None,
) -> None:
    try:
        await run_in_owned_thread(
            _cleanup_upload_transaction,
            staging_path,
            final_path,
        )
    except OSError:
        logger.exception(
            "Failed to clean upload transaction staging=%s final=%s",
            staging_path,
            final_path,
        )


async def _publish_upload_staging(
    staging_path: Path,
    final_path: Path,
) -> None:
    publication = {"linked": False}
    try:
        await run_in_owned_thread(
            _link_upload_staging,
            staging_path,
            final_path,
            publication,
        )
    except FileExistsError as exc:
        await _cleanup_upload_transaction_safely(staging_path)
        raise HTTPException(
            status_code=409,
            detail="A file with this name already exists",
        ) from exc
    except BaseException:
        await _cleanup_upload_transaction_safely(
            staging_path,
            final_path if publication["linked"] else None,
        )
        raise


class InitChunkedUploadRequest(BaseModel):
    """Request to initialize a legacy web chunked upload."""
    filename: str
    total_size: int
    total_chunks: int
    file_type: str  # 'snrna_seq', 'loom', 'pathology', 'biochemistry'
    group: Optional[str] = None  # 'control' or 'treatment' for seq data
    expected_sha256: str = Field(
        description=(
            "Required SHA-256 digest of the complete file as 64 hexadecimal "
            "characters. Clients predating ScAI 0.38.11 must compute and send "
            "this field before initializing an upload."
        ),
    )


# ============== Upload Endpoints ==============

@router.post("/session/{session_id}/upload/data")
async def upload_data(
    session_id: str,
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    section: str = Form(...),
    data_type: str = Form(...),
    group_id: Optional[str] = Form(None),
    sample_number: Optional[int] = Form(None),
    description: Optional[str] = Form(None)
):
    """Unified data file upload endpoint"""
    request_id = uuid.uuid4().hex
    response.headers["X-Request-ID"] = request_id
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    sanitized_name = safe_filename(file.filename)

    # Validate section
    if section not in ("pathology", "sequencing", "biochemistry"):
        raise _upload_validation_error(
            request_id=request_id,
            session_id=session_id,
            error_code="UPLOAD_INVALID_SECTION",
            detail="Section must be 'pathology', 'sequencing', or 'biochemistry'",
            section=section,
            data_type=data_type,
            sanitized_filename=sanitized_name,
        )

    # Validate group_id if provided
    if group_id and group_id != "none":
        group = state.get_group_by_id(group_id)
        if group is None:
            raise _upload_validation_error(
                request_id=request_id,
                session_id=session_id,
                error_code="UPLOAD_GROUP_NOT_FOUND",
                detail="Group not found",
                section=section,
                data_type=data_type,
                sanitized_filename=sanitized_name,
            )
        # Validate sample_number
        if sample_number is not None and not (
            1 <= sample_number <= group.sample_count
        ):
            raise _upload_validation_error(
                request_id=request_id,
                session_id=session_id,
                error_code="UPLOAD_SAMPLE_NUMBER_OUT_OF_RANGE",
                detail=(
                    "Sample number must be between 1 and the group sample "
                    f"count ({group.sample_count})"
                ),
                section=section,
                data_type=data_type,
                sanitized_filename=sanitized_name,
            )
    else:
        group_id = None
        sample_number = None  # No sample distinction when no group

    async with _upload_generation_barrier(session_id, state):
        session_uploads = get_session_uploads(session_id)
        if group_id:
            sample_dir = (
                str(sample_number)
                if sample_number is not None
                else "_unsorted"
            )
            upload_parent = (
                session_uploads
                / section
                / group_id
                / sample_dir
            )
        else:
            upload_parent = (
                session_uploads / section / "_ungrouped"
            )

        if not sanitized_name:
            raise _upload_validation_error(
                request_id=request_id,
                session_id=session_id,
                error_code="UPLOAD_INVALID_FILENAME",
                detail="Invalid filename",
                section=section,
                data_type=data_type,
            )
        file_path = upload_parent / sanitized_name
        # Imported state can carry legacy/non-UUID group IDs. Validate against
        # the session root before mkdir; validating only against the derived
        # parent would bless an already-escaped ``../`` parent.
        validate_path_within(upload_parent, session_uploads)
        validate_path_within(file_path, session_uploads)
        upload_parent.mkdir(parents=True, exist_ok=True)

        data_file = DataFile(
            filename=sanitized_name,
            file_path=str(file_path),
            section=section,
            data_type=data_type,
            group_id=group_id,
            sample_number=sample_number,
            description=description,
        )
        final_resolved = file_path.resolve()
        if os.path.lexists(file_path) or any(
            Path(str(existing.file_path)).resolve() == final_resolved
            for existing in state.data_files
            if getattr(existing, "file_path", None)
        ):
            raise HTTPException(
                status_code=409,
                detail="A file with this name already exists",
            )
        pending_record = None
        if state.status in (TaskStatus.WAITING_INPUT, TaskStatus.COMPLETED):
            pending_record = {
                "type": section,
                "filename": sanitized_name,
                "id": data_file.id,
            }

        staging_path = _upload_staging_path(file_path)
        published = False
        data_file_appended = False
        pending_record_appended = False
        save_committed = False
        save_task: asyncio.Task | None = None
        try:
            await _stream_generic_upload(
                file,
                staging_path,
                max_bytes=_GENERIC_UPLOAD_MAX_BYTES,
            )
            await _publish_upload_staging(staging_path, file_path)
            published = True

            state.data_files.append(data_file)
            data_file_appended = True
            if pending_record is not None:
                state.pending_uploaded_files.append(pending_record)
                pending_record_appended = True

            save_task = asyncio.create_task(
                _async_save_session_under_lifecycle(
                    session_id,
                    state,
                    force=True,
                ),
                name=f"upload-state-save:{session_id}:{data_file.id}",
            )
            try:
                await await_with_deferred_cancellation(save_task)
                save_committed = True
            except asyncio.CancelledError:
                if save_task.done() and not save_task.cancelled():
                    try:
                        save_task.result()
                    except BaseException:
                        pass
                    else:
                        save_committed = True
                raise

            await _cleanup_upload_transaction_safely(staging_path)
        except BaseException:
            if save_committed:
                # The caller disconnected after the forced state write became
                # durable. Keep the committed record/file and retire only the
                # ownership link before preserving cancellation.
                await _cleanup_upload_transaction_safely(staging_path)
            else:
                if pending_record_appended:
                    for index, item in enumerate(state.pending_uploaded_files):
                        if item is pending_record:
                            state.pending_uploaded_files.pop(index)
                            break
                if data_file_appended:
                    for index, item in enumerate(state.data_files):
                        if item is data_file:
                            state.data_files.pop(index)
                            break
                await _cleanup_upload_transaction_safely(
                    staging_path,
                    file_path if published else None,
                )
            raise

        return APIResponse(
            success=True,
            message=f"{section} {data_type} uploaded: {sanitized_name}",
            data={
                "file_id": data_file.id,
                "section": section,
                "data_type": data_type,
                "file_path": str(file_path),
                "group_id": group_id,
                "sample_number": sample_number,
                "request_id": request_id,
            },
        )


@router.post("/session/{session_id}/upload/file")
async def upload_generic_file(
    session_id: str,
    request: Request,
    file: UploadFile = File(...)
):
    """Upload a generic file (for supplementary data)"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    
    # Save file with absolute path (sanitize filename to prevent path traversal)
    sanitized_name = safe_filename(file.filename)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    async with _upload_generation_barrier(session_id, state):
        upload_parent = get_session_uploads(session_id) / "files"
        file_path = upload_parent / sanitized_name
        upload_parent.mkdir(parents=True, exist_ok=True)
        validate_path_within(file_path, upload_parent)

        max_generic_file_size = _GENERIC_UPLOAD_MAX_BYTES
        if os.path.lexists(file_path):
            raise HTTPException(
                status_code=409,
                detail="A file with this name already exists",
            )
        staging_path = _upload_staging_path(file_path)
        await _stream_generic_upload(
            file,
            staging_path,
            max_bytes=max_generic_file_size,
            chunk_size=64 * 1024,
        )
        await _publish_upload_staging(staging_path, file_path)
        await _cleanup_upload_transaction_safely(staging_path)

        return APIResponse(
            success=True,
            message=f"File uploaded: {sanitized_name}",
            data={"filename": sanitized_name},
        )


# ============== Chunked Upload Endpoints (for large files up to 10GB) ==============


@router.post("/session/{session_id}/upload/init-chunked")
async def init_chunked_upload(session_id: str, request: Request, body: InitChunkedUploadRequest):
    """Initialize a chunked upload whose final bytes must match the required digest."""
    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if body.total_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed ({MAX_UPLOAD_SIZE / (1024**3):.1f}GB)"
        )
    if body.total_size <= 0:
        raise HTTPException(status_code=400, detail="total_size must be positive")
    if body.file_type not in _LEGACY_CHUNK_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file_type")
    if body.group not in _LEGACY_CHUNK_GROUPS:
        raise HTTPException(status_code=400, detail="Unsupported upload group")
    expected_sha256 = body.expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise HTTPException(
            status_code=400,
            detail="expected_sha256 must be a 64-character hex digest",
        )
    expected_chunks = (
        body.total_size + UPLOAD_CHUNK_SIZE - 1
    ) // UPLOAD_CHUNK_SIZE
    if body.total_chunks != expected_chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                f"total_chunks must equal {expected_chunks} for the "
                "declared size"
            ),
        )

    async with _upload_generation_barrier(session_id, state):
        try:
            upload_id = await upload_manager.init_upload(
                session_id=session_id,
                filename=body.filename,
                total_size=body.total_size,
                total_chunks=body.total_chunks,
                file_type=body.file_type,
                group=body.group,
                expected_sha256=expected_sha256,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "upload_id": upload_id,
        "chunk_size": UPLOAD_CHUNK_SIZE,
        "total_chunks": body.total_chunks
    }


@router.post("/session/{session_id}/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(
    session_id: str,
    upload_id: str,
    chunk_index: int,
    request: Request,
    file: UploadFile = File(...),
    chunk_hash: Optional[str] = Form(None)
):
    """Upload a single chunk"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(
        session_id=session_id, user_id=user_id,
    )

    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    chunk_data = await read_chunk_with_limit(
        file, max_bytes=UPLOAD_CHUNK_SIZE,
    )

    async with _upload_generation_barrier(session_id, state):
        result = await upload_manager.save_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_data=chunk_data,
            chunk_hash=chunk_hash,
            session_id=session_id,
        )

    if not result["success"]:
        status_code = 404 if result.get("not_found") else (
            413 if result.get("too_large") else 400
        )
        raise HTTPException(status_code=status_code, detail=result["error"])

    return result


@router.post("/session/{session_id}/upload/complete-chunked/{upload_id}")
async def complete_chunked_upload(
    session_id: str,
    upload_id: str,
    request: Request,
    file_type: str = Form(None),
    group: Optional[str] = Form(None)
):
    """Complete chunked upload and assemble file"""
    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    async with _upload_generation_barrier(session_id, state):
        status = await upload_manager.get_upload_status(
            upload_id, session_id=session_id,
        )
        if status is None:
            raise HTTPException(status_code=404, detail="Upload not found")

        file_type = status.get("file_type") or file_type
        group = status.get("group") or group
        if file_type not in _LEGACY_CHUNK_FILE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file_type")
        if group not in _LEGACY_CHUNK_GROUPS:
            raise HTTPException(status_code=400, detail="Unsupported upload group")
        filename = safe_filename(status["filename"])
        if not filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        session_uploads = get_session_uploads(session_id)
        if file_type == "snrna_seq":
            final_path = (
                session_uploads / "snrna_seq" / (group or "control") / filename
            )
        elif file_type == "loom":
            final_path = (
                session_uploads / "loom" / (group or "control") / filename
            )
        elif file_type == "bulk_rna_seq":
            final_path = (
                session_uploads
                / "bulk_rna_seq"
                / (group or "control")
                / filename
            )
        elif file_type == "pathology":
            final_path = session_uploads / "pathology" / filename
        elif file_type == "biochemistry":
            final_path = session_uploads / "biochemistry" / filename
        else:
            final_path = session_uploads / "files" / filename
        final_path = validate_path_within(final_path, session_uploads)

        section_map = {
            "snrna_seq": "sequencing",
            "bulk_rna_seq": "sequencing",
            "loom": "sequencing",
            "pathology": "pathology",
            "biochemistry": "biochemistry",
        }
        type_map = {
            "snrna_seq": DATA_TYPE_SCRNASEQ,
            "bulk_rna_seq": DATA_TYPE_BULK_RNASEQ,
            "loom": "loom",
            "pathology": "uploaded",
            "biochemistry": "file",
        }
        # Generic supplementary files follow /upload/file: they are stored but
        # are not analysis inputs and therefore have no valid DataFile section.
        # Construct every typed record before assembly so schema validation can
        # never consume the upload's resumability artifacts.
        data_file = None
        existing_data_file = None
        if file_type != "file":
            data_file_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"scai:legacy-chunk-upload:{session_id}:{upload_id}",
            ))
            data_file = DataFile(
                id=data_file_id,
                filename=filename,
                file_path=str(final_path.resolve()),
                section=section_map[file_type],
                data_type=type_map[file_type],
                group_id=status.get("group_id"),
                sample_number=status.get("sample_number"),
            )
            existing_data_file = next(
                (item for item in state.data_files if item.id == data_file_id),
                None,
            )
            record_fields = (
                "filename",
                "file_path",
                "section",
                "data_type",
                "group_id",
                "sample_number",
            )
            if existing_data_file is not None and any(
                getattr(existing_data_file, field) != getattr(data_file, field)
                for field in record_fields
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Upload record conflicts with committed session state",
                )

        result = await upload_manager.complete_upload(
            upload_id,
            final_path,
            session_id=session_id,
            defer_cleanup=True,
        )
        if not result["success"]:
            raise HTTPException(
                status_code=404 if result.get("not_found") else 400,
                detail=result["error"],
            )

        def _remove_uncommitted_record() -> None:
            if data_file is None:
                return
            state.data_files[:] = [
                item for item in state.data_files if item is not data_file
            ]

        async def _rollback_assembly() -> None:
            _remove_uncommitted_record()
            rolled_back = await upload_manager.rollback_assembled_upload(
                upload_id,
                final_path,
                session_id=session_id,
            )
            if not rolled_back:
                raise RuntimeError(
                    "Could not roll back an unpublished chunked upload"
                )

        cancellation = None
        save_completed = data_file is None
        if data_file is not None and existing_data_file is None:
            state.data_files.append(data_file)

            async def _persist_record() -> None:
                nonlocal save_completed
                # This is a transaction commit, not a best-effort checkpoint;
                # bypass the normal five-second save throttle.
                await _async_save_session_under_lifecycle(
                    session_id,
                    state,
                    force=True,
                )
                save_completed = True

            save_task = asyncio.create_task(
                _persist_record(),
                name=f"legacy-upload-state-commit:{session_id}:{upload_id}",
            )
            try:
                await await_with_deferred_cancellation(save_task)
            except asyncio.CancelledError as exc:
                if save_completed:
                    cancellation = exc
                else:
                    rollback = asyncio.create_task(
                        _rollback_assembly(),
                        name=f"legacy-upload-rollback:{session_id}:{upload_id}",
                    )
                    await await_with_deferred_cancellation(rollback)
                    raise
            except BaseException:
                rollback = asyncio.create_task(
                    _rollback_assembly(),
                    name=f"legacy-upload-rollback:{session_id}:{upload_id}",
                )
                await await_with_deferred_cancellation(rollback)
                raise
        elif existing_data_file is not None:
            save_completed = True

        cleanup = asyncio.create_task(
            upload_manager.commit_completed_upload(
                upload_id,
                session_id=session_id,
                data_file_id=data_file.id if data_file is not None else None,
            ),
            name=f"legacy-upload-cleanup:{session_id}:{upload_id}",
        )
        try:
            cleanup_complete = await await_with_deferred_cancellation(cleanup)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            cleanup_complete = cleanup.result()
        result["cleanup_pending"] = not cleanup_complete

        if cancellation is not None:
            raise cancellation

        return APIResponse(
            success=True,
            message=f"File uploaded successfully: {filename}",
            data=result,
        )


@router.get("/session/{session_id}/upload/status/{upload_id}")
async def get_chunked_upload_status(session_id: str, upload_id: str, request: Request):
    """Get chunked upload progress"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(
        session_id=session_id, user_id=user_id,
    )

    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    status = await upload_manager.get_upload_status(
        upload_id, session_id=session_id,
    )
    if status is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return status


@router.delete("/session/{session_id}/upload/cancel/{upload_id}")
async def cancel_chunked_upload(session_id: str, upload_id: str, request: Request):
    """Cancel and cleanup a chunked upload"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(
        session_id=session_id, user_id=user_id,
    )

    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    async with _upload_generation_barrier(session_id, state):
        result = await upload_manager.cancel_upload(
            upload_id, session_id=session_id,
        )
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Upload not found"))
    return {"success": True, "message": "Upload cancelled"}
