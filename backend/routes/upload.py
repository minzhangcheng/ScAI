"""Upload routes for session files (data, generic files, and chunked uploads)."""
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import (
    UPLOAD_CHUNK_SIZE, MAX_UPLOAD_SIZE, get_session_uploads,
)
from backend.models.schemas import APIResponse, DataFile, TaskStatus, DATA_TYPE_SCRNASEQ, DATA_TYPE_BULK_RNASEQ
from backend.services.session_store import async_save_session, ensure_session_loaded
from backend.utils.files import safe_filename, validate_path_within
from backend.utils.upload import stream_save_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])


class InitChunkedUploadRequest(BaseModel):
    """Request to initialize a chunked upload"""
    filename: str
    total_size: int
    total_chunks: int
    file_type: str  # 'snrna_seq', 'loom', 'pathology', 'biochemistry'
    group: Optional[str] = None  # 'control' or 'treatment' for seq data


# ============== Upload Endpoints ==============

@router.post("/session/{session_id}/upload/data")
async def upload_data(
    session_id: str,
    file: UploadFile = File(...),
    section: str = Form(...),
    data_type: str = Form(...),
    group_id: Optional[str] = Form(None),
    sample_number: Optional[int] = Form(None),
    description: Optional[str] = Form(None)
):
    """Unified data file upload endpoint"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate section
    if section not in ("pathology", "sequencing", "biochemistry"):
        raise HTTPException(status_code=400, detail="Section must be 'pathology', 'sequencing', or 'biochemistry'")

    # Validate group_id if provided
    if group_id and group_id != "none":
        group = state.get_group_by_id(group_id)
        if group is None:
            raise HTTPException(status_code=400, detail="Group not found")
        # Validate sample_number
        if sample_number is not None and sample_number > group.sample_count:
            raise HTTPException(status_code=400, detail=f"Sample number exceeds group sample count ({group.sample_count})")
    else:
        group_id = None
        sample_number = None  # No sample distinction when no group

    # Build storage path
    if group_id:
        sample_dir = str(sample_number) if sample_number else "_unsorted"
        upload_parent = get_session_uploads(session_id) / section / group_id / sample_dir
    else:
        upload_parent = get_session_uploads(session_id) / section / "_ungrouped"

    sanitized_name = safe_filename(file.filename)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = upload_parent / sanitized_name
    upload_parent.mkdir(parents=True, exist_ok=True)
    validate_path_within(file_path, upload_parent)

    # Check available disk space before saving
    import shutil as _shutil
    disk_usage = _shutil.disk_usage(upload_parent)
    if file.size and disk_usage.free < file.size * 2:
        raise HTTPException(status_code=507, detail="Insufficient disk space for upload")

    # For large sequencing files, use stream save
    try:
        if section == "sequencing":
            await stream_save_file(file, file_path)
        else:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
    except OSError as e:
        logger.warning(f"Failed to save uploaded file {file_path}: {e}")
        # Clean up partial file on failure
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Create DataFile record
    data_file = DataFile(
        filename=sanitized_name,
        file_path=str(file_path),
        section=section,
        data_type=data_type,
        group_id=group_id,
        sample_number=sample_number,
        description=description
    )
    state.data_files.append(data_file)

    # Register as pending if workflow is waiting
    if state.status in (TaskStatus.WAITING_INPUT, TaskStatus.COMPLETED):
        state.pending_uploaded_files.append({
            "type": section,
            "filename": sanitized_name,
            "id": data_file.id
        })

    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=f"{section} {data_type} uploaded: {sanitized_name}",
        data={"file_id": data_file.id, "section": section, "data_type": data_type,
              "file_path": str(file_path), "group_id": group_id, "sample_number": sample_number}
    )


@router.post("/session/{session_id}/upload/file")
async def upload_generic_file(
    session_id: str,
    file: UploadFile = File(...)
):
    """Upload a generic file (for supplementary data)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save file with absolute path (sanitize filename to prevent path traversal)
    sanitized_name = safe_filename(file.filename)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    upload_parent = get_session_uploads(session_id) / "files"
    file_path = upload_parent / sanitized_name
    upload_parent.mkdir(parents=True, exist_ok=True)
    validate_path_within(file_path, upload_parent)

    # Stream-read file content with size limit (100MB for generic files)
    MAX_GENERIC_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    CHUNK_SIZE = 64 * 1024  # 64KB chunks
    total_size = 0
    chunks = []
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_GENERIC_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed ({MAX_GENERIC_FILE_SIZE // (1024 * 1024)}MB)"
            )
        chunks.append(chunk)
    content = b''.join(chunks)
    with open(file_path, "wb") as f:
        f.write(content)

    return APIResponse(
        success=True,
        message=f"File uploaded: {sanitized_name}",
        data={"filename": sanitized_name}
    )


# ============== Chunked Upload Endpoints (for large files up to 10GB) ==============


@router.post("/session/{session_id}/upload/init-chunked")
async def init_chunked_upload(session_id: str, request: InitChunkedUploadRequest):
    """Initialize a chunked upload for large files"""
    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.total_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed ({MAX_UPLOAD_SIZE / (1024**3):.1f}GB)"
        )

    upload_id = await upload_manager.init_upload(
        session_id=session_id,
        filename=request.filename,
        total_size=request.total_size,
        total_chunks=request.total_chunks,
        file_type=request.file_type,
        group=request.group
    )

    return {
        "upload_id": upload_id,
        "chunk_size": UPLOAD_CHUNK_SIZE,
        "total_chunks": request.total_chunks
    }


@router.post("/session/{session_id}/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(
    session_id: str,
    upload_id: str,
    chunk_index: int,
    file: UploadFile = File(...),
    chunk_hash: Optional[str] = Form(None)
):
    """Upload a single chunk"""
    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    chunk_data = await file.read()

    result = await upload_manager.save_chunk(
        upload_id=upload_id,
        chunk_index=chunk_index,
        chunk_data=chunk_data,
        chunk_hash=chunk_hash
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/session/{session_id}/upload/complete-chunked/{upload_id}")
async def complete_chunked_upload(
    session_id: str,
    upload_id: str,
    file_type: str = Form(None),
    group: Optional[str] = Form(None)
):
    """Complete chunked upload and assemble file"""
    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get upload metadata
    status = await upload_manager.get_upload_status(upload_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Use metadata values (set at init), not client-overridable form params
    file_type = status.get("file_type") or file_type
    group = status.get("group") or group

    # Determine final path based on file type
    filename = safe_filename(status["filename"])
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    session_uploads = get_session_uploads(session_id)
    if file_type == "snrna_seq":
        final_path = session_uploads / "snrna_seq" / (group or "control") / filename
    elif file_type == "loom":
        final_path = session_uploads / "loom" / (group or "control") / filename
    elif file_type == "bulk_rna_seq":
        final_path = session_uploads / "bulk_rna_seq" / (group or "control") / filename
    elif file_type == "pathology":
        final_path = session_uploads / "pathology" / filename
    elif file_type == "biochemistry":
        final_path = session_uploads / "biochemistry" / filename
    else:
        final_path = session_uploads / "files" / filename

    # Complete upload
    result = await upload_manager.complete_upload(upload_id, final_path.resolve())

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    # Update session state - create DataFile for all chunked uploads
    section_map = {"snrna_seq": "sequencing", "bulk_rna_seq": "sequencing", "loom": "sequencing", "pathology": "pathology", "biochemistry": "biochemistry"}
    type_map = {"snrna_seq": DATA_TYPE_SCRNASEQ, "bulk_rna_seq": DATA_TYPE_BULK_RNASEQ, "loom": "loom", "pathology": "uploaded", "biochemistry": "file"}
    section = section_map.get(file_type, file_type)
    data_type = type_map.get(file_type, file_type)

    data_file = DataFile(
        filename=filename,
        file_path=str(final_path.resolve()),
        section=section,
        data_type=data_type,
        group_id=status.get("group_id"),
        sample_number=status.get("sample_number"),
    )
    state.data_files.append(data_file)

    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=f"File uploaded successfully: {filename}",
        data=result
    )


@router.get("/session/{session_id}/upload/status/{upload_id}")
async def get_chunked_upload_status(session_id: str, upload_id: str):
    """Get chunked upload progress"""
    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    status = await upload_manager.get_upload_status(upload_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return status


@router.delete("/session/{session_id}/upload/cancel/{upload_id}")
async def cancel_chunked_upload(session_id: str, upload_id: str):
    """Cancel and cleanup a chunked upload"""
    # Import here to avoid circular import
    from backend.services.upload import upload_manager

    result = await upload_manager.cancel_upload(upload_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Upload not found"))
    return {"success": True, "message": "Upload cancelled"}
