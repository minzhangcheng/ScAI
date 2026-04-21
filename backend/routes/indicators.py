"""Session indicator and data file management routes."""
import math
import os
from fastapi import APIRouter, HTTPException

from backend.models.api_requests import BiochemistryInput
from backend.models.schemas import APIResponse, BiochemistryIndicator
from backend.services.session_store import async_save_session, ensure_session_loaded

router = APIRouter(prefix="/api", tags=["indicators"])


@router.post("/session/{session_id}/indicators")
async def add_indicator(session_id: str, request: BiochemistryInput):
    """Add a manual biochemistry indicator"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if math.isnan(request.value) or math.isinf(request.value):
        raise HTTPException(status_code=400, detail="Value must be a finite number (not NaN or Infinity)")
    if not request.indicator.strip():
        raise HTTPException(status_code=400, detail="Indicator name cannot be empty or whitespace")
    if not request.unit.strip():
        raise HTTPException(status_code=400, detail="Unit cannot be empty or whitespace")

    group_id = request.group_id
    sample_number = request.sample_number
    if group_id and group_id != "none":
        group = state.get_group_by_id(group_id)
        if group is None:
            raise HTTPException(status_code=400, detail="Group not found")
        if sample_number is not None and sample_number > group.sample_count:
            raise HTTPException(status_code=400, detail=f"Sample number exceeds group sample count ({group.sample_count})")
    else:
        group_id = None
        sample_number = None

    indicator = BiochemistryIndicator(
        indicator=request.indicator.strip(),
        value=request.value,
        unit=request.unit.strip(),
        group_id=group_id,
        sample_number=sample_number
    )
    state.biochemistry_indicators.append(indicator)
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=f"Biochemistry indicator added: {request.indicator}",
        data={"indicator_id": indicator.id}
    )


@router.delete("/session/{session_id}/indicators/{indicator_id}")
async def delete_indicator(session_id: str, indicator_id: str):
    """Delete a biochemistry indicator"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    deleted_name = next((ind.indicator for ind in state.biochemistry_indicators if ind.id == indicator_id), None)
    if deleted_name is None:
        raise HTTPException(status_code=404, detail="Indicator not found")

    state.biochemistry_indicators = [ind for ind in state.biochemistry_indicators if ind.id != indicator_id]
    await async_save_session(session_id, state)
    return APIResponse(success=True, message=f"Indicator deleted: {deleted_name}")


@router.delete("/session/{session_id}/data/{file_id}")
async def delete_data_file(session_id: str, file_id: str):
    """Delete any data file (pathology, sequencing, or biochemistry)"""
    import logging
    logger = logging.getLogger(__name__)

    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    for i, df in enumerate(state.data_files):
        if df.id == file_id:
            try:
                if os.path.exists(df.file_path):
                    os.remove(df.file_path)
            except OSError as e:
                logger.warning(f"Failed to delete file {df.file_path}: {e}")

            state.data_files.pop(i)
            await async_save_session(session_id, state)

            return APIResponse(
                success=True,
                message=f"Data file deleted: {df.filename}",
                data={"file_id": file_id}
            )

    raise HTTPException(status_code=404, detail="Data file not found")
