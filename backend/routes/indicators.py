"""Session indicator and data file management routes."""
import math
import os
from fastapi import APIRouter, HTTPException, Request

from backend.models.api_requests import BiochemistryInput
from backend.models.schemas import APIResponse, BiochemistryIndicator
from backend.services.session_store import async_save_session, ensure_session_loaded
from backend.utils.auth import require_session_ownership

router = APIRouter(prefix="/api", tags=["indicators"])


@router.post("/session/{session_id}/indicators")
async def add_indicator(session_id: str, request: Request, body: BiochemistryInput):
    """Add a manual biochemistry indicator"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if math.isnan(body.value) or math.isinf(body.value):
        raise HTTPException(status_code=400, detail="Value must be a finite number (not NaN or Infinity)")
    if not body.indicator.strip():
        raise HTTPException(status_code=400, detail="Indicator name cannot be empty or whitespace")
    if not body.unit.strip():
        raise HTTPException(status_code=400, detail="Unit cannot be empty or whitespace")

    group_id = body.group_id
    sample_number = body.sample_number
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
        indicator=body.indicator.strip(),
        value=body.value,
        unit=body.unit.strip(),
        group_id=group_id,
        sample_number=sample_number
    )
    state.biochemistry_indicators.append(indicator)
    await async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message=f"Biochemistry indicator added: {body.indicator}",
        data={"indicator_id": indicator.id}
    )


@router.delete("/session/{session_id}/indicators/{indicator_id}")
async def delete_indicator(session_id: str, indicator_id: str, request: Request):
    """Delete a biochemistry indicator"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    deleted_name = next((ind.indicator for ind in state.biochemistry_indicators if ind.id == indicator_id), None)
    if deleted_name is None:
        raise HTTPException(status_code=404, detail="Indicator not found")

    state.biochemistry_indicators = [ind for ind in state.biochemistry_indicators if ind.id != indicator_id]
    await async_save_session(session_id, state)
    return APIResponse(success=True, message=f"Indicator deleted: {deleted_name}")


@router.delete("/session/{session_id}/data/{file_id}")
async def delete_data_file(session_id: str, file_id: str, request: Request):
    """Delete any data file (pathology, sequencing, or biochemistry)"""
    import logging
    logger = logging.getLogger(__name__)

    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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
