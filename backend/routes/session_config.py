"""Session configuration routes (prompts, models, temperatures, vision, rounds, literature, agent-server)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import AgentTemperatures, PI_AGENT_NAME
from backend.models.api_requests import (
    PromptUpdateRequest, ModelUpdateRequest, TemperatureUpdateRequest,
    VisionSettingUpdateRequest, LiteratureConfigRequest, AgentServerConfigRequest,
)
from backend.models.schemas import APIResponse
from backend.services.session_store import async_save_session, ensure_session_loaded
from backend.state import system_config
from backend.utils.files import validate_session_id

router = APIRouter(prefix="/api", tags=["session-config"])


class MaxRoundsRequest(BaseModel):
    max_rounds: int = Field(ge=1, le=50, default=3)


class RoundWaitRequest(BaseModel):
    round_wait_seconds: int = Field(ge=-1, le=3600, default=300)


@router.post("/session/{session_id}/prompts")
async def update_prompts(session_id: str, request: PromptUpdateRequest):
    """Update agent prompts"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_prompts[request.agent] = request.prompt
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Prompt updated for {request.agent}")


@router.get("/session/{session_id}/prompts")
async def get_prompts(session_id: str):
    """Get current prompts"""
    cfg = system_config["current"]
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    prompts = {
        PI_AGENT_NAME: state.custom_prompts.get(PI_AGENT_NAME, cfg.prompts.pi_prompt if cfg else ""),
        "pathology": state.custom_prompts.get("pathology", cfg.prompts.pathology_prompt if cfg else ""),
        "bioinformatics": state.custom_prompts.get("bioinformatics", cfg.prompts.bioinformatics_prompt if cfg else ""),
        "biochemistry": state.custom_prompts.get("biochemistry", cfg.prompts.biochemistry_prompt if cfg else ""),
        "literature": state.custom_prompts.get("literature", cfg.prompts.literature_prompt if cfg else ""),
        "painter": state.custom_prompts.get("painter", cfg.prompts.painter_prompt if cfg else ""),
        "supervisor": state.custom_prompts.get("supervisor", cfg.prompts.supervisor_prompt if cfg else ""),
        "technique": state.custom_prompts.get("technique", ""),
    }
    return prompts


@router.post("/session/{session_id}/models")
async def update_models(session_id: str, request: ModelUpdateRequest):
    """Update agent models"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_models[request.agent] = request.model
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Model updated for {request.agent}")


@router.get("/session/{session_id}/models")
async def get_models(session_id: str):
    """Get current models"""
    cfg = system_config["current"]
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    models = {
        PI_AGENT_NAME: state.custom_models.get(PI_AGENT_NAME, cfg.models.pi_model if cfg else ""),
        "pathology": state.custom_models.get("pathology", cfg.models.pathology_model if cfg else ""),
        "bioinformatics": state.custom_models.get("bioinformatics", cfg.models.bioinformatics_model if cfg else ""),
        "biochemistry": state.custom_models.get("biochemistry", cfg.models.biochemistry_model if cfg else ""),
        "literature": state.custom_models.get("literature", cfg.models.literature_model if cfg else "google/gemini-3-flash-preview"),
        "painter": state.custom_models.get("painter", cfg.models.painter_model if cfg else "gemini-2.0-flash-exp-image-generation"),
        "supervisor": state.custom_models.get("supervisor", cfg.models.supervisor_model if cfg else "google/gemini-3.1-flash-lite-preview"),
        "technique": state.custom_models.get("technique", cfg.models.technique_model if cfg else "google/gemini-3.1-flash-lite-preview"),
    }
    return models


@router.post("/session/{session_id}/temperatures")
async def update_temperatures(session_id: str, request: TemperatureUpdateRequest):
    """Update agent temperature"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.custom_temperatures[request.agent] = request.temperature
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Temperature updated for {request.agent}: {request.temperature}")


@router.get("/session/{session_id}/temperatures")
async def get_temperatures(session_id: str):
    """Get current temperatures - uses centralized config with custom overrides"""
    cfg = system_config["current"]
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    base_temps = cfg.temperatures if cfg else AgentTemperatures()
    merged = base_temps.merge_custom(state.custom_temperatures)
    return merged.model_dump()


@router.post("/session/{session_id}/vision-settings")
async def update_vision_settings(session_id: str, request: VisionSettingUpdateRequest):
    """Update agent vision capability setting"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.agent_vision_settings[request.agent] = request.vision_enabled
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Vision setting updated for {request.agent}: {request.vision_enabled}")


@router.get("/session/{session_id}/vision-settings")
async def get_vision_settings(session_id: str):
    """Get agent vision capability settings"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"vision_settings": state.agent_vision_settings}


@router.post("/session/{session_id}/max-rounds")
async def update_max_rounds(session_id: str, request: MaxRoundsRequest):
    """Update maximum analysis rounds"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    state.max_rounds = request.max_rounds
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Max rounds updated to {request.max_rounds}")


@router.get("/session/{session_id}/max-rounds")
async def get_max_rounds(session_id: str):
    """Get current max rounds setting"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"max_rounds": state.max_rounds}


@router.post("/session/{session_id}/round-wait")
async def update_round_wait(session_id: str, request: RoundWaitRequest):
    """Update round wait countdown seconds (-1=manual, 0=no wait, positive=countdown)"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    state.round_wait_seconds = request.round_wait_seconds
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Round wait updated to {request.round_wait_seconds}s")


@router.post("/session/{session_id}/literature-config")
async def update_literature_config(session_id: str, request: LiteratureConfigRequest):
    """Update literature agent configuration"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.literature_config = {
        "mode": request.mode,
        "deep_research_model": request.deep_research_model,
        "deep_research_llm_base_url": request.deep_research_llm_base_url,
        "deep_research_api_key": request.deep_research_api_key,
        "google_api_key": request.google_api_key,
        "pubmed_email": request.pubmed_email,
        "max_search_rounds": request.max_search_rounds,
    }
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Literature config updated: mode={request.mode}")


@router.get("/session/{session_id}/literature-config")
async def get_literature_config(session_id: str):
    """Get current literature agent configuration"""
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.literature_config


@router.post("/session/{session_id}/agent-server-config")
async def update_agent_server_config(session_id: str, request: AgentServerConfigRequest):
    """Update per-agent LLM server configuration"""
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.agent_server_configs[request.agent] = {
        "llm_base_url": request.llm_base_url,
        "api_key": request.api_key,
    }
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Server config for {request.agent} updated")


@router.get("/session/{session_id}/agent-server-config")
async def get_agent_server_configs(session_id: str):
    """Get per-agent LLM server configurations"""
    validate_session_id(session_id)
    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.agent_server_configs
