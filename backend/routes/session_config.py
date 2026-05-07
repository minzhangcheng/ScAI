"""Session configuration routes (prompts, models, temperatures, vision, rounds, literature, agent-server)."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.config import AgentTemperatures, PI_AGENT_NAME
from backend.models.api_requests import (
    PromptUpdateRequest, ModelUpdateRequest, TemperatureUpdateRequest,
    VisionSettingUpdateRequest, LiteratureConfigRequest, AgentServerConfigRequest,
)
from backend.models.schemas import APIResponse, DEFAULT_ROUND_WAIT_SECONDS
from backend.services.session_store import async_save_session, ensure_session_loaded
from backend.state import system_config
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id

router = APIRouter(prefix="/api", tags=["session-config"])


class MaxRoundsRequest(BaseModel):
    max_rounds: int = Field(ge=1, le=50, default=3)


class RoundWaitRequest(BaseModel):
    round_wait_seconds: int = Field(ge=-1, le=3600, default=DEFAULT_ROUND_WAIT_SECONDS)


@router.post("/session/{session_id}/prompts")
async def update_prompts(session_id: str, request: Request, body: PromptUpdateRequest):
    """Update agent prompts"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    state.custom_prompts[body.agent] = body.prompt
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Prompt updated for {body.agent}")


@router.get("/session/{session_id}/prompts")
async def get_prompts(session_id: str, request: Request):
    """Get current prompts"""
    cfg = system_config["current"]
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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
async def update_models(session_id: str, request: Request, body: ModelUpdateRequest):
    """Update agent models"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    state.custom_models[body.agent] = body.model
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Model updated for {body.agent}")


@router.get("/session/{session_id}/models")
async def get_models(session_id: str, request: Request):
    """Get current models"""
    cfg = system_config["current"]
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

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
async def update_temperatures(session_id: str, request: Request, body: TemperatureUpdateRequest):
    """Update agent temperature"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    state.custom_temperatures[body.agent] = body.temperature
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Temperature updated for {body.agent}: {body.temperature}")


@router.get("/session/{session_id}/temperatures")
async def get_temperatures(session_id: str, request: Request):
    """Get current temperatures - uses centralized config with custom overrides"""
    cfg = system_config["current"]
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    base_temps = cfg.temperatures if cfg else AgentTemperatures()
    merged = base_temps.merge_custom(state.custom_temperatures)
    return merged.model_dump()


@router.post("/session/{session_id}/vision-settings")
async def update_vision_settings(session_id: str, request: Request, body: VisionSettingUpdateRequest):
    """Update agent vision capability setting"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    state.agent_vision_settings[body.agent] = body.vision_enabled
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Vision setting updated for {body.agent}: {body.vision_enabled}")


@router.get("/session/{session_id}/vision-settings")
async def get_vision_settings(session_id: str, request: Request):
    """Get agent vision capability settings"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    return {"vision_settings": state.agent_vision_settings}


@router.post("/session/{session_id}/max-rounds")
async def update_max_rounds(session_id: str, request: Request, body: MaxRoundsRequest):
    """Update maximum analysis rounds"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    state.max_rounds = body.max_rounds
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Max rounds updated to {body.max_rounds}")


@router.get("/session/{session_id}/max-rounds")
async def get_max_rounds(session_id: str, request: Request):
    """Get current max rounds setting"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return {"max_rounds": state.max_rounds}


@router.post("/session/{session_id}/round-wait")
async def update_round_wait(session_id: str, request: Request, body: RoundWaitRequest):
    """Update round wait countdown seconds (-1=manual, 0=no wait, positive=countdown)"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    state.round_wait_seconds = body.round_wait_seconds
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Round wait updated to {body.round_wait_seconds}s")


@router.post("/session/{session_id}/literature-config")
async def update_literature_config(session_id: str, request: Request, body: LiteratureConfigRequest):
    """Update literature agent configuration"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    state.literature_config = {
        "mode": body.mode,
        "deep_research_model": body.deep_research_model,
        "deep_research_llm_base_url": body.deep_research_llm_base_url,
        "deep_research_api_key": body.deep_research_api_key,
        "google_api_key": body.google_api_key,
        "pubmed_email": body.pubmed_email,
        "max_search_rounds": body.max_search_rounds,
    }
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Literature config updated: mode={body.mode}")


@router.get("/session/{session_id}/literature-config")
async def get_literature_config(session_id: str, request: Request):
    """Get current literature agent configuration"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return state.literature_config


@router.post("/session/{session_id}/agent-server-config")
async def update_agent_server_config(session_id: str, request: Request, body: AgentServerConfigRequest):
    """Update per-agent LLM server configuration"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    state.agent_server_configs[body.agent] = {
        "llm_base_url": body.llm_base_url,
        "api_key": body.api_key,
    }
    await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Server config for {body.agent} updated")


@router.get("/session/{session_id}/agent-server-config")
async def get_agent_server_configs(session_id: str, request: Request):
    """Get per-agent LLM server configurations"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return state.agent_server_configs
