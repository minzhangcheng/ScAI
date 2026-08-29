"""Session configuration routes (prompts, models, temperatures, vision, rounds, literature, agent-server)."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.config import AgentTemperatures, PI_AGENT_NAME, SystemConfig
from backend.models.api_requests import (
    PromptUpdateRequest, ModelUpdateRequest, TemperatureUpdateRequest,
    VisionSettingUpdateRequest, LiteratureConfigRequest, AgentServerConfigRequest,
    AgentSettingsUpdateRequest,
)
from backend.models.schemas import APIResponse, DEFAULT_ROUND_WAIT_SECONDS
from backend.services.session_store import async_save_session, ensure_session_loaded, update_session_cache
from backend.state import get_session_lock, workflows
from backend.utils.auth import require_session_ownership, require_session_read_ownership
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import validate_session_id
from backend.utils.secret_redaction import mask_api_key
from backend.utils.settings_provider import AgentSettingsProvider

router = APIRouter(prefix="/api", tags=["session-config"])

_AGENT_NAMES = (
    "pi", "pathology", "bioinformatics", "biochemistry",
    "literature", "painter", "supervisor", "technique",
)


def _invalidate_active_agent_config(session_id: str, agent_name: str) -> None:
    """Notify the active workflow (if any) that an agent's config changed.

    Clears the provider's per-agent client cache and triggers the agent's
    runtime refresh hook so the next use picks up the new configuration.
    """
    workflow = workflows.get(session_id)
    if workflow is None:
        return

    provider = getattr(workflow, "_settings_provider", None)
    if provider is not None and hasattr(provider, "clear_agent_client_cache"):
        provider.clear_agent_client_cache(agent_name)

    agent_attr = f"{agent_name}_agent"
    agent = getattr(workflow, agent_attr, None)
    if agent is not None and hasattr(agent, "refresh_runtime_config_if_needed"):
        agent.refresh_runtime_config_if_needed()


def _masked_agent_server_configs(state) -> dict:
    masked = {}
    for agent, config in (state.agent_server_configs or {}).items():
        value = dict(config) if config else {}
        if value.get("api_key"):
            value["api_key"] = mask_api_key(value["api_key"])
        masked[agent] = value
    return masked


def _agent_settings_payload(state, config: SystemConfig) -> dict:
    server_configs = _masked_agent_server_configs(state)
    return {
        "prompts": {
            agent: state.custom_prompts.get(
                agent, getattr(config.prompts, f"{agent}_prompt", "")
            )
            for agent in _AGENT_NAMES
        },
        "models": {
            agent: state.custom_models.get(
                agent, getattr(config.models, f"{agent}_model", "")
            )
            for agent in _AGENT_NAMES
        },
        "temperatures": {
            agent: state.custom_temperatures.get(
                agent, getattr(config.temperatures, agent, 0.5)
            )
            for agent in _AGENT_NAMES
        },
        "agent_server_configs": {
            agent: server_configs.get(agent, {"llm_base_url": "", "api_key": ""})
            for agent in _AGENT_NAMES
        },
        "vision_settings": {
            agent: state.agent_vision_settings.get(agent, True)
            for agent in _AGENT_NAMES
        },
        "literature_config": AgentSettingsProvider.mask_literature_config(
            state.literature_config
        ),
    }


class MaxRoundsRequest(BaseModel):
    max_rounds: int = Field(ge=1, le=50, default=3)


class RoundWaitRequest(BaseModel):
    round_wait_seconds: int = Field(ge=-1, le=3600, default=DEFAULT_ROUND_WAIT_SECONDS)


@router.post("/session/{session_id}/prompts")
async def update_prompts(session_id: str, request: Request, body: PromptUpdateRequest):
    """Update agent prompts"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        state.custom_prompts[body.agent] = body.prompt
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    _invalidate_active_agent_config(session_id, body.agent)

    return APIResponse(success=True, message=f"Prompt updated for {body.agent}")


@router.get("/session/{session_id}/prompts")
async def get_prompts(session_id: str, request: Request):
    """Get current prompts"""
    user_id = getattr(request.state, "user_id", "")
    cfg = await resolve_config_for_user(user_id)
    state = await require_session_read_ownership(session_id=session_id, user_id=user_id)

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
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        state.custom_models[body.agent] = body.model
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    _invalidate_active_agent_config(session_id, body.agent)

    return APIResponse(success=True, message=f"Model updated for {body.agent}")


@router.get("/session/{session_id}/models")
async def get_models(session_id: str, request: Request):
    """Get current models"""
    user_id = getattr(request.state, "user_id", "")
    cfg = await resolve_config_for_user(user_id)
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    models = {
        PI_AGENT_NAME: state.custom_models.get(PI_AGENT_NAME, cfg.models.pi_model if cfg else "gpt-5.6-sol"),
        "pathology": state.custom_models.get("pathology", cfg.models.pathology_model if cfg else "gemini-3.1-pro-preview"),
        "bioinformatics": state.custom_models.get("bioinformatics", cfg.models.bioinformatics_model if cfg else "gpt-5.6-sol"),
        "biochemistry": state.custom_models.get("biochemistry", cfg.models.biochemistry_model if cfg else "gemini-3.1-pro-preview"),
        "literature": state.custom_models.get("literature", cfg.models.literature_model if cfg else "gpt-5.4-mini"),
        "painter": state.custom_models.get("painter", cfg.models.painter_model if cfg else "gpt-image-2"),
        "supervisor": state.custom_models.get("supervisor", cfg.models.supervisor_model if cfg else "gpt-5.4-mini"),
        "technique": state.custom_models.get("technique", cfg.models.technique_model if cfg else "gpt-5.4-mini"),
    }
    return models


@router.post("/session/{session_id}/temperatures")
async def update_temperatures(session_id: str, request: Request, body: TemperatureUpdateRequest):
    """Update agent temperature"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        state.custom_temperatures[body.agent] = body.temperature
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    _invalidate_active_agent_config(session_id, body.agent)

    return APIResponse(success=True, message=f"Temperature updated for {body.agent}: {body.temperature}")


@router.get("/session/{session_id}/temperatures")
async def get_temperatures(session_id: str, request: Request):
    """Get current temperatures - uses centralized config with custom overrides"""
    user_id = getattr(request.state, "user_id", "")
    cfg = await resolve_config_for_user(user_id)
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    base_temps = cfg.temperatures if cfg else AgentTemperatures()
    merged = base_temps.merge_custom(state.custom_temperatures)
    return merged.model_dump()


@router.post("/session/{session_id}/vision-settings")
async def update_vision_settings(session_id: str, request: Request, body: VisionSettingUpdateRequest):
    """Update agent vision capability setting"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        state.agent_vision_settings[body.agent] = body.vision_enabled
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    _invalidate_active_agent_config(session_id, body.agent)

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
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        state.max_rounds = body.max_rounds
        await update_session_cache(session_id, state)
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
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        state.round_wait_seconds = body.round_wait_seconds
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)

    return APIResponse(success=True, message=f"Round wait updated to {body.round_wait_seconds}s")


@router.post("/session/{session_id}/literature-config")
async def update_literature_config(session_id: str, request: Request, body: LiteratureConfigRequest):
    """Update literature agent configuration"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        incoming = {
            "mode": body.mode,
            "deep_research_model": body.deep_research_model,
            "deep_research_llm_base_url": body.deep_research_llm_base_url,
            "deep_research_api_key": body.deep_research_api_key,
            "google_api_key": body.google_api_key,
            "pubmed_email": body.pubmed_email,
            "max_search_rounds": body.max_search_rounds,
        }
        state.literature_config = AgentSettingsProvider.resolve_literature_config_update(
            state.literature_config or {},
            incoming,
        )
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    _invalidate_active_agent_config(session_id, "literature")

    return APIResponse(success=True, message=f"Literature config updated: mode={body.mode}")


@router.get("/session/{session_id}/literature-config")
async def get_literature_config(session_id: str, request: Request):
    """Get current literature agent configuration (API keys masked)"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return AgentSettingsProvider.mask_literature_config(state.literature_config)


@router.post("/session/{session_id}/agent-server-config")
async def update_agent_server_config(session_id: str, request: Request, body: AgentServerConfigRequest):
    """Update per-agent LLM server configuration"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        existing = (state.agent_server_configs or {}).get(body.agent, {}) or {}
        incoming_key = body.api_key
        if incoming_key and incoming_key == mask_api_key(existing.get("api_key", "")):
            incoming_key = existing.get("api_key", "")
        state.agent_server_configs[body.agent] = {
            "llm_base_url": body.llm_base_url,
            "api_key": incoming_key,
        }
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    _invalidate_active_agent_config(session_id, body.agent)

    return APIResponse(success=True, message=f"Server config for {body.agent} updated")


@router.get("/session/{session_id}/agent-server-config")
async def get_agent_server_configs(session_id: str, request: Request):
    """Get per-agent LLM server configurations (API keys masked)"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return _masked_agent_server_configs(state)


@router.get("/session/{session_id}/agent-settings")
async def get_agent_settings(session_id: str, request: Request):
    """Return the authoritative settings displayed by the Agents page."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    config = await resolve_config_for_user(user_id) or SystemConfig()
    state = await require_session_read_ownership(session_id=session_id, user_id=user_id)
    return _agent_settings_payload(state, config)


@router.post("/session/{session_id}/agent-settings")
async def update_agent_settings(
    session_id: str,
    request: Request,
    body: AgentSettingsUpdateRequest,
):
    """Atomically persist every editable setting for one agent."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        existing_server = (state.agent_server_configs or {}).get(body.agent, {}) or {}
        incoming_key = body.api_key
        if incoming_key and incoming_key == mask_api_key(existing_server.get("api_key", "")):
            incoming_key = existing_server.get("api_key", "")

        state.custom_prompts[body.agent] = body.prompt
        state.custom_models[body.agent] = body.model
        state.custom_temperatures[body.agent] = body.temperature
        state.agent_server_configs[body.agent] = {
            "llm_base_url": body.llm_base_url,
            "api_key": incoming_key,
        }
        state.agent_vision_settings[body.agent] = body.vision_enabled

        if body.agent == "literature" and body.literature_config is not None:
            literature = body.literature_config
            state.literature_config = AgentSettingsProvider.resolve_literature_config_update(
                state.literature_config or {},
                literature.model_dump(),
            )

        await update_session_cache(session_id, state)
        _invalidate_active_agent_config(session_id, body.agent)
        await async_save_session(session_id, state)

    config = await resolve_config_for_user(user_id) or SystemConfig()
    return _agent_settings_payload(state, config)
