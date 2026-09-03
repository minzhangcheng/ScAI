"""Session configuration routes (prompts, models, temperatures, vision, rounds, literature, agent-server)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.config import AgentTemperatures, PI_AGENT_NAME, SystemConfig
from backend.models.api_requests import (
    PromptUpdateRequest, ModelUpdateRequest, TemperatureUpdateRequest,
    VisionSettingUpdateRequest, LiteratureConfigRequest, AgentServerConfigRequest,
    AgentSettingsUpdateRequest,
)
from backend.models.schemas import (
    AnalysisState,
    APIResponse,
    DEFAULT_MAX_DIRECTIONS_PER_GATE,
    DEFAULT_ROUND_WAIT_SECONDS,
)
from backend.services.agent_default_secrets import (
    agent_default_secrets_lock,
    load_agent_default_secrets,
    save_agent_default_secrets,
)
from backend.services.session_store import async_save_session, ensure_session_loaded, update_session_cache
from backend.state import get_session_lock, workflows
from backend.utils.auth import (
    get_db,
    require_auth,
    require_session_ownership,
    require_session_read_ownership,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import validate_session_id
from backend.utils.secret_redaction import mask_api_key
from backend.utils.settings_provider import AgentSettingsProvider, resolve_context_window
from backend.utils.url_validation import validated_http_url

router = APIRouter(prefix="/api", tags=["session-config"])
logger = logging.getLogger(__name__)

_AGENT_NAMES = (
    "pi", "pathology", "bioinformatics", "biochemistry",
    "literature", "painter", "supervisor", "technique",
)


def _validated_optional_url(value: str, field_name: str) -> str:
    try:
        return validated_http_url(value, field_name, allow_empty=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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


def _masked_agent_server_configs(state, config: SystemConfig | None = None) -> dict:
    masked = {}
    for agent, server_config in (state.agent_server_configs or {}).items():
        value = dict(server_config) if server_config else {}
        if value.get("api_key"):
            value["api_key"] = mask_api_key(value["api_key"])
        raw_window = value.get("context_window")
        value["context_window"] = (
            raw_window
            if isinstance(raw_window, int)
            and not isinstance(raw_window, bool)
            and raw_window >= 8_192
            else None
        )
        if config is not None:
            value["effective_context_window"] = resolve_context_window(
                state.agent_server_configs,
                config,
                agent,
            )
        masked[agent] = value
    return masked


def _agent_settings_payload(state, config: SystemConfig) -> dict:
    server_configs = _masked_agent_server_configs(state, config)
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
            agent: server_configs.get(agent, {
                "llm_base_url": "",
                "api_key": "",
                "context_window": None,
                "effective_context_window": config.context_windows.get(agent),
            })
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


@router.get("/user/agent-default-settings")
async def get_user_agent_default_settings(user_id: str = Depends(require_auth)):
    """Return effective non-secret defaults when no prior browser snapshot exists."""
    config = await resolve_config_for_user(user_id) or SystemConfig()
    empty_state = AnalysisState(
        id="agent-default-settings",
        name="Agent Defaults",
        user_id=user_id,
        initial_instruction="",
    )
    return _agent_settings_payload(empty_state, config)


class MaxRoundsRequest(BaseModel):
    max_rounds: int = Field(ge=1, le=50, default=3)


class RoundWaitRequest(BaseModel):
    round_wait_seconds: int = Field(ge=-1, le=3600, default=DEFAULT_ROUND_WAIT_SECONDS)


class MaxDirectionsPerGateRequest(BaseModel):
    max_directions_per_gate: int = Field(
        ge=0, le=20, default=DEFAULT_MAX_DIRECTIONS_PER_GATE
    )


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


@router.post("/session/{session_id}/max-directions-per-gate")
async def update_max_directions_per_gate(
    session_id: str, request: Request, body: MaxDirectionsPerGateRequest
):
    """Cap how many directions one PI-decide gate may commit to (0 = no cap)."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        state = await require_session_ownership(session_id=session_id, user_id=user_id)
        state.max_directions_per_gate = body.max_directions_per_gate
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)

    limit = body.max_directions_per_gate
    return APIResponse(
        success=True,
        message=(
            "Per-gate direction cap removed" if limit == 0
            else f"Per-gate direction cap updated to {limit}"
        ),
    )


@router.get("/session/{session_id}/max-directions-per-gate")
async def get_max_directions_per_gate(session_id: str, request: Request):
    """Get the current per-gate direction cap (0 = no cap)."""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    return {"max_directions_per_gate": state.max_directions_per_gate}


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
            "deep_research_llm_base_url": _validated_optional_url(
                body.deep_research_llm_base_url, "deep_research_llm_base_url"
            ),
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
        new_config = {
            "llm_base_url": _validated_optional_url(body.llm_base_url, "llm_base_url"),
            "api_key": incoming_key,
        }
        if "context_window" not in body.model_fields_set:
            if "context_window" in existing:
                new_config["context_window"] = existing["context_window"]
        elif body.context_window is not None:
            new_config["context_window"] = body.context_window
        state.agent_server_configs[body.agent] = new_config
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
    config = await resolve_config_for_user(user_id) or SystemConfig()
    return _masked_agent_server_configs(state, config)


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
        new_server = {
            "llm_base_url": _validated_optional_url(body.llm_base_url, "llm_base_url"),
            "api_key": incoming_key,
        }
        if "context_window" not in body.model_fields_set:
            if "context_window" in existing_server:
                new_server["context_window"] = existing_server["context_window"]
        elif body.context_window is not None:
            new_server["context_window"] = body.context_window
        state.agent_server_configs[body.agent] = new_server
        state.agent_vision_settings[body.agent] = body.vision_enabled

        if body.agent == "literature" and body.literature_config is not None:
            literature = body.literature_config
            incoming_literature = literature.model_dump()
            incoming_literature["deep_research_llm_base_url"] = _validated_optional_url(
                literature.deep_research_llm_base_url,
                "deep_research_llm_base_url",
            )
            state.literature_config = AgentSettingsProvider.resolve_literature_config_update(
                state.literature_config or {},
                incoming_literature,
            )

        await update_session_cache(session_id, state)
        _invalidate_active_agent_config(session_id, body.agent)
        await async_save_session(session_id, state)

        default_settings_saved = True
        if body.save_as_user_default:
            try:
                db = await get_db()
                async with agent_default_secrets_lock(user_id):
                    defaults = await load_agent_default_secrets(db, user_id)
                    agent_api_keys = defaults.setdefault("agent_api_keys", {})
                    if incoming_key:
                        agent_api_keys[body.agent] = incoming_key
                    else:
                        agent_api_keys.pop(body.agent, None)
                    if body.agent == "literature":
                        saved_literature = defaults.setdefault("literature", {})
                        current_literature = state.literature_config or {}
                        saved_literature["deep_research_api_key"] = current_literature.get(
                            "deep_research_api_key", ""
                        )
                        saved_literature["google_api_key"] = current_literature.get(
                            "google_api_key", ""
                        )
                    await save_agent_default_secrets(db, user_id, defaults)
            except Exception:
                default_settings_saved = False
                logger.exception("Failed to persist Agent defaults for user %s", user_id)

    config = await resolve_config_for_user(user_id) or SystemConfig()
    response = _agent_settings_payload(state, config)
    response["default_settings_saved"] = default_settings_saved
    return response
