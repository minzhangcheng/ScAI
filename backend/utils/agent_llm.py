"""Per-agent LLM client factory + session-path ownership helper."""
from pathlib import Path

from fastapi import HTTPException, Request

from backend.config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.auth import require_session_ownership
from backend.utils.llm_client import OpenAICompatibleClient


async def validate_file_path_ownership(file_path: str, request: Request) -> None:
    """Extract session_id from a file path prefix and verify ownership.

    Expected path format: sessions/<session_id>/... or data/sessions/<session_id>/...
    Uses require_session_ownership for consistent ownership checking.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=403, detail="Not authenticated")

    parts = Path(file_path).parts
    sid = None
    for i, part in enumerate(parts):
        if part == "sessions" and i + 1 < len(parts):
            sid = parts[i + 1]
            break

    if not sid:
        raise HTTPException(status_code=403, detail="Access denied")

    await require_session_ownership(session_id=sid, user_id=user_id)


def get_agent_llm_client(
    state: AnalysisState,
    cfg: SystemConfig,
    agent_name: str,
    session_id: str,
) -> OpenAICompatibleClient:
    """Get LLM client for an agent, using per-agent override if different from global.

    Always creates a new client (caller should close it when done).
    Uses per-agent config if it differs from the global config, otherwise uses global.
    """
    agent_cfg = state.agent_server_configs.get(agent_name, {})
    agent_url = agent_cfg.get("llm_base_url", "")
    agent_key = agent_cfg.get("api_key", "")
    effective_url = agent_url if (agent_url and agent_url != cfg.api.llm_base_url) else cfg.api.llm_base_url
    effective_key = agent_key if (agent_key and agent_key != cfg.api.api_key) else cfg.api.api_key
    return OpenAICompatibleClient(
        api_key=effective_key,
        base_url=effective_url,
        proxy_url=cfg.api.proxy_url,
        session_id=session_id,
        agent_name=agent_name,
    )
