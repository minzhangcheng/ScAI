"""Per-agent LLM client factory + session-path ownership helper."""
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

from backend.config import SystemConfig
from backend.config._config import get_session_dir
from backend.models.schemas import AnalysisState
from backend.utils.auth import require_session_ownership
from backend.utils.llm_client import OpenAICompatibleClient


async def validate_file_path_ownership(
    file_path: str,
    request: Request,
    *,
    expected_session_id: Optional[str] = None,
) -> None:
    """Verify the requesting user owns the session that a file_path belongs to.

    When `expected_session_id` is provided, path-parsing is skipped — the caller
    knows which session the path should resolve to, and we only verify (a) the
    resolved path lies inside that session's directory and (b) the user owns it.

    Otherwise we extract the session_id from a single `sessions/<sid>/` segment.
    Paths with multiple `sessions/` segments are rejected (nested-path attack:
    `.../sessions/<own>/uploads/sessions/<victim>/file`).

    Defense-in-depth: after determining sid, the resolved path must lie inside
    `get_session_dir(sid)` (which contains uploads, workspace, reports, states,
    logs subdirectories).
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=403, detail="Not authenticated")

    try:
        resolved = Path(file_path).resolve()
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="Access denied")

    sid = expected_session_id
    if sid is None:
        parts = Path(file_path).parts
        sessions_indices = [i for i, p in enumerate(parts) if p == "sessions"]
        if not sessions_indices:
            raise HTTPException(status_code=403, detail="Access denied")
        if len(sessions_indices) > 1:
            raise HTTPException(
                status_code=403,
                detail="Access denied: nested 'sessions/' segments",
            )
        idx = sessions_indices[0]
        if idx + 1 >= len(parts):
            raise HTTPException(status_code=403, detail="Access denied")
        sid = parts[idx + 1]

    session_root = Path(get_session_dir(sid)).resolve()
    try:
        resolved.relative_to(session_root)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access denied: path outside session directory",
        )

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
