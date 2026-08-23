"""Per-agent LLM client factory + session-path ownership helper."""
import logging
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

from backend.config import SystemConfig
from backend.config._config import get_session_dir
from backend.models.schemas import AnalysisState
from backend.utils.auth import require_session_ownership
from backend.utils.llm_client import OpenAICompatibleClient
from backend.utils.settings_provider import agent_endpoint_lacks_credential

logger = logging.getLogger(__name__)


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

    When `user_id` is None (no auth token — e.g. an HTML-embedded <img src> or a
    download link that browsers fetch without an Authorization header), the path
    is still confined to a single existing session directory and ownership is
    delegated to `require_session_ownership`, which falls back to UUID-guard
    (session-exists) verification. This mirrors the auth-optional figure/report
    endpoints so embedded resources render without auth headers.
    """
    user_id = getattr(request.state, "user_id", None)

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
    if agent_endpoint_lacks_credential(
        agent_url,
        agent_key,
        cfg.api.llm_base_url,
        cfg.api.api_key,
    ):
        # Custom endpoint, no credential of its own (the shape an imported
        # session always has, since export blanks credentials but not URLs).
        # Pairing it with the installation key would leak that key to a host
        # chosen by the archive.
        logger.warning(
            "Agent %s has a custom endpoint but no credential of its own; "
            "falling back to the global endpoint.",
            agent_name,
        )
        agent_url = ""
    effective_url = agent_url if (agent_url and agent_url != cfg.api.llm_base_url) else cfg.api.llm_base_url
    effective_key = agent_key if (agent_key and agent_key != cfg.api.api_key) else cfg.api.api_key

    if not effective_key:
        cfg.api.require_key()

    return OpenAICompatibleClient(
        api_key=effective_key,
        base_url=effective_url,
        proxy_url=cfg.api.proxy_url,
        session_id=session_id,
        agent_name=agent_name,
    )
