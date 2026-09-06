"""ScAI configuration package — re-exports for backward compatibility."""
from backend.config._config import *  # noqa: F401,F403
from backend.config.agent_prompts import AgentPrompts  # noqa: F401

__all__ = [
    "DATA_BASE_DIR", "UPLOAD_DIR", "STATE_DIR", "WORKSPACE_DIR", "REPORTS_DIR", "LOG_DIR",
    "SESSION_DIR", "APP_DB_PATH", "UPLOAD_CHUNK_SIZE", "MAX_UPLOAD_SIZE", "UPLOAD_TEMP_DIR", "INTEGRITY_DIR",
    "get_session_dir", "get_session_metadata_path", "resolve_session_metadata_path",
    "get_session_workspace", "get_session_states", "get_session_uploads",
    "get_session_reports", "get_session_logs", "ensure_session_dirs",
    "setup_logging",
    "APIConfig", "ModelConfig", "AgentTemperatures", "AgentContextWindows", "AgentPrompts", "SystemConfig",
    "MINIMUM_CONTEXT_WINDOW", "DEFAULT_AGENT_CONTEXT_WINDOW",
    "default_config", "PI_AGENT_NAME",
    "RUNTIME_SAVE_FLOOR_MINUTES", "RUNTIME_SAVE_FLOOR_MINUTES_DEFAULT",
    "MAX_CACHED_SESSIONS", "MAX_REALTIME_MESSAGES", "SESSION_CACHE_TTL",
    "IMPORT_JOB_TTL", "PREPARED_EXPORT_TTL", "CLEANUP_INTERVAL", "WS_RECEIVE_TIMEOUT", "CHAT_DELEGATION_TIMEOUT",
    "PREPARED_EXPORT_PREFIX", "EXPORT_STREAM_PREFIX",
    "RETIRED_SNAPSHOT_EXPORT_PREFIX", "EXPORT_TEMP_PREFIXES",
    "DEFAULT_LITERATURE_MODE", "DEFAULT_LITERATURE_MAX_SEARCH_ROUNDS",
]
