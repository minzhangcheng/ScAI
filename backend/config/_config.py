"""
Configuration for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from pathlib import Path
import os
import logging
import logging.handlers
from datetime import datetime
from backend.config.agent_prompts import AgentPrompts  # noqa: F811
from backend.models.schemas import AgentType


# Data directories - stored outside the code directory for better separation
# All data is stored in ../data/ relative to the backend directory
# Can be overridden with SCAI_DATA_DIR environment variable
_BACKEND_DIR = Path(__file__).parent
DATA_BASE_DIR = Path(os.environ.get(
    'SCAI_DATA_DIR',
    str((_BACKEND_DIR.parent.parent.parent / "data").resolve())
))

_ENABLE_REPRODUCIBILITY = os.environ.get("SCAI_ENABLE_REPRODUCIBILITY", "").lower() in (
    "1",
    "true",
    "yes",
)

_ENABLE_SNAPSHOT_V2 = os.environ.get("SCAI_ENABLE_SNAPSHOT_V2", "").lower() in (
    "1",
    "true",
    "yes",
)
SNAPSHOT_V2_ENABLED = _ENABLE_SNAPSHOT_V2


def is_snapshot_v2_enabled() -> bool:
    """Return the process-scoped Snapshot V2 operator feature gate."""
    return SNAPSHOT_V2_ENABLED

# Legacy top-level directories (kept for backward compatibility / migration fallback)
UPLOAD_DIR = DATA_BASE_DIR / "uploads"
STATE_DIR = DATA_BASE_DIR / "states"
WORKSPACE_DIR = DATA_BASE_DIR / "workspace"
REPORTS_DIR = DATA_BASE_DIR / "reports"
LOG_DIR = DATA_BASE_DIR / "logs"

# Session-centric directory: all per-session data lives under sessions/{session_id}/
SESSION_DIR = DATA_BASE_DIR / "sessions"

# Application database for users and API keys
APP_DB_PATH = DATA_BASE_DIR / "app.db"

# Upload settings for large files
UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10GB maximum file size
UPLOAD_TEMP_DIR = DATA_BASE_DIR / "upload_temp"


def _ensure_directories() -> None:
    """Create data directory structure (idempotent, called once at import time)."""
    DATA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for d in [UPLOAD_DIR, STATE_DIR, WORKSPACE_DIR, SESSION_DIR, REPORTS_DIR, LOG_DIR]:
        d.mkdir(exist_ok=True)
    UPLOAD_TEMP_DIR.mkdir(exist_ok=True)


_ensure_directories()

# Session cache management
MAX_CACHED_SESSIONS = 50  # Maximum sessions to keep in memory
MAX_REALTIME_MESSAGES = 1000  # Maximum realtime messages per session
SESSION_CACHE_TTL = 3600  # Seconds before inactive session eviction (1 hour)
IMPORT_JOB_TTL = 3600  # Cleanup completed/failed import jobs after (seconds)
CLEANUP_INTERVAL = 300  # Background cleanup task interval (seconds)
WS_RECEIVE_TIMEOUT = 60.0  # WebSocket receive timeout (seconds) — increased from 30s
CHAT_DELEGATION_TIMEOUT = 600  # Chat delegation global timeout (seconds)


# ---------------------------------------------------------------------------
# Per-session path helpers (new session-centric layout)
# ---------------------------------------------------------------------------

def get_session_dir(session_id: str) -> Path:
    """Root directory for a specific session's data."""
    return SESSION_DIR / session_id


def get_session_metadata_path(session_id: str) -> Path:
    """Path to session.json metadata file."""
    return get_session_dir(session_id) / "session.json"


def resolve_session_metadata_path(session_id: str) -> Path:
    """Resolve session metadata path (new layout). Legacy paths handled by callers."""
    return get_session_metadata_path(session_id)


def get_session_workspace(session_id: str) -> Path:
    """Workspace directory for a session (analysis code, figures)."""
    return get_session_dir(session_id) / "workspace"


def get_session_states(session_id: str) -> Path:
    """States directory for a session (checkpoints, notebook, pkl)."""
    return get_session_dir(session_id) / "states"


def get_session_uploads(session_id: str) -> Path:
    """Uploads directory for a session."""
    return get_session_dir(session_id) / "uploads"


def get_session_reports(session_id: str) -> Path:
    """Reports directory for a session."""
    return get_session_dir(session_id) / "reports"


def get_session_logs(session_id: str) -> Path:
    """Logs directory for a session (LLM interaction logs)."""
    return get_session_dir(session_id) / "logs"


def get_integrity_dir(session_id: str) -> Path:
    """Host-scoped integrity root (not sandbox-writable). HMAC files live here."""
    path = DATA_BASE_DIR / "integrity" / session_id / "snapshot-v2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_session_dirs(session_id: str) -> Path:
    """Create all subdirectories for a session, return session root."""
    root = get_session_dir(session_id)
    for sub in ["workspace", "states", "uploads", "reports", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


class _HealthProbeAccessFilter(logging.Filter):
    """Suppress uvicorn access-log records for the unauthenticated health probe.

    uvicorn.access emits one record per request with
    ``record.args == (client_addr, method, path, http_version, status)``.
    We drop records whose path is the health endpoint; everything else passes.
    """

    _SILENCED_PATHS = ("/api/health",)

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            path = args[2]
            if isinstance(path, str) and path.split("?", 1)[0] in self._SILENCED_PATHS:
                return False
        return True


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Setup comprehensive logging for the application.

    Logs are written to:
    - {LOG_DIR}/app.log - Main application log (rotating, 10MB max, 5 backups)
    - {LOG_DIR}/error.log - Error-only log (rotating, 10MB max, 5 backups)
    - {LOG_DIR}/debug.log - Debug log with all details (rotating, 50MB max, 3 backups)
    - Console output
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler - INFO level with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    def _add_file_handler(name, level, max_bytes, backup_count):
        path = LOG_DIR / f"{name}.log"
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        handler.setLevel(level)
        handler.setFormatter(detailed_formatter)
        root_logger.addHandler(handler)

    _add_file_handler("app",   logging.INFO,   10*1024*1024, 5)
    _add_file_handler("error", logging.ERROR,  10*1024*1024, 5)
    _add_file_handler("debug", logging.DEBUG,  50*1024*1024, 3)

    # Drop uvicorn access-log lines for the health probe so container/orchestrator
    # health checks (every ~30s, one per line) don't drown out the real request log.
    logging.getLogger("uvicorn.access").addFilter(_HealthProbeAccessFilter())

    # Log startup message
    logging.info(f"Logging initialized. Log directory: {LOG_DIR}")
    logging.info(f"Log files: app.log, error.log, debug.log")

# Snapshot size policy
MAX_SNAPSHOT_KERNEL_PKL_BYTES = 2 * 1024 * 1024 * 1024
MAX_SESSION_SNAPSHOT_BYTES = 20 * 1024 * 1024 * 1024
SNAPSHOT_EXCLUDED_WORKSPACE_PATTERNS = ("core.*", "*.tmp", "*.lock")


class APIConfig(BaseModel):
    """API configuration for various services"""
    api_key: str = ""  # Main LLM API key (OpenRouter, OpenAI, etc.)
    llm_base_url: str = "https://openrouter.ai/api/v1"  # OpenAI-compatible endpoint
    proxy_url: Optional[str] = None  # HTTP/HTTPS proxy for API requests
    # Note: google_api_key moved to literature_config (per-session, in AnalysisState)

    @field_validator("proxy_url")
    @classmethod
    def _check_proxy(cls, v):
        if v is None or v == "":
            return v
        if not (v.startswith("http://") or v.startswith("https://") or v.startswith("socks5://")):
            raise ValueError("proxy_url must start with http://, https:// or socks5://")
        return v

    def require_key(self) -> None:
        if not self.api_key:
            raise ValueError("LLM api_key is not configured; set it in Config")
    
class ModelConfig(BaseModel):
    """Model configuration for each agent"""
    pi_model: str = "gpt-5.6-sol"
    pathology_model: str = "gemini-3.1-pro-preview"
    bioinformatics_model: str = "gpt-5.6-sol"
    biochemistry_model: str = "gemini-3.1-pro-preview"
    literature_model: str = "gpt-5.4-mini"
    # Painter agent uses image generation model via OpenRouter
    painter_model: str = "gpt-image-2"
    # Supervisor agent uses cheap vision model for real-time monitoring
    supervisor_model: str = "gpt-5.4-mini"
    technique_model: str = "gpt-5.4-mini"

class AgentTemperatures(BaseModel):
    """Default temperatures for each agent - centralized configuration"""
    pi: float = Field(default=0.5, ge=0.0, le=2.0)  # More creative synthesis/planning
    pathology: float = Field(default=0.2, ge=0.0, le=2.0)  # Objective, consistent analysis
    bioinformatics: float = Field(default=0.1, ge=0.0, le=2.0)  # Highly deterministic code generation
    biochemistry: float = Field(default=0.2, ge=0.0, le=2.0)  # Objective interpretation
    literature: float = Field(default=0.6, ge=0.0, le=2.0)  # Broader exploration
    painter: float = Field(default=0.7, ge=0.0, le=2.0)  # Creative image generation
    supervisor: float = Field(default=0.1, ge=0.0, le=2.0)  # High-certainty evaluation
    technique: float = Field(default=0.1, ge=0.0, le=2.0)  # Precise documentation lookup

    def get(self, agent: str, default: float = 0.5) -> float:
        """Get temperature for an agent by name"""
        return getattr(self, agent, default)

    def merge_custom(self, custom: Dict[str, float]) -> 'AgentTemperatures':
        """Return new instance with custom values merged"""
        data = self.model_dump()
        data.update({k: v for k, v in custom.items() if k in data})
        return AgentTemperatures(**data)
    
# Literature agent defaults
DEFAULT_LITERATURE_MODE = "pubmed_llm"
DEFAULT_LITERATURE_MAX_SEARCH_ROUNDS = 20

class SystemConfig(BaseModel):
    """System configuration"""
    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    prompts: AgentPrompts = Field(default_factory=AgentPrompts)
    temperatures: AgentTemperatures = Field(default_factory=AgentTemperatures)
    
    # Image processing settings
    image_tile_size: int = 1024  # Size for pathology image tiles
    image_overlap: int = 64  # Overlap between tiles
    blank_threshold: float = 0.95  # Threshold for detecting blank tiles
    
    # Local Jupyter kernel settings
    sandbox_timeout: int = 5400  # Code execution timeout in seconds (1.5 hours)

    # Bioinformatics agent settings
    bioinfo_max_steps: int = 200  # Max autonomous analysis steps per execution
    
    # Session settings
    max_rounds: int = 3  # Maximum analysis rounds
    convergence_rounds: int = 2  # Rounds without new findings to stop

    enable_reproducibility: bool = Field(
        default=_ENABLE_REPRODUCIBILITY,
        description="Enable reproducibility bundle generation (checksums, environment snapshot, commands, decisions)",
    )

    enable_annotation_validation_round: bool = Field(
        default=True,
        description="Enable the automatic annotation validation round (PI plan -> restricted bioinformatics checks -> PI revision) before user annotation confirmation. Gated by trigger conditions; does not run on every annotation.",
    )
    annotation_validation_timeout_seconds: int = Field(
        default=600,
        ge=1,
        description="Wall-clock timeout (seconds) for the whole annotation validation executor. The default covers several bounded LLM/code iterations on large datasets; on timeout it returns a partial result rather than blocking the annotation dialog.",
    )
    annotation_validation_max_iterations: int = Field(
        default=5,
        ge=1,
        le=8,
        description="Max code-generation iterations for the annotation validation executor (hard-capped at 8 in code).",
    )

    # Snapshot V2 settings (V2 is the authoritative lifecycle; V1 restore cut over)
    snapshot_v2_enabled: bool = Field(
        default=_ENABLE_SNAPSHOT_V2,
        description=(
            "Enable Snapshot V2. Disabled by default; set "
            "SCAI_ENABLE_SNAPSHOT_V2=true before backend startup to opt in."
        ),
    )
    snapshot_safe_point_timeout_seconds: int | None = None
    snapshot_v2_session_quota_bytes: int = 100 * 1024 ** 3
    snapshot_v2_pre_restore_quota_grace_bytes: int = 64 * 1024 ** 2
    snapshot_v2_free_space_reserve_bytes: int = 5 * 1024 ** 3
    snapshot_v1_restore_cutover: bool = True
    snapshot_v2_import_compressed_bytes: int = 20 * 1024 ** 3
    snapshot_v2_import_uncompressed_bytes: int = 100 * 1024 ** 3
    snapshot_v2_import_single_member_bytes: int = 80 * 1024 ** 3
    snapshot_v2_import_member_count: int = 200000
    snapshot_v2_import_manifest_count: int = 10000
    snapshot_v2_import_snapshot_count: int = 10000

    def get_snapshot_safe_point_timeout_seconds(self) -> int:
        if self.snapshot_safe_point_timeout_seconds is not None and self.snapshot_safe_point_timeout_seconds > 0:
            return self.snapshot_safe_point_timeout_seconds
        return self.sandbox_timeout + 120

    # State persistence (deprecated: use get_session_states/get_session_workspace instead)
    # Kept for serialization compatibility only.
    state_dir: str = str(STATE_DIR)
    workspace_dir: str = str(WORKSPACE_DIR)

# Default configuration instance
default_config = SystemConfig()

# Shared agent identifier — single source of truth.
# Derived from the canonical enum so the string value can never drift.
PI_AGENT_NAME = AgentType.PI.value
