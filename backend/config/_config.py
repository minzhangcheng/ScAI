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
# Resolved to an absolute path: session import rewrites the source
# installation's data root out of every stored path, and a relative value such
# as "./data" would degrade that rewrite into a search for the bare word "data".
DATA_BASE_DIR = Path(os.environ.get(
    'SCAI_DATA_DIR',
    str((_BACKEND_DIR.parent.parent.parent / "data").resolve())
)).expanduser().resolve()

_ENABLE_REPRODUCIBILITY = os.environ.get("SCAI_ENABLE_REPRODUCIBILITY", "").lower() in (
    "1",
    "true",
    "yes",
)

_ENABLE_METHOD_DECISION_GUIDANCE = os.environ.get(
    "SCAI_ENABLE_METHOD_DECISION_GUIDANCE", ""
).lower() in (
    "1",
    "true",
    "yes",
)


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
INTEGRITY_DIR = DATA_BASE_DIR / "integrity"


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
# Prepared-but-never-downloaded archives are full-size copies of a session. Nothing
# reclaimed them: the only unlink hung off a successful download, so a user who
# closed the "packaging complete" dialog left the bytes on disk forever.
PREPARED_EXPORT_TTL = 6 * 3600  # Delete prepared-but-undownloaded export archives after (seconds)
# The only filename shapes the export endpoints write into UPLOAD_TEMP_DIR.
# The orphan sweep in tasks/session_cleanup.py reclaims exactly these, so a new
# export producer belongs here rather than naming its own temp file — every
# other name in that directory is an upload chunk or an import archive with its
# own lifecycle, and the sweep must not widen past these prefixes.
PREPARED_EXPORT_PREFIX = "prepared_export_"
EXPORT_STREAM_PREFIX = "export_stream_"
#: No producer writes this any more -- the snapshot export route is gone. It
#: stays in the sweep set because an installation upgraded mid-export can still
#: hold a multi-gigabyte `v2-export-` temp that nothing else would ever reclaim.
RETIRED_SNAPSHOT_EXPORT_PREFIX = "v2-export-"
EXPORT_TEMP_PREFIXES = (
    PREPARED_EXPORT_PREFIX, EXPORT_STREAM_PREFIX, RETIRED_SNAPSHOT_EXPORT_PREFIX,
)
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


class _OwnerOnlyRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Keep the active log owner-only after initial open and every rollover."""

    def _open(self):
        stream = super()._open()
        os.chmod(self.baseFilename, 0o600)
        return stream


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Setup comprehensive logging for the application.

    Logs are written to:
    - {LOG_DIR}/app.log - Main application log (rotating, 10MB max, 5 backups)
    - {LOG_DIR}/error.log - Error-only log (rotating, 10MB max, 5 backups)
    - {LOG_DIR}/debug.log - Debug log with all details (rotating, 50MB max, 3 backups)
    - Console output
    """
    from backend.utils.secret_redaction import SecretRedactionFilter

    LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(LOG_DIR, 0o700)
    for existing_log in LOG_DIR.glob("*.log*"):
        if existing_log.is_file():
            os.chmod(existing_log, 0o600)

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
    console_handler.addFilter(SecretRedactionFilter())
    root_logger.addHandler(console_handler)

    def _add_file_handler(name, level, max_bytes, backup_count):
        path = LOG_DIR / f"{name}.log"
        handler = _OwnerOnlyRotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        os.chmod(path, 0o600)
        handler.setLevel(level)
        handler.setFormatter(detailed_formatter)
        handler.addFilter(SecretRedactionFilter())
        root_logger.addHandler(handler)

    _add_file_handler("app",   logging.INFO,   10*1024*1024, 5)
    _add_file_handler("error", logging.ERROR,  10*1024*1024, 5)
    _add_file_handler("debug", logging.DEBUG,  50*1024*1024, 3)

    # Drop uvicorn access-log lines for the health probe so container/orchestrator
    # health checks (every ~30s, one per line) don't drown out the real request log.
    logging.getLogger("uvicorn.access").addFilter(_HealthProbeAccessFilter())
    # aiosqlite DEBUG records include complete SQL bind parameters. LangGraph
    # checkpoint parameters contain the serialized AnalysisState and therefore
    # must never be copied into application logs.
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    # The openai and anthropic SDKs share a generated base client that logs the
    # complete request body at DEBUG ("Request options: ..."), so one vision call
    # copies its base64 image data URIs to disk verbatim — measured at 8.0-8.5 MB
    # for a single pathology tile batch. That collapses debug.log retention to
    # minutes during an image-heavy round and makes SecretRedactionFilter regex
    # the rendered megabytes on the request path. Both trees must be pinned:
    # ScAI probes three protocols and routes to Anthropic Messages for some
    # base_urls, and the leak is per-SDK.
    # httpcore is deliberately NOT pinned — it logs connection lifecycle events
    # only (connect_tcp, start_tls, proxy handshake), never the body, and those
    # are the only records distinguishing a failed handshake from an application
    # error when running behind proxy_url.
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("anthropic").setLevel(logging.INFO)

    # Log startup message
    logging.info(f"Logging initialized. Log directory: {LOG_DIR}")
    logging.info(f"Log files: app.log, error.log, debug.log")

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


MINIMUM_CONTEXT_WINDOW: int = 8_192
DEFAULT_AGENT_CONTEXT_WINDOW: int = 200_000


class AgentContextWindows(BaseModel):
    """Configured context-window defaults for text dispatches by each agent."""

    pi: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)
    pathology: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)
    bioinformatics: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)
    biochemistry: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)
    literature: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)
    painter: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)
    supervisor: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)
    technique: int = Field(default=DEFAULT_AGENT_CONTEXT_WINDOW, ge=MINIMUM_CONTEXT_WINDOW)

    def get(self, agent: str, default: int = DEFAULT_AGENT_CONTEXT_WINDOW) -> int:
        return getattr(self, agent, default)

    def merge_custom(self, custom: Dict[str, int]) -> "AgentContextWindows":
        data = self.model_dump()
        data.update({key: value for key, value in custom.items() if key in data})
        return AgentContextWindows(**data)
    
# Literature agent defaults
DEFAULT_LITERATURE_MODE = "pubmed_llm"
DEFAULT_LITERATURE_MAX_SEARCH_ROUNDS = 20

class SystemConfig(BaseModel):
    """System configuration"""
    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    prompts: AgentPrompts = Field(default_factory=AgentPrompts)
    temperatures: AgentTemperatures = Field(default_factory=AgentTemperatures)
    context_windows: AgentContextWindows = Field(default_factory=AgentContextWindows)
    
    # Image processing settings
    image_tile_size: int = 1024  # Size for pathology image tiles
    image_overlap: int = 64  # Overlap between tiles
    blank_threshold: float = 0.95  # Threshold for detecting blank tiles
    
    # Local Jupyter kernel settings
    sandbox_timeout: int = 5400  # Code execution timeout in seconds (1.5 hours)
    # Runtime persistence memory ceilings default to ``None`` (automatic):
    # workflow construction derives them from the Docker cgroup or host
    # capacity. They are deployment-scoped rather than database configuration;
    # the corresponding SCAI_RUNTIME_* environment variables may install
    # stricter absolute caps.
    runtime_max_anndata_total_bytes: Optional[int] = Field(
        default=None, ge=1, exclude=True
    )
    runtime_max_rss_bytes: Optional[int] = Field(
        default=None, ge=1, exclude=True
    )
    # Bioinformatics agent settings
    bioinfo_max_steps: int = 200  # Max autonomous analysis steps per execution
    
    # Session settings
    max_rounds: int = 3  # Maximum analysis rounds
    convergence_rounds: int = 2  # Rounds without new findings to stop

    enable_reproducibility: bool = Field(
        default=_ENABLE_REPRODUCIBILITY,
        description="Enable reproducibility bundle generation (checksums, environment snapshot, commands, decisions)",
    )

    enable_method_decision_guidance: bool = Field(
        default=_ENABLE_METHOD_DECISION_GUIDANCE,
        description=(
            "Enable the experimental, non-binding method decision card pilot. "
            "Disabled by default; set SCAI_ENABLE_METHOD_DECISION_GUIDANCE=true "
            "before backend startup to opt in."
        ),
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

    enable_post_execution_review: bool = Field(
        default=True,
        description="Enable the gated post-execution review (plan-versus-evidence audit -> PI triage -> restricted supplementary execution -> PI re-assessment) at the end of a bioinformatics execution. The plan audit itself always runs; this flag only controls the supplementary pass.",
    )
    post_execution_review_max_tasks: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Max supplementary tasks the PI may commission per execution. This is the cost ceiling: a PI asked what else could be done proposes something every time, so the budget is fixed here rather than left to its judgment. 0 disables commissioning while keeping the audit.",
    )
    post_execution_review_max_iterations: int = Field(
        default=5,
        ge=1,
        le=8,
        description="Max code-generation iterations for the supplementary executor (hard-capped at 8 in code).",
    )
    post_execution_review_timeout_seconds: int = Field(
        default=0,
        ge=0,
        description="Wall-clock budget (seconds) for ONE supplementary step — its code generation, its execution and the main-object invariant check. 0 (default) derives it from sandbox_timeout plus an LLM allowance, so a heavy analysis (GRN inference, deconvolution training, full ligand-receptor aggregation) gets its whole allowance rather than inheriting what earlier steps left over. Individual cells are also bounded and interrupted by sandbox_timeout itself. A step that overruns interrupts the kernel and is reported back to the executor, which then picks a cheaper route — it does not abandon the pass.",
    )
    post_execution_review_total_timeout_seconds: int = Field(
        default=0,
        ge=0,
        description="Wall-clock ceiling (seconds) for the whole supplementary pass. 0 (default) derives it as the sum of the per-step budgets, i.e. it never becomes the thing that cuts a step off and only backstops a pass that never converges. Lower it to cap the worst case (all steps running to their limit); it is always raised to at least one step's budget.",
    )

    # State persistence (deprecated: use get_session_states/get_session_workspace instead)
    # Kept for serialization compatibility only.
    state_dir: str = str(STATE_DIR)
    workspace_dir: str = str(WORKSPACE_DIR)

# Default configuration instance
default_config = SystemConfig()

# Shared agent identifier — single source of truth.
# Derived from the canonical enum so the string value can never drift.
PI_AGENT_NAME = AgentType.PI.value
