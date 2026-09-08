"""Chat routes for PI Agent free-form chat with analysis sessions."""
import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from backend.config import get_session_workspace, get_session_reports, get_session_uploads, get_session_states, PI_AGENT_NAME
from backend.models.schemas import WebSocketMessage, AgentType, DataFile, AnalysisState
from backend.services.session_store import ensure_session_loaded, async_save_session, update_session_cache
from backend.state import (
    chat_tasks, chat_task_lock, chat_sandboxes,
    websockets, workflows, state_lock,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL

# Kept verbatim from the pre-split module, as every other facade/parts split in
# this repo does, so `backend.routes.chat.<name>` stays a live attribute for
# outside importers. test_pi_agent_name_constant.py reads PI_AGENT_NAME off this
# module, so pruning is not free.

# Implementations live in ./chat_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from .chat_parts.core import (  # noqa: F401
    logger,
    router,
    _build_chat_context,
)
from .chat_parts.figures import (  # noqa: F401
    _build_figure_setup_code,
    _generate_figures_in_chat,
    _collect_session_figures,
)
from .chat_parts.delegation import (  # noqa: F401
    _send_chat_ws,
    _try_parse_delegation,
    _build_synthesis_prompt,
    _execute_chat_agent,
    _run_chat_delegation,
)
from .chat_parts.endpoints import (  # noqa: F401
    chat_with_pi,
    cancel_chat,
    get_chat_history,
)
