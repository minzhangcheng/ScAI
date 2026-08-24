"""
Supervisor Agent - Lightweight real-time monitoring of bioinformatics analysis.

Uses a cheap vision LLM to evaluate every analysis step, detect technical issues,
identify biological checkpoints, and escalate to PI when needed. Also generates
the detailed report from its accumulated supervision log.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Awaitable
import asyncio
import json
import logging
import os
import re
import hashlib
import uuid
import io
import base64
import shutil
from pathlib import Path
from urllib.parse import quote as url_quote

from ..utils.formatting import safe_parse_json, format_revision_history
from ..utils.files import resolve_figure_in_workspace
from ..utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ..constants import REASONING_MODEL_MIN_TOKENS

logger = logging.getLogger(__name__)

from openai import APIConnectionError

from ..models.schemas import AgentType, AnalysisState, AnalysisStep, PlanStage
from ..utils.error_parsing import API_ERROR_TYPES, CODE_GEN_ERROR_TYPES, extract_error_type, TECHNICAL_ERROR_TYPES
from ..utils.llm_client import OpenAICompatibleClient, UpstreamExhaustedError
from ..utils.settings_provider import AgentSettingsProvider
from ..utils.image_utils import save_graphic_abstract_image
from ..services.report.interaction_records import write_interaction_record_files
from ..services.report.artifacts import artifact_stems, eligible_artifacts
from .prompt_blocks import build_integration_block

# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ..utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402

# Re-exported so `backend.agents.supervisor_agent.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .supervisor.shared import (  # noqa: F401
    _SPECIFIC_SECTION_IDS,
    _SECTION_ID_TO_DEFAULT_TITLE,
    _PLACEHOLDER_SECTION_TITLES,
    SupervisorRecord,
    SupervisorEvalResult,
    _CRITICAL_OUTPUT_PATTERNS,
    _PERCENTAGE_PATTERN,
    _INF_NAN_PATTERN,
    _SUPERVISOR_SYSTEM_PROMPT,
    _serialize_record,
    persist_graphic_abstract,
)
from .supervisor import (  # noqa: F401
    RuntimeMixin,
    EvaluationMixin,
    ReportContentMixin,
    ReportHtmlMixin,
    DiagramsMixin,
)
class SupervisorAgent(
    RuntimeMixin,
    EvaluationMixin,
    ReportContentMixin,
    ReportHtmlMixin,
    DiagramsMixin,
):
    """Lightweight analysis supervisor using a cheap vision-capable LLM."""

    def __init__(
        self,
        settings_provider: 'AgentSettingsProvider',
        pi_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
        message_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
        session_id: str = "default",
        technique_agent=None,
    ):
        self._provider = settings_provider
        self.session_id = session_id
        self.pi_callback = pi_callback
        self.message_callback = message_callback
        self.technique_agent = technique_agent
        self._sandbox = None
        self._vision_disabled_runtime: bool = False
        self._last_config_revision: Optional[int] = None

        # Current round state
        self.supervision_log: List[SupervisorRecord] = []
        self._step_count: int = 0
        self._recent_errors: List[str] = []
        self._pi_question_hashes: set = set()  # De-duplicate PI escalations
        self._recent_rule_feedback_hashes: set = set()  # De-duplicate rule-based feedback
        self._diagnosed_packages: set = set()
        self.planning_stage: PlanStage = "full"

        # All rounds archive
        self.all_rounds_log: List[List[SupervisorRecord]] = []

    # --- Properties (hot-reloadable via settings provider) ---










    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------






    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Internal: Error classification
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Internal: Rule-based checks
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Internal: LLM evaluation
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Text-mode graphic abstract review (fallback reviewer)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Internal: PI escalation
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # Internal: Utilities
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # Report generation helpers
    # ------------------------------------------------------------------














