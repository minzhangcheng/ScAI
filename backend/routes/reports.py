"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import asyncio
import io
import json
import logging
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from backend.config import (
    SESSION_DIR, REPORTS_DIR, WORKSPACE_DIR, UPLOAD_DIR, UPLOAD_TEMP_DIR,
    get_session_dir, get_session_reports, get_session_workspace, PI_AGENT_NAME,
)
from backend.models.schemas import AnalysisState, APIResponse, TaskStatus
from backend.workflow import _retry_backoff_delay

from backend.state import detailed_reports, report_websockets
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import safe_filename, validate_path_within, validate_session_id, slugify_session_name, temp_file_on_error
from backend.utils.auth import require_session_ownership
from .serialization import serialize_round_results
from backend.utils.settings_provider import AgentSettingsProvider
from backend.services.session_store import async_save_session, update_session_cache as _update_session_cache
from backend.services.report.paths import resolve_report_output_dir, resolve_report_target_lineage
from backend.services.report.lineage import build_lineage_report_state

logger = logging.getLogger(__name__)

# Implementations live in ./reports_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from .reports_parts.core import (  # noqa: F401
    REPORT_ARCHIVE_PREFIX,
    _should_skip_in_report_zip,
    _branch_query,
    _report_cache_key,
    _report_cache_key_for_lineage,
    router,
    _report_generation_pending,
    _pending_lock,
    _try_begin_generation,
    clear_generation_pending,
    clear_detailed_report_cache,
    _sanitize_nan,
    _load_report_json,
    _next_archive_version,
    _create_report_archive,
    _enforce_archive_retention,
)
from .reports_parts.html_views import (  # noqa: F401
    get_exploration_report_html,
    get_annotation_report_html,
    get_integration_report_html,
    get_detailed_report_html,
    get_round_report_html,
    get_detailed_report,
    generate_detailed_report,
    _generate_report_task,
)
from .reports_parts.downloads import (  # noqa: F401
    download_all_figures,
    download_all_tables,
    download_reproducibility_bundle,
    download_full_report,
    list_report_archives,
    download_report_archive,
)
from .reports_parts.assets import (  # noqa: F401
    get_report_html,
    preview_report,
    get_report_figure,
    get_report_figure_alt,
    get_report_table,
    get_report_table_alt,
    _resolve_report_file,
    get_report_file,
)
from .reports_parts.publication import (  # noqa: F401
    get_graphic_abstract,
    get_publication_figures,
    get_publication_tables,
    download_publication_figure_pdf,
    download_publication_figure_png,
    get_integrated_report,
    get_code_history,
    get_report,
    approve_partial_report,
)



























# ============================================
# Exploration and Annotation Report HTML
# ============================================









# ============================================
# Detailed Report Generation
# ============================================







# ============================================
# Report Download Endpoints
# ============================================













# ============================================
# Report HTML and Preview
# ============================================

















# ============================================
# Publication Figures and Graphic Abstract Endpoints
# ============================================















# ============================================
# Code History and Report Summary
# ============================================




