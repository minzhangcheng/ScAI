"""Report variant generators - HTML report builders for different analysis phases."""
import os
import html as html_module
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.utils.report_html import (
    markdown_to_html,
    _make_figure_block,
    _build_figure_description_map,
)
from backend.utils.image_utils import compress_image_to_base64, categorize_figures
from backend.config import get_session_dir, get_session_reports
from backend.utils.blob_store import inline_blobs
from backend.utils.formatting import as_list
from backend.services.report.score_section import build_score_definitions_section_html
from backend.services.report.artifacts import artifact_stems, eligible_artifacts
from backend.models.schemas import AnalysisState

logger = __import__('logging').getLogger(__name__)


# Implementations live in ./variants_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from .variants_parts.renderers import (  # noqa: F401
    _SPECIES_TISSUE_LABELS,
    _render_score_badge,
    _render_direction_evidence_section_from_lists,
    _render_direction_evidence_section,
    _render_direction_candidates_rows,
)
from .variants_parts.exploration import (  # noqa: F401
    generate_exploration_report_html,
)
from .variants_parts.detailed import (  # noqa: F401
    generate_detailed_report_segment_html,
    _deduplicate_detailed_figure_blocks,
    generate_combined_detailed_report_html,
)
from .variants_parts.integration import (  # noqa: F401
    _inline_session_text,
    _render_integration_section,
    _render_novelty_score,
    _render_evidence_strength,
    _render_integration_candidates_rows,
    _render_integration_evidence_section,
    generate_integration_report_html,
)
from .variants_parts.annotation import (  # noqa: F401
    generate_annotation_context_report_html,
)
































