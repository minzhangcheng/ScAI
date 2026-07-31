"""
Supervisor Agent - Lightweight real-time monitoring of bioinformatics analysis.

Uses a cheap vision LLM to evaluate every analysis step, detect technical issues,
identify biological checkpoints, and escalate to PI when needed. Also generates
the detailed report from its accumulated supervision log.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional
import logging
import re


logger = logging.getLogger(__name__)


from ...utils.image_utils import save_graphic_abstract_image

# Section IDs that represent specific/targeted analysis
_SPECIFIC_SECTION_IDS = frozenset({
    "differential_expression", "key_findings", "integrated_findings",
    "exploration_findings",
})

# Section ID → default fallback title (used in prompt guidance and parse-time safety net)
_SECTION_ID_TO_DEFAULT_TITLE: dict[str, str] = {
    "abstract": "Abstract",
    "analysis_overview": "Analysis Methodology Overview",
    "data_qc": "Data Overview and Quality Control",
    "clustering_annotation": "Clustering and Cell Type Annotation",
    "differential_expression": "Differential Expression Analysis",
    "exploration_findings": "Phase 1 Exploration Findings Summary",
    "integrated_findings": "Integrated Findings Across All Phases",
    "key_findings": "Key Biological Findings",
    "conclusions": "Conclusions and Future Directions",
}

# Placeholder title strings that LLMs sometimes emit verbatim instead of generating titles
_PLACEHOLDER_SECTION_TITLES = frozenset({
    "section title",
    "your descriptive scientific title here",
})

# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ...utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SupervisorRecord:
    """One record per monitored step."""
    step_number: int
    step_type: str
    timestamp: datetime
    technical_issues: List[str] = field(default_factory=list)
    checkpoint_type: Optional[str] = None
    llm_severity: Optional[str] = None  # "none"|"info"|"warning"|"critical"
    pi_escalated: bool = False
    pi_response: Optional[str] = None
    feedback_injected: Optional[str] = None
    # Literature fields
    literature_topic: Optional[str] = None
    literature_result: Optional[str] = None
    # Raw data for report generation
    code: Optional[str] = None
    code_output: Optional[str] = None
    code_error: Optional[str] = None
    saved_figure_filenames: List[str] = field(default_factory=list)
    agent_reasoning: Optional[str] = None


@dataclass
class SupervisorEvalResult:
    """Return type for SupervisorAgent.evaluate_step().

    Attributes:
        feedback: Advisory feedback string to inject into agent messages, or None.
        force_skip: When True, the caller MUST treat the current step as
            non-recoverable and force-skip it. Set when the agent is stuck
            AND PI has already intervened but the agent still cannot fix it.
    """
    feedback: Optional[str] = None
    force_skip: bool = False


# ---------------------------------------------------------------------------
# Rule-based patterns
# ---------------------------------------------------------------------------

_CRITICAL_OUTPUT_PATTERNS = [
    re.compile(r"\b0\s+cells?\b", re.IGNORECASE),
    re.compile(r"\bEmpty DataFrame\b"),
    re.compile(r"\bMemoryError\b"),
    re.compile(r"\bn_obs\s*[:=]\s*([0-9]+)"),
    re.compile(r"obs_names are not unique", re.IGNORECASE),
    re.compile(r"InvalidIndexError"),
]

# Patterns for data integrity checks on code output
_PERCENTAGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE
)
_INF_NAN_PATTERN = re.compile(
    r"\b(inf|-inf|nan)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# System prompt for the cheap LLM evaluator
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM_PROMPT = """\
You are a bioinformatics analysis supervisor. Your job is to evaluate each analysis step \
for technical correctness and identify important biological decision points.

Respond ONLY with a JSON object (no markdown, no explanation):
{
  "severity": "none|info|warning|critical",
  "issues": ["list of specific technical issues found, empty if none"],
  "action": "none|warn|redirect",
  "message": "short feedback for the analyst (1-2 sentences), empty string if no issues",
  "redirect_instruction": "specific instruction if action=redirect, empty string otherwise",
  "issue_type": "technical|biological|ambiguous",
  "is_biological_checkpoint": false,
  "pi_question": "question to ask PI if is_biological_checkpoint=true, empty string otherwise"
}

IMPORTANT: All responses must be in English.

Severity guide:
- none: step looks fine
- info: minor note, no action needed
- warning: potential issue that analyst should be aware of, including a figure whose \
labels, legend or annotations are illegible
- critical: blank/broken figure, a figure where text is covered by another element and \
cannot be read at all, repeated errors, zero cells, memory errors, obviously wrong results

IMPORTANT: Code and output may be truncated for display (only the last 2000 characters are shown for long content). If content is marked as truncated, do NOT report it as syntactically incomplete or missing. Only report syntax errors if the Execution status is FAILED and the Error indicates a SyntaxError.

Biological checkpoint criteria (set is_biological_checkpoint=true when):
- A major analysis phase completed (clustering, DE, annotation, trajectory, batch correction)
- Results require biological interpretation or domain expertise
- Analysis reached a decision fork (multiple directions possible)
- PI guidance would improve the next analysis direction

IMPORTANT: Do NOT set is_biological_checkpoint=true for purely technical issues such as:
- Sandbox security blocks (e.g., "Dangerous pattern detected")
- Code validation failures or syntax errors
- Missing package / ModuleNotFoundError
- Memory errors or kernel crashes
These are technical problems the analyst can fix on their own without PI guidance.

Issue type guide (only relevant when severity is "warning" or "critical"):
- technical: coding/programming problem that the analyst can fix alone. Examples: \
wrong column name used for plotting (blank plot), incorrect API usage, data wrangling \
logic error, missing data transformation step, wrong plot type parameters.
- biological: data quality or biological interpretation issue needing PI expertise. \
Examples: all cells filtered out (QC thresholds too strict), suspicious clustering \
patterns, unexpected cell type proportions, batch effect concerns, annotation disagreement.
- ambiguous: unclear whether technical or biological. Default to this when unsure.

If images are provided, check for:
- Blank or empty plots (no data visible)
- Axis labels missing or nonsensical
- Plots with no meaningful variation (flat lines, uniform colors)
- Legends or annotations that are cut off or overlapping
- Text drawn on top of other text: a legend covering the tick labels of a neighbouring \
panel, a panel letter sitting on top of an axis label, a stray annotation over the title
- Tick labels running into each other, or text too small to read
- Two series in one legend sharing the same colour
Report these as issues. A figure whose text cannot be read conveys nothing, however \
correct the analysis behind it is.
"""


def _serialize_record(r, round_number: int) -> Dict:
    d = asdict(r)
    d["timestamp"] = r.timestamp.isoformat()
    d["round_number"] = round_number
    return d


def persist_graphic_abstract(graphic_abstract, figures_dir) -> "str | None":
    """Write the (image-mode) graphic abstract to figures_dir/graphic_abstract.jpg
    and return its path. Returns None for text-mode (no image_base64) or absence."""
    if graphic_abstract is None:
        return None
    b64 = getattr(graphic_abstract, "image_base64", "") or ""
    return save_graphic_abstract_image(b64, figures_dir)


