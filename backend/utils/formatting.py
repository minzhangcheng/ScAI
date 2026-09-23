"""Shared formatting and parsing utilities for reports and HTML generation."""

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Optional import for json_repair (handles malformed JSON from LLM responses)
try:
    from json_repair import repair_json as _repair_json
    _JSON_REPAIR_AVAILABLE = True
except ImportError:
    _repair_json = None  # type: ignore[assignment]
    _JSON_REPAIR_AVAILABLE = False


def as_list(value: Any) -> List[Any]:
    """Return an independent list for scalar or unstable LLM-produced values."""
    if value is None or (isinstance(value, str) and value == ""):
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def normalize_text_list(
    value: Any,
    *,
    limit: int = 20,
    max_chars: Optional[int] = 1000,
) -> List[str]:
    """Normalize a scalar-or-iterable text field without iterating strings."""
    item_limit = max(0, int(limit))
    if item_limit == 0:
        return []
    if isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    ):
        raw_items = list(value)
    else:
        raw_items = as_list(value)
    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if max_chars is not None:
            text = text[:max(0, int(max_chars))]
        if text and text not in normalized:
            normalized.append(text)
        if len(normalized) >= item_limit:
            break
    return normalized


def coerce_json_text(value: Any, *, default: str = "") -> str:
    """Preserve planner text while serializing structured values deterministically."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, OverflowError):
            pass
    return str(value).strip()


def fmt_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_server(url: str) -> str:
    """Extract hostname from a URL for display."""
    if not url:
        return "-"
    try:
        return urlparse(url).hostname or url
    except Exception:
        return url


def loads_llm_json(response: str) -> Any:
    """Decode model JSON, preserving literal control characters inside strings.

    Models sometimes emit raw newlines or tabs in instructions and code. This
    only relaxes string escaping; broken JSON structure still raises. Keep
    persisted files and external request validation on the strict decoder.
    """
    return json.loads(response, strict=False)


def safe_parse_json(response: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response using balanced bracket matching.

    Handles common LLM output patterns where JSON is embedded in markdown
    code blocks or surrounded by explanatory text.

    Args:
        response: Raw LLM response string (may be None/empty).
        default: Value to return when parsing fails. If *None*, returns
                 ``None`` on failure.

    Returns:
        Parsed dict, or *default* when extraction/parsing fails.
    """
    if not response:
        return default

    start = response.find('{')
    if start == -1:
        return default

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i, c in enumerate(response[start:], start):
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if depth == 0 and end > start:
        try:
            return loads_llm_json(response[start:end])
        except json.JSONDecodeError:
            # Try to repair malformed JSON from LLM responses
            if _JSON_REPAIR_AVAILABLE:
                try:
                    repaired = _repair_json(response[start:end], return_objects=True)
                    if isinstance(repaired, dict):
                        return repaired
                except Exception:
                    pass
            logger.debug("JSON extraction from LLM response failed at position %d-%d", start, end)

    if _JSON_REPAIR_AVAILABLE:
        # Skip response[start:] when it's identical to the already-tried
        # response[start:end] (i.e., bracket extraction reached the string end).
        candidates = []
        if end < len(response):
            candidates.append(response[start:])
        candidates.append(response)
        for candidate in candidates:
            try:
                repaired = _repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict):
                    return repaired
            except Exception:
                pass

    return default


def safe_parse_json_array(response: str, default: Optional[List[Any]] = None) -> Optional[List[Any]]:
    """Parse a JSON array from LLM response using balanced bracket matching.

    Similar to safe_parse_json but targets ``[...]`` instead of ``{...}``.

    Args:
        response: Raw LLM response string (may be None/empty).
        default: Value to return when parsing fails. If *None*, returns
                 ``None`` on failure.

    Returns:
        Parsed list, or *default* when extraction/parsing fails.
    """
    if not response:
        return default

    start = response.find('[')
    if start == -1:
        return default

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i, c in enumerate(response[start:], start):
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if depth == 0 and end > start:
        try:
            result = loads_llm_json(response[start:end])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            if _JSON_REPAIR_AVAILABLE:
                try:
                    repaired = _repair_json(response[start:end], return_objects=True)
                    if isinstance(repaired, list):
                        return repaired
                except Exception:
                    pass
            logger.debug("JSON array extraction from LLM response failed at position %d-%d", start, end)

    return default


def truncate_for_llm(text: str, max_chars: int = 128 * 1024) -> str:
    """Truncate sandbox output for LLM consumption, keeping head and tail halves.

    When text exceeds max_chars, retains the first and last (max_chars // 2)
    characters with an omission marker in between so the LLM sees both the
    start (context/imports/setup) and the end (errors/results).
    """
    if not text or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return (
        text[:half]
        + f"\n\n[... truncated {omitted} chars from sandbox output ...]\n\n"
        + text[-half:]
    )


def format_revision_history(entries: List[Dict[str, Any]], include_reviewer: bool = True) -> str:
    """Format revision history entries as a human-readable text block.

    Args:
        entries: List of revision entries with 'round', 'reviewer', 'feedback' keys.
        include_reviewer: Whether to include reviewer name in the output.

    Returns:
        Formatted history text with "Previous feedback:" header, or empty string if no entries.
    """
    if not entries:
        return ""

    lines = []
    for entry in entries:
        r = entry.get("round", "?")
        fb = entry.get("feedback", "")
        if include_reviewer:
            rev = entry.get("reviewer", "reviewer")
            lines.append(f"  Round {r} ({rev}): {fb}")
        else:
            lines.append(f"  Round {r}: {fb}")

    return "Previous feedback:\n" + "\n".join(lines)


# --- analysis-plan step rendering -------------------------------------------

# Field caps tried largest-first until the rendered block fits its budget.
PLAN_STEP_FIELD_CAPS = (600, 400, 260, 160, 90)

_PLAN_STEP_FIELDS = (
    ("Method", "method"),
    ("Input", "input"),
    ("Output", "expected_output"),
    ("Required figures", "figure_requirements"),
    ("Text-only exception", "allow_text_only"),
    ("Text-only reason", "text_only_reason"),
    ("Analysis scope", "analysis_scope"),
    ("Rationale", "rationale"),
)


def _json_model_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _collapse(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, (dict, list)):
        value = json.dumps(
            value,
            default=_json_model_default,
            ensure_ascii=False,
            sort_keys=True,
        )
    return " ".join(str(value or "").split())


def render_plan_steps(steps: Any, tag: Any = None, *, budget: int = 24_000) -> str:
    """Render analysis-plan steps with their full field payload for a prompt.

    Never drops a step. Clipping a joined step list at a character budget
    truncates the tail, and the tail is where a plan's interpretation and
    grading steps live; shrinking the per-field cap instead keeps every step's
    identity and intent visible. That distinction is the whole point — a step
    rendered as ``title (method)`` tells a revising model nothing about what it
    analyses, so the model regenerates the missing fields generically.

    ``tag`` labels each step: a fixed string, or ``None`` to use the step's own
    ``status``.
    """
    step_list = list(steps or [])
    if not step_list:
        return ""

    # Normalize once. The cap only decides where to cut, so re-collapsing
    # whitespace for every candidate cap is pure repeat work.
    normalized = [
        (
            str(tag) if tag is not None else str(getattr(s, "status", "pending")),
            s.id,
            _collapse(getattr(s, "title", "")),
            [(label, _collapse(getattr(s, attr, ""))) for label, attr in _PLAN_STEP_FIELDS],
            list(getattr(s, "depends_on", None) or []),
        )
        for s in step_list
    ]

    def render(cap: Optional[int]) -> str:
        blocks = []
        for label, step_id, title, fields, depends in normalized:
            lines = [f"  [{label}] Step {step_id}: {title}"]
            if cap is None:
                if fields[0][1]:
                    lines[0] += f" ({fields[0][1][:120]})"
            else:
                for name, value in fields:
                    if value:
                        clipped = value if len(value) <= cap else value[: cap - 1].rstrip() + "..."
                        lines.append(f"    {name}: {clipped}")
                if depends:
                    lines.append(f"    Depends on: {depends}")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    for cap in PLAN_STEP_FIELD_CAPS:
        block = render(cap)
        if len(block) <= budget:
            return block
    logger.warning(
        "%d plan steps exceed the %d-char budget even at the tightest field cap; "
        "rendering titles and methods only", len(step_list), budget,
    )
    return render(None)
