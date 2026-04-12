"""Shared formatting and parsing utilities for reports and HTML generation."""

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Optional import for json_repair (handles malformed JSON from LLM responses)
try:
    from json_repair import repair_json as _repair_json
    _JSON_REPAIR_AVAILABLE = True
except ImportError:
    _JSON_REPAIR_AVAILABLE = False


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
            return json.loads(response[start:end])
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

    return default


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
