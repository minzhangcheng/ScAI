"""Shared formatting and parsing utilities for reports and HTML generation."""

import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


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
            logger.debug("JSON extraction from LLM response failed at position %d-%d", start, end)

    return default
