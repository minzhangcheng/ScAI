"""Bounded, domain-neutral notes for stable runtime API mismatches."""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


_MODULE_ATTRIBUTE = re.compile(
    r"AttributeError: module '([A-Za-z0-9_.]+)' has no attribute '([A-Za-z0-9_]+)'"
)
_UNEXPECTED_KEYWORD = re.compile(
    r"TypeError: ([A-Za-z0-9_.]+)\(\) got an unexpected keyword argument '([A-Za-z0-9_]+)'"
)
_POSITIONAL_ARITY = re.compile(
    r"TypeError: ([A-Za-z0-9_.]+)\(\) takes \d+ positional arguments? but \d+ (?:were|was) given"
)


def api_change_signature(error_text: Any) -> str:
    """Return a stable signature for a narrow, recognizable API mismatch."""
    text = str(error_text or "")
    match = _MODULE_ATTRIBUTE.search(text)
    if match:
        return f"{match.group(1)}.{match.group(2)} (missing attribute)"
    match = _UNEXPECTED_KEYWORD.search(text)
    if match:
        return f"{match.group(1)}({match.group(2)}=...) (unexpected keyword)"
    match = _POSITIONAL_ARITY.search(text)
    if match:
        return f"{match.group(1)}() (positional arity)"
    return ""


def recorded_api_note(state: Any, signature: str) -> Optional[str]:
    if state is None or not signature:
        return ""
    notes = getattr(state, "environment_api_notes", None) or {}
    if not isinstance(notes, dict):
        return ""
    return str(notes.get(signature) or "").strip()[:2000]


def record_api_note(state: Any, signature: str, note: str) -> None:
    if state is None or not signature or not str(note or "").strip():
        return
    try:
        notes = dict(getattr(state, "environment_api_notes", None) or {})
        notes[signature] = str(note).strip()[:2000]
        state.environment_api_notes = notes
        logger.info("Recorded environment API note for %s", signature)
    except Exception as exc:
        logger.warning("Could not record the environment API note: %s", exc)


def format_api_note(signature: str, note: Optional[str]) -> str:
    if not signature or not str(note or "").strip():
        return ""
    api_name = signature.split(" (")[0]
    bounded_note = str(note).strip()[:2000]
    return (
        f"\n\n[Environment API] `{api_name}` failed the same way earlier in "
        f"this session ({signature}): the installed library version does not "
        "support that call. Apply the guidance below instead of retrying the "
        "original call:\n"
        f"{bounded_note}"
    )
