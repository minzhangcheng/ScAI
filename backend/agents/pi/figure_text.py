"""Pure text helpers for the PI figure prompts.

Kept out of figures.py, which is at its size limit. Neither touches the agent.
"""
from __future__ import annotations

from typing import Any, Dict, List


def build_ga_revision_history_text(revision_history: List[Dict[str, Any]]) -> str:
    """Summarise graphic-abstract revision feedback for a prompt."""
    if not revision_history:
        return ""
    lines = []
    for entry in revision_history:
        round_num = entry.get("round", "?")
        hist_feedback = entry.get("feedback", "")
        lines.append(f"  Round {round_num}: {hist_feedback}")
    return "- Previous revision feedback:\n" + "\n".join(lines)


def describe_candidate(figure: Dict[str, Any]) -> str:
    """One prompt line per candidate, using whatever the caller supplied.

    Every field beyond name and type is optional: this is also called with bare
    ``{name, type, is_publication}`` dicts, so nothing may be required.
    """
    parts = [f"- {figure['name']} (type: {figure['type']}"]
    if figure.get("round"):
        parts.append(f", round: {figure['round']}")
    if figure.get("diagnostic"):
        parts.append(", diagnostic")
    parts.append(")")
    description = (figure.get("description") or "").strip()
    if description:
        parts.append(f" \u2014 {description}")
    return "".join(parts)
