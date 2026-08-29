# Loading and querying the methodological rule corpus.
"""Access to backend/data/method_rules.yaml.

The corpus supplies non-blocking statistical observations and on-demand method
guidance to the PI. Rules are keyed by id so each observation carries its
mechanism and literature without host code deciding whether a result passes.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "method_rules.yaml"

#: Emitted in code as a non-blocking observation for PI review.
ENFORCEMENT_ADVISORY = "advisory"
ENFORCEMENT_GUIDANCE = "guidance"

ENFORCEMENT_VALUES = (ENFORCEMENT_ADVISORY, ENFORCEMENT_GUIDANCE)


@lru_cache(maxsize=1)
def _load() -> dict:
    """Parse the corpus once. A malformed corpus must not break an analysis."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml ships with the stack
        logger.warning("pyyaml unavailable; method rules disabled")
        return {"meta": {}, "rules": []}
    try:
        data = yaml.safe_load(_RULES_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("could not load method rules: %s", exc)
        return {"meta": {}, "rules": []}
    rules = data.get("rules") or []
    return {"meta": data.get("meta") or {}, "rules": rules}


def all_rules() -> list[dict]:
    return list(_load()["rules"])


def corpus_meta() -> dict:
    return dict(_load()["meta"])


def get_rule(rule_id: str) -> Optional[dict]:
    return next((r for r in all_rules() if r.get("id") == rule_id), None)


def format_rule(rule: dict, *, include_python: bool = True) -> str:
    """Render one rule for injection into a tool result.

    Kept compact on purpose: this text is appended to an observation the PI
    must assess, and a wall of prose would compete with the observation itself.
    """
    if not rule:
        return ""
    parts = [f"[method rule: {rule.get('id', '')}] {rule.get('rule', '').strip()}"]
    why = (rule.get("why") or "").strip()
    if why:
        parts.append(f"Why: {why}")
    if include_python:
        python = (rule.get("python") or "").strip()
        if python:
            parts.append(f"In this stack: {python}")
    citations = rule.get("citations") or []
    if citations:
        parts.append("See: " + "; ".join(str(c) for c in citations[:2]))
    return " ".join(parts)


def rule_text(rule_id: str, *, include_python: bool = True) -> str:
    """Formatted text for one rule id, or '' when the id is unknown."""
    return format_rule(get_rule(rule_id) or {}, include_python=include_python)


def search_rules(query: str, limit: int = 5) -> list[dict]:
    """Rules matching a free-text query, ranked by token overlap.

    Deliberately simple: the corpus is small enough that lexical matching beats
    anything needing an index, and a wrong match here is cheap because the model
    reads the rule before applying it.
    """
    tokens = {t for t in re.split(r"[^a-z0-9]+", str(query).lower()) if len(t) > 2}
    if not tokens:
        return []
    scored: list[tuple[int, dict]] = []
    for rule in all_rules():
        haystack = " ".join(str(rule.get(field, "")) for field in
                            ("id", "title", "rule", "why", "applies_when", "python")).lower()
        hay_tokens = {t for t in re.split(r"[^a-z0-9]+", haystack) if len(t) > 2}
        overlap = len(tokens & hay_tokens)
        if overlap:
            scored.append((overlap, rule))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("id", "")))
    return [rule for _, rule in scored[:limit]]


def attribution() -> str:
    """One-line provenance for the report methodology section."""
    meta = corpus_meta()
    if not meta:
        return ""
    return (
        f"Methodological rules distilled from {meta.get('source', 'bioSkills')} "
        f"({meta.get('source_url', '')}) at commit "
        f"{str(meta.get('source_commit', ''))[:12]}, {meta.get('source_license', '')} licence, "
        f"{meta.get('source_copyright', '')}; translated for this stack."
    )
