"""Fail-open access to non-binding method decision cards."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .method_guidance_search import search_by_token_overlap

logger = logging.getLogger(__name__)

_CARDS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "method_decision_cards.yaml"
)
MAX_GUIDANCE_CARDS = 3
MAX_GUIDANCE_CHARS = 6_000

_CARD_FIELDS = (
    "id",
    "title",
    "enforcement",
    "applies_when",
    "aliases",
    "facts_to_consider",
    "candidate_approaches",
    "selection_considerations",
    "parameters_to_justify",
    "diagnostics",
    "expected_outputs",
    "related_rule_ids",
    "citations",
    "source_provenance",
)
_LIST_FIELDS = (
    "aliases",
    "facts_to_consider",
    "candidate_approaches",
    "selection_considerations",
    "parameters_to_justify",
    "diagnostics",
    "expected_outputs",
    "related_rule_ids",
    "citations",
)
_APPROACH_FIELDS = (
    "name",
    "answers",
    "consider_when",
    "limitations",
    "python_stack",
)
_FORBIDDEN_FIELDS = {
    "preferred",
    "rank",
    "score",
    "winner",
    "blocking",
    "fallback_on_failure",
}
_SEARCH_WEIGHTS = {
    "id": 4.0,
    "title": 4.0,
    "aliases": 3.0,
    "applies_when": 2.0,
    "facts_to_consider": 1.0,
    "candidate_approaches": 2.0,
    "selection_considerations": 1.0,
    "parameters_to_justify": 1.0,
    "diagnostics": 1.0,
}


def _has_forbidden_field(value) -> bool:
    if isinstance(value, dict):
        return bool(_FORBIDDEN_FIELDS & set(value)) or any(
            _has_forbidden_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_has_forbidden_field(item) for item in value)
    return False


def _validate_cards(cards) -> list[dict]:
    if not isinstance(cards, list) or not cards:
        raise ValueError("cards must be a non-empty list")
    ids: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            raise ValueError("each card must be an object")
        missing = [field for field in _CARD_FIELDS if not card.get(field)]
        if missing:
            raise ValueError(f"card {card.get('id', '<unknown>')} missing {missing}")
        if card.get("enforcement") != "guidance":
            raise ValueError(f"card {card.get('id')} must use guidance enforcement")
        if _has_forbidden_field(card):
            raise ValueError(f"card {card.get('id')} contains a ranking/gate field")
        card_id = str(card["id"])
        if card_id in ids:
            raise ValueError(f"duplicate card id: {card_id}")
        ids.add(card_id)
        for field in _LIST_FIELDS:
            if not isinstance(card.get(field), list) or not card[field]:
                raise ValueError(f"card {card_id} field {field} must be a list")
        for approach in card["candidate_approaches"]:
            if not isinstance(approach, dict) or any(
                not approach.get(field) for field in _APPROACH_FIELDS
            ):
                raise ValueError(f"card {card_id} has an incomplete approach")
            for field in ("consider_when", "limitations", "python_stack"):
                if not isinstance(approach[field], list) or not approach[field]:
                    raise ValueError(
                        f"card {card_id} approach field {field} must be a list"
                    )
        for citation in card["citations"]:
            if not isinstance(citation, dict) or not citation.get(
                "citation"
            ) or not citation.get("supports"):
                raise ValueError(f"card {card_id} has an incomplete citation")
    return cards


@lru_cache(maxsize=1)
def _load() -> dict:
    """Load and validate once; any failure disables this optional corpus."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml ships with the stack
        logger.warning("pyyaml unavailable; method decision cards disabled")
        return {"meta": {}, "cards": []}
    try:
        data = yaml.safe_load(_CARDS_PATH.read_text(encoding="utf-8")) or {}
        cards = _validate_cards(data.get("cards"))
    except Exception as exc:
        logger.warning("could not load method decision cards: %s", exc)
        return {"meta": {}, "cards": []}
    return {"meta": data.get("meta") or {}, "cards": cards}


def all_cards() -> list[dict]:
    return list(_load()["cards"])


def corpus_meta() -> dict:
    return dict(_load()["meta"])


def get_card(card_id: str) -> Optional[dict]:
    return next((card for card in all_cards() if card.get("id") == card_id), None)


def search_cards(query: str, limit: int = MAX_GUIDANCE_CARDS) -> list[dict]:
    return search_by_token_overlap(
        all_cards(),
        query,
        field_weights=_SEARCH_WEIGHTS,
        limit=min(max(int(limit), 0), MAX_GUIDANCE_CARDS),
    )


def _bullets(values: list) -> str:
    return "\n".join(f"- {value}" for value in values)


def format_card(card: dict) -> str:
    """Render the card fields authorized for prompt and tool context."""
    if not card:
        return ""
    lines = [
        f"[decision card: {card['id']}] {card['title']}",
        f"Analytical question: {card['applies_when']}",
        "Facts to consider:",
        _bullets(card["facts_to_consider"]),
        "Unranked approaches:",
    ]
    for approach in card["candidate_approaches"]:
        lines.append(f"- {approach['name']}: {approach['answers']}")
        lines.append(
            "  Consider when: " + "; ".join(approach["consider_when"])
        )
        lines.append("  Limitations: " + "; ".join(approach["limitations"]))
    lines.extend([
        "Selection considerations:",
        _bullets(card["selection_considerations"]),
        "Parameters to justify:",
        _bullets(card["parameters_to_justify"]),
        "Diagnostics:",
        _bullets(card["diagnostics"]),
        "Related rule IDs: " + ", ".join(card["related_rule_ids"]),
    ])
    citations = card["citations"][:2]
    lines.append(
        "Citations: "
        + "; ".join(
            f"{item['citation']} ({item['supports']})" for item in citations
        )
    )
    return "\n".join(lines)


def guidance_for_query(
    query: str,
    *,
    limit: int = MAX_GUIDANCE_CARDS,
    max_chars: int = MAX_GUIDANCE_CHARS,
) -> str:
    """Return a bounded card block, or an empty string on no match/failure."""
    if max_chars <= 0:
        return ""
    try:
        rendered = [format_card(card) for card in search_cards(query, limit=limit)]
    except Exception as exc:
        logger.warning("method decision card retrieval failed: %s", exc)
        return ""
    text = "\n\n".join(item for item in rendered if item)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"
