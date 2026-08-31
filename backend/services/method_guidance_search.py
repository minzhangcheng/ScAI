"""Shared lexical retrieval mechanics for methodological guidance corpora."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Iterable


def tokenize(value: Any) -> set[str]:
    """Return the legacy method-rule tokens for an arbitrary field value."""
    if isinstance(value, Mapping):
        text = " ".join(
            f"{key} {_field_text(item)}" for key, item in value.items()
        )
    else:
        text = _field_text(value)
    return {
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) > 2
    }


def _field_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {_field_text(item)}" for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_field_text(item) for item in value)
    return str(value or "")


def search_by_token_overlap(
    records: Iterable[dict],
    query: str,
    *,
    field_weights: Mapping[str, float],
    limit: int,
    id_field: str = "id",
    ignored_tokens: Iterable[str] = (),
) -> list[dict]:
    """Rank records by weighted query-token overlap with deterministic ties.

    A token occurring in several fields receives the largest applicable field
    weight, not the sum. Equal field weights therefore reproduce the previous
    method-rule union-of-fields overlap exactly. Scores can only grow as a
    query gains distinct matching tokens, so callers should pass focused
    queries and may exclude corpus-specific generic tokens.
    """
    ignored = {str(token).lower() for token in ignored_tokens}
    query_tokens = tokenize(query) - ignored
    if not query_tokens or limit <= 0:
        return []

    scored: list[tuple[float, dict]] = []
    for record in records:
        token_weights: dict[str, float] = {}
        for field, raw_weight in field_weights.items():
            weight = float(raw_weight)
            if weight <= 0:
                continue
            for token in tokenize(record.get(field, "")):
                token_weights[token] = max(token_weights.get(token, 0.0), weight)
        score = sum(token_weights.get(token, 0.0) for token in query_tokens)
        if score:
            scored.append((score, record))

    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get(id_field, ""))))
    return [record for _, record in scored[:limit]]
