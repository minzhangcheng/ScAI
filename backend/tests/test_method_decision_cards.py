"""Structural and retrieval contracts for optional method decision cards."""
import json
import re

import pytest

from backend.services import method_decision_cards as cards_mod
from backend.services.method_decision_cards import (
    MAX_GUIDANCE_CARDS,
    MAX_GUIDANCE_CHARS,
    all_cards,
    build_planning_guidance_query,
    format_card,
    get_card,
    guidance_for_query,
    search_cards,
)
from backend.services.method_rules import all_rules, get_rule, search_rules


REQUIRED_IDS = {
    "latent-representation-and-batch-integration",
    "differential-expression-method-choice",
    "clustering-resolution-and-stability",
    "cell-annotation-strategy",
    "trajectory-connectivity-and-velocity",
    "gene-set-and-regulatory-activity-method-choice",
    "composition-abundance-and-expression-choice",
}
REQUIRED_FIELDS = {
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
}
FORBIDDEN_FIELDS = {
    "preferred",
    "rank",
    "score",
    "winner",
    "blocking",
    "fallback_on_failure",
}


def _keys(value):
    if isinstance(value, dict):
        yield from value
        for item in value.values():
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_required_card_ids_are_a_subset_and_ids_are_unique():
    ids = [card["id"] for card in all_cards()]
    assert REQUIRED_IDS <= set(ids)
    assert len(ids) == len(set(ids))


def test_cards_are_complete_cited_unranked_guidance():
    for card in all_cards():
        assert REQUIRED_FIELDS <= set(card)
        assert card["enforcement"] == "guidance"
        assert card["candidate_approaches"]
        assert card["citations"]
        assert all(item.get("citation") and item.get("supports")
                   for item in card["citations"])
        assert not (FORBIDDEN_FIELDS & set(_keys(card)))


def test_every_related_rule_id_resolves():
    for card in all_cards():
        for rule_id in card["related_rule_ids"]:
            assert get_rule(rule_id), f"{card['id']} references {rule_id}"


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("Harmony PCA scVI batch correction", "latent-representation-and-batch-integration"),
        ("pseudobulk marker sample inference", "differential-expression-method-choice"),
        ("PAGA DPT RNA velocity direction", "trajectory-connectivity-and-velocity"),
        ("ORA module TF activity background", "gene-set-and-regulatory-activity-method-choice"),
        ("composition differential abundance", "composition-abundance-and-expression-choice"),
    ],
)
def test_card_search_finds_the_named_decision_surface(query, expected_id):
    assert get_card(expected_id) in search_cards(query)


def test_card_search_does_not_rank_on_verbose_rendering_fields(monkeypatch):
    monkeypatch.setattr(
        cards_mod,
        "all_cards",
        lambda: [
            {
                "id": "unrelated-card",
                "title": "Unrelated card",
                "aliases": [],
                "applies_when": "unrelated question",
                "facts_to_consider": ["body-only-retrieval-token"],
            }
        ],
    )

    assert search_cards("body-only-retrieval-token") == []


def test_card_search_ignores_generic_method_prose():
    assert search_cards(
        "and the for with between from data method choice"
    ) == []


def test_rendering_respects_card_and_character_caps_without_partial_cards():
    text = guidance_for_query(
        "Harmony PCA scVI batch correction reference mapping marker annotation"
    )
    assert text.count("[decision card:") == MAX_GUIDANCE_CARDS
    assert len(text) <= MAX_GUIDANCE_CHARS
    assert "Unranked approaches:" in text
    assert "...[truncated]" not in text
    for rendered_card in text.split("\n\n"):
        assert rendered_card.startswith("[decision card:")
        assert "Selection considerations:" in rendered_card
        assert "Citations:" in rendered_card


def test_planning_guidance_query_contains_only_explicit_decision_context():
    query = build_planning_guidance_query(
        overview="choose annotation strategy",
        tasks="compare reference mapping with marker evidence",
        stage="full",
        modality="single-cell and bulk",
    )

    assert "choose annotation strategy" in query
    assert "compare reference mapping with marker evidence" in query
    assert "full" in query
    assert "single-cell and bulk" in query
    assert "Data Context" not in query


def test_no_match_returns_no_optional_guidance():
    assert guidance_for_query("zzz") == ""


def test_formatter_excludes_expected_outputs_and_python_stack_detail():
    text = format_card(get_card("differential-expression-method-choice"))
    assert "Expected outputs:" not in text
    assert "python_stack" not in text


def _legacy_search(query: str, limit: int = 5) -> list[dict]:
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(query).lower())
        if len(token) > 2
    }
    if not tokens:
        return []
    scored = []
    for rule in all_rules():
        haystack = " ".join(
            str(rule.get(field, ""))
            for field in ("id", "title", "rule", "why", "applies_when", "python")
        ).lower()
        hay_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", haystack)
            if len(token) > 2
        }
        overlap = len(tokens & hay_tokens)
        if overlap:
            scored.append((overlap, rule))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("id", "")))
    return [rule for _, rule in scored[:limit]]


@pytest.mark.parametrize(
    "query",
    [
        "multiple testing family q-value",
        "leiden clustering resolution choice",
        "pseudobulk biological replicate",
        "trajectory direction velocity root",
        "gene set background enrichment",
        "zzz",
    ],
)
def test_shared_search_helper_preserves_legacy_rule_order(query):
    assert [rule["id"] for rule in search_rules(query)] == [
        rule["id"] for rule in _legacy_search(query)
    ]


def _rendered_corpus() -> str:
    return json.dumps(all_cards())


@pytest.mark.parametrize(
    "expected_substring",
    [
        "scvi.model.SCANVI",
        "celltypist",
        "CytoTRACEKernel",
        "decoupler.get_pseudobulk",
    ],
)
def test_python_stack_names_installed_tools(expected_substring):
    assert expected_substring in _rendered_corpus()


def test_python_stack_drops_vague_reference_mapping_entry():
    assert "ingest-compatible reference mapping" not in _rendered_corpus()


def test_malformed_corpus_fails_open(monkeypatch, tmp_path):
    malformed = tmp_path / "cards.yaml"
    malformed.write_text("cards:\n  - id: incomplete\n", encoding="utf-8")
    cards_mod._load.cache_clear()
    monkeypatch.setattr(cards_mod, "_CARDS_PATH", malformed)
    try:
        assert cards_mod.all_cards() == []
        assert cards_mod.guidance_for_query("differential expression") == ""
    finally:
        cards_mod._load.cache_clear()
