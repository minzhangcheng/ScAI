"""The methodological rule corpus must stay loadable, attributed and neutral.

The corpus is distilled from bioSkills (MIT) and translated for this stack; see
THIRD_PARTY_NOTICES.md. These tests pin the parts that would silently rot: the
attribution, the gate/guidance split that other code keys off, and the
domain-neutrality that ScAI's general-purpose remit requires.
"""
import re

import pytest

from backend.services.method_rules import (
    ENFORCEMENT_ADVISORY,
    ENFORCEMENT_GATE,
    ENFORCEMENT_GUIDANCE,
    ENFORCEMENT_VALUES,
    all_rules,
    attribution,
    corpus_meta,
    format_rule,
    gates,
    get_rule,
    rule_text,
    search_rules,
)

_REQUIRED_FIELDS = ("id", "title", "enforcement", "applies_when", "rule",
                    "why", "python", "citations", "source_skill")


def test_corpus_loads_and_is_not_empty():
    rules = all_rules()
    assert len(rules) >= 10, "corpus looks truncated"


def test_every_rule_has_the_required_fields():
    for rule in all_rules():
        missing = [f for f in _REQUIRED_FIELDS if not rule.get(f)]
        assert not missing, f"{rule.get('id')} is missing {missing}"


def test_rule_ids_are_unique():
    ids = [r["id"] for r in all_rules()]
    assert len(ids) == len(set(ids))


def test_enforcement_values_are_known():
    for rule in all_rules():
        assert rule["enforcement"] in ENFORCEMENT_VALUES


def test_there_are_both_gates_and_guidance():
    """A corpus of pure guidance would enforce nothing; pure gates, unusable."""
    kinds = {r["enforcement"] for r in all_rules()}
    assert ENFORCEMENT_GATE in kinds
    assert ENFORCEMENT_GUIDANCE in kinds
    assert kinds <= set(ENFORCEMENT_VALUES)


def test_gates_subset_matches_enforcement_field():
    assert {r["id"] for r in gates()} == {
        r["id"] for r in all_rules() if r["enforcement"] == ENFORCEMENT_GATE
    }


def test_every_rule_cites_something():
    for rule in all_rules():
        citations = rule.get("citations") or []
        assert citations, f"{rule['id']} has no citation"
        assert all(str(c).strip() for c in citations)


# ------------------------------------------------------------- attribution

def test_attribution_names_source_licence_and_commit():
    """bioSkills is MIT; this project is Apache-2.0. Attribution is required."""
    text = attribution()
    assert "bioSkills" in text
    assert "MIT" in text
    assert "GPTomics" in text
    meta = corpus_meta()
    assert re.fullmatch(r"[0-9a-f]{40}", str(meta["source_commit"])), (
        "the source commit must be a full hash so the derivation is checkable"
    )


def test_meta_records_that_rules_were_translated():
    """bioSkills targets R; the python field is this project's own work."""
    assert corpus_meta().get("translated_for")


# --------------------------------------------------------------- retrieval

def test_get_rule_by_id():
    """minimum-gene-set-size is guidance: in code it produces a caveat on the
    score, not a refusal. The refusable consequence -- a negative conclusion
    from such a panel -- is a separate rule."""
    rule = get_rule("minimum-gene-set-size")
    assert rule and rule["enforcement"] == ENFORCEMENT_GUIDANCE


def test_get_rule_unknown_id_returns_none():
    assert get_rule("no-such-rule") is None


def test_rule_text_is_compact_and_actionable():
    text = rule_text("minimum-gene-set-size")
    assert text.startswith("[method rule: minimum-gene-set-size]")
    assert "Why:" in text
    assert "In this stack:" in text


def test_rule_text_unknown_id_is_empty_not_an_error():
    assert rule_text("no-such-rule") == ""


def test_format_rule_can_omit_the_python_translation():
    rule = get_rule("minimum-gene-set-size")
    assert "In this stack:" not in format_rule(rule, include_python=False)


def test_search_finds_the_family_rule():
    hits = search_rules("multiple testing family q-value")
    assert any(r["id"] == "testing-family-must-be-declared" for r in hits)


def test_search_finds_the_resolution_rule():
    hits = search_rules("leiden clustering resolution choice")
    assert any(r["id"] == "cluster-resolution-must-be-swept" for r in hits)


def test_search_with_no_overlap_returns_empty():
    assert search_rules("zzz") == []


# ------------------------------------------------ the observed failure modes

@pytest.mark.parametrize("rule_id", [
    "testing-family-must-be-declared",          # q-value depended on the call
    "minimum-gene-set-size",                    # 5-gene panel
    "no-negative-claim-from-an-underpowered-panel",
    "biological-replicate-is-the-unit",         # cells treated as replicates
    "one-sample-per-condition-is-unidentifiable",
    "cluster-resolution-must-be-swept",         # resolution decided the analysis
    "gene-permutation-is-anticonservative",     # prerank at n=3 vs 3
])
def test_rule_exists_for_each_observed_failure_mode(rule_id):
    """Each of these corresponds to a defect seen in production runs."""
    assert get_rule(rule_id) is not None


# ------------------------------------------------------- domain neutrality

def test_rules_name_no_specific_biology():
    """Rules are selected by capability surface, never by past analyses.

    Distilling rules from the runs that motivated them would overfit a
    general-purpose platform to whatever biology those runs involved.
    """
    banned_substrings = ("ferroptos", "apoptos", "cuproptos", "necroptos",
                         "pyroptos", "nrf2", "hepatocyte", "kupffer")
    banned_words = ("liver", "mouse", "human", "cancer", "tumor", "tumour",
                    "copper", "iron", "zinc")
    for rule in all_rules():
        blob = " ".join(str(v) for v in rule.values()).lower()
        for term in banned_substrings:
            assert term not in blob, f"{rule['id']} mentions {term!r}"
        for word in banned_words:
            assert not re.search(rf"\b{word}\b", blob), (
                f"{rule['id']} mentions {word!r}"
            )


def test_rules_do_not_prescribe_r_only_solutions_without_saying_so():
    """bioSkills is R-centric; a rule whose only fix is R must admit it."""
    for rule in all_rules():
        python = str(rule.get("python", ""))
        assert python.strip(), f"{rule['id']} has no Python translation"
