"""A scored gene panel must say where it came from, and how big it really is.

``register_score`` previously accepted free-text provenance ("curated gene set")
and validated nothing, so a five-gene panel assembled from memory was recorded
identically to a versioned database collection. Downstream -- report, supervisor,
a later reader -- could not tell them apart or re-fetch either one.

These tests pin the controlled vocabulary, the power assessment, and the fact
that the module stays domain-neutral (ScAI is a general-purpose platform; see
CLAUDE.md).
"""
import pytest

from backend.services.gene_set_provenance import (
    KIND_BESPOKE,
    KIND_DATABASE,
    KIND_DERIVED,
    KIND_LITERATURE,
    MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM,
    assess_panel_power,
    parse_provenance,
    power_caveat,
)


# --------------------------------------------------------------------- accept

def test_database_provenance_parses():
    prov, err = parse_provenance("database:msigdb:2025.1.Mm:SOME_SET_ID")
    assert err is None
    assert prov.kind == KIND_DATABASE
    assert prov.resource == "msigdb"
    assert prov.version == "2025.1.Mm"
    assert prov.set_id == "SOME_SET_ID"
    assert prov.is_reproducible
    assert not prov.is_bespoke


def test_database_set_id_may_contain_colons():
    prov, err = parse_provenance("database:reactome:v90:R-MMU-12345:sub")
    assert err is None
    assert prov.set_id == "R-MMU-12345:sub"


def test_derived_provenance_parses():
    prov, err = parse_provenance("derived:deg:groupA_vs_groupB")
    assert err is None
    assert prov.kind == KIND_DERIVED
    assert prov.resource == "deg"
    assert prov.detail == "groupA_vs_groupB"
    assert prov.is_reproducible


def test_literature_provenance_parses():
    prov, err = parse_provenance("literature:PMID:12345678")
    assert err is None
    assert prov.kind == KIND_LITERATURE
    assert prov.set_id == "12345678"
    # A publication is citable but not machine re-fetchable as a gene list.
    assert not prov.is_reproducible


def test_bespoke_requires_justification_but_is_allowed():
    prov, err = parse_provenance(
        "bespoke:no versioned collection covers this combination of components"
    )
    assert err is None
    assert prov.kind == KIND_BESPOKE
    assert prov.is_bespoke
    assert not prov.is_reproducible


# --------------------------------------------------------------------- reject

def test_empty_source_rejected():
    prov, err = parse_provenance("")
    assert prov is None and "required" in err


def test_none_source_rejected():
    prov, err = parse_provenance(None)
    assert prov is None and err


def test_freetext_source_rejected():
    """This is the exact string the old schema offered as an example."""
    prov, err = parse_provenance("curated gene set")
    assert prov is None
    assert "must start with one of" in err


def test_unpinned_database_version_rejected():
    prov, err = parse_provenance("database:msigdb:latest:SOME_SET")
    assert prov is None
    assert "does not pin anything" in err


def test_incomplete_database_provenance_rejected():
    prov, err = parse_provenance("database:msigdb")
    assert prov is None
    assert "database:<resource>:<version>:<set_id>" in err


def test_short_bespoke_justification_rejected():
    prov, err = parse_provenance("bespoke:because")
    assert prov is None
    assert "justification" in err


def test_bad_literature_scheme_rejected():
    prov, err = parse_provenance("literature:ISBN:123")
    assert prov is None
    assert "PMID" in err


def test_error_messages_are_actionable():
    """Every rejection must show the model the accepted forms."""
    for bad in ["", "curated gene set", "database:msigdb:latest:X", "bespoke:x"]:
        _, err = parse_provenance(bad)
        assert "database:<resource>:<version>:<set_id>" in err


# ---------------------------------------------------------------------- power

def test_small_panel_is_underpowered():
    """The real failure: a 5-gene panel used to justify a negative conclusion."""
    power = assess_panel_power(5, 5)
    assert power["underpowered"]
    assert not power["supports_negative_claim"]
    assert "not evidence of absence" in power_caveat(power)


def test_adequate_panel_is_not_underpowered():
    power = assess_panel_power(200, 180)
    assert not power["underpowered"]
    assert power["supports_negative_claim"]
    assert power_caveat(power) == ""
    assert power["match_rate"] == 0.9


def test_matched_count_governs_power_not_nominal_size():
    """A large panel whose genes are absent from the matrix is still underpowered."""
    power = assess_panel_power(200, 4)
    assert power["effective_n"] == 4
    assert power["underpowered"]
    assert power["basis"] == "matched"


def test_nominal_size_used_when_match_count_unknown():
    power = assess_panel_power(50, None)
    assert power["effective_n"] == 50
    assert power["basis"] == "nominal"
    assert power["match_rate"] is None


def test_threshold_matches_mainstream_tooling_default():
    """clusterProfiler and fgsea both default minGSSize to 10."""
    assert MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM == 10


def test_empty_panel_supports_no_claim():
    power = assess_panel_power(0, None)
    assert not power["supports_negative_claim"]
    assert not power["underpowered"]  # nothing to be underpowered about


# ------------------------------------------------------------ domain neutrality

def test_module_names_no_specific_biology():
    """ScAI is general-purpose: no tissue/disease/organism/pathway names here."""
    import inspect

    from backend.services import gene_set_provenance

    import re

    src = inspect.getsource(gene_set_provenance).lower()
    # Substring terms: these have no innocent occurrence.
    for term in ("ferroptos", "apoptos", "cuproptos", "necroptos", "nrf2",
                 "hepatocyte", "msigdb", "collectri"):
        assert term not in src, f"domain-specific term {term!r} leaked into the validator"
    # Word terms: matched on word boundaries so "human-readable" does not trip
    # the organism check.
    for word in ("liver", "mouse", "cancer", "tumor", "tumour"):
        assert not re.search(rf"\b{word}\b", src), (
            f"domain-specific term {word!r} leaked into the validator"
        )
