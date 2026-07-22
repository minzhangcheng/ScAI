"""Focused-query helpers must use supplied evidence, not built-in biology terms."""
from backend.agents.literature_agent import (
    _gene_case_variants,
    _sanitize_synonyms,
    build_focused_query,
    plausibility_terms_for_tissue,
)


def test_no_static_tissue_plausibility_vocabulary():
    assert plausibility_terms_for_tissue("organ A") == []
    assert plausibility_terms_for_tissue("organ B") == []


def test_contamination_tier2_uses_only_observed_markers_and_tissue():
    item = {"candidate_celltype": "candidate A", "top_markers": ["GeneA", "GeneB"]}
    query = build_focused_query(
        item,
        organism="species A",
        tissue="organ A",
        tier=2,
        bucket="suspected_contamination",
    )
    assert '"GeneA"[tiab]' in query
    assert "organ A" in query
    assert "candidate A" not in query


def test_explicit_plausibility_terms_are_respected():
    item = {"candidate_celltype": None, "top_markers": ["GeneA"]}
    query = build_focused_query(
        item,
        organism="species A",
        tissue="organ A",
        tier=2,
        bucket="suspected_contamination",
        plausibility_terms=['"supplied anatomical context"[tiab]'],
    )
    assert '"supplied anatomical context"[tiab]' in query


def test_gene_case_variants_dual():
    assert _gene_case_variants("GeneA") == ["GeneA", "GENEA"]
    assert _gene_case_variants("GENEA") == ["GENEA"]
    assert _gene_case_variants("") == []


def test_sanitize_synonyms():
    raw = ["Alias A", "Alias A", "  Alias B ", "", "x" * 80, 'bad"quote']
    out = _sanitize_synonyms(raw)
    assert out.count("Alias A") == 1
    assert "Alias B" in out
    assert all(len(s) <= 60 for s in out)
    assert all('"' not in s for s in out)
    assert "" not in out


def test_rare_tier1_includes_supplied_tissue_candidate_and_markers():
    item = {
        "candidate_celltype": "candidate cell type A",
        "synonyms": ["Alias A"],
        "top_markers": ["GeneA", "GeneB"],
    }
    query = build_focused_query(
        item, organism="species A", tissue="organ A", tier=1, bucket="rare_unresolved"
    )
    assert "organ A" in query
    assert "candidate cell type A" in query
    assert "Alias A" in query
    assert '"GeneA"[tiab]' in query and '"GENEA"[tiab]' in query


def test_rare_tier2_drops_tissue():
    item = {
        "candidate_celltype": "candidate cell type A",
        "synonyms": [],
        "top_markers": ["GeneA"],
    }
    query = build_focused_query(
        item, organism="species A", tissue="organ A", tier=2, bucket="rare_unresolved"
    )
    assert "candidate cell type A" in query
    assert "organ A" not in query


def test_contamination_uses_markers_not_candidate():
    item = {"candidate_celltype": "candidate A", "top_markers": ["GeneA", "GeneB"]}
    discovery = build_focused_query(
        item, organism="species A", tissue="organ A", tier=1, bucket="suspected_contamination"
    )
    assert "candidate A" not in discovery
    assert '"GeneA"[tiab]' in discovery
    assert "organ A" not in discovery


def test_empty_terms_returns_empty_query():
    item = {"candidate_celltype": None, "synonyms": [], "top_markers": []}
    assert build_focused_query(item, "species A", "organ A", 1, "rare_unresolved") == ""


def test_contamination_without_markers_returns_empty_query():
    item = {"candidate_celltype": "candidate A", "top_markers": []}
    assert build_focused_query(item, "species A", "organ A", 1, "suspected_contamination") == ""
    assert build_focused_query(item, "species A", "organ A", 2, "suspected_contamination") == ""
