"""focused query 纯函数: 双拼、synonyms 卫生、两级两桶、空 terms 防御、tissue plausibility 词表。"""
from backend.agents.literature_agent import (
    _gene_case_variants, _sanitize_synonyms, build_focused_query,
    plausibility_terms_for_tissue,
)


def test_liver_plausibility_terms_include_portal_context():
    terms = plausibility_terms_for_tissue("liver")
    assert '"portal tract"[tiab]' in terms
    assert 'Schwann[tiab]' in terms


def test_non_liver_plausibility_terms_exclude_portal_context():
    terms = plausibility_terms_for_tissue("kidney")
    joined = " ".join(terms).lower()
    assert "portal tract" not in joined
    assert "portal triad" not in joined
    assert 'Schwann[tiab]' in terms


def test_contamination_tier2_uses_tissue_specific_terms():
    item = {"candidate_celltype": "neuronal", "top_markers": ["Nrxn1", "Reln"]}
    liver_query = build_focused_query(item, organism="mouse", tissue="liver", tier=2, bucket="suspected_contamination")
    kidney_query = build_focused_query(item, organism="mouse", tissue="kidney", tier=2, bucket="suspected_contamination")
    assert '"portal tract"[tiab]' in liver_query
    assert '"portal tract"[tiab]' not in kidney_query
    assert "kidney" in kidney_query.lower()
    assert 'Schwann[tiab]' in kidney_query


def test_explicit_plausibility_terms_override_tissue_defaults():
    item = {"candidate_celltype": None, "top_markers": ["Nrxn1"]}
    query = build_focused_query(
        item, organism="mouse", tissue="liver", tier=2, bucket="suspected_contamination",
        plausibility_terms=['"renal capsule"[tiab]'],
    )
    assert '"renal capsule"[tiab]' in query
    assert '"portal tract"[tiab]' not in query


def test_gene_case_variants_dual():
    assert _gene_case_variants("Pkhd1") == ["Pkhd1", "PKHD1"]
    assert _gene_case_variants("PKHD1") == ["PKHD1"]   # 已全大写不重复
    assert _gene_case_variants("") == []


def test_sanitize_synonyms():
    raw = ["pDC", "pDC", "  bile duct cell ", "", "x" * 80, 'bad"quote']
    out = _sanitize_synonyms(raw)
    assert out.count("pDC") == 1
    assert "bile duct cell" in out
    assert all(len(s) <= 60 for s in out)
    assert all('"' not in s for s in out)
    assert "" not in out


def test_rare_tier1_includes_tissue_and_candidate():
    item = {"candidate_celltype": "plasmacytoid dendritic cell",
            "synonyms": ["pDC"], "top_markers": ["Siglech", "Tcf4"]}
    q = build_focused_query(item, organism="mouse", tissue="liver", tier=1, bucket="rare_unresolved")
    assert "liver" in q.lower() and "plasmacytoid dendritic cell" in q and "pDC" in q
    assert '"Siglech"[tiab]' in q and '"SIGLECH"[tiab]' in q
    assert "mouse" in q.lower()


def test_rare_tier2_drops_tissue():
    item = {"candidate_celltype": "cholangiocyte", "synonyms": [], "top_markers": ["Pkhd1"]}
    q = build_focused_query(item, organism="mouse", tissue="liver", tier=2, bucket="rare_unresolved")
    assert "cholangiocyte" in q and "liver" not in q.lower()


def test_contamination_uses_markers_not_candidate():
    item = {"candidate_celltype": "neuronal", "top_markers": ["Nrxn1", "Reln"]}
    discovery = build_focused_query(item, organism="mouse", tissue="liver", tier=1, bucket="suspected_contamination")
    assert "neuronal" not in discovery and '"Nrxn1"[tiab]' in discovery and "liver" not in discovery.lower()
    plausibility = build_focused_query(item, organism="mouse", tissue="liver", tier=2, bucket="suspected_contamination")
    assert "liver" in plausibility.lower() and '"Nrxn1"[tiab]' in plausibility
    assert "Schwann" in plausibility or "nerve fiber" in plausibility   # tier2 含组织合理性词


def test_empty_terms_returns_empty_query():
    item = {"candidate_celltype": None, "synonyms": [], "top_markers": []}
    assert build_focused_query(item, "mouse", "liver", tier=1, bucket="rare_unresolved") == ""


def test_contamination_without_markers_returns_empty_query():
    item = {"candidate_celltype": "neuronal", "top_markers": []}
    assert build_focused_query(item, "mouse", "liver", tier=1, bucket="suspected_contamination") == ""
    assert build_focused_query(item, "mouse", "liver", tier=2, bucket="suspected_contamination") == ""
