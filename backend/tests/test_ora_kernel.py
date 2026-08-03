import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import pytest
from backend.services.ora_kernel import compute_ora_priors


def _toy_adata(tmp_path):
    genes = [f"G{i}" for i in range(47)] + ["Irf8", "Siglech", "Tcf4", "Alb"]
    X = np.random.RandomState(0).poisson(1, size=(40, len(genes))).astype(float)
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    adata.obs["leiden"] = (["0"] * 20 + ["1"] * 20)
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    names0 = ["Irf8", "Siglech", "Tcf4"] + [f"G{i}" for i in range(47)]
    names1 = [f"G{i}" for i in range(47)] + ["Alb", "Irf8", "Siglech"]
    n_rank = 50
    dtype = [("0", "U20"), ("1", "U20")]
    names_arr = np.array(list(zip(names0, names1)), dtype=dtype)
    adata.uns["rank_genes_groups"] = {
        "params": {"groupby": "leiden", "method": "t-test"},
        "names": names_arr,
        "scores": np.zeros(n_rank, dtype=[("0", float), ("1", float)]),
        "logfoldchanges": np.zeros(n_rank, dtype=[("0", float), ("1", float)]),
        "pvals": np.ones(n_rank, dtype=[("0", float), ("1", float)]),
        "pvals_adj": np.ones(n_rank, dtype=[("0", float), ("1", float)]),
    }
    db = pd.DataFrame([
        {"cell_type": "pDC", "gene_symbol": "Irf8", "source": "PanglaoDB"},
        {"cell_type": "pDC", "gene_symbol": "Siglech", "source": "PanglaoDB"},
        {"cell_type": "pDC", "gene_symbol": "Tcf4", "source": "PanglaoDB"},
        {"cell_type": "hepatocyte", "gene_symbol": "Alb", "source": "PanglaoDB"},
    ])
    db_path = tmp_path / "marker_priors_input.tsv"
    db.to_csv(db_path, sep="\t", index=False)
    return adata, db_path


def test_pdc_cluster_hits_pdc_with_overlap_and_padj(tmp_path):
    adata, db_path = _toy_adata(tmp_path)
    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50, min_overlap=2, top_k=3)
    assert "0" in res
    top = res["0"][0]
    assert top["cell_type"] == "pDC"
    assert set(top["overlap"]) >= {"IRF8", "SIGLECH"}
    assert "p_adj" in top and top["p_adj"] <= 1.0
    assert top["universe_source"] == "adata.var_names"
    assert top["marker_source"] == "rank_genes_groups"


def test_min_overlap_filters_single_gene_hits(tmp_path):
    adata, db_path = _toy_adata(tmp_path)
    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50, min_overlap=2, top_k=3)
    all_celltypes = {h["cell_type"] for hits in res.values() for h in hits}
    assert "hepatocyte" not in all_celltypes
    for hits in res.values():
        for h in hits:
            assert len(h["overlap"]) >= 2


def test_explicit_markers_fallback_when_rank_key_is_missing(tmp_path):
    adata, db_path = _toy_adata(tmp_path)
    del adata.uns["rank_genes_groups"]

    res = compute_ora_priors(
        adata, str(db_path), cluster_key="leiden",
        rank_key="rank_genes_groups", top_n=50, min_overlap=2, top_k=3,
        cluster_markers={
            "0": ["Irf8", "Siglech", "Tcf4"],
            "1": ["Alb"],
            "not_observed": ["Irf8", "Siglech"],
        },
    )

    assert res["0"][0]["cell_type"] == "pDC"
    assert res["0"][0]["marker_source"] == "provided_cluster_markers"
    assert "not_observed" not in res


def test_missing_rank_key_without_explicit_markers_is_not_silenced(tmp_path):
    adata, db_path = _toy_adata(tmp_path)
    del adata.uns["rank_genes_groups"]

    with pytest.raises(KeyError):
        compute_ora_priors(
            adata, str(db_path), cluster_key="leiden",
            rank_key="rank_genes_groups", cluster_markers={})


def _toy_db_with_tissue(tmp_path):
    db = pd.DataFrame([
        {"cell_type": "pDC", "gene_symbol": "Irf8", "source": "CellMarker",
         "tissue": "Immune system"},
        {"cell_type": "pDC", "gene_symbol": "Siglech", "source": "CellMarker",
         "tissue": "Immune system"},
        {"cell_type": "pDC", "gene_symbol": "Tcf4", "source": "CellMarker",
         "tissue": "Blood"},
        {"cell_type": "hepatocyte", "gene_symbol": "Alb", "source": "PanglaoDB",
         "tissue": "Liver"},
    ])
    path = tmp_path / "marker_priors_input_tissue.tsv"
    db.to_csv(path, sep="\t", index=False)
    return path


def test_priors_flag_whether_the_cell_type_matches_the_analysed_tissue(tmp_path):
    """Reported, never filtered: a free-text miss must not drop a correct hit."""
    adata, _ = _toy_adata(tmp_path)
    db_path = _toy_db_with_tissue(tmp_path)

    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50,
                             min_overlap=2, top_k=3, tissue="Immune system")
    top = res["0"][0]
    assert top["cell_type"] == "pDC"
    assert top["tissue_match"] is True
    assert top["n_tissues"] == 2          # immune system + blood

    # A mismatching tissue flags but does not remove the hit.
    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50,
                             min_overlap=2, top_k=3, tissue="liver")
    top = res["0"][0]
    assert top["cell_type"] == "pDC"
    assert top["tissue_match"] is False


def test_no_tissue_argument_reports_breadth_not_a_match(tmp_path):
    adata, _ = _toy_adata(tmp_path)
    db_path = _toy_db_with_tissue(tmp_path)
    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50,
                             min_overlap=2, top_k=3)
    top = res["0"][0]
    assert top["tissue_match"] is None
    assert top["n_tissues"] == 2


def test_tissue_less_table_still_works(tmp_path):
    """A revision-2 table has no tissue column; priors must degrade, not crash."""
    adata, db_path = _toy_adata(tmp_path)
    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50,
                             min_overlap=2, top_k=3, tissue="liver")
    top = res["0"][0]
    assert top["tissue_match"] is None
    assert top["n_tissues"] == 0


#: Genes the fixture below puts at the head of cluster 0's ranking.
_TOP = ["Irf8", "Siglech", "Tcf4", "Alb"]


def _discriminating_adata(universe_size=500, depth=50):
    """A universe large enough for the hypergeometric to separate hits.

    ``_toy_adata`` draws 50 of 51 genes, which degenerates the distribution: every
    p-value collapses towards the same value. That is fine for "was pDC found at
    all" but useless for comparing two hits, which is what ranking is about.

    Cluster 0's top-``depth`` is ``_TOP`` plus filler; genes named ``F1xx`` are
    deliberately outside it, so a db entry can be given a large gene set without
    growing its overlap.
    """
    filler = [f"F{i}" for i in range(universe_size - len(_TOP))]
    genes = _TOP + filler
    adata = ad.AnnData(X=np.zeros((4, len(genes)), dtype=np.float32),
                       var=pd.DataFrame(index=genes))
    adata.obs["leiden"] = pd.Categorical(["0", "0", "1", "1"])
    names0 = _TOP + filler[:depth - len(_TOP)]
    names1 = filler[depth:depth * 2]
    dtype = [("0", "U20"), ("1", "U20")]
    names_arr = np.array(list(zip(names0, names1)), dtype=dtype)
    zeros = np.zeros(depth, dtype=[("0", float), ("1", float)])
    adata.uns["rank_genes_groups"] = {
        "params": {"groupby": "leiden", "method": "t-test"},
        "names": names_arr, "scores": zeros, "logfoldchanges": zeros,
        "pvals": np.ones(depth, dtype=[("0", float), ("1", float)]),
        "pvals_adj": np.ones(depth, dtype=[("0", float), ("1", float)]),
    }
    return adata


def _write_db(tmp_path, name, entries):
    """entries: {cell_type: (tissue, [genes])}"""
    rows = [{"cell_type": ct, "gene_symbol": g, "source": "CellMarker", "tissue": tis}
            for ct, (tis, genes) in entries.items() for g in genes]
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_synonymous_hits_collapse_so_they_cannot_fill_top_k(tmp_path):
    """The bug this exists for: three intestinal synonyms buried Hepatocytes.

    All three fired on the identical two-gene overlap with an identical p-value,
    took all of top_k=3, and the correct coarse label never reached the prompt.
    """
    adata = _discriminating_adata()
    db_path = _write_db(tmp_path, "collapse.tsv", {
        "Brush cell": ("Intestine", ["Irf8", "Siglech"]),
        "Goblet cell": ("Intestine", ["Irf8", "Siglech"]),
        "Tuft cell": ("Intestine", ["Irf8", "Siglech"]),
        "hepatocyte": ("Liver", ["Tcf4", "Alb"]),
    })

    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50,
                             min_overlap=2, top_k=3, tissue="liver")
    hits = res["0"]
    assert len(hits) <= 3
    synonym_hit = next(h for h in hits if h["cell_type"] in
                       ("Brush cell", "Goblet cell", "Tuft cell"))
    assert len(synonym_hit["also_matched"]) == 2, "the three must share one slot"
    assert "hepatocyte" in [h["cell_type"] for h in hits], (
        "collapsing must leave room for the distinct hit that used to be pushed out"
    )


def test_tissue_match_only_breaks_ties_never_overrides_p_value(tmp_path):
    """A stronger hit from another organ must still outrank a weaker local one."""
    adata = _discriminating_adata()
    db_path = _write_db(tmp_path, "tie.tsv", {
        # 3 of 3 genes overlap -> strong, but the wrong organ.
        "Gut cell": ("Intestine", ["Irf8", "Siglech", "Tcf4"]),
        # 2 overlapping genes diluted by 5 that cluster 0 never ranks -> weaker.
        "Liver cell": ("Liver", ["Irf8", "Siglech",
                                 "F100", "F101", "F102", "F103", "F104"]),
    })

    hits = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                              rank_key="rank_genes_groups", top_n=50,
                              min_overlap=2, top_k=3, tissue="liver")["0"]
    gut = next(h for h in hits if h["cell_type"] == "Gut cell")
    liver = next(h for h in hits if h["cell_type"] == "Liver cell")
    assert gut["tissue_match"] is False and liver["tissue_match"] is True
    assert gut["p_adj"] < liver["p_adj"], "fixture must actually favour the gut cell"
    assert hits[0]["cell_type"] == "Gut cell", (
        "tissue preference must not reorder a genuinely stronger off-tissue hit"
    )


def test_tissue_match_does_break_an_actual_tie(tmp_path):
    """Equal p-values are where the preference is allowed to act."""
    adata = _discriminating_adata()
    db_path = _write_db(tmp_path, "equal.tsv", {
        "Gut cell": ("Intestine", ["Irf8", "Siglech", "Tcf4"]),
        "Liver cell": ("Liver", ["Siglech", "Tcf4", "Alb"]),
    })

    hits = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                              rank_key="rank_genes_groups", top_n=50,
                              min_overlap=2, top_k=3, tissue="liver")["0"]
    assert hits[0]["p_adj"] == hits[1]["p_adj"], "fixture must produce a real tie"
    assert hits[0]["cell_type"] == "Liver cell"


def test_kernel_splits_the_multi_valued_tissue_field(tmp_path):
    """marker_priors ";"-joins organs; the kernel must match against each."""
    adata = _discriminating_adata()
    db_path = _write_db(tmp_path, "multi.tsv", {
        "Macrophage": ("Spleen;Liver;Lung", ["Irf8", "Siglech"]),
    })

    hit = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50,
                             min_overlap=2, top_k=3, tissue="liver")["0"][0]
    assert hit["tissue_match"] is True, "a match on any listed organ counts"
    assert hit["n_tissues"] == 3

    miss = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                              rank_key="rank_genes_groups", top_n=50,
                              min_overlap=2, top_k=3, tissue="kidney")["0"][0]
    assert miss["tissue_match"] is False
    assert miss["n_tissues"] == 3
