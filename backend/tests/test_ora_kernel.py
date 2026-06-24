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


def test_min_overlap_filters_single_gene_hits(tmp_path):
    adata, db_path = _toy_adata(tmp_path)
    res = compute_ora_priors(adata, str(db_path), cluster_key="leiden",
                             rank_key="rank_genes_groups", top_n=50, min_overlap=2, top_k=3)
    all_celltypes = {h["cell_type"] for hits in res.values() for h in hits}
    assert "hepatocyte" not in all_celltypes
    for hits in res.values():
        for h in hits:
            assert len(h["overlap"]) >= 2
