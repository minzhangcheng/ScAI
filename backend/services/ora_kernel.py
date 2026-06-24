"""自包含 ORA 计算（hypergeom + BH-FDR）。

禁止 import backend.*：本文件源码经 inspect.getsource 注入 sandbox kernel 执行。
"""
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def compute_ora_priors(adata, db_path, cluster_key, rank_key,
                       top_n=50, min_overlap=2, top_k=3):
    db = pd.read_csv(db_path, sep="\t", comment="#")
    db["gene_symbol"] = db["gene_symbol"].astype(str).str.upper()

    try:
        universe = set(pd.Index(adata.var_names).str.upper())
        universe_source = "adata.var_names"
    except Exception:
        universe = None
        universe_source = "marker_union"

    if universe is not None:
        db = db[db["gene_symbol"].isin(universe)]
        N = int(adata.n_vars)

    cluster_top = {}
    for cl in adata.obs[cluster_key].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=str(cl), key=rank_key)
        cluster_top[str(cl)] = set(df.head(top_n)["names"].astype(str).str.upper())

    if universe is None:
        universe = set().union(*cluster_top.values()) if cluster_top else set()
        N = len(universe)
        db = db[db["gene_symbol"].isin(universe)]

    celltype_genes = (db.groupby("cell_type")["gene_symbol"]
                        .apply(lambda s: set(s)).to_dict())
    celltype_source = (db.groupby("cell_type")["source"]
                         .apply(lambda s: "+".join(dict.fromkeys(s))).to_dict()
                       if "source" in db.columns else {})

    records, pvals = [], []
    for cl, markers in cluster_top.items():
        n_drawn = len(markers)
        for ct, genes in celltype_genes.items():
            overlap = sorted(markers & genes)
            if len(overlap) < min_overlap:
                continue
            K = len(genes)
            x = len(overlap)
            p = float(hypergeom.sf(x - 1, N, K, n_drawn))
            records.append({"cluster": cl, "cell_type": ct, "overlap": overlap,
                            "p_value": p, "source": celltype_source.get(ct, "")})
            pvals.append(p)

    if not records:
        return {}

    padj = multipletests(pvals, method="fdr_bh")[1]
    by_cluster = {}
    for rec, pa in zip(records, padj):
        rec["p_adj"] = float(pa)
        rec["strong"] = bool(pa < 0.05)
        rec["universe_source"] = universe_source
        by_cluster.setdefault(rec["cluster"], []).append(rec)

    result = {}
    for cl, hits in by_cluster.items():
        hits.sort(key=lambda r: r["p_adj"])
        result[cl] = hits[:top_k]
    return result
