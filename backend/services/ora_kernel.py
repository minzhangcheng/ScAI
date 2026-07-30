"""自包含 ORA 计算（hypergeom + BH-FDR）。

禁止 import backend.*：本文件源码经 inspect.getsource 注入 sandbox kernel 执行。
"""
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def _collapse_synonymous(hits):
    """Merge hits that fired on exactly the same genes into one entry.

    The marker databases carry many near-synonymous cell types, and they all
    score identically on a shared overlap. On a real mouse-liver cluster,
    "Brush cell", "Early enterocyte precursor cell" and "Goblet cell" each hit
    the same two Cyp genes with the same p-value and filled all of top_k --
    pushing out Hepatocytes, which was the answer. Collapsing by overlap keeps
    the alternatives visible while spending one slot on them.
    """
    merged = []
    by_overlap = {}
    for h in hits:
        key = tuple(h["overlap"])
        first = by_overlap.get(key)
        if first is None:
            by_overlap[key] = h
            h["also_matched"] = []
            merged.append(h)
        else:
            first["also_matched"].append(h["cell_type"])
    return merged


def _tissue_relevance(tissues, target):
    """Does this cell type's annotation include the tissue under analysis?

    Reported, never used to filter: organ names are free text on both sides, so
    a miss must not be able to drop a correct hit. Naming the breadth matters as
    much as the match -- a cell type annotated across dozens of organs carries no
    tissue information either way, and listing an arbitrary three of them (an
    earlier attempt) actively misled, labelling CD8 T cells as "Kidney".
    """
    n = len(tissues)
    if not n:
        return {"tissue_match": None, "n_tissues": 0}
    if not target:
        return {"tissue_match": None, "n_tissues": n}
    matched = any(target in t or t in target for t in tissues)
    return {"tissue_match": bool(matched), "n_tissues": n}


def compute_ora_priors(adata, db_path, cluster_key, rank_key,
                       top_n=50, min_overlap=2, top_k=3, tissue=""):
    db = pd.read_csv(db_path, sep="\t", comment="#")
    db["gene_symbol"] = db["gene_symbol"].astype(str).str.upper()
    target_tissue = str(tissue or "").strip().lower()

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
    # Set of organs each cell type is annotated in, lowercased for matching. The
    # field is multi-valued (";"-joined by marker_priors) because one marker is
    # typically annotated in several tissues.
    def _tissue_set(series):
        out = set()
        for value in series:
            for part in str(value).split(";"):
                part = part.strip().lower()
                if part and part != "nan":
                    out.add(part)
        return out

    celltype_tissues = (db.groupby("cell_type")["tissue"].apply(_tissue_set).to_dict()
                        if "tissue" in db.columns else {})

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
            rec = {"cluster": cl, "cell_type": ct, "overlap": overlap,
                   "p_value": p, "source": celltype_source.get(ct, "")}
            rec.update(_tissue_relevance(celltype_tissues.get(ct, set()),
                                         target_tissue))
            records.append(rec)
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
        # Prefer the tissue under analysis on ties. Sorting only breaks ties --
        # never overrides a stronger p-value -- so a genuinely better hit from
        # another organ still wins and a tissue mismatch cannot hide anything.
        hits.sort(key=lambda r: (r["p_adj"], r["tissue_match"] is False))
        result[cl] = _collapse_synonymous(hits)[:top_k]
    return result
