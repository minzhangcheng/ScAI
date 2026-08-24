"""Domain-neutral execution guidance for installed analysis APIs."""


PYDESEQ2_RUNTIME_GUIDANCE = (
    "For the installed PyDESeq2 API, import DeseqDataSet from pydeseq2.dds and "
    "DeseqStats from pydeseq2.ds. Build a samples-by-genes count DataFrame and "
    "sample-indexed metadata, run dds.deseq2(), then construct "
    "DeseqStats(dds, contrast=['condition', '<numerator>', '<denominator>']), "
    "call summary(), and read results_df. Do not import WaldTest or "
    "pydeseq2.stats, and do not substitute Scanpy rank_genes_groups or a "
    "per-gene t-test for bulk differential expression."
)

GENE_INDEX_ALIGNMENT_GUIDANCE = (
    "After mapping feature IDs to gene symbols, first check index.is_unique. "
    "Collapse duplicate raw-count rows with groupby(level=0).sum(); for already "
    "normalized/log-scale values use a scientifically justified aggregation "
    "such as mean. Align matrices explicitly with DataFrame.align(join='inner', "
    "axis=0), then assert that both indexes are unique and equal before building "
    "or applying a positional boolean mask. Do not use set intersection plus "
    ".loc on an index that may contain duplicate symbols."
)

LIANA_RUNTIME_GUIDANCE = (
    "For installed LIANA 1.8.x, use `import liana as li` and call "
    "`li.mt.rank_aggregate(adata, groupby='<annotation column>', "
    "resource_name='<resource>', key_added='liana_res', n_jobs=1)`. Use "
    "`mouseconsensus` for mouse and `consensus` for human unless an approved "
    "plan names a validated custom resource. Do not call legacy/nonexistent "
    "`singlecellpipe` or `run_liana`; inspect "
    "`liana.method.rank_aggregate` before changing the API."
)
