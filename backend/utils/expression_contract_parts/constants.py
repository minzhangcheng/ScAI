"""Constants shared by expression-contract owners."""

from __future__ import annotations


_SENSITIVE_CALLS = frozenset({
    "rank_genes_groups",
    "score_genes",
    "score_genes_cell_cycle",
    "non_negative_factorization",
})
_REPRESENTATION_MUTATORS = frozenset({
    "normalize_total", "log1p", "scale", "normalize_per_cell",
})
# Module roots whose ``log1p``/``scale``/``normalize_*`` are array maths, not an
# operation on the canonical AnnData. ``np.log1p(counts / total * 1e6)`` shares a
# name with ``sc.pp.log1p(adata)`` and nothing else; matched on the bare name it
# was indistinguishable, and because such a call has no resolvable target root it
# took the "unknown source, assume canonical" branch and always reported True.
# Measured over 211 real detector firings, 68 (32%) were numpy-only calls of
# exactly this shape. A call whose module is unknown still counts as canonical.
_NON_EXPRESSION_CALL_ROOTS = frozenset({
    "np", "numpy", "pd", "pandas", "sp", "scipy", "sk", "sklearn",
    "math", "torch", "cp", "cupy", "jnp", "stats",
})
_MARKER = "__SCAI_EXPRESSION_CONTRACT__:"
_RUNTIME_CALL_HELPER = "_scai_expression_contract_call"
_RUNTIME_CALL_RESOLVER = "_scai_expression_contract_target"


# Names whose call produces a brand-new AnnData object. Mutating such an object
# cannot retroactively change a figure rendered from a different object.
#
# Deliberately a conservative, closed set: a call whose name is absent (e.g. a
# user-defined loader like ``create_anndata()``) is treated as binding the
# canonical object, so a later mutation of that name is flagged.
#
# That bias is only cheap where the consequence is a refresh. It is not: every
# consumer of ``invalidated_artifact_stems`` reads it as "hide this from output",
# so a false positive deletes a good figure rather than re-rendering it. Keep new
# entries here generous and keep ``_NON_EXPRESSION_CALL_ROOTS`` current; the
# blacklist those feed is bounded by round (``unchanged_artifact_stems``) but a
# same-round false positive still costs the user a figure.
_FRESH_OBJECT_CALLS = frozenset({
    "copy", "AnnData", "concat", "concatenate",
    "read", "read_h5ad", "read_10x_h5", "read_10x_mtx", "read_mtx",
    "read_csv", "read_text", "read_visium", "read_loom", "read_excel",
})
