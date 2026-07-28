# Make an AnnData object safe to write as .h5ad.
# Moved verbatim out of backend/services/snapshot_v2/runtime.py.
"""Runtime capture for Snapshot V2.

Owns kernel introspection, required-set assembly via the canonical registry,
and authoritative serializer success (h5ad + dill).
"""

from __future__ import annotations


# Re-export for existing imports / tests.


def sanitize_anndata_for_h5ad(adata):
    """Return a copy of *adata* safe for ``AnnData.write_h5ad``.

    Converts unwritable matrix-like slots (e.g. formulaic ModelMatrix /
    ObjectProxy in obsm/varm/layers/obsp/varp) and preserves scipy sparse
    matrices and exact native ``ndarray`` / ``DataFrame`` / ``Series``.
    Does not mutate *adata*. Also recursively sanitizes ``uns``: nested
    metadata dicts and native leaves are preserved, but leaves anndata cannot
    write to h5ad (a bare ``pd.Series`` such as PyDESeq2 ``uns['trend_coeffs']``,
    ``pd.Index``/``pd.Categorical``, formulaic ModelMatrix, ...) are coerced to
    ``ndarray`` or dropped. Also coerces ``obs``/``var`` object columns holding
    non-string objects (bool/None, e.g. scrublet ``predicted_doublet``) to str,
    which anndata otherwise rejects with "Can't implicitly convert non-string
    objects to strings". Covers the same object-column issue in DataFrame values
    under obsm/varm/layers and inside uns, coerces an object-dtype dense ``X`` to
    float, and rebuilds ``raw`` from a sanitized view so the raw layer's own
    columns/varm cannot block capture.

    Kernel capture uses an equivalent algorithm in ``_SAVE_ANNDATA_TEMPLATE``
    (kernel cannot import this host module); keep both in sync.
    """
    import numpy as np
    import pandas as pd

    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover
        sp = None

    to_write = adata.copy()

    def _is_formulaic_like(v) -> bool:
        t = type(v)
        mod = getattr(t, "__module__", "") or ""
        qn = getattr(t, "__qualname__", "") or getattr(t, "__name__", "") or ""
        return mod.startswith("formulaic") or ("ModelMatrix" in qn)

    def _usable_asarray(arr) -> bool:
        if getattr(arr, "ndim", None) == 0 and getattr(arr, "dtype", None) == object:
            return False
        return True

    def _convert_value(v):
        if hasattr(v, "to_numpy"):
            return np.asarray(v.to_numpy())
        if hasattr(v, "values") and not isinstance(v, dict):
            return np.asarray(v.values)
        return np.asarray(v)

    def _sanitize_mapping(mapping, slot_name: str) -> None:
        keys = list(mapping.keys())
        for k in keys:
            v = mapping[k]
            try:
                # 1) scipy sparse: keep unchanged
                if sp is not None and sp.issparse(v):
                    continue
                # 2) DataFrame: keep, but sanitize its object columns (a bad
                #    object column would fail write_dataframe under this slot).
                if type(v) is pd.DataFrame:
                    _sanitize_frame_columns(v)
                    continue
                # 3) exact native types only (not isinstance — ObjectProxy)
                if type(v) is np.ndarray or type(v) is pd.Series:
                    continue
                # 4) formulaic / ModelMatrix: force convert
                if _is_formulaic_like(v):
                    try:
                        mapping[k] = _convert_value(v)
                        continue
                    except Exception:
                        del mapping[k]
                        continue
                # 4) to_numpy / values / asarray with usable check
                if hasattr(v, "to_numpy"):
                    arr = np.asarray(v.to_numpy())
                    if _usable_asarray(arr):
                        mapping[k] = arr
                        continue
                elif hasattr(v, "values") and not isinstance(v, dict):
                    arr = np.asarray(v.values)
                    if _usable_asarray(arr):
                        mapping[k] = arr
                        continue
                else:
                    arr = np.asarray(v)
                    if _usable_asarray(arr):
                        mapping[k] = arr
                        continue
                del mapping[k]
            except Exception:
                try:
                    del mapping[k]
                except Exception:
                    pass

    def _sanitize_frame_columns(df) -> None:
        # Coerce object-dtype columns anndata cannot write as a vlen-string
        # array (non-string objects: bool / None / NaN / arrays, e.g. scrublet
        # 'predicted_doublet') to plain str. All-string object columns and
        # numeric/bool/categorical columns are left untouched.
        try:
            cols = list(df.columns)
        except Exception:
            return
        for col in cols:
            try:
                s = df[col]
                dtype = getattr(s, "dtype", None)
                kind = getattr(dtype, "kind", None)
                if kind in ("M", "m"):
                    # datetime64 / timedelta64: no h5ad writer — stringify.
                    df[col] = s.astype(str)
                    continue
                if dtype != object:
                    continue
                vals = s.to_numpy()
                if all(isinstance(x, str) for x in vals):
                    continue
                try:
                    df[col] = s.astype(str)
                except Exception:
                    df[col] = np.asarray([str(x) for x in vals], dtype=object)
            except Exception:
                try:
                    del df[col]
                except Exception:
                    pass

    def _sanitize_uns_mapping(mapping) -> None:
        # Recursively coerce/drop uns leaves anndata cannot write to h5ad,
        # while preserving nested Scanpy metadata dicts and native leaves.
        # A bare pd.Series (e.g. PyDESeq2 uns['trend_coeffs']) has no
        # registered h5ad writer under /uns and must be coerced to ndarray.
        for k in list(mapping.keys()):
            v = mapping[k]
            try:
                # h5py group names must be str. A cluster-id -> label map such as
                # uns['cluster_labels'] = {0: 'T cell'} (or numpy integer keys)
                # otherwise fails inside h5py with "'int' object has no attribute
                # 'split'". Kept in sync with _SAVE_ANNDATA_TEMPLATE.
                if not isinstance(k, str):
                    new_k = str(k)
                    del mapping[k]
                    if new_k in mapping:
                        # Renaming would clobber a real entry; drop instead.
                        continue
                    mapping[new_k] = v
                    k = new_k
                # Nested metadata dict: recurse, keep structure.
                if isinstance(v, dict):
                    _sanitize_uns_mapping(v)
                    continue
                # scipy sparse: keep.
                if sp is not None and sp.issparse(v):
                    continue
                # Native h5ad-writable leaves: keep untouched. Use isinstance for
                # ndarray to preserve structured/recarray/object-string arrays
                # (rank_genes_groups etc.); exact type for DataFrame so a
                # formulaic ModelMatrix (ObjectProxy) falls through to coercion.
                if type(v) is pd.DataFrame:
                    _sanitize_frame_columns(v)
                    continue
                if isinstance(v, np.ndarray):
                    continue
                if v is None or isinstance(
                    v, (str, bytes, bool, int, float, complex, np.generic)
                ):
                    continue
                # Unwritable-in-uns (bare Series/Index/Categorical, ModelMatrix,
                # list/tuple, ...): coerce to ndarray or drop.
                if hasattr(v, "to_numpy"):
                    arr = np.asarray(v.to_numpy())
                elif hasattr(v, "values") and not isinstance(v, dict):
                    arr = np.asarray(v.values)
                else:
                    arr = np.asarray(v)
                # Keep only if the coerced array is h5ad-writable: reject 0-d
                # object and object arrays that are not entirely strings (e.g.
                # a coerced list[dict] / list of arbitrary objects).
                if _usable_asarray(arr) and (
                    arr.dtype != object
                    or all(isinstance(x, str) for x in arr.ravel().tolist())
                ):
                    mapping[k] = arr
                    continue
                del mapping[k]
            except Exception:
                try:
                    del mapping[k]
                except Exception:
                    pass

    # X: coerce an object-dtype dense matrix to float (numeric-in-object);
    # leave sparse and genuinely non-numeric X untouched.
    X = getattr(to_write, "X", None)
    if (
        X is not None
        and not (sp is not None and sp.issparse(X))
        and getattr(X, "dtype", None) == object
    ):
        try:
            to_write.X = np.asarray(X, dtype=float)
        except Exception:
            pass

    # Matrix-like slots: convert unwritable matrix-like values.
    for slot, mapping in (
        ("obsm", getattr(to_write, "obsm", None)),
        ("varm", getattr(to_write, "varm", None)),
        ("layers", getattr(to_write, "layers", None)),
        ("obsp", getattr(to_write, "obsp", None)),
        ("varp", getattr(to_write, "varp", None)),
    ):
        if mapping is None:
            continue
        try:
            _sanitize_mapping(mapping, slot)
        except Exception:
            pass

    # obs / var: coerce object columns anndata cannot write as vlen-str.
    for frame in (getattr(to_write, "obs", None), getattr(to_write, "var", None)):
        if frame is None:
            continue
        try:
            _sanitize_frame_columns(frame)
        except Exception:
            pass

    # uns: recursively coerce/drop leaves anndata cannot write.
    uns = getattr(to_write, "uns", None)
    if uns is not None:
        try:
            _sanitize_uns_mapping(uns)
        except Exception:
            pass

    # raw: sanitize the raw layer's own var columns / varm (the pipeline sets
    # adata.raw = adata.copy() before HVG subsetting, so raw carries the same
    # unwritable object columns). Rebuild raw from a sanitized AnnData view.
    raw = getattr(to_write, "raw", None)
    if raw is not None:
        try:
            raw_ad = raw.to_adata()
            _sanitize_frame_columns(raw_ad.var)
            raw_varm = getattr(raw_ad, "varm", None)
            if raw_varm is not None:
                try:
                    _sanitize_mapping(raw_varm, "raw.varm")
                except Exception:
                    pass
            to_write.raw = raw_ad
        except Exception:
            pass

    return to_write
