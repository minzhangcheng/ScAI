# Source templates executed inside the Jupyter kernel to introspect and save runtime state.
# Moved verbatim out of backend/services/snapshot_v2/runtime.py.
"""Runtime capture for Snapshot V2.

Owns kernel introspection, required-set assembly via the canonical registry,
and authoritative serializer success (h5ad + dill).
"""

from __future__ import annotations


# Re-export for existing imports / tests.


_INTROSPECT_CODE = """
import anndata as _runtime_ad
import json as _runtime_json
_globals = {}
for _name, _obj in list(globals().items()):
    if _name.startswith('_'):
        continue
    if isinstance(_obj, _runtime_ad.AnnData):
        _shape = str(getattr(_obj, 'shape', '?'))
        _globals[_name] = 'AnnData(' + _shape + ')'
    else:
        _globals[_name] = type(_obj).__module__ + '.' + type(_obj).__qualname__
print('__RUNTIME_GLOBALS_BEGIN__')
print(_runtime_json.dumps(_globals))
print('__RUNTIME_GLOBALS_END__')
"""

_SAVE_ANNDATA_TEMPLATE = """
import anndata as _runtime_ad
import numpy as _runtime_np
import pandas as _runtime_pd
_adata = globals().get({var_name_repr!r})
if _adata is not None:
    # Never mutate the live analysis object; sanitize a copy for h5ad.
    # Algorithm must stay in sync with host sanitize_anndata_for_h5ad().
    _to_write = _adata.copy()
    def _is_formulaic_like(_v):
        _t = type(_v)
        _mod = getattr(_t, '__module__', '') or ''
        _qn = getattr(_t, '__qualname__', '') or getattr(_t, '__name__', '') or ''
        return _mod.startswith('formulaic') or ('ModelMatrix' in _qn)
    def _usable_asarray(_arr):
        # Reject 0-d object holding an opaque type (failed cast of sparse/proxy).
        if getattr(_arr, 'ndim', None) == 0 and getattr(_arr, 'dtype', None) == object:
            return False
        return True
    def _convert_value(_v):
        if hasattr(_v, 'to_numpy'):
            return _runtime_np.asarray(_v.to_numpy())
        if hasattr(_v, 'values') and not isinstance(_v, dict):
            return _runtime_np.asarray(_v.values)
        return _runtime_np.asarray(_v)
    def _sanitize_mapping(_mapping, _slot_name):
        try:
            import scipy.sparse as _sp
        except Exception:
            _sp = None
        _keys = list(_mapping.keys())
        for _k in _keys:
            _v = _mapping[_k]
            try:
                # 1) scipy sparse: keep (do NOT asarray — yields 0-d object)
                if _sp is not None and _sp.issparse(_v):
                    continue
                # 2) DataFrame: keep, but sanitize its object columns.
                if type(_v) is _runtime_pd.DataFrame:
                    _sanitize_frame_columns(_v)
                    continue
                # 3) exact native types only (formulaic ModelMatrix is ObjectProxy;
                #    isinstance(mm, DataFrame) is True but write still fails)
                if type(_v) is _runtime_np.ndarray or type(_v) is _runtime_pd.Series:
                    continue
                # 4) formulaic / ModelMatrix: force convert
                if _is_formulaic_like(_v):
                    try:
                        _mapping[_k] = _convert_value(_v)
                        continue
                    except Exception:
                        del _mapping[_k]
                        print('__H5AD_SANITIZE_DROP__:' + _slot_name + '/' + str(_k) + ':' + type(_v).__module__ + '.' + type(_v).__qualname__)
                        continue
                # 4) to_numpy / values / asarray
                if hasattr(_v, 'to_numpy'):
                    _arr = _runtime_np.asarray(_v.to_numpy())
                elif hasattr(_v, 'values') and not isinstance(_v, dict):
                    _arr = _runtime_np.asarray(_v.values)
                else:
                    _arr = _runtime_np.asarray(_v)
                if _usable_asarray(_arr):
                    _mapping[_k] = _arr
                    continue
                del _mapping[_k]
                print('__H5AD_SANITIZE_DROP__:' + _slot_name + '/' + str(_k) + ':' + type(_v).__module__ + '.' + type(_v).__qualname__)
            except Exception as _se:
                try:
                    del _mapping[_k]
                except Exception:
                    pass
                print('__H5AD_SANITIZE_DROP__:' + _slot_name + '/' + str(_k) + ':' + type(_v).__module__ + '.' + type(_v).__qualname__)
    def _sanitize_frame_columns(_df):
        try:
            _cols = list(_df.columns)
        except Exception:
            return
        for _col in _cols:
            try:
                _s = _df[_col]
                _dtype = getattr(_s, 'dtype', None)
                _kind = getattr(_dtype, 'kind', None)
                if _kind in ('M', 'm'):
                    _df[_col] = _s.astype(str)
                    continue
                if _dtype != object:
                    continue
                _vals = _s.to_numpy()
                if all(isinstance(_x, str) for _x in _vals):
                    continue
                try:
                    _df[_col] = _s.astype(str)
                except Exception:
                    _df[_col] = _runtime_np.asarray([str(_x) for _x in _vals], dtype=object)
            except Exception:
                try:
                    del _df[_col]
                except Exception:
                    pass
    def _sanitize_uns_mapping(_mapping):
        try:
            import scipy.sparse as _sp
        except Exception:
            _sp = None
        for _k in list(_mapping.keys()):
            _v = _mapping[_k]
            try:
                # h5py group names must be str. A cluster-id -> label map under
                # uns (integer or numpy-integer keys, which scanpy workflows
                # produce routinely) otherwise fails deep inside h5py with
                # "'int' object has no attribute 'split'" and takes the whole
                # required capture down with it.
                # NB: this is a .format() template — never write a literal brace
                # here; str.format reads it as a replacement field.
                if not isinstance(_k, str):
                    _new_k = str(_k)
                    del _mapping[_k]
                    if _new_k in _mapping:
                        # Renaming would clobber a real entry; drop instead of
                        # silently overwriting the user's data.
                        print('__H5AD_SANITIZE_DROP__:uns/' + _new_k + ':key-collision')
                        continue
                    _mapping[_new_k] = _v
                    _k = _new_k
                # Nested metadata dict (neighbors, log1p, pca,
                # rank_genes_groups, ...): recurse, keep structure.
                if isinstance(_v, dict):
                    _sanitize_uns_mapping(_v)
                    continue
                if _sp is not None and _sp.issparse(_v):
                    continue
                # Native h5ad-writable leaves kept untouched (isinstance for
                # ndarray to preserve structured/recarray; exact DataFrame so a
                # ModelMatrix ObjectProxy falls through to coercion).
                if type(_v) is _runtime_pd.DataFrame:
                    _sanitize_frame_columns(_v)
                    continue
                if isinstance(_v, _runtime_np.ndarray):
                    continue
                if _v is None or isinstance(_v, (str, bytes, bool, int, float, complex, _runtime_np.generic)):
                    continue
                # Unwritable-in-uns (bare Series/Index/Categorical, ModelMatrix,
                # list/tuple, ...): coerce to ndarray or drop.
                if hasattr(_v, 'to_numpy'):
                    _arr = _runtime_np.asarray(_v.to_numpy())
                elif hasattr(_v, 'values') and not isinstance(_v, dict):
                    _arr = _runtime_np.asarray(_v.values)
                else:
                    _arr = _runtime_np.asarray(_v)
                # Keep only if h5ad-writable: reject 0-d object and object
                # arrays that are not entirely strings (e.g. coerced list[dict]).
                if _usable_asarray(_arr) and (
                    _arr.dtype != object
                    or all(isinstance(_x, str) for _x in _arr.ravel().tolist())
                ):
                    _mapping[_k] = _arr
                    continue
                del _mapping[_k]
                print('__H5AD_SANITIZE_DROP__:uns/' + str(_k) + ':' + type(_v).__module__ + '.' + type(_v).__qualname__)
            except Exception:
                try:
                    del _mapping[_k]
                except Exception:
                    pass
                print('__H5AD_SANITIZE_DROP__:uns/' + str(_k) + ':' + type(_v).__module__ + '.' + type(_v).__qualname__)
    # X: coerce an object-dtype dense matrix to float; leave sparse and
    # genuinely non-numeric X untouched.
    try:
        import scipy.sparse as _spx
    except Exception:
        _spx = None
    _X = getattr(_to_write, 'X', None)
    if _X is not None and not (_spx is not None and _spx.issparse(_X)) and getattr(_X, 'dtype', None) == object:
        try:
            _to_write.X = _runtime_np.asarray(_X, dtype=float)
        except Exception:
            pass
    # Matrix-like slots: convert unwritable matrix-like values.
    for _slot, _mapping in (
        ('obsm', getattr(_to_write, 'obsm', None)),
        ('varm', getattr(_to_write, 'varm', None)),
        ('layers', getattr(_to_write, 'layers', None)),
        ('obsp', getattr(_to_write, 'obsp', None)),
        ('varp', getattr(_to_write, 'varp', None)),
    ):
        if _mapping is None:
            continue
        try:
            _sanitize_mapping(_mapping, _slot)
        except Exception:
            pass
    # obs / var: coerce object columns anndata cannot write as vlen-str
    # (non-string objects like scrublet 'predicted_doublet' bool/None).
    for _frame in (getattr(_to_write, 'obs', None), getattr(_to_write, 'var', None)):
        if _frame is None:
            continue
        try:
            _sanitize_frame_columns(_frame)
        except Exception:
            pass
    # uns: recursively coerce/drop leaves anndata cannot write, preserving
    # nested Scanpy metadata dicts (a bare pd.Series such as PyDESeq2
    # uns['trend_coeffs'] has no registered h5ad writer under /uns).
    _uns = getattr(_to_write, 'uns', None)
    if _uns is not None:
        try:
            _sanitize_uns_mapping(_uns)
        except Exception:
            pass
    # raw: rebuild from a sanitized AnnData view (raw carries the same
    # unwritable object columns as the pre-HVG snapshot of adata).
    _raw = getattr(_to_write, 'raw', None)
    if _raw is not None:
        try:
            _raw_ad = _raw.to_adata()
            _sanitize_frame_columns(_raw_ad.var)
            _raw_varm = getattr(_raw_ad, 'varm', None)
            if _raw_varm is not None:
                try:
                    _sanitize_mapping(_raw_varm, 'raw.varm')
                except Exception:
                    pass
            _to_write.raw = _raw_ad
        except Exception:
            pass
    try:
        _to_write.write({h5ad_path_repr!r})
        import os as _runtime_os
        _size = _runtime_os.path.getsize({h5ad_path_repr!r})
        print('__H5AD_SAVED__:' + str(_size))
    finally:
        # This cell runs at the kernel's __main__ scope, so every name bound
        # above is a kernel global that outlives the capture. _to_write is a
        # full copy of the dataset and _X / _uns / _mapping / _frame are
        # references into it, so leaving any of them bound keeps a second
        # dataset resident for the life of the kernel: save_session skips
        # _-prefixed names so it is never persisted, no probe measures kernel
        # RSS, and only a kernel restart — a failure path — reclaims it. The
        # next capture then evaluates _adata.copy() before rebinding, peaking
        # at 3x the dataset where the algorithm needs 2x.
        # It has to be a finally, not a trailing statement: a failed write is
        # exactly when the host retries and builds another copy, and IPython
        # keeps the failing frames alive on top of that.
        for _cleanup_name in (
            '_to_write', '_raw_ad', '_raw_varm', '_raw', '_adata',
            '_X', '_uns', '_mapping', '_frame', '_slot',
        ):
            globals().pop(_cleanup_name, None)
        globals().pop('_cleanup_name', None)
else:
    print('__H5AD_NOT_FOUND__:' + {var_name_repr!r})
"""

_SAVE_DILL_TEMPLATE = """
import dill as _runtime_dill
import json as _runtime_json
_names = {names_repr!r}
import os as _runtime_os
_saved_count = 0
_skipped = []
_saved = []
try:
    with open({dill_path_repr!r}, 'wb') as _f:
        for _n in _names:
            _obj = globals().get(_n)
            if _obj is None and _n not in globals():
                _skipped.append(_n)
                continue
            _checkpoint = _f.tell()
            try:
                _runtime_dill.dump((_n, _obj), _f)
                _saved_count += 1
                _saved.append(_n)
            except Exception as _e:
                # dill writes incrementally. Remove the incomplete frame before
                # appending another variable or its memo references can bind to
                # objects leaked from the failed frame during restore.
                _f.seek(_checkpoint)
                _f.truncate()
                _skipped.append(_n)
    _verified = []
    with open({dill_path_repr!r}, 'rb') as _verify_f:
        while True:
            try:
                _verified_name, _verified_obj = _runtime_dill.load(_verify_f)
                _verified.append(_verified_name)
            except EOFError:
                break
    if _verified != _saved:
        raise RuntimeError(
            'Residual dill readback mismatch: saved=' + repr(_saved)
            + ' verified=' + repr(_verified)
        )
    _size = _runtime_os.path.getsize({dill_path_repr!r})
    print('__DILL_SAVED__:' + str(_saved_count) + '/' + str(len(_names)) + ' size=' + str(_size))
    print('__DILL_SAVED_NAMES__:' + _runtime_json.dumps(_saved))
    print('__DILL_VERIFIED_NAMES__:' + _runtime_json.dumps(_verified))
    if _skipped:
        print('__DILL_SKIPPED__:' + _runtime_json.dumps(_skipped))
finally:
    # Same kernel-scope rule as the h5ad template: these are __main__ globals,
    # not locals. _obj only aliases the live residual value, but _verified_obj
    # is a second, freshly deserialized copy of the last verified one, and the
    # readback exists precisely because these values can be large. The readback
    # mismatch above raises, so the cleanup has to be a finally to cover it.
    for _cleanup_name in ('_obj', '_verified_obj', '_verified_name'):
        globals().pop(_cleanup_name, None)
    globals().pop('_cleanup_name', None)
"""
