"""Runtime capture for Snapshot V2.

Owns kernel introspection, required-set assembly via the canonical registry,
and authoritative serializer success (h5ad + dill).
"""

from __future__ import annotations

import ast
import json
import shutil
import uuid
from pathlib import Path

from backend.services.snapshot_v2.errors import SnapshotCaptureError, SnapshotIntegrityError
from backend.utils.error_parsing import strip_ansi
from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeDependency,
    RuntimeDependencyManifest,
    RuntimeManifest,
    RuntimeSkip,
    empty_complete_runtime,
)
from backend.services.snapshot_v2.object_store import ObjectStore
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.runtime_names import (
    runtime_h5ad_path,
    validate_runtime_global_name,
)
from backend.services.snapshot_v2.runtime_registry import (
    CANONICAL_RUNTIME_ARTIFACTS,
    build_required_runtime_manifest,
    build_required_runtime_set,
    is_known_unpersistable_residual,
)


# Re-export for existing imports / tests.
__all__ = [
    "CANONICAL_RUNTIME_ARTIFACTS",
    "RuntimeCapture",
    "RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES",
    "build_required_runtime_manifest",
    "build_required_runtime_set",
    "sanitize_anndata_for_h5ad",
    "validate_runtime_dependency_proof",
    "_parse_dill_skipped",
]


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
    _to_write.write({h5ad_path_repr!r})
    import os as _runtime_os
    _size = _runtime_os.path.getsize({h5ad_path_repr!r})
    print('__H5AD_SAVED__:' + str(_size))
else:
    print('__H5AD_NOT_FOUND__:' + {var_name_repr!r})
"""


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

_SAVE_DILL_TEMPLATE = """
import dill as _runtime_dill
import json as _runtime_json
_names = {names_repr!r}
import os as _runtime_os
_saved_count = 0
_skipped = []
_saved = []
with open({dill_path_repr!r}, 'wb') as _f:
    for _n in _names:
        _obj = globals().get(_n)
        if _obj is None and _n not in globals():
            _skipped.append(_n)
            continue
        try:
            _runtime_dill.dump((_n, _obj), _f)
            _saved_count += 1
            _saved.append(_n)
        except Exception as _e:
            _skipped.append(_n)
_size = _runtime_os.path.getsize({dill_path_repr!r})
print('__DILL_SAVED__:' + str(_saved_count) + '/' + str(len(_names)) + ' size=' + str(_size))
print('__DILL_SAVED_NAMES__:' + _runtime_json.dumps(_saved))
if _skipped:
    print('__DILL_SKIPPED__:' + _runtime_json.dumps(_skipped))
"""


# No hard size cap. Capacity is gated by free disk space minus reserve
# (aligned with SystemConfig.snapshot_v2_free_space_reserve_bytes default).
RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES: int = 5 * 1024 * 1024 * 1024


def _parse_dill_skipped(output: str) -> list[str]:
    """Structurally parse ``__DILL_SKIPPED__`` lines (JSON or Python literal)."""
    skipped: list[str] = []
    for line in (output or "").splitlines():
        if not line.startswith("__DILL_SKIPPED__:"):
            continue
        raw = line[len("__DILL_SKIPPED__:"):].strip()
        parsed = _parse_name_list(raw)
        if parsed is not None:
            skipped = parsed
    return skipped


def _parse_dill_saved_names(output: str) -> list[str]:
    for line in (output or "").splitlines():
        if line.startswith("__DILL_SAVED_NAMES__:"):
            raw = line[len("__DILL_SAVED_NAMES__:"):].strip()
            parsed = _parse_name_list(raw)
            if parsed is not None:
                return parsed
    return []


def _parse_name_list(raw: str) -> list[str] | None:
    """Parse a list of names from JSON or a Python literal (ast.literal_eval)."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def validate_runtime_dependency_proof(runtime: RuntimeManifest) -> RuntimeManifest:
    """Re-validate a staged runtime's completeness dependency proof.

    Ensures required names are covered by adata_refs / captured residual names,
    residual_ref is present when residual names are declared, and every
    ObjectRef in the runtime is self-consistent with status.
    """
    try:
        validated = RuntimeManifest.model_validate(runtime.model_dump(mode="json"))
    except Exception as exc:
        raise SnapshotIntegrityError(
            "RUNTIME_PROOF_INVALID",
            f"Runtime dependency proof failed validation: {exc}",
            details={"error": str(exc)},
        ) from exc

    if validated.status == "complete":
        captured = set(validated.adata_refs) | set(validated.captured_residual_names)
        missing = set(validated.dependencies.required) - captured
        if missing:
            raise SnapshotIntegrityError(
                "RUNTIME_PROOF_INCOMPLETE",
                f"complete runtime missing required values: {sorted(missing)}",
                details={"missing": sorted(missing)},
            )
        if validated.captured_residual_names and validated.residual_ref is None:
            raise SnapshotIntegrityError(
                "RUNTIME_PROOF_INCOMPLETE",
                "captured residual names require residual_ref",
            )
    return validated


def _validate_h5ad_file(path: Path) -> None:
    """Ensure *path* is a readable, structurally valid h5ad before put_file.

    Uses ``backed="r"`` mode for memory-efficient structural validation.
    Raises SnapshotCaptureError with code ``required_runtime_unpersistable`` on failure.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise SnapshotCaptureError(
            "required_runtime_unpersistable",
            f"h5ad file missing or empty: {path.name}",
            details={"path": str(path), "serializer": "h5ad"},
        )
    try:
        import anndata as ad

        # backed="r" validates structure without loading data into memory
        _ = ad.read_h5ad(path, backed="r")
    except SnapshotCaptureError:
        raise
    except Exception as e:
        raise SnapshotCaptureError(
            "required_runtime_unpersistable",
            f"h5ad validation failed for {path.name}: {e}",
            details={"path": str(path), "serializer": "h5ad", "error": str(e)},
        ) from e


class RuntimeCapture:
    FREE_SPACE_RESERVE_BYTES: int = RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES

    def __init__(self, object_store: ObjectStore, paths: SnapshotV2Paths):
        self._object_store = object_store
        self._paths = paths

    def mark_runtime_committed(self, sandbox, runtime_revision: int | None = None) -> None:
        """Signals that the current runtime generation has been committed.

        Called after a successful snapshot capture transaction. Sets
        ``saved_runtime_revision`` on the sandbox to the frozen
        ``runtime_revision`` (or current if not provided) so subsequent
        captures can detect drift.
        """
        if runtime_revision is None:
            runtime_revision = sandbox.runtime_revision
        sandbox.saved_runtime_revision = runtime_revision

    async def _run_kernel(
        self,
        sandbox,
        code: str,
        *,
        exec_lock_held: bool = False,
    ):
        """Execute trusted capture code; use unlocked path when lock already held."""
        if exec_lock_held:
            unlocked = getattr(sandbox, "_execute_unlocked", None)
            if unlocked is not None and callable(unlocked):
                # MagicMock auto-attrs are callable but not coroutine functions;
                # only use when the call returns an awaitable.
                try:
                    maybe = unlocked(
                        code, save_to_history=False, skip_safety_check=True
                    )
                except TypeError:
                    maybe = None
                if hasattr(maybe, "__await__"):
                    return await maybe
                # Non-awaitable (e.g. MagicMock default) — fall through to execute.
        return await sandbox.execute(
            code, save_to_history=False, skip_safety_check=True
        )



    async def _introspect_globals(self, sandbox, *, exec_lock_held: bool = False) -> dict[str, str]:
        result = await self._run_kernel(
            sandbox, _INTROSPECT_CODE, exec_lock_held=exec_lock_held
        )
        if not result.success:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_FAILED",
                f"Kernel globals introspection failed: {strip_ansi(result.error or '')}",
            )
        output = result.output or ""
        begin_marker = "__RUNTIME_GLOBALS_BEGIN__"
        end_marker = "__RUNTIME_GLOBALS_END__"
        start = output.find(begin_marker)
        if start == -1:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                f"Cannot find globals marker in output: {output[:500]}",
            )
        start += len(begin_marker)
        end = output.find(end_marker, start)
        if end == -1:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                "Missing end marker in introspection output",
            )
        json_str = output[start:end].strip()
        if not json_str:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                "Empty globals payload between markers",
            )
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                f"Failed to parse globals JSON: {e}",
                details={"json_snippet": json_str[:500]},
            ) from e
        if not isinstance(parsed, dict):
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                f"Globals payload is not a dict: {type(parsed).__name__}",
            )
        return {str(k): str(v) for k, v in parsed.items()}

    async def _capture_adata(
        self, sandbox, var_name: str, type_str: str, capture_dir: Path,
        *, exec_lock_held: bool = False,
    ) -> ObjectRef:
        var_name = validate_runtime_global_name(var_name)
        h5ad_path = runtime_h5ad_path(capture_dir, var_name)
        code = _SAVE_ANNDATA_TEMPLATE.format(
            var_name_repr=var_name,
            h5ad_path_repr=str(h5ad_path),
        )
        result = await self._run_kernel(
            sandbox, code, exec_lock_held=exec_lock_held
        )
        if not result.success:
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    f"AnnData variable '{var_name}' (type={type_str or 'unknown'}, "
                    f"serializer=h5ad) write failed: {strip_ansi(result.error or '')}"
                ),
                details={
                    "name": var_name,
                    "type": type_str,
                    "serializer": "h5ad",
                },
            )
        if not h5ad_path.exists():
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    f"AnnData variable '{var_name}' (type={type_str or 'unknown'}, "
                    f"serializer=h5ad) could not be serialized to h5ad"
                ),
                details={
                    "name": var_name,
                    "type": type_str,
                    "serializer": "h5ad",
                },
            )
        _validate_h5ad_file(h5ad_path)
        try:
            put_result = self._object_store.put_file(h5ad_path, "application/x-hdf5")
            ref = put_result.ref
        except SnapshotCaptureError:
            raise
        except Exception as e:
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    f"AnnData variable '{var_name}' (type={type_str or 'unknown'}, "
                    f"serializer=h5ad) object store put failed: {e}"
                ),
                details={
                    "name": var_name,
                    "type": type_str,
                    "serializer": "h5ad",
                },
            ) from e
        h5ad_path.unlink(missing_ok=True)
        return ref


    async def _capture_residual_dill(
        self,
        sandbox,
        var_names: list[str],
        type_by_name: dict[str, str],
        capture_dir: Path,
        *,
        exec_lock_held: bool = False,
    ) -> tuple[ObjectRef | None, list[str], list[str]]:
        """Return (ref, saved_names, skipped_names)."""
        if not var_names:
            return None, [], []
        var_names = [validate_runtime_global_name(name) for name in var_names]
        dill_path = capture_dir / "residual.pkl"
        code = _SAVE_DILL_TEMPLATE.format(
            names_repr=var_names,
            dill_path_repr=str(dill_path),
        )
        result = await self._run_kernel(
            sandbox, code, exec_lock_held=exec_lock_held
        )
        output = result.output or ""
        skipped = _parse_dill_skipped(output)
        saved = _parse_dill_saved_names(output)
        if not result.success:
            # Treat whole batch as unpersistable for required names.
            return None, saved, list(var_names)
        if not dill_path.exists():
            return None, saved, list(set(var_names) - set(saved)) or list(var_names)
        if not saved and not skipped:
            # Older kernel output without structural saved list: infer from skip.
            skipped = skipped or []
            saved = [n for n in var_names if n not in skipped]
        put_result = self._object_store.put_file(dill_path, "application/octet-stream")
        ref = put_result.ref
        dill_path.unlink(missing_ok=True)
        return ref, saved, skipped

    def _raise_unpersistable(
        self, name: str, type_str: str, serializer: str, reason: str
    ) -> None:
        raise SnapshotCaptureError(
            "required_runtime_unpersistable",
            (
                f"Required variable '{name}' (type={type_str or 'unknown'}, "
                f"serializer={serializer}) {reason}"
            ),
            details={
                "name": name,
                "type": type_str,
                "serializer": serializer,
            },
        )

    async def capture(
        self,
        sandbox,
        workflow_state: dict | None = None,
        declared_outputs: set[str] | None = None,
        checkpoint_dependencies: set[str] | None = None,
        kernel_present: bool | None = None,
        *,
        exec_lock_held: bool = False,
    ) -> RuntimeManifest:
        """Capture required runtime dependencies from a live sandbox.

        Returns a :class:`RuntimeManifest` directly (no transitional wrapper).

        ``kernel_present``:
          - ``False``: empty required set is accepted as complete (no runtime refs).
          - ``True``/``None`` with live kernel: empty required set is fail-closed.

        ``exec_lock_held``:
          When True, caller already holds ``sandbox._exec_lock``; use
          ``_execute_unlocked`` to avoid non-reentrant asyncio.Lock deadlock.
        """
        if kernel_present is None:
            kernel_present = getattr(sandbox, "km", None) is not None

        # No live kernel: do not introspect; empty complete is valid.
        if kernel_present is False:
            # Still allow declared/checkpoint names to force a required set, but
            # without a kernel they cannot be captured.
            dep_manifest = build_required_runtime_manifest(
                globals_meta={},
                persistent_outputs=declared_outputs,
                checkpoint_dependencies=checkpoint_dependencies,
            )
            if dep_manifest.required:
                missing = sorted(dep_manifest.required)
                raise SnapshotCaptureError(
                    "required_runtime_unpersistable",
                    (
                        "Required runtime values declared but no live kernel "
                        f"present to capture them: {missing}"
                    ),
                    details={"names": missing},
                )
            return empty_complete_runtime()

        globals_meta = await self._introspect_globals(
            sandbox, exec_lock_held=exec_lock_held
        )

        dep_manifest = build_required_runtime_manifest(
            globals_meta=globals_meta,
            persistent_outputs=declared_outputs,
            checkpoint_dependencies=checkpoint_dependencies,
        )
        required = set(dep_manifest.required)

        if not required:
            # Live kernel + empty required set is fail-closed: registry and
            # introspection must be authoritative; empty is only valid when
            # kernel_present=False.
            raise SnapshotCaptureError(
                "NO_RUNTIME_VARIABLES",
                "No runtime variables found to capture from live kernel",
            )

        capture_dir = self._paths.staging / f"rtcap_{uuid.uuid4().hex[:12]}"
        capture_dir.mkdir(parents=True, exist_ok=True)

        captured_adata_refs: dict[str, ObjectRef] = {}
        captured_residual_vars: list[str] = []
        captured_residual_ref: ObjectRef | None = None
        incidental_skips: list[RuntimeSkip] = []

        # Names that are continuation-critical (checkpoint dependencies) must
        # fail closed when absent from globals. Advisory declared names (e.g. an
        # LLM-declared persistent output that was later `del`-eted) cannot be
        # captured if they are gone, and failing closed on them would permanently
        # brick every snapshot. Those are soft-skipped instead.
        mandatory_names = set(checkpoint_dependencies or set())

        # Pruned required set drives the completeness proof; soft-skipped absent
        # advisory names are removed from it and recorded as incidental skips.
        pruned_required = dict(dep_manifest.required)

        def _soft_skip_absent(name: str, serializer: str) -> None:
            type_str = dep_manifest.required[name].type_name
            incidental_skips.append(
                RuntimeSkip(
                    name=name,
                    type_name=type_str,
                    reason=(
                        f"declared runtime dependency absent from kernel globals "
                        f"at capture (serializer={serializer}); soft-skipped"
                    ),
                )
            )
            pruned_required.pop(name, None)

        try:
            # Partition required names into h5ad vs dill based on dependency serializer.
            h5ad_names: list[str] = []
            dill_names: list[str] = []
            for name, dep in dep_manifest.required.items():
                if dep.serializer == "h5ad":
                    h5ad_names.append(name)
                else:
                    dill_names.append(name)

            for var_name in sorted(h5ad_names):
                type_str = globals_meta.get(var_name, dep_manifest.required[var_name].type_name)
                if var_name not in globals_meta:
                    if var_name in mandatory_names:
                        self._raise_unpersistable(
                            var_name, type_str, "h5ad", "not found in kernel globals"
                        )
                    _soft_skip_absent(var_name, "h5ad")
                    continue
                try:
                    ref = await self._capture_adata(
                        sandbox, var_name, type_str, capture_dir,
                        exec_lock_held=exec_lock_held,
                    )
                except SnapshotCaptureError:
                    raise
                except Exception as e:
                    self._raise_unpersistable(
                        var_name, type_str, "h5ad", f"capture failed: {e}"
                    )
                captured_adata_refs[var_name] = ref

            # Drop absent advisory dill names (soft-skip); keep absent mandatory
            # names so the fail-closed check below still raises for them.
            # Also pre-skip present advisory names with known-unpersistable types
            # (e.g. LoomConnection file handles) so they never brick capture.
            present_dill_names: list[str] = []
            for var_name in sorted(dill_names):
                type_str = globals_meta.get(
                    var_name, dep_manifest.required[var_name].type_name
                )
                if var_name not in globals_meta and var_name not in mandatory_names:
                    _soft_skip_absent(var_name, "dill")
                    continue
                if (
                    var_name not in mandatory_names
                    and var_name in globals_meta
                    and is_known_unpersistable_residual(type_str)
                ):
                    incidental_skips.append(
                        RuntimeSkip(
                            name=var_name,
                            type_name=type_str,
                            reason=(
                                "known unpersistable residual type; soft-skipped "
                                f"(serializer=dill): {type_str}"
                            ),
                        )
                    )
                    pruned_required.pop(var_name, None)
                    continue
                present_dill_names.append(var_name)

            residual_names = sorted(present_dill_names)
            if residual_names:
                for var_name in residual_names:
                    if var_name not in globals_meta:
                        type_str = dep_manifest.required[var_name].type_name
                        self._raise_unpersistable(
                            var_name, type_str, "dill", "not found in kernel globals"
                        )
                type_by_name = {
                    n: globals_meta.get(n, dep_manifest.required[n].type_name)
                    for n in residual_names
                }
                try:
                    ref, saved, dill_skipped = await self._capture_residual_dill(
                        sandbox, residual_names, type_by_name, capture_dir,
                        exec_lock_held=exec_lock_held,
                    )
                except SnapshotCaptureError:
                    raise
                except Exception as e:
                    raise SnapshotCaptureError(
                        "required_runtime_unpersistable",
                        f"Residual dill capture failed: {e}",
                        details={"names": residual_names},
                    ) from e

                required_skipped = set(residual_names) & set(dill_skipped)
                # Also treat required names that neither saved nor skipped as missing.
                if saved is not None:
                    not_accounted = set(residual_names) - set(saved) - set(dill_skipped)
                    required_skipped |= not_accounted

                advisory_skipped = {
                    n for n in required_skipped if n not in mandatory_names
                }
                mandatory_skipped = {
                    n for n in required_skipped if n in mandatory_names
                }
                for n in sorted(advisory_skipped):
                    type_str = type_by_name.get(n, "unknown")
                    incidental_skips.append(
                        RuntimeSkip(
                            name=n,
                            type_name=type_str,
                            reason=(
                                "dill serialization failed for advisory residual; "
                                "soft-skipped"
                            ),
                        )
                    )
                    pruned_required.pop(n, None)
                if mandatory_skipped:
                    first = sorted(mandatory_skipped)[0]
                    self._raise_unpersistable(
                        first,
                        type_by_name.get(first, "unknown"),
                        "dill",
                        f"could not be serialized via dill "
                        f"(also missing: {sorted(mandatory_skipped)})",
                    )

                # Only treat successfully saved residual names as captured
                residual_names_saved = [
                    n for n in residual_names if n in set(saved or [])
                ]
                if not residual_names_saved:
                    if residual_names and not mandatory_names.intersection(
                        residual_names
                    ):
                        # all advisory — ok to have no residual pickle
                        captured_residual_vars = []
                        captured_residual_ref = None
                    elif residual_names:
                        # mandatory names remained but none saved (should have raised)
                        first = sorted(
                            mandatory_names.intersection(residual_names)
                        )[0]
                        self._raise_unpersistable(
                            first,
                            type_by_name.get(first, "unknown"),
                            "dill",
                            "residual pickle was not produced",
                        )
                else:
                    if ref is None:
                        self._raise_unpersistable(
                            residual_names_saved[0],
                            type_by_name.get(residual_names_saved[0], "unknown"),
                            "dill",
                            "residual pickle was not produced",
                        )
                    captured_residual_vars = list(residual_names_saved)
                    captured_residual_ref = ref

            # Capacity gate (post-write): objects are already in the store, so
            # disk free already accounts for payload. Require remaining free
            # space to stay above reserve. Fail closed if we cannot probe.
            total_size = sum(ref.size for ref in captured_adata_refs.values())
            if captured_residual_ref is not None:
                total_size += captured_residual_ref.size

            probe = getattr(self._paths, "staging", None) or getattr(self._paths, "root", None)
            if probe is None:
                raise SnapshotCaptureError(
                    "RUNTIME_CAPTURE_DISK_FULL",
                    (
                        "Cannot verify free disk space for runtime capture: "
                        "no staging/root path available"
                    ),
                    details={"total_size": total_size},
                )
            probe_path = Path(probe)
            while not probe_path.exists() and probe_path != probe_path.parent:
                probe_path = probe_path.parent
            if not probe_path.exists():
                raise SnapshotCaptureError(
                    "RUNTIME_CAPTURE_DISK_FULL",
                    (
                        "Cannot verify free disk space for runtime capture: "
                        f"no existing path from {probe}"
                    ),
                    details={"total_size": total_size, "probe": str(probe)},
                )
            try:
                usage = shutil.disk_usage(probe_path)
            except OSError as e:
                raise SnapshotCaptureError(
                    "RUNTIME_CAPTURE_DISK_FULL",
                    f"Failed to check disk usage for runtime capture: {e}",
                    details={"total_size": total_size},
                ) from e
            # After objects are written, free already accounts for payload.
            # Require remaining free space to stay above reserve.
            if usage.free < self.FREE_SPACE_RESERVE_BYTES:
                raise SnapshotCaptureError(
                    "RUNTIME_CAPTURE_DISK_FULL",
                    (
                        f"Insufficient disk space for runtime capture: "
                        f"{usage.free} free after write, "
                        f"{total_size} captured, "
                        f"{self.FREE_SPACE_RESERVE_BYTES} reserve"
                    ),
                    details={
                        "free": usage.free,
                        "total_size": total_size,
                        "reserve": self.FREE_SPACE_RESERVE_BYTES,
                    },
                )

            # Build complete RuntimeManifest with dependency proof.
            # Required is the registry-declared set minus soft-skipped absent
            # advisory names, so the completeness proof holds against what was
            # actually captured.
            runtime = RuntimeManifest(
                status="complete",
                dependencies=RuntimeDependencyManifest(
                    required=pruned_required,
                    incidental_skips=incidental_skips,
                ),
                residual_ref=captured_residual_ref,
                captured_residual_names=list(captured_residual_vars),
                adata_refs=dict(captured_adata_refs),
            )

            # Return RuntimeManifest directly.
            return runtime
        finally:
            try:
                if capture_dir.exists():
                    for p in capture_dir.iterdir():
                        p.unlink(missing_ok=True)
                    capture_dir.rmdir()
            except OSError:
                pass
