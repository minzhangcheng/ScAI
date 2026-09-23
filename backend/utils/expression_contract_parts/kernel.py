"""Trusted kernel source builders for expression contracts."""

from __future__ import annotations

import json

from .constants import _MARKER, _RUNTIME_CALL_HELPER, _RUNTIME_CALL_RESOLVER


def build_expression_preflight_code(specs: list[dict[str, str | None]]) -> str:
    """Build trusted kernel code that rejects raw integer count inputs."""
    payload = json.dumps(specs, separators=(",", ":"))
    return build_expression_runtime_code() + f"""
import json as _scai_json
_scai_specs = _scai_json.loads({payload!r})
_scai_contracts = []
for _scai_spec in _scai_specs:
    _scai_name = _scai_spec['name']
    if _scai_name not in globals():
        continue
    _scai_layer = _scai_spec.get('layer')
    _scai_contracts.append(_scai_validate_expression_contract(
        globals()[_scai_name], _scai_name, _scai_layer, emit=False
    ))
print({_MARKER!r} + _scai_json.dumps(_scai_contracts, sort_keys=True))
if not _scai_contracts:
    raise ValueError(
        'Expression contract could not resolve the analysis input in the current runtime. '
        'Create and normalize the input in a prior execute_code step, declare it as a '
        'persistent output, then run DE, gene scoring, or NMF.'
    )
_scai_invalid = [c for c in _scai_contracts if c['scale'] != 'normalized_continuous']
if _scai_invalid:
    _scai_raise_expression_scale_error(_scai_invalid)
"""


_EXPRESSION_RUNTIME_CODE = f"""
import json as _scai_expression_json
import numpy as _scai_expression_np

_scai_expression_chunk_size = 65536

def _scai_expression_dtype_error(_scai_label, _scai_layer, _scai_dtype_class):
    _scai_subject = str(_scai_label)
    if _scai_layer is not None:
        _scai_subject += " layer " + repr(str(_scai_layer))
    raise ValueError(
        "Expression contract unsupported dtype for " + _scai_subject + ": "
        + str(_scai_dtype_class)
        + ". DE, gene scoring, and NMF require a real numeric matrix; "
          "boolean, object, string, datetime, timedelta, and complex values are not supported."
    )

def _scai_new_expression_stats():
    return {{
        "finite_count": 0,
        "nonzero_count": 0,
        "integer_count": 0,
        "min": None,
        "max": None,
        "has_positive_integer": False,
        "has_continuous": False,
    }}

def _scai_expression_chunk_stats(_scai_chunk):
    # Reduce one bounded chunk to scalars without retaining its values.
    _scai_values = _scai_expression_np.asarray(
        _scai_chunk, dtype=float
    ).reshape(-1)
    _scai_finite_mask = _scai_expression_np.isfinite(_scai_values)
    _scai_finite_count = int(
        _scai_expression_np.count_nonzero(_scai_finite_mask)
    )
    if not _scai_finite_count:
        return 0, 0, 0, None, None
    if _scai_finite_count == int(_scai_values.size):
        _scai_finite_values = _scai_values
    else:
        _scai_finite_values = _scai_values[_scai_finite_mask]
    _scai_min = float(_scai_finite_values.min())
    _scai_max = float(_scai_finite_values.max())
    _scai_zero_mask = _scai_expression_np.isclose(
        _scai_finite_values, 0.0
    )
    _scai_nonzero_count = _scai_finite_count - int(
        _scai_expression_np.count_nonzero(_scai_zero_mask)
    )
    if not _scai_nonzero_count:
        return _scai_finite_count, 0, 0, _scai_min, _scai_max
    if _scai_nonzero_count == _scai_finite_count:
        _scai_nonzero = _scai_finite_values
    else:
        _scai_expression_np.logical_not(
            _scai_zero_mask, out=_scai_zero_mask
        )
        _scai_nonzero = _scai_finite_values[_scai_zero_mask]
    _scai_integer_count = int(
        _scai_expression_np.count_nonzero(
            _scai_expression_np.isclose(
                _scai_nonzero,
                _scai_expression_np.rint(_scai_nonzero),
                atol=1e-7,
            )
        )
    )
    return (
        _scai_finite_count,
        _scai_nonzero_count,
        _scai_integer_count,
        _scai_min,
        _scai_max,
    )

def _scai_accumulate_expression_stats(_scai_stats, _scai_values):
    # Inspect every element through fixed-size NumPy iterator buffers.
    _scai_array = _scai_expression_np.asarray(_scai_values)
    for _scai_chunk in _scai_expression_np.nditer(
        _scai_array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=_scai_expression_chunk_size,
    ):
        (
            _scai_finite_count,
            _scai_nonzero_count,
            _scai_integer_count,
            _scai_min,
            _scai_max,
        ) = _scai_expression_chunk_stats(_scai_chunk)
        if not _scai_finite_count:
            continue
        _scai_stats["finite_count"] += _scai_finite_count
        _scai_stats["nonzero_count"] += _scai_nonzero_count
        _scai_stats["integer_count"] += _scai_integer_count
        if (
            _scai_stats["min"] is None
            or _scai_min < _scai_stats["min"]
        ):
            _scai_stats["min"] = _scai_min
        if (
            _scai_stats["max"] is None
            or _scai_max > _scai_stats["max"]
        ):
            _scai_stats["max"] = _scai_max
        if _scai_integer_count:
            _scai_stats["has_positive_integer"] = True
        if _scai_integer_count < _scai_nonzero_count:
            _scai_stats["has_continuous"] = True

def _scai_accumulate_pandas_series(_scai_stats, _scai_series):
    # Convert extension dtypes to float one bounded slice at a time.
    _scai_before_nonzero = _scai_stats["nonzero_count"]
    _scai_before_integer = _scai_stats["integer_count"]
    _scai_length = int(len(_scai_series))
    for _scai_start in range(0, _scai_length, _scai_expression_chunk_size):
        _scai_stop = min(
            _scai_start + _scai_expression_chunk_size, _scai_length
        )
        if hasattr(_scai_series, "iloc"):
            _scai_piece = _scai_series.iloc[_scai_start:_scai_stop]
        else:
            _scai_piece = _scai_series[_scai_start:_scai_stop]
        try:
            _scai_piece_values = _scai_piece.to_numpy(
                dtype=float, na_value=_scai_expression_np.nan
            )
        except TypeError:
            _scai_piece_values = _scai_piece.to_numpy(dtype=float)
        _scai_accumulate_expression_stats(_scai_stats, _scai_piece_values)
    _scai_column_nonzero = (
        _scai_stats["nonzero_count"] - _scai_before_nonzero
    )
    _scai_column_integer = (
        _scai_stats["integer_count"] - _scai_before_integer
    )
    return bool(
        _scai_column_nonzero
        and _scai_column_integer == _scai_column_nonzero
    )

def _scai_expression_real_values(_scai_matrix, _scai_label, _scai_layer):
    _scai_module = type(_scai_matrix).__module__.split(".", 1)[0]
    _scai_stats = _scai_new_expression_stats()
    _scai_has_declared_integer = False
    _scai_has_integer_valued_component = False
    if _scai_module == "pandas":
        import pandas as _scai_expression_pd
        _scai_dtypes_obj = getattr(_scai_matrix, "dtypes", None)
        if _scai_dtypes_obj is None:
            _scai_dtypes = [getattr(_scai_matrix, "dtype", None)]
        else:
            try:
                _scai_dtypes = list(_scai_dtypes_obj)
            except TypeError:
                _scai_dtypes = [_scai_dtypes_obj]
        _scai_supported = bool(_scai_dtypes) and all(
            _scai_expression_pd.api.types.is_numeric_dtype(_scai_dtype)
            and not _scai_expression_pd.api.types.is_bool_dtype(_scai_dtype)
            and not _scai_expression_pd.api.types.is_complex_dtype(_scai_dtype)
            for _scai_dtype in _scai_dtypes
        )
        if not _scai_supported:
            _scai_unsupported_classes = []
            for _scai_dtype in _scai_dtypes:
                if _scai_expression_pd.api.types.is_bool_dtype(_scai_dtype):
                    _scai_class = "boolean"
                elif _scai_expression_pd.api.types.is_complex_dtype(_scai_dtype):
                    _scai_class = "complex"
                elif _scai_expression_pd.api.types.is_datetime64_any_dtype(_scai_dtype):
                    _scai_class = "datetime"
                elif _scai_expression_pd.api.types.is_timedelta64_dtype(_scai_dtype):
                    _scai_class = "timedelta"
                elif _scai_expression_pd.api.types.is_object_dtype(_scai_dtype):
                    _scai_class = "object"
                elif _scai_expression_pd.api.types.is_string_dtype(_scai_dtype):
                    _scai_class = "string"
                else:
                    _scai_class = "unsupported"
                _scai_unsupported_classes.append(_scai_class)
            _scai_expression_dtype_error(
                _scai_label,
                _scai_layer,
                ",".join(sorted(set(_scai_unsupported_classes))),
            )
        _scai_integer_dtypes = [
            _scai_expression_pd.api.types.is_integer_dtype(_scai_dtype)
            for _scai_dtype in _scai_dtypes
        ]
        _scai_has_declared_integer = any(_scai_integer_dtypes)
        _scai_dtype_class = (
            "integer" if all(_scai_integer_dtypes)
            else "mixed_numeric" if _scai_has_declared_integer
            else "floating"
        )
        # A DataFrame can hide one raw-count column among several continuous
        # columns.  A single aggregate dtype/fraction loses that information,
        # so retain an aggregate signal per column without persisting values.
        if getattr(_scai_matrix, "ndim", 1) > 1:
            for _scai_name, _scai_series in _scai_matrix.items():
                if _scai_accumulate_pandas_series(
                    _scai_stats, _scai_series
                ):
                    _scai_has_integer_valued_component = True
        else:
            _scai_has_integer_valued_component = (
                _scai_accumulate_pandas_series(
                    _scai_stats, _scai_matrix
                )
            )
    else:
        _scai_values = (
            _scai_matrix.data if hasattr(_scai_matrix, "tocsr") else _scai_matrix
        )
        _scai_values = _scai_expression_np.asarray(_scai_values)
        _scai_dtype = getattr(_scai_values, "dtype", None)
        _scai_dtype_kind = getattr(_scai_dtype, "kind", "")
        if _scai_dtype_kind not in "iuf":
            if _scai_dtype_kind == "b":
                _scai_unsupported_class = "boolean"
            elif _scai_dtype_kind == "O":
                _scai_unsupported_class = "object"
            elif _scai_dtype_kind in "SU":
                _scai_unsupported_class = "string"
            elif _scai_dtype_kind == "M":
                _scai_unsupported_class = "datetime"
            elif _scai_dtype_kind == "m":
                _scai_unsupported_class = "timedelta"
            elif _scai_dtype_kind == "c":
                _scai_unsupported_class = "complex"
            else:
                _scai_unsupported_class = "unsupported"
            _scai_expression_dtype_error(
                _scai_label, _scai_layer, _scai_unsupported_class
            )
        _scai_dtype_class = (
            "integer" if _scai_dtype_kind in "iu" else "floating"
        )
        _scai_has_declared_integer = _scai_dtype_class == "integer"
        _scai_accumulate_expression_stats(_scai_stats, _scai_values)
    return (
        _scai_stats,
        _scai_dtype_class,
        _scai_has_declared_integer,
        _scai_has_integer_valued_component,
    )

def _scai_raise_expression_scale_error(_scai_invalid):
    raise ValueError(
        "Expression contract violation: DE, gene scoring, and NMF require an explicitly "
        "normalized continuous matrix; raw integer counts or centered/scaled values detected in "
        + ", ".join(
            str(_scai_contract.get("name", "runtime input"))
            for _scai_contract in _scai_invalid
        )
        + ". Normalize a copy (for example CP10K + log1p) and rerun."
    )

def _scai_validate_expression_contract(
    _scai_obj, _scai_label, _scai_layer=None, emit=True
):
    if hasattr(_scai_obj, "X"):
        _scai_matrix = (
            _scai_obj.layers[_scai_layer] if _scai_layer else _scai_obj.X
        )
        _scai_shape = list(_scai_obj.shape)
    else:
        _scai_matrix = _scai_obj
        _scai_shape = list(getattr(_scai_matrix, "shape", ()))
    (
        _scai_stats,
        _scai_dtype_class,
        _scai_has_declared_integer,
        _scai_has_integer_valued_component,
    ) = _scai_expression_real_values(
        _scai_matrix, _scai_label, _scai_layer
    )
    _scai_finite_count = int(_scai_stats["finite_count"])
    _scai_nonzero_count = int(_scai_stats["nonzero_count"])
    _scai_integer_fraction = (
        float(_scai_stats["integer_count"]) / _scai_nonzero_count
        if _scai_nonzero_count else 1.0
    )
    _scai_min = (
        float(_scai_stats["min"]) if _scai_finite_count else 0.0
    )
    _scai_has_positive_integer = bool(
        _scai_stats["has_positive_integer"]
    )
    _scai_has_continuous = bool(_scai_stats["has_continuous"])
    if not _scai_finite_count or not _scai_nonzero_count:
        _scai_scale = "insufficient_finite_continuous_evidence"
    elif _scai_min < 0:
        _scai_scale = "centered_or_scaled"
    elif _scai_has_declared_integer or _scai_integer_fraction >= 0.99:
        _scai_scale = "raw_counts"
    elif (
        _scai_has_continuous
        and _scai_has_integer_valued_component
    ):
        _scai_scale = "mixed_raw_and_continuous"
    else:
        _scai_scale = "normalized_continuous"
    _scai_contract = {{
        "name": str(_scai_label),
        "layer": _scai_layer,
        "shape": _scai_shape,
        "scale": _scai_scale,
        "dtype_class": _scai_dtype_class,
    }}
    if _scai_scale != "normalized_continuous":
        _scai_raise_expression_scale_error([_scai_contract])
    if emit:
        print(
            {_MARKER!r}
            + _scai_expression_json.dumps([_scai_contract], sort_keys=True)
        )
    return _scai_contract

def _scai_expression_callable_kind(_scai_callable, _scai_hint=""):
    _scai_sensitive_names = {{
        "rank_genes_groups",
        "score_genes",
        "score_genes_cell_cycle",
        "non_negative_factorization",
    }}
    _scai_seen = set()
    _scai_candidates = [_scai_callable]
    while _scai_candidates:
        _scai_candidate = _scai_candidates.pop()
        if id(_scai_candidate) in _scai_seen:
            continue
        _scai_seen.add(id(_scai_candidate))
        _scai_name = str(getattr(_scai_candidate, "__name__", "") or "")
        _scai_module = str(getattr(_scai_candidate, "__module__", "") or "")
        # A user helper may legitimately share a short name such as
        # ``score_genes``.  A name by itself is not authority to classify it as
        # a Scanpy/sklearn operation; imported library callables carry their
        # defining module, while aliases and monkeypatches are resolved by API
        # slot identity below.
        if (
            _scai_name in _scai_sensitive_names
            and (
                _scai_module == "scanpy"
                or _scai_module.startswith("scanpy.")
                or _scai_module == "sklearn"
                or _scai_module.startswith("sklearn.")
            )
        ):
            return _scai_name
        _scai_owner = getattr(_scai_candidate, "__self__", None)
        _scai_qualname = str(
            getattr(_scai_candidate, "__qualname__", "") or ""
        )
        if _scai_name == "fit_transform" and (
            type(_scai_owner).__name__ == "NMF"
            or ".NMF.fit_transform" in ("." + _scai_qualname)
        ):
            return "non_negative_factorization"
        for _scai_link in ("__func__", "__wrapped__", "func"):
            _scai_inner = getattr(_scai_candidate, _scai_link, None)
            if callable(_scai_inner):
                _scai_candidates.append(_scai_inner)

    # Resolve aliases by runtime object identity, not by assignment syntax.
    # This also covers test doubles whose callable is a nameless lambda stored
    # at the same API slot as the real scanpy function.
    for _scai_root_name in ("sc", "scanpy"):
        _scai_root = globals().get(_scai_root_name)
        _scai_tl = getattr(_scai_root, "tl", None)
        for _scai_name in _scai_sensitive_names:
            _scai_known = getattr(_scai_tl, _scai_name, None)
            if _scai_known is _scai_callable:
                return _scai_name
    return None

def {_RUNTIME_CALL_HELPER}(
    _scai_callable, _scai_call_hint, *args, **kwargs
):
    _scai_call_name = _scai_expression_callable_kind(
        _scai_callable, _scai_call_hint
    )
    if _scai_call_name is None:
        return _scai_callable(*args, **kwargs)
    if args:
        _scai_obj = args[0]
    else:
        _scai_obj = None
        for _scai_key in ("adata", "X", "data"):
            if _scai_key in kwargs:
                _scai_obj = kwargs[_scai_key]
                break
    if _scai_obj is None:
        raise ValueError(
            "Expression contract could not resolve the runtime input for "
            + str(_scai_call_name) + "."
        )
    _scai_validate_expression_contract(
        _scai_obj, _scai_call_name, kwargs.get("layer"), emit=True
    )
    return _scai_callable(*args, **kwargs)

def {_RUNTIME_CALL_RESOLVER}(_scai_callable):
    # Ordinary callables must run in the user's frame, especially namespace
    # builtins reached through aliases. Keep the existing dispatcher for
    # sensitive operations and for previously instrumented runtime functions.
    if _scai_expression_callable_kind(_scai_callable) is None:
        return _scai_callable

    def _scai_checked_call(*args, **kwargs):
        return {_RUNTIME_CALL_HELPER}(_scai_callable, None, *args, **kwargs)

    return _scai_checked_call
"""


def build_expression_runtime_code() -> str:
    """Return trusted helpers that validate the actual sensitive-call input."""
    return _EXPRESSION_RUNTIME_CODE
