"""Runtime scale guard and provenance tracking for expression-based analyses."""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

from backend.services.report.artifacts import invalidate_current_artifacts


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


def _root_name(node: ast.AST | None) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def extract_sensitive_expression_inputs(code: str) -> list[dict[str, str | None]]:
    """Return global inputs used by DE, gene scoring, or NMF calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    aliases: dict[str, str] = {}
    nmf_instances: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            source = _root_name(value)
            if isinstance(value, ast.Call) and value.args:
                source = _root_name(value.args[0]) or source
            for target in targets:
                if isinstance(target, ast.Name) and source:
                    aliases[target.id] = aliases.get(source, source)
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Call)
                    and (
                        getattr(value.func, "id", None) == "NMF"
                        or getattr(value.func, "attr", None) == "NMF"
                    )
                ):
                    nmf_instances.add(target.id)

    specs: dict[tuple[str, str | None], dict[str, str | None]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        is_nmf_fit = (
            call_name == "fit_transform"
            and isinstance(node.func, ast.Attribute)
            and (
                getattr(node.func.value, "id", None) in nmf_instances
                or (
                    isinstance(node.func.value, ast.Call)
                    and (
                        getattr(node.func.value.func, "id", None) == "NMF"
                        or getattr(node.func.value.func, "attr", None) == "NMF"
                    )
                )
            )
        )
        if call_name not in _SENSITIVE_CALLS and not is_nmf_fit:
            continue
        input_node = node.args[0] if node.args else None
        if input_node is None:
            for keyword in node.keywords:
                if keyword.arg in {"adata", "X", "data"}:
                    input_node = keyword.value
                    break
        name = _root_name(input_node)
        if name:
            name = aliases.get(name, name)
        if not name:
            continue
        layer = None
        for keyword in node.keywords:
            if keyword.arg == "layer" and isinstance(keyword.value, ast.Constant):
                layer = str(keyword.value.value) if keyword.value.value is not None else None
        specs[(name, layer)] = {"name": name, "layer": layer}
    return list(specs.values())


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


def build_expression_runtime_code() -> str:
    """Return trusted helpers that validate the actual sensitive-call input."""
    return f"""
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
        and (
            _scai_has_integer_valued_component
            or _scai_has_positive_integer
        )
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
"""


def instrument_expression_contract_calls(code: str) -> tuple[str, int]:
    """Route calls through runtime callable classification exactly once.

    Alias syntax is intentionally irrelevant: every invocation expression is
    routed through the same lightweight dispatcher, which recognizes sensitive
    functions and bound NMF methods from the runtime object. Surgical source
    edits preserve comments and line numbers, and the original callable plus
    every argument are evaluated once and forwarded unchanged.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0

    # Most calls used to construct another call's callable
    # (``getattr(...)(...)`` or ``NMF().fit_transform(...)``) are covered by
    # the outer dispatcher.  A sensitive operation can itself be the callable
    # for a chained operation, though (``NMF().fit_transform(X).sum()``); that
    # inner operation must retain its own guard.  The insertion-only rewrite
    # below allows those nested wrappers to share a source start without
    # overlapping replacement ranges.
    callable_expression_calls: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callable_expression_calls.update(
                id(inner)
                for inner in ast.walk(node.func)
                if isinstance(inner, ast.Call)
            )

    source = code.encode("utf-8")
    line_offsets: list[int] = []
    offset = 0
    for line in code.splitlines(keepends=True):
        line_offsets.append(offset)
        offset += len(line.encode("utf-8"))

    def absolute(lineno: int, col: int) -> int:
        return line_offsets[lineno - 1] + col

    edits: list[tuple[int, int, bytes]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        raw_name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if (
            id(node) in callable_expression_calls
            and raw_name not in _SENSITIVE_CALLS
            and raw_name != "fit_transform"
        ):
            continue
        if raw_name == _RUNTIME_CALL_HELPER:
            continue
        start = absolute(node.func.lineno, node.func.col_offset)
        func_end = absolute(node.func.end_lineno, node.func.end_col_offset)
        call_end = absolute(node.end_lineno, node.end_col_offset)
        opening = source.find(b"(", func_end, call_end)
        if opening < 0:
            continue
        edits.append((start, start, _RUNTIME_CALL_HELPER.encode("ascii") + b"("))
        edits.append((opening, opening + 1, b", '', "))

    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source.decode("utf-8"), len(edits) // 2


def has_direct_expression_contract_call(code: str) -> bool:
    """Return whether trusted helpers can be installed before the user cell.

    This is only a round-trip optimization and never decides enforcement: all
    invocation expressions are still dispatched at runtime. A syntactically
    direct Scanpy call lets callers distinguish helper-setup failure from user
    execution failure; aliases and dynamic callables receive the same helper
    inline with their user cell.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _SENSITIVE_CALLS
        and _root_name(node.func) in {"sc", "scanpy"}
        for node in ast.walk(tree)
    )


def strip_expression_contract_markers(output: str | None) -> str | None:
    """Remove trusted provenance marker lines from user-visible output."""
    if output is None:
        return None
    return "\n".join(
        line for line in output.splitlines() if not line.startswith(_MARKER)
    )


def parse_expression_contracts(output: str | None) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for line in (output or "").splitlines():
        if line.startswith(_MARKER):
            try:
                value = json.loads(line[len(_MARKER):])
                if isinstance(value, list):
                    contracts.extend(
                        item for item in value if isinstance(item, dict)
                    )
            except (TypeError, json.JSONDecodeError):
                continue
    return contracts


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


def _call_module_root(node: ast.Call) -> str | None:
    """Leftmost module name of a dotted call, or None for a bare call."""
    if isinstance(node.func, ast.Attribute):
        return _root_name(node.func.value)
    return None


def _mutation_target_root(node: ast.Call) -> str | None:
    """Root variable an in-place expression mutator operates on."""
    arg = node.args[0] if node.args else None
    if arg is None:
        for keyword in node.keywords:
            if keyword.arg in {"adata", "data", "X"}:
                arg = keyword.value
                break
    return _root_name(arg)


def _value_makes_fresh_object(value: ast.AST | None, local: set[str]) -> bool:
    """Whether an assignment RHS binds a throwaway object distinct from the
    canonical AnnData: an explicit ``.copy()``/reader/constructor call, an alias
    of an already-local object, or a subset of one."""
    if isinstance(value, ast.Call):
        fn = getattr(value.func, "attr", None) or getattr(value.func, "id", None)
        return fn in _FRESH_OBJECT_CALLS
    if isinstance(value, ast.Name):
        return value.id in local
    if isinstance(value, ast.Subscript):
        return _root_name(value) in local
    return False


def mutates_expression_representation(code: str) -> bool:
    """Detect normalization/scaling or matrix replacement of the *canonical*
    expression matrix.

    Only mutations of the shared AnnData object invalidate previously generated
    figures. Normalizing or replacing the matrix of a throwaway copy created
    within the same execution (for example ``work = adata.copy();
    sc.pp.log1p(work)``) leaves the canonical representation -- and every figure
    derived from it -- intact, so it must not be treated as a representation
    change. Small models routinely re-normalize such copies in nearly every
    step; flagging those would invalidate the entire Results gallery.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Collect local-object assignments and mutation events, then replay them in
    # source order so a copy created earlier in the block shields its own later
    # mutations. Tie-break assignments (0) before mutations (1) at one position.
    events: list[tuple[tuple[int, int], int, str, Any]] = []
    for node in ast.walk(tree):
        pos = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        # Only plain assignment binds a name to local-copy status here. Other
        # binding forms -- walrus ``(w := adata.copy())``, ``for w in ...``,
        # ``with ... as w``, and tuple/list unpacking targets -- are NOT tracked,
        # so a copy bound that way is later seen as the canonical object and its
        # mutation is flagged. These forms are vanishingly rare in generated
        # scanpy code, and the miss is a safe false positive (an extra figure
        # refresh), never a stale-figure false negative.
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if names and node.value is not None:
                events.append((pos, 0, "assign", (names, node.value)))
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == "X":
                    events.append((pos, 1, "mutate", _root_name(target)))
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Attribute) and node.target.attr == "X":
                events.append((pos, 1, "mutate", _root_name(node.target)))
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if (
                name in _REPRESENTATION_MUTATORS
                and _call_module_root(node) not in _NON_EXPRESSION_CALL_ROOTS
            ):
                events.append((pos, 1, "mutate", _mutation_target_root(node)))

    local: set[str] = set()
    for _pos, _tie, kind, data in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == "assign":
            names, value = data
            if _value_makes_fresh_object(value, local):
                local.update(names)
            else:
                # Rebinding a name to the canonical object (or an unknown source)
                # drops its local-copy status.
                local.difference_update(names)
        elif data is None or data not in local:
            return True
    return False


def prepares_expression_before_sensitive_call(code: str) -> bool:
    """Return whether an explicit representation mutation precedes analysis.

    Intentionally asymmetric with ``mutates_expression_representation``: this
    function does NOT distinguish canonical mutations from mutations of a local
    copy. The question it answers -- "was the expression prepared before a
    sensitive call?" -- is answered "yes" even when the preparation happens on a
    copy, so the naive any-mutation scan is correct for its purpose. Its result
    is informational/heuristic and does not drive figure invalidation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    mutation_lines: list[int] = []
    sensitive_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _REPRESENTATION_MUTATORS:
                mutation_lines.append(node.lineno)
            if name in _SENSITIVE_CALLS or name == "fit_transform":
                sensitive_lines.append(node.lineno)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Attribute) and t.attr == "X" for t in targets):
                mutation_lines.append(node.lineno)
    return bool(
        mutation_lines and sensitive_lines
        and min(mutation_lines) < min(sensitive_lines)
    )


def record_expression_contract(state: Any, contracts: list[dict[str, Any]]) -> None:
    # Contract markers are emitted into the kernel stream before the agent can
    # strip them from its tool response, so their payload is also present in a
    # successfully persisted notebook cell.  Keep only provenance metadata;
    # raw extrema (emitted by the first implementation) are neither required
    # for later policy decisions nor appropriate for a durable execution log.
    safe_contracts = [
        {
            key: contract.get(key)
            for key in ("name", "layer", "shape", "scale", "dtype_class")
            if key in contract
        }
        for contract in contracts
        if isinstance(contract, dict)
    ]
    if not safe_contracts:
        return
    sandbox_state = state.sandbox_state or {}
    sandbox_state["expression_contract_v1"] = {
        "round_number": int(state.current_round or 0),
        "analysis_generation": int(sandbox_state.get("analysis_generation", 0) or 0),
        "inputs": safe_contracts,
    }
    state.sandbox_state = sandbox_state


def record_representation_change(state: Any, code: str) -> int:
    """Bump provenance generation and invalidate same-round derived artifacts."""
    if not mutates_expression_representation(code):
        return 0
    sandbox_state = state.sandbox_state or {}
    generation = int(sandbox_state.get("analysis_generation", 0) or 0) + 1
    sandbox_state["analysis_generation"] = generation
    sandbox_state.pop("expression_contract_v1", None)
    state.sandbox_state = sandbox_state
    reason = (
        f"Expression representation changed in round {state.current_round}; "
        f"derived artifact generation {generation - 1} is stale."
    )
    return (
        invalidate_current_artifacts(
            state.publication_figures,
            round_number=int(state.current_round or 0),
            reason=reason,
            lineage_id=(getattr(state, "restore_branch_id", None) or "main"),
        )
        + invalidate_current_artifacts(
            state.publication_tables,
            round_number=int(state.current_round or 0),
            reason=reason,
        )
    )


def record_executed_representation_change(
    state: Any,
    code: str,
    working_dir: str | Path,
    before: dict[str, int],
    *,
    execution_started: bool,
) -> tuple[int, set[str], str]:
    """Invalidate derived artifacts after a dispatched, non-transactional cell.

    A Jupyter cell can mutate ``adata.X`` and then fail.  A failed result is
    therefore not evidence that the canonical representation stayed intact.
    Once the user's cell has actually been dispatched, conservatively apply the
    same generation and file-stem invalidation used for successful cells.  The
    explicit flag keeps failures in trusted setup helpers from invalidating
    anything when the user's cell never ran.
    """
    if not execution_started or not mutates_expression_representation(code):
        return 0, set(), ""
    invalidated_count = record_representation_change(state, code)
    stale_stems = unchanged_artifact_stems(
        working_dir,
        before,
        since_ns=round_start_ns(state),
    )
    record_invalidated_file_stems(state, stale_stems)
    return (
        invalidated_count,
        stale_stems,
        describe_invalidated_artifacts(stale_stems, invalidated_count),
    )


def workspace_artifact_versions(working_dir: str | Path) -> dict[str, int]:
    """Snapshot derived-file mtimes before a representation change."""
    root = Path(working_dir)
    versions: dict[str, int] = {}
    for directory in (root, root / "figures", root / "tables"):
        if not directory.exists():
            continue
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.pdf", "*.csv"):
            for path in directory.glob(suffix):
                versions[path.stem] = max(
                    versions.get(path.stem, 0), path.stat().st_mtime_ns
                )
    return versions


#: Where ``record_round_start`` parks the stamp. Lives in ``sandbox_state``
#: rather than on a dedicated field so the mechanism needs no schema change and
#: travels with the rest of this module's provenance record.
ROUND_START_KEY = "round_started_ns_v1"


def record_round_start(state: Any) -> int:
    """Stamp the wall-clock start of the round in progress and return it.

    Idempotent per round: both stages of a round re-enter the node that calls
    this, and a stage-2 representation change invalidates stage-1 artifacts too,
    so the round keeps the stamp it was given when it opened.
    """
    sandbox_state = state.sandbox_state or {}
    current = int(getattr(state, "current_round", 0) or 0)
    stamp = sandbox_state.get(ROUND_START_KEY)
    if isinstance(stamp, dict) and int(stamp.get("round", -1) or -1) == current:
        started = int(stamp.get("started_ns", 0) or 0)
        if started > 0:
            return started
    started = time.time_ns()
    sandbox_state[ROUND_START_KEY] = {"round": current, "started_ns": started}
    state.sandbox_state = sandbox_state
    return started


def round_start_ns(state: Any) -> int:
    """Wall-clock start of the round in progress, in ns.

    Read from the stamp ``record_round_start`` writes when the round opens. The
    analysis log cannot answer this while the round is running and never could:
    the bioinformatics agent keeps its steps in a local list and they are merged
    into ``state.analysis_log`` -- and stamped with ``round_number`` -- only
    after the round's execution has finished. So on every main pass the
    current-round lookup found nothing and the function returned
    ``time.time_ns()``, a bound nothing predates, which made
    ``unchanged_artifact_stems`` return the empty set: no figure was ever
    withheld, while ``describe_invalidated_artifacts`` still told the agent that
    N of them had been.

    The remaining fallbacks are ordered so an unknown start includes artifacts
    rather than excluding them. Withholding too much is visible to the user and
    the agent can regenerate; withholding nothing is silent and ships a figure
    that no longer matches its data.
    """
    current = int(getattr(state, "current_round", 0) or 0)
    sandbox_state = getattr(state, "sandbox_state", None) or {}
    stamp = sandbox_state.get(ROUND_START_KEY) if isinstance(sandbox_state, dict) else None
    if isinstance(stamp, dict) and int(stamp.get("round", -1) or -1) == current:
        started = int(stamp.get("started_ns", 0) or 0)
        if started > 0:
            return started

    steps = list(getattr(state, "analysis_log", None) or [])
    starts = [
        step.timestamp
        for step in steps
        if int(getattr(step, "round_number", 0) or 0) == current
        and getattr(step, "timestamp", None) is not None
    ]
    if starts:
        return int(min(starts).timestamp() * 1_000_000_000)

    # No stamp (a session that predates it, or a node that never opened the
    # round) and nothing logged for this round yet. The last step of an earlier
    # round is the boundary this one started after: it still excludes finished
    # rounds' artifacts, which is what the round scoping exists for.
    ends = [
        step.timestamp
        for step in steps
        if 0 < int(getattr(step, "round_number", 0) or 0) < current
        and getattr(step, "timestamp", None) is not None
    ]
    if ends:
        return int(max(ends).timestamp() * 1_000_000_000)

    # Nothing to bound with at all -- the first round of a session, where every
    # artifact on disk does belong to the round in progress.
    return 0


def unchanged_artifact_stems(
    working_dir: str | Path, before: dict[str, int], since_ns: int = 0,
) -> set[str]:
    """Return prior artifacts not overwritten by the correcting execution.

    ``since_ns`` bounds the answer to artifacts produced during the round that is
    mutating, matching the rule ``invalidate_current_artifacts`` already applies
    to the model-level record (``artifact_round not in (0, round_number)``). The
    file-stem path never got that scoping, so one mutating cell in round N
    blacklisted every figure from rounds 1..N-1 for the rest of the session --
    and since the set is a monotonic union with a single narrow subtraction, the
    only way back out was regenerating that exact figure. Measured across 27 real
    sessions it removed 76% of all gallery candidates.

    Defaulted rather than required: this function is re-exported by nine
    ``backend.agents.bioinformatics`` modules.
    """
    after = workspace_artifact_versions(working_dir)
    return {
        stem for stem, version in before.items()
        if after.get(stem) == version and version >= since_ns
    }


_MAX_NAMED_STALE_ARTIFACTS = 12


def describe_invalidated_artifacts(stems: set[str], model_count: int) -> str:
    """One line telling the agent which of its own artifacts just went stale.

    Invalidation was silent: seven consumers read the record as "hide this from
    the report" and none as "regenerate this", and nothing in any prompt, tool
    result, or context block ever mentioned it. The producer could therefore
    never repair the deficit it had just caused. This is that missing signal --
    a statement of fact routed back to the party able to act on it, not an
    instruction, since whether a figure is still worth having is the agent's
    call.
    """
    if not stems and model_count <= 0:
        return ""
    names = sorted(stems)
    shown = ", ".join(names[:_MAX_NAMED_STALE_ARTIFACTS])
    if len(names) > _MAX_NAMED_STALE_ARTIFACTS:
        shown += f", and {len(names) - _MAX_NAMED_STALE_ARTIFACTS} more"
    total = len(names) or model_count
    return (
        f"[Provenance] This step changed the expression representation, so {total} "
        f"artifact(s) produced earlier in this round no longer match the data they "
        f"describe and have been withheld from the report"
        + (f": {shown}." if shown else ".")
        + " Regenerate any of them you want reported."
    )


def record_invalidated_file_stems(state: Any, stems: set[str]) -> None:
    if not stems:
        return
    state.invalidated_artifact_stems = sorted(
        set(state.invalidated_artifact_stems or []) | set(stems)
    )
