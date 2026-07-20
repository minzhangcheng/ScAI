"""Runtime scale guard and provenance tracking for expression-based analyses."""

from __future__ import annotations

import ast
import json
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
_MARKER = "__SCAI_EXPRESSION_CONTRACT__:"


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
    return f"""
import json as _scai_json
import numpy as _scai_np
_scai_specs = _scai_json.loads({payload!r})
_scai_contracts = []
for _scai_spec in _scai_specs:
    _scai_name = _scai_spec['name']
    if _scai_name not in globals():
        continue
    _scai_obj = globals()[_scai_name]
    _scai_layer = _scai_spec.get('layer')
    if hasattr(_scai_obj, 'X'):
        _scai_matrix = _scai_obj.layers[_scai_layer] if _scai_layer else _scai_obj.X
        _scai_shape = list(_scai_obj.shape)
    else:
        _scai_matrix = _scai_obj
        _scai_shape = list(getattr(_scai_matrix, 'shape', ()))
    _scai_values = _scai_matrix.data if hasattr(_scai_matrix, 'tocsr') else _scai_matrix
    _scai_values = _scai_np.asarray(_scai_values).reshape(-1)
    if _scai_values.size > 10000:
        _scai_values = _scai_values[_scai_np.linspace(0, _scai_values.size - 1, 10000, dtype=int)]
    _scai_values = _scai_values[_scai_np.isfinite(_scai_values)]
    _scai_nonzero = _scai_values[~_scai_np.isclose(_scai_values, 0.0)]
    _scai_scale_values = _scai_nonzero if _scai_nonzero.size else _scai_values
    _scai_min = float(_scai_values.min()) if _scai_values.size else 0.0
    _scai_max = float(_scai_values.max()) if _scai_values.size else 0.0
    _scai_integer_fraction = float(
        _scai_np.mean(_scai_np.isclose(_scai_scale_values, _scai_np.rint(_scai_scale_values), atol=1e-7))
    ) if _scai_scale_values.size else 1.0
    if _scai_min >= 0 and _scai_max > 1 and _scai_integer_fraction >= 0.99:
        _scai_scale = 'raw_counts'
    elif _scai_min < 0:
        _scai_scale = 'centered_or_scaled'
    else:
        _scai_scale = 'normalized_continuous'
    _scai_contracts.append({{
        'name': _scai_name, 'layer': _scai_layer, 'shape': _scai_shape,
        'scale': _scai_scale, 'min': _scai_min, 'max': _scai_max,
    }})
print({_MARKER!r} + _scai_json.dumps(_scai_contracts, sort_keys=True))
if not _scai_contracts:
    raise ValueError(
        'Expression contract could not resolve the analysis input in the current runtime. '
        'Create and normalize the input in a prior execute_code step, declare it as a '
        'persistent output, then run DE, gene scoring, or NMF.'
    )
_scai_invalid = [c for c in _scai_contracts if c['scale'] != 'normalized_continuous']
if _scai_invalid:
    raise ValueError(
        'Expression contract violation: DE, gene scoring, and NMF require an explicitly '
        'normalized continuous matrix; raw integer counts or centered/scaled values detected in '
        + ', '.join(c['name'] for c in _scai_invalid)
        + '. Normalize a copy (for example CP10K + log1p) and rerun.'
    )
"""


def parse_expression_contracts(output: str | None) -> list[dict[str, Any]]:
    for line in reversed((output or "").splitlines()):
        if line.startswith(_MARKER):
            try:
                value = json.loads(line[len(_MARKER):])
                return value if isinstance(value, list) else []
            except (TypeError, json.JSONDecodeError):
                return []
    return []


def mutates_expression_representation(code: str) -> bool:
    """Detect explicit expression normalization/scaling or matrix replacement."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _REPRESENTATION_MUTATORS:
                return True
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == "X":
                    return True
    return False


def prepares_expression_before_sensitive_call(code: str) -> bool:
    """Return whether an explicit representation mutation precedes analysis."""
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
    if not contracts:
        return
    sandbox_state = state.sandbox_state or {}
    sandbox_state["expression_contract_v1"] = {
        "round_number": int(state.current_round or 0),
        "analysis_generation": int(sandbox_state.get("analysis_generation", 0) or 0),
        "inputs": contracts,
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
        )
        + invalidate_current_artifacts(
            state.publication_tables,
            round_number=int(state.current_round or 0),
            reason=reason,
        )
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


def unchanged_artifact_stems(
    working_dir: str | Path, before: dict[str, int],
) -> set[str]:
    """Return prior artifacts not overwritten by the correcting execution."""
    after = workspace_artifact_versions(working_dir)
    return {
        stem for stem, version in before.items()
        if after.get(stem) == version
    }


def record_invalidated_file_stems(state: Any, stems: set[str]) -> None:
    if not stems:
        return
    state.invalidated_artifact_stems = sorted(
        set(state.invalidated_artifact_stems or []) | set(stems)
    )
