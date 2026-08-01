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
