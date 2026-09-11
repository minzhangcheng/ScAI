"""Technical integrity boundary for the accepted Stage 1 AnnData object.

The durable runtime remains the only persistence mechanism.  This module stores
only a host-side fingerprint and a small cross-cell binding ledger in
``AnalysisState.sandbox_state``; it never writes or restores a second AnnData.
"""
from __future__ import annotations

import ast
import math
from typing import Any


FOUNDATION_KEY = "stage1_foundation_v1"
_BINDINGS_KEY = "stage1_foundation_bindings_v1"

_REQUIRED_SIGNATURE_KEYS = {
    "n_obs", "n_vars", "obs_index_hash", "var_hash", "x_sum", "x_nnz",
    "x_fingerprint", "obs_fingerprints", "raw_fingerprint",
}

_INPLACE_METHODS = {
    "obs_names_make_unique", "var_names_make_unique", "_inplace_subset_obs",
    "_inplace_subset_var",
}
_FOUNDATION_MUTATORS = {
    "filter_cells", "filter_genes", "normalize_total", "log1p", "scale",
    "regress_out", "highly_variable_genes", "pca", "neighbors", "umap",
    "leiden", "louvain", "scrublet", "combat", "harmony_integrate",
}
_KEYED_CONTAINERS = {"obs", "var", "layers", "obsm", "varm", "obsp", "uns"}


def _sandbox_state(state: Any) -> dict[str, Any]:
    value = getattr(state, "sandbox_state", None) or {}
    state.sandbox_state = value
    return value


def foundation_manifest(state: Any) -> dict[str, Any] | None:
    value = (getattr(state, "sandbox_state", None) or {}).get(FOUNDATION_KEY)
    return value if isinstance(value, dict) and value.get("status") == "sealed" else None


def foundation_is_sealed(state: Any) -> bool:
    return foundation_manifest(state) is not None


def seal_foundation(state: Any, signature: dict[str, Any], *, canonical_name: str = "adata") -> str | None:
    """Seal the live canonical object, returning an objective error on failure."""
    if not isinstance(signature, dict) or not signature.get("id"):
        return f"Stage 1 cannot be sealed: live canonical AnnData '{canonical_name}' is unavailable."
    missing = sorted(_REQUIRED_SIGNATURE_KEYS.difference(signature))
    if missing:
        return "Stage 1 cannot be sealed: canonical fingerprint is incomplete (" + ", ".join(missing) + ")."
    store = _sandbox_state(state)
    store[FOUNDATION_KEY] = {
        "status": "sealed",
        "canonical_name": canonical_name,
        "round_number": int(getattr(state, "current_round", 0) or 0),
        "signature": signature,
    }
    store[_BINDINGS_KEY] = {"derived": [], "canonical_aliases": [], "component_aliases": []}
    return None


def registered_derived_names(state: Any) -> set[str]:
    ledger = (getattr(state, "sandbox_state", None) or {}).get(_BINDINGS_KEY) or {}
    return {str(name) for name in ledger.get("derived", []) if str(name).isidentifier()}


def _root_name(node: ast.AST | None) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _bound_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _copy_root(value: ast.AST | None) -> str | None:
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "copy"
    ):
        return _root_name(value.func.value)
    return None


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _container_write(node: ast.AST) -> tuple[str | None, str | None, str | None]:
    """Return root, keyed container and literal key for an assignment target."""
    if not isinstance(node, ast.Subscript):
        return _root_name(node), None, None
    base = node.value
    if isinstance(base, ast.Attribute) and base.attr in _KEYED_CONTAINERS:
        return _root_name(base), base.attr, _literal_key(node.slice)
    return _root_name(base), None, None


def _baseline_keys(signature: dict[str, Any], container: str) -> set[str]:
    mapping = {
        "obs": "obs_fingerprints", "var": "var_fingerprints",
        "layers": "layers_fingerprint", "obsm": "obsm_fingerprints",
        "varm": "varm_fingerprints", "obsp": "obsp_fingerprints",
        "uns": "uns_fingerprints",
    }
    value = signature.get(mapping[container], {})
    return set(value) if isinstance(value, dict) else set()


def _function_scope_locals(tree: ast.AST) -> dict[int, set[str]]:
    """Map nested-scope nodes to names that cannot resolve to kernel globals."""
    scoped: dict[int, set[str]] = {}

    class _LocalCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()
            self.globals: set[str] = set()

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                self.names.add(node.id)

        def visit_Global(self, node: ast.Global) -> None:
            self.globals.update(node.names)

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            return

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.names.add(node.name)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.names.add(node.name)

    def _arguments(node: ast.arguments) -> set[str]:
        values = {
            arg.arg
            for arg in (*node.posonlyargs, *node.args, *node.kwonlyargs)
        }
        if node.vararg:
            values.add(node.vararg.arg)
        if node.kwarg:
            values.add(node.kwarg.arg)
        return values

    def _walk(node: ast.AST, inherited: set[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            collector = _LocalCollector()
            body = node.body if not isinstance(node, ast.Lambda) else [node.body]
            for child in body:
                collector.visit(child)
            locals_here = (
                collector.names | _arguments(node.args)
            ) - collector.globals
            inherited = (inherited | locals_here) - collector.globals
        scoped[id(node)] = inherited
        for child in ast.iter_child_nodes(node):
            _walk(child, inherited)

    _walk(tree, set())
    return scoped


def foundation_write_guard(code: str, state: Any) -> str | None:
    """Reject statically visible writes to a sealed canonical foundation."""
    manifest = foundation_manifest(state)
    if manifest is None:
        return None
    if manifest.get("integrity_error"):
        return str(manifest["integrity_error"])
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    canonical = str(manifest.get("canonical_name") or "adata")
    ledger = (getattr(state, "sandbox_state", None) or {}).get(_BINDINGS_KEY) or {}
    canonical_names = {canonical, *ledger.get("canonical_aliases", [])}
    derived = set(ledger.get("derived", []))
    component_aliases = set(ledger.get("component_aliases", []))
    signature = manifest.get("signature") or {}
    scoped_locals = _function_scope_locals(tree)

    events = sorted(
        ast.walk(tree),
        key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
    )
    for node in events:
        local_names = scoped_locals.get(id(node), set())
        active_canonical_names = canonical_names - local_names
        active_component_aliases = component_aliases - local_names
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name):
                    if target.id in active_canonical_names:
                        return f"FoundationIntegrityError: rebinding sealed canonical object '{target.id}' is blocked."
                else:
                    root, container, key = _container_write(target)
                    if root in active_component_aliases:
                        return f"FoundationIntegrityError: mutation through foundation component alias '{root}' is blocked."
                    if root in active_canonical_names:
                        if container and key is not None and key not in _baseline_keys(signature, container):
                            continue
                        return "FoundationIntegrityError: mutation of a sealed Stage 1 component is blocked; create a derived copy or append a new result key."

                if active_canonical_names.intersection(_bound_names(target)):
                    return f"FoundationIntegrityError: assignment target cannot rebind sealed canonical object '{canonical}'."

            if local_names:
                continue
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if names:
                root = _root_name(value)
                copy_root = _copy_root(value)
                if copy_root in active_canonical_names | derived:
                    derived.update(names)
                    canonical_names.difference_update(names)
                    component_aliases.difference_update(names)
                elif isinstance(value, ast.Name) and value.id in derived:
                    derived.update(names)
                    canonical_names.difference_update(names)
                    component_aliases.difference_update(names)
                elif root in active_canonical_names and isinstance(value, (ast.Attribute, ast.Subscript)):
                    component_aliases.update(names)
                    canonical_names.difference_update(names)
                    derived.difference_update(names)
                elif isinstance(value, ast.Name) and value.id in active_canonical_names:
                    canonical_names.update(names)
                    derived.difference_update(names)
                    component_aliases.difference_update(names)
                else:
                    canonical_names.difference_update(names)
                    derived.difference_update(names)
                    component_aliases.difference_update(names)
        elif isinstance(node, (ast.AugAssign, ast.Delete)):
            targets = [node.target] if isinstance(node, ast.AugAssign) else list(node.targets)
            for target in targets:
                if _root_name(target) in active_canonical_names | active_component_aliases:
                    return "FoundationIntegrityError: mutation of the sealed Stage 1 object is blocked."
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if any(name.id in active_canonical_names for name in ast.walk(node.target) if isinstance(name, ast.Name)):
                return f"FoundationIntegrityError: loop target cannot rebind sealed canonical object '{canonical}'."
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            receiver = _root_name(node.func.value) if isinstance(node.func, ast.Attribute) else None
            target = _root_name(node.args[0]) if node.args else receiver
            if name in _INPLACE_METHODS and receiver in active_canonical_names | active_component_aliases:
                return f"FoundationIntegrityError: '{name}' would mutate the sealed Stage 1 object."
            if name in _FOUNDATION_MUTATORS and target in active_canonical_names | active_component_aliases:
                return f"FoundationIntegrityError: '{name}' would revise the sealed Stage 1 foundation; use an explicit derived copy."
    return None


def commit_foundation_bindings(state: Any, code: str) -> None:
    """Commit only successful-cell aliases needed to recognize derived copies."""
    if not foundation_is_sealed(state):
        return
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    store = _sandbox_state(state)
    ledger = store.setdefault(_BINDINGS_KEY, {"derived": [], "canonical_aliases": [], "component_aliases": []})
    canonical = str(foundation_manifest(state).get("canonical_name") or "adata")
    canonical_names = {canonical, *ledger.get("canonical_aliases", [])}
    derived = set(ledger.get("derived", []))
    components = set(ledger.get("component_aliases", []))
    scoped_locals = _function_scope_locals(tree)
    for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if scoped_locals.get(id(node)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not names:
            continue
        value = node.value
        root = _root_name(value)
        copy_root = _copy_root(value)
        if (copy_root in canonical_names | derived) or (isinstance(value, ast.Name) and value.id in derived):
            derived.update(names)
            canonical_names.difference_update(names)
            components.difference_update(names)
        elif isinstance(value, ast.Name) and value.id in canonical_names:
            canonical_names.update(names)
            derived.difference_update(names)
            components.difference_update(names)
        elif root in canonical_names and isinstance(value, (ast.Attribute, ast.Subscript)):
            components.update(names)
            canonical_names.difference_update(names)
            derived.difference_update(names)
        else:
            canonical_names.difference_update(names)
            derived.difference_update(names)
            components.difference_update(names)
    ledger["derived"] = sorted(derived)
    ledger["canonical_aliases"] = sorted(canonical_names - {canonical})
    ledger["component_aliases"] = sorted(components)


def clear_foundation_bindings(state: Any) -> None:
    if foundation_is_sealed(state):
        _sandbox_state(state)[_BINDINGS_KEY] = {"derived": [], "canonical_aliases": [], "component_aliases": []}


def compare_foundation_signature(state: Any, current: dict[str, Any]) -> list[str]:
    """Return protected-component drift; new result keys are allowed."""
    manifest = foundation_manifest(state)
    if manifest is None:
        return []
    baseline = manifest.get("signature") or {}
    if not current:
        return ["canonical AnnData is missing or unreadable"]
    changes: list[str] = []
    # Python object ids necessarily change when the ordinary persisted runtime
    # is loaded into a new kernel. Content identity, not allocator identity, is
    # the durable invariant.
    def same(left: Any, right: Any) -> bool:
        return left == right or (
            isinstance(left, float) and isinstance(right, float)
            and math.isnan(left) and math.isnan(right)
        )

    for key in (
        "n_obs", "n_vars", "obs_index_hash", "var_hash", "x_sum", "x_nnz",
        "x_fingerprint", "raw_fingerprint",
    ):
        if not same(current.get(key), baseline.get(key)):
            changes.append(key)
    for key in ("obs_fingerprints", "var_fingerprints", "layers_fingerprint", "obsm_fingerprints", "varm_fingerprints", "obsp_fingerprints", "uns_fingerprints"):
        before = baseline.get(key) or {}
        after = current.get(key) or {}
        for name, fingerprint in before.items():
            if after.get(name) != fingerprint:
                changes.append(f"{key}.{name}")
    return changes


def classify_foundation_signature(
    state: Any, current: dict[str, Any],
) -> tuple[str, list[str]]:
    """Distinguish unavailable verification from confirmed content drift."""
    if (
        not isinstance(current, dict)
        or not current.get("id")
        or _REQUIRED_SIGNATURE_KEYS.difference(current)
    ):
        return "unavailable", []
    changes = compare_foundation_signature(state, current)
    return ("drift", changes) if changes else ("verified", [])


def accept_foundation_extensions(state: Any, current: dict[str, Any]) -> None:
    manifest = foundation_manifest(state)
    if manifest is not None:
        manifest["signature"] = current


def mark_foundation_drift(state: Any, error: str) -> None:
    manifest = foundation_manifest(state)
    if manifest is not None:
        manifest["integrity_error"] = error
