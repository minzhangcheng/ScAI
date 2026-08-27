"""AST analysis and instrumentation for expression contracts."""

from __future__ import annotations

import ast
from typing import Any

from .constants import (
    _FRESH_OBJECT_CALLS,
    _NON_EXPRESSION_CALL_ROOTS,
    _REPRESENTATION_MUTATORS,
    _RUNTIME_CALL_HELPER,
    _SENSITIVE_CALLS,
)


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
    instrumented_call_count = 0
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
        generator = node.args[0] if len(node.args) == 1 else None
        sole_unparenthesized_generator = (
            isinstance(generator, ast.GeneratorExp)
            and absolute(generator.lineno, generator.col_offset) == opening
            and absolute(generator.end_lineno, generator.end_col_offset)
            == call_end
        )
        if sole_unparenthesized_generator:
            edits.append((opening, opening + 1, b", None, ("))
            edits.append((call_end, call_end, b")"))
        else:
            edits.append((opening, opening + 1, b", None, "))
        instrumented_call_count += 1

    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source.decode("utf-8"), instrumented_call_count


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
                    events.append((
                        pos,
                        1,
                        "mutate",
                        (_root_name(target), None, None),
                    ))
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Attribute) and node.target.attr == "X":
                events.append((
                    pos,
                    1,
                    "mutate",
                    (_root_name(node.target), None, None),
                ))
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if (
                name in _REPRESENTATION_MUTATORS
                and _call_module_root(node) not in _NON_EXPRESSION_CALL_ROOTS
            ):
                events.append((
                    pos,
                    1,
                    "mutate",
                    (_mutation_target_root(node), _call_module_root(node), name),
                ))

    local: set[str] = set()
    table_names: set[str] = set()
    for _pos, _tie, kind, data in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == "assign":
            names, value = data
            if _value_makes_fresh_object(value, local):
                local.update(names)
            else:
                # Rebinding a name to the canonical object (or an unknown source)
                # drops its local-copy status.
                local.difference_update(names)
            is_table_result = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "table"
            )
            is_table_alias = (
                isinstance(value, ast.Name) and value.id in table_names
            )
            if is_table_result or is_table_alias:
                table_names.update(names)
            else:
                table_names.difference_update(names)
        elif kind == "mutate":
            target, receiver, mutator = data
            if mutator == "scale" and receiver in table_names:
                continue
            if target is None or target not in local:
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
