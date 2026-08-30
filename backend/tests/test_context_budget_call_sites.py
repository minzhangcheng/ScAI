"""Known text-message client dispatches pass through the shared budget once."""

import ast
from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
DISPATCH_METHODS = {"chat", "chat_with_tools", "chat_stream"}
BUDGET_CALLS = {
    "_apply_context_budget",
    "_budgeted_text_call",
    "budget_text_dispatch",
    "context_budget_callback",
}
IMPLEMENTATION_SITES = {
    ("backend/utils/llm_client.py", "chat"),
    ("backend/utils/llm_client.py", "chat_with_tools"),
    ("backend/utils/llm_client.py", "chat_stream"),
}


def _function_owners(tree: ast.AST):
    owners = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                if hasattr(node, "lineno"):
                    owners.setdefault(node.lineno, function.name)
    return owners


def _budget_calls_in(tree: ast.AST):
    found = {}
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = set()
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            name = (
                called.attr
                if isinstance(called, ast.Attribute)
                else getattr(called, "id", "")
            )
            if name in BUDGET_CALLS:
                names.add(name)
        if names:
            found[function.name] = names
    return found


def _receiver_name(call: ast.Call) -> str:
    value = call.func.value
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Name):
        return value.id
    return type(value).__name__


def _scan():
    rows = []
    for path in sorted(BACKEND.rglob("*.py")):
        relative = path.relative_to(BACKEND.parent)
        if relative.parts[1] == "tests":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owners = _function_owners(tree)
        budgeting = _budget_calls_in(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in DISPATCH_METHODS:
                continue
            if _receiver_name(node) in {"self", "agent"}:
                continue
            owner = owners.get(node.lineno, "<module>")
            rows.append((str(relative), owner, owner in budgeting))
    return rows


def test_every_text_client_dispatch_has_one_shared_budget_owner():
    unbudgeted = [
        (module, owner)
        for module, owner, budgeted in _scan()
        if not budgeted and (module, owner) not in IMPLEMENTATION_SITES
    ]
    assert not unbudgeted, "Unbudgeted text dispatches:\n" + "\n".join(
        f"  {module}::{owner}" for module, owner in sorted(set(unbudgeted))
    )


def test_blocking_archive_budget_mechanisms_are_absent():
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in BACKEND.rglob("*.py")
        if "tests" not in path.parts
    )
    assert "ContextBudgetExceeded" not in production
    assert 'on_overflow="raise"' not in production
    assert "_OVERFLOW_COUNTS" not in production
