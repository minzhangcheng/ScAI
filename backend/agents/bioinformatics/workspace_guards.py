# Static AST guard limiting how many .h5ad files generated code may write.
# Moved verbatim out of backend/agents/bioinformatics_agent.py.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Optional
import logging
import ast
from pathlib import Path


import inspect
from backend.services import ora_kernel

_ORA_KERNEL_SRC = inspect.getsource(ora_kernel)

# Configure logger
logger = logging.getLogger(__name__)

MAX_WORKSPACE_H5AD_FILES = 2


def _static_h5ad_path(expr: ast.AST) -> Optional[str]:
    """Resolve the filename portion of a statically visible h5ad path."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value if expr.value.casefold().endswith(".h5ad") else None
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        return _static_h5ad_path(expr.right)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "Path":
        return _static_h5ad_path(expr.args[0]) if expr.args else None
    if isinstance(expr, ast.JoinedStr) and all(
        isinstance(part, ast.Constant) for part in expr.values
    ):
        value = "".join(str(part.value) for part in expr.values)
        return value if value.casefold().endswith(".h5ad") else None
    return None

def workspace_h5ad_write_guard(code: str, workspace_dir: str | Path) -> Optional[str]:
    """Reject creation of more than two full AnnData workspace checkpoints.

    Existing files may be overwritten. This is intentionally scoped to explicit
    ``write_h5ad`` calls and ``write`` calls whose target visibly ends in h5ad;
    unrelated file writes are unaffected.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    known_targets: set[str] = set()
    unresolved_write_h5ad = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in {"write", "write_h5ad"}:
            continue
        target_expr = node.args[0] if node.args else None
        if target_expr is None:
            target_expr = next(
                (kw.value for kw in node.keywords if kw.arg in {"filename", "path"}),
                None,
            )
        target = _static_h5ad_path(target_expr) if target_expr is not None else None
        if target:
            known_targets.add(target)
        elif method == "write_h5ad":
            unresolved_write_h5ad = True

    if not known_targets and not unresolved_write_h5ad:
        return None

    workspace = Path(workspace_dir)
    existing = {path.resolve() for path in workspace.rglob("*.h5ad") if path.is_file()}
    resolved_targets = {
        (Path(target) if Path(target).is_absolute() else workspace / target).resolve()
        for target in known_targets
    }
    new_targets = resolved_targets - existing
    if len(existing) + len(new_targets) > MAX_WORKSPACE_H5AD_FILES or (
        unresolved_write_h5ad and len(existing) >= MAX_WORKSPACE_H5AD_FILES
    ):
        return (
            f"Workspace already contains {len(existing)} h5ad checkpoint file(s). "
            f"At most {MAX_WORKSPACE_H5AD_FILES} are allowed to prevent multi-GB "
            "duplicate datasets. Overwrite an existing checkpoint or delete an "
            "obsolete one before creating another."
        )
    return None
