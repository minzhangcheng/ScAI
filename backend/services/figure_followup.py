"""Technical scope, capability, and provenance contracts for figure follow-up."""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Iterable

from backend.services.execution_evidence import (
    PLAN_LINKED_RECEIPT_STATUSES,
    execution_is_current_receipt,
)


SCOPE_KEY = "figure_followup_scopes_v1"
ANALYSIS_REQUESTS_KEY = "figure_followup_analysis_requests_v1"
MAX_FOLLOWUP_SCOPES = 8
MAX_SCOPE_TASKS = 12
MAX_SCOPE_RECEIPTS = 256
MAX_PENDING_ANALYSIS_REQUESTS = 40

# The source session predates schema-v2 figure provenance. This exact artifact
# was audited as a post-hoc Scrublet rerun that conflicts with the accepted
# Round 1 cohort. Preserve the record, but never let an import/restore/report
# silently treat it as a current-plan result.
_KNOWN_INVALID_LEGACY_FIGURES = frozenset({
    (
        "14fc685e-9ffa-4bd2-bc03-2d92da0a9c87",
        "014b3726-7baa-44b2-a0e4-d78366063654",
        "a4989f32-064d-4c1d-b2ef-d47e153dd6ee",
    ),
})

_PROTECTED_ADATA_CONTAINERS = frozenset({
    "obs", "var", "layers", "obsm", "varm", "obsp", "varp", "uns", "raw", "X",
})
_READERS = frozenset({
    "open", "read", "read_text", "read_bytes", "read_csv", "read_table",
    "read_parquet", "read_pickle", "read_excel", "read_hdf", "read_json",
    "read_feather", "read_fwf", "read_sas", "read_stata", "read_sql",
    "read_h5ad", "read_loom", "read_mtx", "read_10x_mtx", "read_10x_h5",
    "read_visium", "load", "loadtxt", "genfromtxt", "fromfile", "memmap",
    "glob", "iglob", "listdir", "scandir", "iterdir", "rglob",
})
_WRITERS = frozenset({
    "save", "savefig", "write", "write_h5ad", "write_csv", "write_loom",
    "to_csv", "to_table", "to_parquet", "to_pickle", "to_excel", "to_hdf",
    "to_json", "to_feather", "dump", "dumps", "savetxt", "savez", "savez_compressed",
    "write_text", "write_bytes", "unlink", "rmdir", "mkdir", "touch",
    "copyfile", "copy2", "move", "rename",
})
_ANALYSIS_CALLS = frozenset({
    "fit", "fit_transform", "fit_predict", "partial_fit", "predict",
    "predict_proba", "transform", "score",
    "cluster", "scrublet", "neighbors", "rank_genes_groups", "leiden",
    "louvain", "umap", "pca", "tsne", "diffmap", "dpt", "paga",
    "score_genes", "score_genes_cell_cycle", "wilcoxon", "ttest_ind",
    "mannwhitneyu", "kruskal", "f_oneway", "pearsonr", "spearmanr",
    "multipletests", "permutation_test", "bootstrap", "ttest_rel",
    "chi2_contingency", "fisher_exact", "linregress", "gridsearchcv",
    "randomizedsearchcv", "optuna", "cross_val_score",
    "corr", "correlate", "correlation", "corrcoef", "cov", "covariance",
    "svd", "eig", "eigh", "eigvals", "eigvalsh", "qr", "pinv",
    "lstsq", "matrix_rank", "cholesky", "polyfit",
})
_ANALYSIS_MODULE_PREFIXES = (
    "numpy.linalg.", "scipy.linalg.", "scipy.stats.",
)
_DYNAMIC_CALLS = frozenset({
    "exec", "eval", "compile", "getattr", "setattr", "delattr", "__import__",
    "globals", "locals", "vars", "__setattr__", "__delattr__", "__setitem__",
})
_MUTATORS = frozenset({
    "drop", "dropna", "fillna", "replace", "rename", "sort_values", "sort_index",
    "set_index", "reset_index", "update", "insert", "pop", "__setitem__",
    "__delitem__", "append", "extend", "clear", "remove",
})


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _lineage_id(state: Any) -> str:
    return str(getattr(state, "restore_branch_id", None) or "main")


def _active_preparation_id(state: Any) -> str | None:
    value = str(
        getattr(getattr(state, "current_plan", None), "adata_source_preparation_id", "")
        or ""
    ).strip()
    return value or None


def _plan_step_ids(plan: Any, *, completed_only: bool = False) -> list[int]:
    result = []
    for step in getattr(plan, "steps", None) or []:
        if completed_only and str(getattr(step, "status", "")) != "completed":
            continue
        try:
            step_id = int(getattr(step, "id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if step_id > 0:
            result.append(step_id)
    return sorted(set(result))


def figure_followup_source_prompt(state: Any, scope: dict[str, Any]) -> str:
    """Describe exact persisted task capabilities without inventing source IDs."""
    import json

    receipts = _receipt_map(state)
    tasks = []
    for task in scope.get("tasks", []):
        allowed = list(task.get("eligible_execution_ids") or [])
        tasks.append({
            "figure_task_id": task["task_id"],
            "logical_figure_id": task["logical_figure_id"],
            "plan_step_id": task["plan_step_id"],
            "eligible_execution_ids": allowed,
            "source_receipts": [{
                "execution_id": aid,
                "plan_step_ids": _value(receipts.get(aid), "plan_step_ids", []),
                "description": str(_value(receipts.get(aid), "description", "") or "")[:240],
            } for aid in allowed],
        })
    return (
        "Figure follow-up source contract: copy evidence_execution_id exactly from "
        "the current task's eligible_execution_ids. figure_task_id identifies the task; "
        "logical_figure_id identifies the figure, NOT its source execution. Never extract "
        "a UUID from logical_figure_id or borrow another task's execution IDs. "
        "Choose the listed receipt supporting the plotted result. If none supports it, "
        "report the task unresolved rather than inventing an ID.\n"
        + json.dumps(tasks, ensure_ascii=False)
    )


def _receipt_map(state: Any, current_receipts: Iterable[Any] = ()) -> dict[str, Any]:
    receipts = [
        *(getattr(state, "code_history", None) or []),
        *(current_receipts or []),
    ]
    return {
        str(_value(receipt, "id", "")): receipt
        for receipt in receipts
        if str(_value(receipt, "id", ""))
    }


def eligible_figure_source_ids(
    state: Any, *, plan_step_id: int | None = None, pi_authorized_redraw: bool = False,
) -> list[str]:
    """Return successful receipts for completed steps or an explicit PI redraw.

    A step can remain pending because its figure is missing. The PI's redraw
    authorization permits rendering its existing receipts without granting the
    step scientific completion or authorizing new analysis.
    """
    plan = getattr(state, "current_plan", None)
    plan_id = str(getattr(plan, "id", "") or "")
    round_number = int(getattr(state, "current_round", 0) or 0)
    lineage_id = _lineage_id(state)
    source_step_ids = set(_plan_step_ids(
        plan, completed_only=not (pi_authorized_redraw and plan_step_id is not None),
    ))
    if plan_step_id is not None:
        source_step_ids &= {int(plan_step_id)}
    if not plan_id or round_number < 1 or not source_step_ids:
        return []
    preparation_id = _active_preparation_id(state)
    eligible = []
    for receipt in getattr(state, "code_history", None) or []:
        if str(getattr(receipt, "executor_run_kind", "") or "") != "ordinary":
            continue
        if not execution_is_current_receipt(
            receipt,
            round_number=round_number,
            plan_id=plan_id,
            lineage_id=lineage_id,
        ):
            continue
        if (
            str(getattr(receipt, "plan_binding_status", "") or "")
            not in PLAN_LINKED_RECEIPT_STATUSES
        ):
            continue
        receipt_steps = {
            int(raw)
            for raw in (getattr(receipt, "plan_step_ids", None) or [])
            if str(raw).isdigit() and int(raw) > 0
        }
        if not receipt_steps.intersection(source_step_ids):
            continue
        source_preparation_id = str(
            getattr(receipt, "adata_source_preparation_id", "") or ""
        ).strip() or None
        if preparation_id and source_preparation_id != preparation_id:
            continue
        if not preparation_id and source_preparation_id:
            continue
        eligible.append(str(getattr(receipt, "id", "")))
    return sorted(set(eligible))[:MAX_SCOPE_RECEIPTS]


def build_figure_followup_scope(
    state: Any,
    tasks: Iterable[Any],
    *,
    run_scope_id: str,
) -> dict[str, Any]:
    """Build the bounded immutable pre-dispatch scope used by one redraw pass."""
    plan = getattr(state, "current_plan", None)
    plan_id = str(getattr(plan, "id", "") or "")
    round_number = int(getattr(state, "current_round", 0) or 0)
    lineage_id = _lineage_id(state)
    if not plan_id or round_number < 1 or not run_scope_id:
        raise ValueError("Figure follow-up scope has no current plan, round, or run identity")
    completed_ids = set(_plan_step_ids(plan, completed_only=True))
    plan_step_ids = set(_plan_step_ids(plan))
    receipt_map = _receipt_map(state)
    task_entries = []
    seen_task_ids = set()
    task_list = list(tasks)
    if len(task_list) > MAX_SCOPE_TASKS:
        raise ValueError(
            f"Figure follow-up scope supports at most {MAX_SCOPE_TASKS} tasks"
        )
    for task in task_list:
        task_id = str(_value(task, "id", "") or "").strip()
        try:
            step_id = int(_value(task, "plan_step_id", 0) or 0)
        except (TypeError, ValueError):
            step_id = 0
        authorized_redraw = (
            _value(task, "pi_disposition") == "targeted_redraw"
            and _value(task, "plan_id") == plan_id
            and int(_value(task, "round_number", 0) or 0) == round_number
            and (_value(task, "lineage_id") or "main") == lineage_id
        )
        if (
            not task_id or task_id in seen_task_ids or step_id not in plan_step_ids
            or (step_id not in completed_ids and not authorized_redraw)
        ):
            continue
        allowed_ids = eligible_figure_source_ids(
            state, plan_step_id=step_id, pi_authorized_redraw=authorized_redraw,
        )
        task_entries.append({
            "task_id": task_id[:240],
            "logical_figure_id": str(_value(task, "logical_figure_id", "") or "")[:240],
            "plan_requirement_key": str(
                _value(task, "plan_requirement_key", "") or ""
            )[:300] or None,
            "plan_step_id": step_id,
            "eligible_execution_ids": [
                execution_id for execution_id in allowed_ids
                if execution_id in receipt_map
            ][:MAX_SCOPE_RECEIPTS],
        })
        seen_task_ids.add(task_id)
    all_receipt_ids = sorted({
        execution_id
        for entry in task_entries
        for execution_id in entry["eligible_execution_ids"]
    })[:MAX_SCOPE_RECEIPTS]
    scope = {
        "schema_version": 1,
        "run_scope_id": str(run_scope_id)[:240],
        "plan_id": plan_id[:240],
        "round_number": round_number,
        "lineage_id": lineage_id[:240],
        "source_preparation_id": _active_preparation_id(state),
        "completed_plan_step_ids": sorted(completed_ids)[:128],
        "eligible_execution_ids": all_receipt_ids,
        "tasks": task_entries,
        "captured_at": datetime.now().isoformat(),
    }
    digest_source = {key: value for key, value in scope.items() if key != "captured_at"}
    payload = json.dumps(digest_source, sort_keys=True, separators=(",", ":"))
    scope["scope_digest"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return scope


def persist_figure_followup_scope(state: Any, scope: dict[str, Any]) -> dict[str, Any]:
    """Persist one bounded scope idempotently before any executor dispatch."""
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    scopes = dict(sandbox_state.get(SCOPE_KEY) or {})
    scope_id = str(scope.get("run_scope_id") or "")
    if not scope_id:
        raise ValueError("Figure follow-up scope requires a run scope ID")
    existing = scopes.get(scope_id)
    if isinstance(existing, dict):
        for identity_key in (
            "plan_id", "round_number", "lineage_id", "scope_digest",
        ):
            if existing.get(identity_key) != scope.get(identity_key):
                raise ValueError("Figure follow-up run scope was reused with different identity")
        scope = existing
    else:
        scopes[scope_id] = dict(scope)
    while len(scopes) > MAX_FOLLOWUP_SCOPES:
        oldest = next(iter(scopes))
        if oldest == scope_id:
            break
        scopes.pop(oldest, None)
    sandbox_state[SCOPE_KEY] = scopes
    state.sandbox_state = sandbox_state
    return dict(scope)


def get_figure_followup_scope(state: Any, run_scope_id: str | None) -> dict[str, Any] | None:
    if not run_scope_id:
        return None
    sandbox_state = getattr(state, "sandbox_state", None) or {}
    scopes = sandbox_state.get(SCOPE_KEY) or {}
    scope = scopes.get(str(run_scope_id)) if isinstance(scopes, Mapping) else None
    return dict(scope) if isinstance(scope, Mapping) else None


def create_needs_new_analysis_request(state: Any, task: Any) -> dict[str, Any]:
    """Persist a bounded PI request without dispatching analytical execution."""
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    requests = list(sandbox_state.get(ANALYSIS_REQUESTS_KEY) or [])
    task_id = str(_value(task, "id", "") or "")[:240]
    request_id = "figure_analysis_" + hashlib.sha256(
        f"{getattr(getattr(state, 'current_plan', None), 'id', '')}|{task_id}".encode()
    ).hexdigest()[:32]
    existing = next(
        (item for item in requests if isinstance(item, dict) and item.get("id") == request_id),
        None,
    )
    request = existing or {
        "id": request_id,
        "task_id": task_id,
        "plan_id": str(getattr(getattr(state, "current_plan", None), "id", "") or "")[:240],
        "round_number": int(getattr(state, "current_round", 0) or 0),
        "lineage_id": _lineage_id(state),
        "plan_step_id": _value(task, "plan_step_id", None),
        "plan_requirement_key": str(_value(task, "plan_requirement_key", "") or "")[:300] or None,
        "logical_figure_id": str(_value(task, "logical_figure_id", "") or "")[:240],
        "request": str(_value(task, "pi_instruction", "") or "")[:2000],
        "rationale": str(_value(task, "pi_rationale", "") or "")[:1000],
        "status": "pending_review",
        "created_at": datetime.now().isoformat(),
    }
    if existing is None:
        requests.append(request)
    sandbox_state[ANALYSIS_REQUESTS_KEY] = requests[-MAX_PENDING_ANALYSIS_REQUESTS:]
    state.sandbox_state = sandbox_state
    return dict(request)


def pending_figure_analysis_context(state: Any) -> str:
    """Bounded planning projection of PI-declared work that needs analysis."""
    sandbox_state = getattr(state, "sandbox_state", None) or {}
    requests = [
        item for item in (sandbox_state.get(ANALYSIS_REQUESTS_KEY) or [])
        if isinstance(item, dict) and item.get("status") == "pending_review"
        and str(item.get("lineage_id") or "main") == _lineage_id(state)
    ][-12:]
    if not requests:
        return ""
    return (
        "\n\n## Pending PI figure requests requiring separately approved analysis\n"
        "These requests are durable planning context only. Do not execute them as "
        "figure rendering; include them in the next normal plan/review if appropriate.\n"
        + json.dumps(requests, ensure_ascii=False, sort_keys=True)[:5000]
    )


def record_figure_followup_completion(
    state: Any,
    run_scope_id: str,
    *,
    delivered_task_ids: list[str],
    unresolved_task_ids: list[str],
    summary: str,
) -> dict[str, Any]:
    """Store a bounded completion record without changing the frozen scope."""
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    scopes = dict(sandbox_state.get(SCOPE_KEY) or {})
    scope = scopes.get(str(run_scope_id))
    if not isinstance(scope, dict):
        raise ValueError("Figure follow-up scope is not persisted")
    expected = {
        str(item.get("task_id"))
        for item in (scope.get("tasks") or [])
        if isinstance(item, Mapping) and item.get("task_id")
    }
    delivered = [str(item)[:240] for item in delivered_task_ids]
    unresolved = [str(item)[:240] for item in unresolved_task_ids]
    if len(delivered) != len(set(delivered)) or len(unresolved) != len(set(unresolved)):
        raise ValueError("Figure completion task IDs must be unique")
    if set(delivered) & set(unresolved) or set(delivered) | set(unresolved) != expected:
        raise ValueError("Figure completion must partition the persisted task scope")
    completion = {
        "delivered_task_ids": delivered,
        "unresolved_task_ids": unresolved,
        "summary": " ".join(str(summary or "").split())[:1200],
        "completed_at": datetime.now().isoformat(),
    }
    scope = dict(scope)
    scope["completion"] = completion
    scopes[str(run_scope_id)] = scope
    sandbox_state[SCOPE_KEY] = scopes
    state.sandbox_state = sandbox_state
    return completion


def _dotted_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _target_names(item)]
    return []


def _container_reference(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, ast.Attribute):
        return node.attr in _PROTECTED_ADATA_CONTAINERS or _container_reference(node.value, aliases)
    if isinstance(node, ast.Subscript):
        return _container_reference(node.value, aliases)
    return False


class _PlottingOnlyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.error: str | None = None
        self.aliases: set[str] = set()
        self.import_aliases: dict[str, str] = {}

    def reject(self, message: str) -> None:
        if self.error is None:
            self.error = message

    def visit_Import(self, node: ast.Import) -> None:
        restricted_modules = {
            "pickle", "joblib", "h5py", "sqlite3", "pathlib", "os", "shutil",
            "glob", "tempfile", "sklearn", "statsmodels", "gseapy",
            "decoupler", "pertpy",
        }
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            self.import_aliases[local] = (
                alias.name if alias.asname else alias.name.split(".", 1)[0]
            )
            if (
                alias.name.split(".", 1)[0] in restricted_modules
                or alias.name.startswith("scipy.stats")
            ):
                self.reject(f"figure follow-up cannot import filesystem/data reader module {alias.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        restricted_modules = {
            "pickle", "joblib", "h5py", "sqlite3", "pathlib", "os", "shutil",
            "glob", "tempfile", "sklearn", "statsmodels", "gseapy",
            "decoupler", "pertpy",
        }
        for alias in node.names:
            local = alias.asname or alias.name
            target = f"{module}.{alias.name}" if module else alias.name
            self.import_aliases[local] = target
            if (
                alias.name in _READERS | _WRITERS | _ANALYSIS_CALLS | _DYNAMIC_CALLS
                or self._is_explicit_analysis_capability(target)
            ):
                self.reject(f"figure follow-up cannot import capability {target!r}")
            if (
                module.split(".", 1)[0] in restricted_modules
                or module.startswith("scipy.stats")
            ):
                self.reject(f"figure follow-up cannot import filesystem/data reader module {module!r}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        resolved = self._resolved_value(node.value)
        if resolved:
            for target in node.targets:
                for name in _target_names(target):
                    self.import_aliases[name] = resolved
        if isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript)):
            if _container_reference(node.value, self.aliases):
                for target in node.targets:
                    self.aliases.update(_target_names(target))
        for target in node.targets:
            if self._is_protected_store(target):
                self.reject("figure follow-up cannot assign into a pre-existing AnnData container")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        resolved = self._resolved_value(node.value) if node.value is not None else ""
        if resolved:
            for name in _target_names(node.target):
                self.import_aliases[name] = resolved
        if self._is_protected_store(node.target):
            self.reject("figure follow-up cannot assign into a pre-existing AnnData container")
        if isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript)) and _container_reference(node.value, self.aliases):
            self.aliases.update(_target_names(node.target))
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if self._is_protected_store(target):
                self.reject(
                    "figure follow-up cannot delete from a pre-existing AnnData container"
                )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self._is_protected_store(node.target):
            self.reject(
                "figure follow-up cannot assign loop values into a pre-existing AnnData container"
            )
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        if self._is_protected_store(node.target):
            self.reject(
                "figure follow-up cannot assign loop values into a pre-existing AnnData container"
            )
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._check_comprehension_targets(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._check_comprehension_targets(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._check_comprehension_targets(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._check_comprehension_targets(node)

    def _check_comprehension_targets(self, node: ast.AST) -> None:
        for generator in getattr(node, "generators", ()):
            if self._is_protected_store(generator.target):
                self.reject(
                    "figure follow-up cannot assign comprehension values into a "
                    "pre-existing AnnData container"
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._is_protected_store(node.target):
            self.reject("figure follow-up cannot mutate a pre-existing AnnData container")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        leaf = name.rsplit(".", 1)[-1].casefold()
        resolved = self._resolved_value(node.func)
        resolved_leaf = resolved.rsplit(".", 1)[-1].casefold()
        components = resolved.casefold().split(".")
        plot_namespace_call = (
            len(components) >= 2
            and components[-2] == "pl"
            and leaf in {
                "umap", "pca", "tsne", "diffmap", "dpt", "paga",
                "rank_genes_groups", "rank_genes_groups_dotplot",
            }
        )
        if (
            self._is_explicit_analysis_capability(resolved)
            and not plot_namespace_call
        ):
            self.reject(
                "figure follow-up cannot call explicit statistical or numerical-analysis capabilities"
            )
        if leaf in _DYNAMIC_CALLS or resolved_leaf in _DYNAMIC_CALLS:
            self.reject(f"figure follow-up cannot use dynamic capability {leaf!r}")
        if leaf in _READERS or resolved_leaf in _READERS:
            self.reject(f"figure follow-up cannot read session or input files via {leaf!r}")
        if leaf in _WRITERS or resolved_leaf in _WRITERS:
            self.reject(f"figure follow-up cannot persist data or render files directly via {leaf!r}")
        if (
            (leaf in _ANALYSIS_CALLS or resolved_leaf in _ANALYSIS_CALLS)
            and not plot_namespace_call
        ):
            self.reject(f"figure follow-up cannot perform analysis/model call {leaf!r}")
        if len(components) >= 2 and components[-2] in {"pp", "tl"}:
            self.reject(f"figure follow-up cannot call analysis namespace {'.'.join(components[-2:])!r}")
        if isinstance(node.func, ast.Attribute) and node.func.attr.casefold() in _MUTATORS:
            if _container_reference(node.func.value, self.aliases):
                self.reject("figure follow-up cannot mutate a pre-existing AnnData container")
        if any(keyword.arg == "inplace" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords):
            self.reject("figure follow-up cannot request in-place mutation")
        self.generic_visit(node)

    def _resolved_value(self, node: ast.AST | None) -> str:
        if isinstance(node, ast.Name):
            return self.import_aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            root, separator, suffix = dotted.partition(".")
            resolved_root = self.import_aliases.get(root, root)
            return resolved_root + (separator + suffix if separator else "")
        return ""

    @staticmethod
    def _is_explicit_analysis_capability(name: str) -> bool:
        normalized = str(name or "").casefold()
        return (
            normalized.startswith(_ANALYSIS_MODULE_PREFIXES)
            or normalized.rsplit(".", 1)[-1] in _ANALYSIS_CALLS
        )

    def _is_protected_store(self, target: ast.AST) -> bool:
        if isinstance(target, ast.Attribute):
            return target.attr in _PROTECTED_ADATA_CONTAINERS or self._is_protected_store(target.value)
        if isinstance(target, ast.Subscript):
            return self._is_protected_store(target.value) or _container_reference(target.value, self.aliases)
        return isinstance(target, ast.Name) and target.id in self.aliases


def validate_plotting_only_code(code: str) -> str | None:
    """Reject analysis, input/output, dynamic, and pre-existing AnnData mutation."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError) as exc:
        return f"Figure plotting code is not valid Python: {exc}"
    visitor = _PlottingOnlyVisitor()
    visitor.visit(tree)
    return visitor.error


def validate_publication_figure_provenance(
    state: Any,
    figure: Any,
    *,
    followup_scope: dict[str, Any] | None = None,
    current_receipts: Iterable[Any] = (),
) -> str | None:
    """Revalidate plan-bound figure provenance against durable receipts."""
    version = _value(figure, "provenance_schema_version")
    plan_id = str(_value(figure, "plan_id", "") or "").strip()
    purpose = str(_value(figure, "purpose", "result") or "result")
    if not plan_id or purpose != "result":
        return None
    source_id = str(_value(figure, "source_execution_id", "") or "").strip()
    if version is None:
        signature = (
            str(_value(figure, "id", "") or ""),
            plan_id,
            source_id,
        )
        if signature in _KNOWN_INVALID_LEGACY_FIGURES:
            return (
                "Known legacy figure was invalidated because its post-hoc analysis "
                "does not represent the accepted foundation."
            )
        source = _receipt_map(state, current_receipts).get(source_id) if source_id else None
        if str(_value(source, "executor_run_kind", "") or "") == "figure_followup":
            return (
                "A legacy figure-follow-up execution cannot source a scientific "
                "result figure."
            )
        return str(_value(figure, "provenance_validation_reason", "") or "") or None
    if int(version or 0) != 2:
        return "Figure provenance schema version is not supported."
    if not source_id:
        return "Schema-v2 plan-bound figure has no source execution ID."
    source = _receipt_map(state, current_receipts).get(source_id)
    if source is None:
        return f"Figure source execution {source_id} is missing from durable code history."
    round_number = int(_value(figure, "round_number", 0) or 0)
    figure_lineage = str(_value(figure, "lineage_id", "main") or "main")
    source_lineage = str(
        _value(figure, "inherited_from_lineage_id", None) or figure_lineage
    )
    if not execution_is_current_receipt(
        source,
        round_number=round_number,
        plan_id=plan_id,
        lineage_id=source_lineage,
    ):
        return "Figure source execution is failed, legacy, or no longer current."
    if str(_value(source, "plan_id", "") or "") != plan_id:
        return "Figure plan identity disagrees with its source receipt."
    source_run_kind = str(_value(source, "executor_run_kind", "") or "")
    persisted_source_kind = str(_value(figure, "source_execution_run_kind", "") or "")
    if source_run_kind == "figure_followup":
        return "A figure-follow-up execution cannot source a scientific result figure."
    if source_run_kind != "ordinary":
        return "Figure source must be an ordinary current-plan execution; supplementary sources require explicit adoption."
    if persisted_source_kind != source_run_kind:
        return "Persisted source run kind disagrees with the durable execution receipt."
    if _value(figure, "source_execution_validated") is not True:
        return "Figure source execution was not validated at creation."
    if (
        str(_value(source, "plan_binding_status", "") or "")
        not in PLAN_LINKED_RECEIPT_STATUSES
    ):
        return "Figure source receipt is not bound to the approved plan."
    try:
        plan_step_id = int(_value(figure, "plan_step_id", 0) or 0)
    except (TypeError, ValueError):
        plan_step_id = 0
    source_steps = {
        int(raw)
        for raw in (_value(source, "plan_step_ids", None) or [])
        if str(raw).isdigit() and int(raw) > 0
    }
    if plan_step_id <= 0 or plan_step_id not in source_steps:
        return "Figure plan step is not linked to its source execution receipt."
    source_preparation = str(_value(source, "adata_source_preparation_id", "") or "") or None
    figure_preparation = str(_value(figure, "source_preparation_id", "") or "") or None
    if figure_preparation != source_preparation:
        return "Persisted figure preparation identity disagrees with the source receipt."
    if int(_value(source, "round_number", 0) or 0) != round_number:
        return "Figure round identity disagrees with its source receipt."
    if str(_value(source, "lineage_id", "main") or "main") != source_lineage:
        return "Figure source lineage disagrees with its durable receipt."
    for field_name in ("analysis_generation", "cohort_generation", "cohort_digest"):
        figure_value = _value(figure, field_name, None)
        source_value = _value(source, field_name, None)
        if (
            figure_value is not None
            and source_value is not None
            and figure_value != source_value
        ):
            return f"Figure {field_name.replace('_', ' ')} disagrees with its source receipt."

    from backend.services.report.artifacts import has_frozen_publication_pair

    if not has_frozen_publication_pair(figure):
        return "Figure JPG/PDF pair is missing or no longer matches its frozen digests."

    producer_kind = str(_value(figure, "producer_run_kind", "") or "")
    if producer_kind not in {"ordinary", "supplementary", "figure_followup", "platform"}:
        return "Figure producer run kind is missing or unsupported."
    if producer_kind == "figure_followup":
        run_scope_id = str(_value(figure, "producer_run_scope_id", "") or "")
        scope = followup_scope or get_figure_followup_scope(state, run_scope_id)
        if not scope or scope.get("run_scope_id") != run_scope_id:
            return "Figure follow-up scope is missing from durable state."
        if (
            scope.get("plan_id") != plan_id
            or int(scope.get("round_number", 0) or 0) != round_number
            or scope.get("lineage_id") != figure_lineage
            or scope.get("source_preparation_id") != figure_preparation
        ):
            return "Figure follow-up scope does not match persisted plan/round/lineage/preparation."
        tasks = scope.get("tasks") or []
        task_id = str(_value(figure, "figure_followup_task_id", "") or "")
        task = next((item for item in tasks if isinstance(item, Mapping) and item.get("task_id") == task_id), None)
        if task is None or source_id not in (task.get("eligible_execution_ids") or []):
            return "Figure source was not eligible for the persisted pre-follow-up task scope."
        if int(task.get("plan_step_id", 0) or 0) != int(_value(figure, "plan_step_id", 0) or 0):
            return "Figure plan step does not match its persisted follow-up task."
        if str(task.get("plan_requirement_key") or "") != str(_value(figure, "plan_requirement_key", "") or ""):
            return "Figure requirement does not match its persisted follow-up task."
    return None


def revalidate_publication_figures(state: Any) -> list[dict[str, str]]:
    """Annotate invalid figures while keeping their audit records."""
    failures = []
    for figure in getattr(state, "publication_figures", None) or []:
        reason = validate_publication_figure_provenance(state, figure)
        if reason:
            if isinstance(figure, dict):
                figure["provenance_validation_reason"] = reason[:1000]
            else:
                figure.provenance_validation_reason = reason[:1000]
            failures.append({"figure_id": str(_value(figure, "id", "")), "reason": reason[:1000]})
        elif isinstance(figure, dict):
            figure.pop("provenance_validation_reason", None)
        elif _value(figure, "provenance_validation_reason"):
            figure.provenance_validation_reason = None
    return failures
