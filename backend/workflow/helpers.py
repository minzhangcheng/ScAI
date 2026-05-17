"""Pure helper functions for the workflow — no side effects, no class dependencies."""
import random
from typing import Dict, List


def _retry_backoff_delay(attempt: int) -> float:
    """Compute exponential backoff delay with jitter (shared by @with_retry and inline retries)."""
    return min(2 ** attempt + random.uniform(0, 1), 30)


# Stage1 keyword set for filtering bioinformatics tasks
STAGE1_KEYWORDS = frozenset([
    "qc", "quality", "filter", "normaliz", "cluster", "umap", "pca",
    "dimension", "annotat", "cell type", "doublet", "batch", "harmony",
    "hvg", "preprocess", "load", "inspect", "merg", "integrat",
])

_DIFF_LABELS = {
    'added': '[ADDED]',
    'removed': '[REMOVED]',
    'modified': '[MODIFIED]',
    'unchanged': '',
}


def _diff_plan_steps(prev_steps: List[Dict], curr_steps: List[Dict]) -> Dict[int, str]:
    """Compare steps between two plan versions. Returns {step_id: 'added'|'removed'|'modified'|'unchanged'}."""
    prev_by_id = {s['id']: s for s in prev_steps}
    curr_by_id = {s['id']: s for s in curr_steps}
    diff = {}
    all_ids = set(prev_by_id.keys()) | set(curr_by_id.keys())
    for sid in all_ids:
        if sid in curr_by_id and sid not in prev_by_id:
            diff[sid] = 'added'
        elif sid in prev_by_id and sid not in curr_by_id:
            diff[sid] = 'removed'
        elif sid in curr_by_id and sid in prev_by_id:
            prev = prev_by_id[sid]
            curr = curr_by_id[sid]
            if (prev.get('title') != curr.get('title') or
                prev.get('method') != curr.get('method')):
                diff[sid] = 'modified'
            else:
                diff[sid] = 'unchanged'
    return diff


def _get_task_text(t) -> str:
    """Extract searchable text from a task (string or dict)."""
    if isinstance(t, dict):
        return " ".join(str(v) for v in t.values()).lower()
    return str(t).lower()


def _normalize_tasks(tasks: list) -> List[str]:
    """Convert task items to strings — LLMs sometimes return dicts instead of plain strings."""
    result = []
    for t in tasks:
        if isinstance(t, str):
            result.append(t)
        elif isinstance(t, dict):
            parts = []
            for key in ("task", "description", "topic", "name"):
                if key in t:
                    parts.append(str(t[key]))
                    break
            if not parts:
                parts.append(str(t))
            if "rationale" in t:
                parts.append(f"(Rationale: {t['rationale']})")
            if "recommended_methods" in t and t["recommended_methods"]:
                methods = t["recommended_methods"]
                if isinstance(methods, list):
                    parts.append(f"[Methods: {'; '.join(str(m) for m in methods)}]")
                else:
                    parts.append(f"[Methods: {methods}]")
            if "target_readouts" in t and t["target_readouts"]:
                readouts = t["target_readouts"]
                if isinstance(readouts, list):
                    parts.append(f"[Readouts: {'; '.join(str(r) for r in readouts)}]")
                else:
                    parts.append(f"[Readouts: {readouts}]")
            result.append(" ".join(parts))
        else:
            result.append(str(t))
    return result
