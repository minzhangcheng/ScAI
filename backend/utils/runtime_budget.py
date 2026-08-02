"""Configurable live-runtime resource budgets and deterministic evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MEMORY_FRACTIONS = {
    "max_single_object_bytes": 0.25,
    "max_anndata_total_bytes": 0.45,
    "max_rss_bytes": 0.60,
    "max_pkl_bytes": 0.30,
}
_UNBOUNDED_CGROUP_SENTINEL = 1 << 60


def _read_positive_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value == "max":
            return None
        parsed = int(value)
    except (OSError, ValueError):
        return None
    return parsed if 0 < parsed < _UNBOUNDED_CGROUP_SENTINEL else None


def detect_memory_capacity_bytes() -> tuple[int | None, str]:
    """Return the effective cgroup/system memory capacity and its source."""
    for path, source in (
        (Path("/sys/fs/cgroup/memory.max"), "cgroup_v2"),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "cgroup_v1",
        ),
    ):
        value = _read_positive_int(path)
        if value is not None:
            return value, source
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024, "system_memtotal"
    except (OSError, ValueError, IndexError):
        pass
    return None, "configured_only"


def resolve_effective_runtime_budgets(
    configured: dict[str, int | float],
    *,
    memory_capacity_bytes: int | None = None,
    capacity_source: str | None = None,
) -> dict[str, Any]:
    """Clamp administrator ceilings to pre-OOM memory-safe effective limits."""
    if memory_capacity_bytes is None:
        memory_capacity_bytes, detected_source = detect_memory_capacity_bytes()
        capacity_source = capacity_source or detected_source
    effective: dict[str, Any] = dict(configured)
    if memory_capacity_bytes and memory_capacity_bytes > 0:
        for key, fraction in _MEMORY_FRACTIONS.items():
            configured_limit = configured.get(key)
            if configured_limit is None or float(configured_limit) <= 0:
                continue
            dynamic_limit = max(1, int(memory_capacity_bytes * fraction))
            effective[key] = min(int(configured_limit), dynamic_limit)
    effective["effective_memory_capacity_bytes"] = memory_capacity_bytes
    effective["effective_limit_basis"] = capacity_source or "configured_only"
    return effective


@dataclass(frozen=True)
class RuntimeBudgetViolation:
    resource: str
    observed: int | float
    limit: int | float

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "observed": self.observed,
            "limit": self.limit,
        }


class RuntimeBudgetExceeded(RuntimeError):
    def __init__(self, violations, top_offenders=None):
        self.violations = list(violations)
        self.top_offenders = list(top_offenders or [])[:20]
        names = ", ".join(v.resource for v in self.violations)
        super().__init__(
            f"Runtime persistence resource budget exceeded: {names}. "
            "Remove disposable objects or explicitly recompute a smaller continuation state."
        )


def evaluate_runtime_budget(
    metrics: dict[str, Any],
    budgets: dict[str, int | float],
) -> list[RuntimeBudgetViolation]:
    """Compare normalized metrics with configured positive limits."""
    mapping = {
        "namespace_count": "max_namespace_count",
        "largest_object_bytes": "max_single_object_bytes",
        "anndata_total_bytes": "max_anndata_total_bytes",
        "rss_bytes": "max_rss_bytes",
        "pkl_bytes": "max_pkl_bytes",
        "save_seconds": "max_save_seconds",
    }
    violations = []
    for resource, budget_key in mapping.items():
        limit = budgets.get(budget_key)
        observed = metrics.get(resource)
        if limit is None or observed is None or float(limit) <= 0:
            continue
        if float(observed) > float(limit):
            violations.append(RuntimeBudgetViolation(resource, observed, limit))
    return violations
