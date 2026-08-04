"""Configurable live-runtime resource budgets and deterministic evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any


_MEMORY_FRACTIONS = {
    "max_single_object_bytes": 0.25,
    "max_anndata_total_bytes": 0.45,
    "max_rss_bytes": 0.60,
    "max_pkl_bytes": 0.30,
}
_UNBOUNDED_CGROUP_SENTINEL = 1 << 60

_CAPACITY_ENV_VARS = {
    "max_single_object_bytes": "SCAI_RUNTIME_MAX_SINGLE_OBJECT_GB",
    "max_anndata_total_bytes": "SCAI_RUNTIME_MAX_ANNDATA_TOTAL_GB",
    "max_rss_bytes": "SCAI_RUNTIME_MAX_RSS_GB",
    "max_pkl_bytes": "SCAI_RUNTIME_MAX_PKL_GB",
}
_POSITIVE_INT_ENV_VARS = {
    "max_namespace_count": "SCAI_RUNTIME_MAX_NAMESPACE_COUNT",
}
_POSITIVE_FLOAT_ENV_VARS = {
    "max_save_seconds": "SCAI_RUNTIME_MAX_SAVE_SECONDS",
}


def apply_runtime_budget_environment(
    configured: dict[str, int | float | None],
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, int | float | None]:
    """Overlay deployment-only runtime persistence budget settings.

    Capacity variables are expressed in GiB. Missing values preserve the
    supplied configuration, while ``auto`` explicitly removes an absolute
    ceiling so :func:`resolve_effective_runtime_budgets` derives one from the
    Docker cgroup or host memory capacity. Invalid values are ignored with a
    warning instead of making the backend unavailable because of an operator
    typo.
    """
    env = os.environ if environ is None else environ
    result: dict[str, int | float | None] = dict(configured)
    logger = logging.getLogger(__name__)

    for key, name in _CAPACITY_ENV_VARS.items():
        raw = str(env.get(name, "") or "").strip()
        if not raw:
            continue
        if raw.lower() == "auto":
            result[key] = None
            continue
        try:
            value_gib = float(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not a number of GiB or 'auto'; keeping %r",
                name,
                raw,
                result.get(key),
            )
            continue
        if not math.isfinite(value_gib) or value_gib <= 0:
            logger.warning(
                "%s=%r must be a positive finite number of GiB; keeping %r",
                name,
                raw,
                result.get(key),
            )
            continue
        result[key] = max(1, int(value_gib * 1024 ** 3))

    for key, name in _POSITIVE_INT_ENV_VARS.items():
        raw = str(env.get(name, "") or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not a positive integer; keeping %r",
                name,
                raw,
                result.get(key),
            )
            continue
        if value <= 0:
            logger.warning(
                "%s=%r must be a positive integer; keeping %r",
                name,
                raw,
                result.get(key),
            )
            continue
        result[key] = value

    for key, name in _POSITIVE_FLOAT_ENV_VARS.items():
        raw = str(env.get(name, "") or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not a positive number; keeping %r",
                name,
                raw,
                result.get(key),
            )
            continue
        if not math.isfinite(value) or value <= 0:
            logger.warning(
                "%s=%r must be a positive finite number; keeping %r",
                name,
                raw,
                result.get(key),
            )
            continue
        result[key] = value

    return result


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
    configured: dict[str, int | float | None],
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
            dynamic_limit = max(1, int(memory_capacity_bytes * fraction))
            if configured_limit is None:
                effective[key] = dynamic_limit
            elif float(configured_limit) > 0:
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
