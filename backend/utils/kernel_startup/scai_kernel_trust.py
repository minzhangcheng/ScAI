"""Kernel-internal publication finalization capabilities.

The persistent IPython ``user_ns`` is intentionally mutable and therefore
cannot own renderer verdict dependencies. Kernel bootstrap captures the
pristine callables here once; publication wrappers retrieve an immutable tuple
from this trusted startup module before running generated plotting code.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class RuntimeTrust(NamedTuple):
    builtins_map: dict[str, Any]
    print_fn: Any
    json_dumps: Any
    stdout: Any
    get_parent_header: Any
    set_stdout_parent: Any
    plt_close: Any
    plt_show: Any
    plt_get_fignums: Any
    plt_gcf: Any
    figure_savefig: Any
    runtime_savefig: Any


_RUNTIME_TRUST: RuntimeTrust | None = None


def capture_runtime(**capabilities: Any) -> None:
    """Capture trusted runtime callables exactly once during kernel bootstrap."""
    global _RUNTIME_TRUST
    if _RUNTIME_TRUST is not None:
        raise RuntimeError("kernel runtime trust was already initialized")
    _RUNTIME_TRUST = RuntimeTrust(**capabilities)


def get_runtime_trust() -> RuntimeTrust:
    """Return the bootstrap-owned capability tuple or fail closed."""
    if _RUNTIME_TRUST is None:
        raise RuntimeError("kernel runtime trust is not initialized")
    return _RUNTIME_TRUST
