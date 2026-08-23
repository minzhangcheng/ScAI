"""Trusted first-stage launcher for ScAI Jupyter kernels.

The launcher runs with ``-I -S`` from an absolute path.  It establishes the
native write boundary before executing the ordinary ipykernel interpreter, so
a workspace-provided ``sitecustomize.py`` cannot run ahead of Landlock.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

try:  # pragma: no cover - POSIX only; the kernel boundary is Linux-first
    import resource
except ImportError:  # pragma: no cover - Windows has no rlimits
    resource = None  # type: ignore[assignment]


_STARTUP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STARTUP_DIR))

from scai_kernel_isolation import activate_from_environment  # noqa: E402

ALLOW_CORE_DUMP_ENV = "SCAI_KERNEL_ALLOW_CORE_DUMP"


def suppress_core_dumps() -> None:
    """Prevent a faulting kernel from writing its address space to the session."""
    if resource is None:
        return
    if os.environ.get(ALLOW_CORE_DUMP_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        # Core suppression is diagnostic hardening and must not block startup.
        pass


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("missing kernel command")
    suppress_core_dumps()
    activate_from_environment()
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ.copy())


if __name__ == "__main__":
    main()
