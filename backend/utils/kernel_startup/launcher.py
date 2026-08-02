"""Trusted first-stage launcher for ScAI Jupyter kernels.

The launcher runs with ``-I -S`` from an absolute path.  It establishes the
native write boundary before executing the ordinary ipykernel interpreter, so
a workspace-provided ``sitecustomize.py`` cannot run ahead of Landlock.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


_STARTUP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STARTUP_DIR))

from scai_kernel_isolation import activate_from_environment  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("missing kernel command")
    activate_from_environment()
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ.copy())


if __name__ == "__main__":
    main()
