"""Validated runtime variable names used by Snapshot V2 sidecars."""

from __future__ import annotations

import re

_RUNTIME_GLOBAL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def validate_runtime_global_name(name: str) -> str:
    """Return a safe kernel global name or raise ValueError.

    Runtime names are used both as Python globals and legacy h5ad sidecar
    filenames.  Keeping the accepted set to normal identifiers avoids path
    traversal, hidden files, ambiguous stems, and names the sandbox cannot
    assign back into globals safely.
    """
    if not isinstance(name, str):
        raise ValueError("runtime variable name must be a string")
    if not _RUNTIME_GLOBAL_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid runtime variable name: {name!r}")
    return name
