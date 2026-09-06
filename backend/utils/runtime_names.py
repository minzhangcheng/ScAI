"""Validated runtime variable names used by Snapshot V2 sidecars."""

from __future__ import annotations

import re
from pathlib import Path

_RUNTIME_GLOBAL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def validate_runtime_global_name(name: str) -> str:
    """Return a safe kernel global name or raise ValueError.

    Snapshot runtime names are used both as Python globals and h5ad sidecar
    filenames.  Keeping the accepted set to normal identifiers avoids path
    traversal, hidden files, ambiguous stems, and names the sandbox cannot
    assign back into globals safely.
    """
    if not isinstance(name, str):
        raise ValueError("runtime variable name must be a string")
    if not _RUNTIME_GLOBAL_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid runtime variable name: {name!r}")
    return name


def runtime_h5ad_filename(name: str) -> str:
    """Return the h5ad sidecar filename for a validated runtime variable."""
    safe_name = validate_runtime_global_name(name)
    return f"{safe_name}.h5ad"


def runtime_h5ad_path(root: Path, name: str) -> Path:
    """Return a confined h5ad sidecar path below root."""
    return root / runtime_h5ad_filename(name)


def resolve_confined_runtime_h5ad_path(
    states_dir: Path, name: str, entry_path: str
) -> Path:
    """Resolve a manifest entry to a path confined under *states_dir*.

    Fail closed: only the exact basename ``{name}.h5ad`` is accepted.
    Absolute paths, nested paths, and path traversal are rejected.
    """
    safe_name = validate_runtime_global_name(name)
    expected = runtime_h5ad_filename(safe_name)
    if not isinstance(entry_path, str) or entry_path != expected:
        raise ValueError(
            f"h5ad sidecar path must be exactly {expected!r}, got {entry_path!r}"
        )
    root = states_dir.resolve()
    path = (root / expected).resolve()
    if path.parent != root:
        raise ValueError(f"h5ad sidecar path escapes states dir: {path}")
    return path
