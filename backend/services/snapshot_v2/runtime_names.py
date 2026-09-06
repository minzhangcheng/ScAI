"""Deprecated location. Snapshot V2 is deleted in A2-5; import from
``backend.utils.runtime_names`` instead."""

from backend.utils.runtime_names import (  # noqa: F401
    resolve_confined_runtime_h5ad_path,
    runtime_h5ad_filename,
    runtime_h5ad_path,
    validate_runtime_global_name,
)
