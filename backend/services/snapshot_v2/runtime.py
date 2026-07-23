"""Runtime capture for Snapshot V2.

Owns kernel introspection, required-set assembly via the canonical registry,
and authoritative serializer success (h5ad + dill).
"""

from __future__ import annotations

import ast
import json
import shutil
import uuid
from pathlib import Path

from backend.services.snapshot_v2.errors import SnapshotCaptureError, SnapshotIntegrityError
from backend.utils.error_parsing import strip_ansi
from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeDependency,
    RuntimeDependencyManifest,
    RuntimeManifest,
    RuntimeSkip,
    empty_complete_runtime,
)
from backend.services.snapshot_v2.object_store import ObjectStore
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.runtime_names import (
    runtime_h5ad_path,
    validate_runtime_global_name,
)
from backend.services.snapshot_v2.runtime_registry import (
    CANONICAL_RUNTIME_ARTIFACTS,
    build_required_runtime_manifest,
    build_required_runtime_set,
    is_known_unpersistable_residual,
)


# Re-export for existing imports / tests.

# Implementations live in ./runtime_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from .runtime_parts.kernel_templates import (  # noqa: F401
    _INTROSPECT_CODE,
    _SAVE_ANNDATA_TEMPLATE,
    _SAVE_DILL_TEMPLATE,
)
from .runtime_parts.anndata_sanitize import (  # noqa: F401
    sanitize_anndata_for_h5ad,
)
from .runtime_parts.proof import (  # noqa: F401
    RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES,
    _parse_dill_skipped,
    _parse_dill_name_line,
    _parse_dill_saved_names,
    _parse_dill_verified_names,
    _parse_name_list,
    validate_runtime_dependency_proof,
    _validate_h5ad_file,
)
from .runtime_parts.capture import (  # noqa: F401
    RuntimeCapture,
)
__all__ = [
    "CANONICAL_RUNTIME_ARTIFACTS",
    "RuntimeCapture",
    "RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES",
    "build_required_runtime_manifest",
    "build_required_runtime_set",
    "sanitize_anndata_for_h5ad",
    "validate_runtime_dependency_proof",
    "_parse_dill_skipped",
]
























