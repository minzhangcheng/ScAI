"""Canonical required-runtime registry and declaration tracking for Snapshot V2.

The registry is the single source of truth for which kernel globals must be
captured. It unions:
  1. Canonical artifact names present in live globals
  2. AnnData-typed globals (any name)
  3. Explicit tool-declared persistent_outputs
  4. Continuation checkpoint dependencies
  5. Cross-cell promotions (created then later loaded)
"""

from __future__ import annotations

from backend.services.snapshot_v2.models import (
    RuntimeDependency,
    RuntimeDependencyManifest,
)

CANONICAL_RUNTIME_ARTIFACTS: frozenset[str] = frozenset({
    "adata",
    "adata_control",
    "adata_treatment",
    "adata_merged",
    "ldata",
    "ldata_control",
    "ldata_treatment",
    "results",
    "markers",
    "deg_results",
    "pathway_results",
    "cell_types",
    "cluster_markers",
    "differential_genes",
})


def _is_anndata_type(type_str: str) -> bool:
    if not type_str:
        return False
    lowered = type_str.lower()
    return "anndata" in lowered or "AnnData" in type_str


_KNOWN_UNPERSISTABLE_RESIDUAL_MARKERS: tuple[str, ...] = (
    "loompy.loompy.LoomConnection",
    "loompy.LoomConnection",
    "h5py._hl.files.File",
    "h5py._hl.group.Group",
    "h5py._hl.dataset.Dataset",
    "tables.file.File",
    "sqlite3.Connection",
    "_io.BufferedReader",
    "_io.BufferedWriter",
    "_io.TextIOWrapper",
)


def is_known_unpersistable_residual(type_str: str) -> bool:
    """Return True for residual types that cannot be dill-serialized safely.

    Used to soft-skip advisory residual globals (open file handles, loom
    connections) instead of permanently blocking required snapshots.
    """
    if not type_str:
        return False
    t = type_str.strip()
    if t in _KNOWN_UNPERSISTABLE_RESIDUAL_MARKERS:
        return True
    lowered = t.lower()
    # Substring matches for common file-handle patterns
    for marker in (
        "loomconnection",
        "h5py._hl.files.file",
        "sqlite3.connection",
        "textiowrapper",
        "bufferedreader",
        "bufferedwriter",
    ):
        if marker in lowered:
            return True
    return False


def _serializer_for(name: str, type_str: str) -> str:
    if _is_anndata_type(type_str):
        return "h5ad"
    return "dill"


def build_required_runtime_manifest(
    globals_meta: dict[str, str] | None,
    persistent_outputs: set[str] | None = None,
    checkpoint_dependencies: set[str] | None = None,
) -> RuntimeDependencyManifest:
    """Build the authoritative required-set proof from live globals + declarations.

    A name is required when it is:
    - present in globals and is AnnData-typed, or
    - present in globals and listed in CANONICAL_RUNTIME_ARTIFACTS, or
    - explicitly declared via persistent_outputs / checkpoint_dependencies
      (even if not yet present in globals — capture will fail closed if missing).
    """
    globals_meta = dict(globals_meta or {})
    required_names: set[str] = set()

    for name, type_str in globals_meta.items():
        if _is_anndata_type(type_str) or name in CANONICAL_RUNTIME_ARTIFACTS:
            required_names.add(name)

    if persistent_outputs:
        required_names.update(n for n in persistent_outputs if n)
    if checkpoint_dependencies:
        required_names.update(n for n in checkpoint_dependencies if n)

    required: dict[str, RuntimeDependency] = {}
    for name in sorted(required_names):
        type_str = globals_meta.get(name, "")
        required[name] = RuntimeDependency(
            name=name,
            serializer=_serializer_for(name, type_str),  # type: ignore[arg-type]
            type_name=type_str or "unknown",
        )
    return RuntimeDependencyManifest(required=required)


def build_required_runtime_set(
    globals_meta: dict[str, str] | None,
    persistent_outputs: set[str] | None = None,
    checkpoint_dependencies: set[str] | None = None,
) -> set[str]:
    """Convenience set view of :func:`build_required_runtime_manifest`."""
    manifest = build_required_runtime_manifest(
        globals_meta=globals_meta,
        persistent_outputs=persistent_outputs,
        checkpoint_dependencies=checkpoint_dependencies,
    )
    return set(manifest.required)


class RuntimeDependencyTracker:
    """Tracks created/loaded/declared user-globals across successful cells.

    A value created by an earlier cell and loaded by a later cell is promoted
    into ``required_outputs`` before the next snapshot boundary. Explicit
    ``declared`` names are required immediately. Dynamic namespace reads still
    depend on explicit ``persistent_outputs`` (declared).
    """

    def __init__(self) -> None:
        self._created: set[str] = set()
        self._required: set[str] = set()

    @property
    def required_outputs(self) -> set[str]:
        return set(self._required)

    @property
    def created_outputs(self) -> set[str]:
        return set(self._created)

    def record_execution(
        self,
        *,
        created: set[str] | None = None,
        loaded: set[str] | None = None,
        declared: set[str] | None = None,
        removed: set[str] | None = None,
    ) -> set[str]:
        created = set(created or set())
        loaded = set(loaded or set())
        declared = set(declared or set())
        removed = set(removed or set())

        self._created.update(created)
        # Declared outputs are required immediately.
        self._required.update(declared)
        # Cross-cell promotion: prior-created name later loaded becomes required.
        promoted = loaded & self._created
        self._required.update(promoted)
        # A name removed from the namespace (del / rebinding out of existence) is
        # no longer capturable and must drop out of the required set. Without this
        # a transient object (e.g. a loompy connection) declared as a persistent
        # output then deleted would stay permanently required and fail every
        # subsequent snapshot capture (required_runtime_unpersistable).
        if removed:
            self._required.difference_update(removed)
            self._created.difference_update(removed)
        return set(self._required)

    def merge_declared(self, names: set[str] | list[str] | None) -> None:
        if names:
            self._required.update(n for n in names if n)

    def to_state_dict(self) -> dict:
        return {
            "created": sorted(self._created),
            "required": sorted(self._required),
        }

    @classmethod
    def from_state_dict(cls, data: dict | None) -> "RuntimeDependencyTracker":
        tracker = cls()
        if not data:
            return tracker
        tracker._created = set(data.get("created") or [])
        tracker._required = set(data.get("required") or [])
        return tracker
