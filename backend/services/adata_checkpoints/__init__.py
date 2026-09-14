"""AnnData checkpoint lineage service.

Boundary capture and immutable blob storage live here.  This package does not
import the workflow package; workflow hooks call it through
``backend.workflow.adata_checkpoint_hooks``.
"""
from .capture import CaptureOutcome, capture_boundary, capture_boundary_locked
from .errors import (
    AdataCaptureError,
    AdataCheckpointError,
    AdataSourcePreparationError,
    AdataStorageError,
)
from .layout import (
    BlobIdentity,
    blob_path,
    blob_relative_path,
    canonical_blob_relative_path,
    checkpoint_root,
    verify_blob,
)
from .reconcile import (
    mark_runtime_reconciliation_pending,
    rebase_inherited_source_state,
    reconcile_catalog_availability,
    scrub_runtime_dependent_state,
)
from .projection import (
    build_checkpoint_projection,
    build_prompt_catalog_text,
)
from .selection import ensure_adata_source_prepared, selection_signature
from .storage import (
    StoragePublishResult,
    prune_unreferenced_checkpoint_blobs,
    publish_or_reuse,
)

__all__ = [
    "CaptureOutcome",
    "capture_boundary",
    "capture_boundary_locked",
    "AdataCaptureError",
    "AdataCheckpointError",
    "AdataSourcePreparationError",
    "AdataStorageError",
    "BlobIdentity",
    "blob_path",
    "blob_relative_path",
    "canonical_blob_relative_path",
    "checkpoint_root",
    "verify_blob",
    "StoragePublishResult",
    "publish_or_reuse",
    "prune_unreferenced_checkpoint_blobs",
    "ensure_adata_source_prepared",
    "selection_signature",
    "build_checkpoint_projection",
    "build_prompt_catalog_text",
    "reconcile_catalog_availability",
    "scrub_runtime_dependent_state",
    "mark_runtime_reconciliation_pending",
    "rebase_inherited_source_state",
]
