"""Decision-point snapshots. Part B of the plain-persistence design.

A snapshot is a restore point taken immediately before one of the four
decisions an analysis asks about itself -- confirm the cell annotation, choose
the research direction after Stage 2, choose it again after each integration
cycle, accept a round's result. Restoring one creates a *new* session parked on
that same decision; the source session is never modified. Nothing else is
captured, nothing is captured on request, and an error or incident dialog is
not a decision.

The layer performs no serialisation of its own: it copies files the ordinary
save already committed and verifies them against the digests that commit
signed. See ``docs/coding/plain-persistence-part-b-spec.md``.
"""
from __future__ import annotations

from .gates import (
    DECISION_GATES,
    deletion_sort_key,
    in_deletion_order,
    mark_superseded,
    next_node_for,
    restored_status_for,
    retention_rank,
    round_boundary_gate,
    with_deletion_order,
)
from .layout import (
    GATE_WORDS,
    WORD_GATES,
    SnapshotsDisabled,
    failed_record_path,
    manifest_path,
    parse_snapshot_id,
    session_root,
    snapshot_dir,
    snapshot_id,
    snapshot_storage_root,
    snapshots_root,
    staging_root,
    uploads_root,
)
from .listing import read_session_listing
from .manifest import (
    SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    SnapshotManifestError,
    build_continuation,
    build_snapshot_manifest,
    manifest_sig_path,
    read_snapshot_manifest,
    validate_continuation,
    write_snapshot_manifest,
)

__all__ = [
    "DECISION_GATES",
    "GATE_WORDS",
    "SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "SnapshotManifestError",
    "SnapshotsDisabled",
    "WORD_GATES",
    "build_continuation",
    "build_snapshot_manifest",
    "deletion_sort_key",
    "failed_record_path",
    "in_deletion_order",
    "manifest_path",
    "manifest_sig_path",
    "mark_superseded",
    "next_node_for",
    "parse_snapshot_id",
    "read_session_listing",
    "read_snapshot_manifest",
    "restored_status_for",
    "retention_rank",
    "round_boundary_gate",
    "session_root",
    "snapshot_dir",
    "snapshot_id",
    "snapshot_storage_root",
    "snapshots_root",
    "staging_root",
    "uploads_root",
    "validate_continuation",
    "with_deletion_order",
    "write_snapshot_manifest",
]
