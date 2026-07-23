"""Crash-recoverable multi-artifact restore coordinator for Snapshot V2.

Provides ``RestoreCoordinator`` with four-phase lifecycle::

    stage(snapshot_id, ...) -> StagedRestore
    apply(staged, paths, ...) -> AppliedRestore
    commit(transaction_id, paths) -> None
    cleanup(transaction_id, paths) -> None

Every transition is journaled under ``journals/<transaction_id>.json`` with
HMAC integrity, so ``recover_snapshot_transactions`` in ``recovery.py`` can
complete or roll back an interrupted transaction.
"""

import hashlib
import hmac as _hmac
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.models import ObjectRef
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.services.snapshot_v2.runtime import validate_runtime_dependency_proof
from backend.services.snapshot_v2.runtime_names import runtime_h5ad_path
from backend.services.snapshot_v2.workspace import WorkspaceScope

logger = logging.getLogger(__name__)


# -- Result types ------------------------------------------------------------


# Implementations live in ./restore_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from .restore_parts.journal import (  # noqa: F401
    StagedRestore,
    AppliedRestore,
    RestoreJournal,
    _new_transaction_id,
    _journal_path,
    _journal_hmac,
    _canonicalize_journal_payload,
    _restore_dict_from_envelope_payload,
    _restore_journal_from_body,
    _verify_flat_legacy_restore_hmac,
    read_verified_restore_journal,
    _read_journal,
    _write_journal_hmac_signed,
    migrate_verified_restore_v0,
)
from .restore_parts.staging import (  # noqa: F401
    _stream_copy,
    _atomic_copy,
    _atomic_replace,
    _require_declared_object,
    _stage_object_ref,
    _validate_staged_analysis_state,
    _validate_staged_notebook,
    _validate_staged_checkpoint,
    _validate_checkpoint_capsule_compatibility,
    _iter_manifest_object_refs,
    _runtime_status_from_manifest,
)
from .restore_parts.coordinator import (  # noqa: F401
    RestoreCoordinator,
)







# -- Journal helpers ---------------------------------------------------------












































# -- RestoreCoordinator ------------------------------------------------------




