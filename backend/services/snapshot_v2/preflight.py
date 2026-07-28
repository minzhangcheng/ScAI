"""Read-only preflight verification for Snapshot V2 restore."""

import json

from pydantic import BaseModel, Field

from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.models import (
    GraphNodeCapsule,
    SnapshotManifestV2,
    SnapshotRestoreRequest,
)
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.utils.runtime_apply import RuntimeChoiceRequired


class RestorePreflightResult(BaseModel):
    status: str = "ready"
    format_version: int = 2
    compatibility_errors: list[str] = Field(default_factory=list)
    objects_total: int = 0
    objects_verified: int = 0
    objects_failed: list[str] = Field(default_factory=list)
    required_staging_bytes: int = 0
    capsule: dict = Field(default_factory=dict)
    runtime_choice_supported: bool = True
    runtime_choice_required: bool = False


class RestorePreflight:
    """Read-only verification that a snapshot is ready to be restored.

    Loads the manifest, verifies integrity (HMAC, root hash, objects),
    checks compatibility (format version, capsule, runtime), and computes
    storage requirements. Never touches the live workflow or session files.
    """

    def __init__(self, paths):
        self._paths = paths
        self._manifest_store = ManifestStore(paths)
        self._object_store = ObjectStore(paths)

    def run(
        self,
        snapshot_id: str,
        request: SnapshotRestoreRequest | None = None,
    ) -> RestorePreflightResult:
        result = RestorePreflightResult()
        result.capsule = {}

        # 1. Check format version from raw JSON first
        committed_path = self._paths.committed / f"{snapshot_id}.json"
        if not committed_path.exists():
            return RestorePreflightResult(
                status="corrupted",
                compatibility_errors=[f"Manifest {snapshot_id} not found"],
            )

        try:
            raw = json.loads(committed_path.read_bytes())
        except (json.JSONDecodeError, OSError):
            return RestorePreflightResult(
                status="corrupted",
                compatibility_errors=[f"Manifest {snapshot_id} is not valid JSON"],
            )

        fmt = raw.get("format_version", 0)
        if fmt != 2:
            return RestorePreflightResult(
                status="incompatible",
                format_version=fmt,
                compatibility_errors=[
                    f"Snapshot format v{fmt} is not supported. "
                    f"Only format v2 manifests can be restored.",
                ],
            )

        # 2. Load and verify manifest integrity (HMAC + root hash)
        try:
            manifest: SnapshotManifestV2 = self._manifest_store.load(snapshot_id)
        except SnapshotIntegrityError as e:
            return RestorePreflightResult(
                status="corrupted",
                compatibility_errors=[f"Manifest integrity check failed: {e}"],
            )

        # Populate capsule info
        capsule_data = manifest.continuation
        if hasattr(manifest.continuation, "model_dump"):
            result.capsule = manifest.continuation.model_dump()
        elif isinstance(capsule_data, dict):
            result.capsule = capsule_data

        # 3. Verify all objects
        objects_total = len(manifest.objects)
        result.objects_total = objects_total

        objects_failed: list[str] = []
        objects_verified = 0
        total_bytes = 0

        for obj_hash, obj_ref in manifest.objects.items():
            try:
                self._object_store.verify(obj_ref)
                objects_verified += 1
                total_bytes += obj_ref.size
            except SnapshotIntegrityError:
                objects_failed.append(obj_hash)

        result.objects_verified = objects_verified
        result.objects_failed = objects_failed
        result.required_staging_bytes = total_bytes

        # 4. Verify workflow state content before declaring the snapshot ready.
        if isinstance(manifest.continuation, GraphNodeCapsule):
            state_path = (
                self._paths.objects
                / manifest.continuation.workflow_state_ref.sha256
            )
            if state_path.exists():
                try:
                    json.loads(state_path.read_bytes())
                except (json.JSONDecodeError, OSError):
                    result.compatibility_errors.append(
                        "Workflow state content is not valid JSON"
                    )

        # 4b. The capsule's resume target must exist in THIS build. Both checks
        # are read-only and every input is already in hand, but until now they
        # only ran in ContinuationDispatcher.stage_checkpoint — which happens
        # AFTER restore_coordinator.apply() has already replaced the workspace,
        # notebook, session.json, checkpoint and runtime. An archive imported
        # from another instance, or a snapshot taken before an in-place upgrade
        # that renamed a node, therefore passed preflight, destroyed the live
        # session, and only then was rejected as UNKNOWN_NEXT_NODE /
        # UNKNOWN_GATE, leaving the session recovery-blocked. Reject it while
        # rejecting is still free.
        from backend.services.snapshot_v2.dispatcher import (
            _KNOWN_GATES,
            _KNOWN_GRAPH_NODES,
        )

        capsule = manifest.continuation
        capsule_kind = getattr(capsule, "kind", None)
        if capsule_kind == "graph_node":
            node = getattr(capsule, "next_node", None)
            if not node:
                result.compatibility_errors.append(
                    "Graph-node capsule has no next_node"
                )
            elif node not in _KNOWN_GRAPH_NODES:
                result.compatibility_errors.append(
                    f"Graph-node capsule targets unknown graph node: {node}"
                )
        elif capsule_kind == "semantic_gate":
            gate = getattr(capsule, "gate", None)
            if not gate:
                result.compatibility_errors.append(
                    "Semantic-gate capsule has no gate"
                )
            elif gate not in _KNOWN_GATES:
                result.compatibility_errors.append(
                    f"Semantic-gate capsule targets unknown gate: {gate}"
                )

        # 5. Determine status
        if objects_failed:
            result.status = "corrupted"
        elif result.compatibility_errors:
            result.status = "corrupted"
        else:
            result.status = "ready"

        # 6. Runtime choice capability / enforcement. A complete runtime with
        # no payload is an exact empty runtime, so it needs no reset/keep choice.
        runtime_status = manifest.runtime.status
        needs_explicit_runtime_choice = runtime_status != "complete"
        result.runtime_choice_supported = True
        result.runtime_choice_required = needs_explicit_runtime_choice
        # Only enforce runtime_choice on restorable snapshots. Corrupted /
        # non-ready preflight must return its status instead of masking it.
        if (
            result.status == "ready"
            and request is not None
            and needs_explicit_runtime_choice
            and request.runtime_choice is None
        ):
            raise RuntimeChoiceRequired(runtime_status)

        return result
