from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError
from typing import Any, Literal, Annotated
from pathlib import PurePosixPath

from backend.models.schemas import SnapshotPhase

FORMAT_VERSION = 2

JournalKind = Literal[
    "restore_apply",
    "restore_job",
    "delete",
    "import",
    "export_protection",
]


class SnapshotJournalEnvelope(BaseModel):
    """Signed discriminated journal envelope (schema version 1).

    Canonical HMAC excludes ``hmac``, signs sorted compact JSON of the rest.
    Kind-specific fields live in ``payload``; top-level status is for dispatch.
    """

    schema_version: Literal[1] = 1
    kind: JournalKind
    transaction_id: str
    session_id: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    hmac: str = ""


class ObjectRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    media_type: str


class WorkspaceEntry(BaseModel):
    path: str
    object_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    mode: int = Field(ge=0)
    kind: Literal["file", "empty_dir"] = "file"

    @field_validator("path")
    @classmethod
    def confined_relative_path(cls, value: str) -> str:
        normalized = PurePosixPath(value)
        if normalized.is_absolute() or ".." in normalized.parts or value != normalized.as_posix():
            raise ValueError("workspace path must be normalized and relative")
        return value


class GraphNodeCapsule(BaseModel):
    kind: Literal["graph_node"] = "graph_node"
    graph_version: str
    next_node: str
    workflow_state_ref: ObjectRef
    planning_stage: Literal["stage1", "stage2", "full"]


class SemanticGateCapsule(BaseModel):
    kind: Literal["semantic_gate"] = "semantic_gate"
    graph_version: str
    gate: str
    wait_reason: str | None = None
    payload_ref: ObjectRef


class TerminalCapsule(BaseModel):
    kind: Literal["terminal"] = "terminal"
    graph_version: str
    status: Literal["completed", "stopped"]


class IdleCapsule(BaseModel):
    kind: Literal["idle"] = "idle"
    graph_version: str


ContinuationCapsule = Annotated[
    GraphNodeCapsule | SemanticGateCapsule | TerminalCapsule | IdleCapsule,
    Field(discriminator="kind"),
]


class RuntimeDependency(BaseModel):
    """One required runtime value and how it must be serialized."""

    name: str
    serializer: Literal["dill", "h5ad"]
    type_name: str = ""
    version: str = ""


class RuntimeSkip(BaseModel):
    """Incidental unpicklable global that was intentionally skipped."""

    name: str
    type_name: str = ""
    reason: str = ""


class RuntimeDependencyManifest(BaseModel):
    """Authoritative required-set proof for a runtime capture."""

    required: dict[str, RuntimeDependency] = Field(default_factory=dict)
    incidental_skips: list[RuntimeSkip] = Field(default_factory=list)


class RuntimeManifest(BaseModel):
    """Runtime availability and completeness proof for a snapshot."""

    status: Literal["complete", "missing", "state_only", "corrupt"]
    dependencies: RuntimeDependencyManifest
    residual_ref: ObjectRef | None = None
    captured_residual_names: list[str] = Field(default_factory=list)
    adata_refs: dict[str, ObjectRef] = Field(default_factory=dict)

    @model_validator(mode="after")
    def prove_complete(self):
        if self.residual_ref is not None and not self.captured_residual_names:
            raise ValueError("residual_ref requires captured names")
        if self.status == "complete":
            captured = set(self.adata_refs) | set(self.captured_residual_names)
            if self.captured_residual_names and self.residual_ref is None:
                raise ValueError("captured residual names require residual_ref")
            missing = set(self.dependencies.required) - captured
            if missing:
                raise ValueError(
                    f"complete runtime missing required values: {sorted(missing)}"
                )
        return self


# Compatibility aliases used by capture pipeline until Task 2 migrates producers.
class RuntimeDependencyEntry(BaseModel):
    name: str
    version: str
    type: str


class RuntimeGenerationManifest(BaseModel):
    """Legacy capture-side runtime generation record (Task 2 will replace usage)."""

    capture_command: str
    capture_timestamp: str
    kernel_id: str
    captured_adata_vars: list[str] = Field(default_factory=list)
    captured_residual_vars: list[str] = Field(default_factory=list)
    captured_adata_refs: dict[str, ObjectRef] = Field(default_factory=dict)
    captured_residual_ref: ObjectRef | None = None
    warnings: list[str] = Field(default_factory=list)


def runtime_manifest_from_generation(
    generation: RuntimeGenerationManifest | None,
    *,
    status: Literal["complete", "missing", "state_only", "corrupt"] = "complete",
    required: dict[str, RuntimeDependency] | None = None,
) -> RuntimeManifest:
    """Bridge legacy RuntimeGenerationManifest into RuntimeManifest."""
    if generation is None:
        return RuntimeManifest(
            status=status if status != "complete" else "missing",
            dependencies=RuntimeDependencyManifest(required=required or {}),
            residual_ref=None,
            captured_residual_names=[],
            adata_refs={},
        )
    residual_names = list(generation.captured_residual_vars or [])
    adata_refs = dict(generation.captured_adata_refs or {})
    # When required is not provided, treat all captured names as the required set
    # so a complete status remains self-consistent for migrated fixtures.
    if required is None:
        required = {}
        for name in adata_refs:
            required[name] = RuntimeDependency(
                name=name, serializer="h5ad", type_name="anndata.AnnData",
            )
        for name in residual_names:
            required[name] = RuntimeDependency(
                name=name, serializer="dill", type_name="unknown",
            )
    return RuntimeManifest(
        status=status,
        dependencies=RuntimeDependencyManifest(required=required),
        residual_ref=generation.captured_residual_ref,
        captured_residual_names=residual_names,
        adata_refs=adata_refs,
    )


def empty_complete_runtime() -> RuntimeManifest:
    """Complete runtime with empty required set (no live kernel dependency)."""
    return RuntimeManifest(
        status="complete",
        dependencies=RuntimeDependencyManifest(required={}),
        residual_ref=None,
        captured_residual_names=[],
        adata_refs={},
    )


def validate_system_commit(runtime: RuntimeManifest) -> RuntimeManifest:
    """System capture may publish only runtime status ``complete``."""
    if runtime.status != "complete":
        raise ValidationError.from_exception_data(
            "RuntimeManifest",
            [
                {
                    "type": "value_error",
                    "loc": ("status",),
                    "input": runtime.status,
                    "ctx": {
                        "error": (
                            "system capture may publish only runtime status "
                            f"'complete', got {runtime.status!r}"
                        ),
                    },
                }
            ],
        )
    # Re-validate completeness proof
    return RuntimeManifest.model_validate(runtime.model_dump(mode="json"))


class SnapshotManifestV2(BaseModel):
    """Authoritative committed Snapshot V2 manifest.

    Design fields only — no ambiguous ``branch``/``generation``/``captured_at``
    aliases. System capture requires analysis_state_ref, notebook_ref, and a
    complete runtime proof.
    """

    format_version: Literal[2] = 2
    snapshot_id: str
    session_id: str
    name: str = ""
    description: str = ""
    phase: SnapshotPhase | str = "manual"
    created_at: str
    round_number: int = 0
    parent_id: str | None = None
    branch_id: str = "main"
    branch_label: str = "main"
    source_restore_transaction_id: str | None = None
    analysis_state_ref: ObjectRef
    notebook_ref: ObjectRef
    checkpoint_ref: ObjectRef | None = None
    runtime: RuntimeManifest
    workspace: list[WorkspaceEntry] = Field(default_factory=list)
    continuation: ContinuationCapsule
    logical_size_bytes: int = Field(default=0, ge=0)
    physical_added_bytes: int = Field(default=0, ge=0)
    root_hash: str = Field(default="")
    objects: dict[str, ObjectRef] = Field(default_factory=dict)
    # Capture idempotency metadata (session_id, phase, key) uniqueness.
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def check_workspace_unique(self):
        paths = [e.path for e in self.workspace]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate workspace paths are not allowed")
        return self


class SnapshotCaptureRequest(BaseModel):
    session_id: str
    reason: str = "manual"
    name: str = ""
    description: str = ""
    phase: SnapshotPhase | str = "manual"
    round_number: int = 0
    parent_id: str | None = None
    branch_id: str = "main"
    branch_label: str = "main"
    source_restore_transaction_id: str | None = None
    continuation: ContinuationCapsule | None = None
    # Legacy field kept during Task 1 migration of producers; prefer continuation.
    capsule: ContinuationCapsule | None = None
    workspace: list[WorkspaceEntry] = Field(default_factory=list)
    objects: dict[str, ObjectRef] = Field(default_factory=dict)
    analysis_state_ref: ObjectRef | None = None
    notebook_ref: ObjectRef | None = None
    checkpoint_ref: ObjectRef | None = None
    runtime: RuntimeManifest | None = None
    # Legacy bridge fields until Task 2 capture pipeline lands.
    runtime_dependencies: RuntimeDependencyManifest | None = None
    runtime_snapshot_capture: RuntimeGenerationManifest | None = None
    # Idempotency key for required captures (session_id, phase, key) uniqueness.
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def normalize_continuation(self):
        if self.continuation is None and self.capsule is not None:
            self.continuation = self.capsule
        if self.continuation is None:
            raise ValueError("continuation (or legacy capsule) is required")
        return self


class SnapshotFailureRecord(BaseModel):
    snapshot_id: str
    session_id: str
    failed_at: str
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    allowed_actions: list[str] = Field(
        default_factory=lambda: ["retry", "manage", "terminate"]
    )


class DeleteResult(BaseModel):
    deleted_ids: list[str]
    freed_bytes: int
    surviving_manifests: list[str]


class DeleteJournalRecord(BaseModel):
    transaction_id: str
    session_id: str
    delete_set: list[str]
    surviving_manifest_ids: list[str]
    previous_lineage_head: str | None = None
    new_lineage_head: str | None = None
    status: Literal["STAGED", "APPLYING", "COMMITTED", "FAILED"] = "STAGED"
    error: str | None = None


class SnapshotRecordV2(BaseModel):
    snapshot_id: str
    manifest: SnapshotManifestV2 | None = None
    failure: SnapshotFailureRecord | None = None
    committed: bool | None = None
    session_reconciliation_pending: bool = False

    @model_validator(mode="after")
    def _default_committed(self):
        if self.committed is None:
            self.committed = self.failure is None and self.manifest is not None
        return self


class AuditBundleRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    audit_id: str
    snapshot_id: str
    manifest: SnapshotManifestV2
    failure: SnapshotFailureRecord


class SnapshotRestoreRequest(BaseModel):
    snapshot_id: str
    runtime_choice: Literal["reset", "keep"] | None = None
    current_state_choice: Literal["abort", "discard_current"] = "abort"
    discard_confirmation: bool = False

    @model_validator(mode="after")
    def reject_discard_keep(self):
        if self.current_state_choice == "discard_current" and self.runtime_choice == "keep":
            raise ValueError("discard_current cannot be combined with runtime keep")
        return self

    @model_validator(mode="after")
    def require_discard_confirmation(self):
        if self.current_state_choice == "discard_current" and not self.discard_confirmation:
            raise ValueError(
                "discard_current requires discard_confirmation=True"
            )
        return self


class SnapshotTransactionRecord(BaseModel):
    transaction_id: str
    session_id: str
    snapshot_id: str
    status: Literal[
        "queued", "waiting_safe_point", "staging", "staged", "applying",
        "committed", "failed", "cancelled", "timed_out",
    ]
    recovery_mode: Literal["rollback", "forward_only"] = "rollback"
    failure: dict | None = None


class SanitizedImportPlan(BaseModel):
    """Plan of objects and rebound manifests allowed to reach the object store.

    ``object_hashes`` is the only set permitted to be written via ``put_bytes``.
    Runtime hashes omitted when ``trust_runtime=False`` must not appear here.
    """

    object_hashes: set[str] = Field(default_factory=set)
    manifests: list[SnapshotManifestV2] = Field(default_factory=list)
    omitted_runtime_src_ids: set[str] = Field(default_factory=set)
    skipped_runtime_hashes: set[str] = Field(default_factory=set)
