"""Persisted models for AnnData checkpoint lineage and source preparation.

These models are ordinary backward-compatible ``AnalysisState`` payloads.  They
record analysis-data lineage only: they are not a runtime restore mechanism and
they do not judge scientific mutations.  ``blob_relative_path`` and application
tokens are Host-internal and must never be added to the public session payload.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_OPAQUE_ID = 240


def canonical_id_set_digest(values: Any) -> str:
    """Return the canonical digest of a complete, order-independent ID set.

    The capture service and the model validator must agree on this exactly, so
    it lives beside the model rather than being re-implemented in each service.
    """
    normalized = sorted({str(value) for value in (values or []) if str(value)})
    payload = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_id_set(values: Any) -> List[str]:
    """Return a deduplicated, lexicographically sorted string ID list."""
    return sorted({str(value) for value in (values or []) if str(value)})


def _validate_opaque_id(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > _MAX_OPAQUE_ID or "\x00" in text or "/" in text or "\\" in text:
        raise ValueError(f"{field_name} is not a valid opaque identifier")
    return text


def _validate_sha256(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None or str(value) == "":
        return None
    text = str(value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex string")
    return text


def _validate_complete_id_set(
    *,
    values: Any,
    count: int,
    digest: str,
    prefix: str,
) -> None:
    normalized = normalize_id_set(values)
    has_payload = bool(normalized) or bool(count) or bool(str(digest or ""))
    if not has_payload:
        return
    if int(count) != len(normalized):
        raise ValueError(f"{prefix}_count does not match the complete id set")
    expected = canonical_id_set_digest(normalized)
    if str(digest or "") != expected:
        raise ValueError(f"{prefix}_set_digest does not match the complete id set")


def _validate_bounded_id_sample(
    *,
    values: Any,
    count: int,
    digest: str,
    prefix: str,
) -> None:
    """Validate a bounded navigation sample against a complete-set claim.

    The complete set is deliberately not persisted; only its count and digest
    are.  Therefore this validator must *not* recompute the digest from the
    sample: it proves syntactic validity, uniqueness, a subset relationship by
    size, and a well-formed digest claim.
    """
    raw = [str(value) for value in (values or []) if str(value)]
    normalized = normalize_id_set(raw)
    if raw != normalized:
        raise ValueError(f"{prefix} sample must be sorted and unique")
    if len(normalized) > 200:
        raise ValueError(f"{prefix} sample exceeds the 200-id bound")
    for value in normalized:
        _validate_opaque_id(value, prefix)
    if len(normalized) > int(count):
        raise ValueError(f"{prefix} sample exceeds its complete count")
    if str(digest or ""):
        _validate_sha256(str(digest), f"{prefix}_set_digest")
        if int(count) == len(normalized):
            expected = canonical_id_set_digest(normalized)
            if str(digest) != expected:
                raise ValueError(
                    f"{prefix}_set_digest does not match the complete sample"
                )
    elif int(count) or normalized:
        raise ValueError(f"{prefix}_set_digest is required when a set is claimed")


class AdataCheckpointRecord(BaseModel):
    """One logical checkpoint boundary and its immutable blob identity."""

    checkpoint_id: str = Field(
        default_factory=lambda: f"cp_{uuid.uuid4().hex}",
        min_length=1,
        max_length=240,
    )
    status: Literal["available", "failed", "unavailable"]

    boundary: Literal["stage1_complete", "round_complete", "supplementary_complete"]
    boundary_event_id: str = Field(min_length=1, max_length=240)
    round_number: int = Field(default=0, ge=0)
    planning_stage: Literal["stage1", "stage2", "full"] = "full"
    boundary_revision: int = Field(default=1, ge=1)
    phase: str = ""

    session_lineage_id: str = "main"
    adata_lineage_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None
    supersedes_checkpoint_id: Optional[str] = None

    source_plan_id: Optional[str] = None
    active_source_preparation_id: Optional[str] = None
    source_preparation_ids_at_capture: List[str] = Field(default_factory=list)
    source_preparation_id_count_at_capture: int = Field(default=0, ge=0)
    source_preparation_id_set_digest_at_capture: str = ""
    source_scope_id_at_capture: Optional[str] = None
    executor_settlement_scope_id_at_capture: Optional[str] = None
    supplementary_request_id: Optional[str] = None
    round_result_identity_key: Optional[str] = None
    direction: str = ""
    research_label: str = Field(default="", max_length=240)
    summary_at_capture: str = Field(default="", max_length=600)
    execution_ids_at_capture: List[str] = Field(default_factory=list)
    attempt_execution_ids_at_capture: List[str] = Field(default_factory=list)
    execution_id_count_at_capture: int = Field(default=0, ge=0)
    execution_id_set_digest_at_capture: str = ""
    attempt_execution_id_count_at_capture: int = Field(default=0, ge=0)
    attempt_execution_id_set_digest_at_capture: str = ""
    outcome_at_capture: str = ""

    captured_runtime_instance_id: Optional[str] = None
    captured_runtime_revision: Optional[int] = Field(default=None, ge=0)

    blob_sha256: Optional[str] = None
    blob_relative_path: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)
    n_obs: Optional[int] = Field(default=None, ge=0)
    n_vars: Optional[int] = Field(default=None, ge=0)
    same_as_parent: bool = False

    error_summary: Optional[str] = None
    unavailable_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator(
        "checkpoint_id",
        "boundary_event_id",
        "source_plan_id",
        "active_source_preparation_id",
        "source_scope_id_at_capture",
        "executor_settlement_scope_id_at_capture",
        "supplementary_request_id",
        "parent_checkpoint_id",
        "supersedes_checkpoint_id",
        "round_result_identity_key",
        "adata_lineage_id",
        "session_lineage_id",
        mode="before",
    )
    @classmethod
    def _strip_opaque_fields(cls, value: Any, info) -> Any:
        if value is None:
            return None
        return str(value).strip()

    @model_validator(mode="after")
    def _validate_technical_coherence(self) -> "AdataCheckpointRecord":
        _validate_complete_id_set(
            values=self.source_preparation_ids_at_capture,
            count=self.source_preparation_id_count_at_capture,
            digest=self.source_preparation_id_set_digest_at_capture,
            prefix="source_preparation_id",
        )
        _validate_bounded_id_sample(
            values=self.execution_ids_at_capture,
            count=self.execution_id_count_at_capture,
            digest=self.execution_id_set_digest_at_capture,
            prefix="execution_id",
        )
        _validate_bounded_id_sample(
            values=self.attempt_execution_ids_at_capture,
            count=self.attempt_execution_id_count_at_capture,
            digest=self.attempt_execution_id_set_digest_at_capture,
            prefix="attempt_execution_id",
        )

        complete_source_ids = set(normalize_id_set(self.source_preparation_ids_at_capture))
        if (
            self.active_source_preparation_id
            and complete_source_ids
            and self.active_source_preparation_id not in complete_source_ids
        ):
            raise ValueError("active source preparation is not in the capture set")
        if self.boundary == "supplementary_complete":
            if not self.supplementary_request_id:
                raise ValueError("supplementary_complete requires a request id")
            if not self.executor_settlement_scope_id_at_capture:
                raise ValueError(
                    "supplementary_complete requires an executor settlement scope"
                )

        if self.status == "available":
            if not self.adata_lineage_id:
                raise ValueError("available checkpoint requires an AnnData lineage")
            if self.blob_sha256 is None:
                raise ValueError("available checkpoint requires a blob digest")
            self.blob_sha256 = _validate_sha256(self.blob_sha256, "blob_sha256")
            expected_path = (
                f"adata-checkpoints/blobs/{self.blob_sha256}.h5ad"
                if self.blob_sha256 else None
            )
            if self.blob_relative_path != expected_path:
                raise ValueError(
                    "available checkpoint requires its validated session-relative path"
                )
            if self.size_bytes is None or self.n_obs is None or self.n_vars is None:
                raise ValueError(
                    "available checkpoint requires size and AnnData dimensions"
                )
        elif self.status == "failed":
            if any(
                value not in (None, "")
                for value in (self.blob_sha256, self.blob_relative_path)
            ):
                raise ValueError("failed capture must not claim a blob")
            if self.parent_checkpoint_id is not None:
                raise ValueError("failed capture must not claim a parent lineage")
            if self.adata_lineage_id is not None:
                # A failed attempt may be recorded in the active lineage's
                # history, but it is not itself an available ancestor.
                self.parent_checkpoint_id = None
        elif self.status == "unavailable":
            if self.blob_sha256:
                self.blob_sha256 = _validate_sha256(self.blob_sha256, "blob_sha256")
                expected_path = f"adata-checkpoints/blobs/{self.blob_sha256}.h5ad"
                if self.blob_relative_path != expected_path:
                    raise ValueError(
                        "unavailable checkpoint has an invalid historical blob path"
                    )
        return self


class AdataSourceSelection(BaseModel):
    """One PI plan's explicit or inherited AnnData source choice."""

    mode: Literal[
        "current",
        "checkpoint_copy",
        "checkpoint_branch",
        "live_binding",
        "promote_live",
    ] = "current"
    checkpoint_id: Optional[str] = None
    live_binding_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_selector_combination(self) -> "AdataSourceSelection":
        checkpoint_id = _validate_opaque_id(self.checkpoint_id, "checkpoint_id")
        live_binding_id = _validate_opaque_id(self.live_binding_id, "live_binding_id")
        self.checkpoint_id = checkpoint_id
        self.live_binding_id = live_binding_id

        if self.mode in {"checkpoint_copy", "checkpoint_branch"}:
            if not checkpoint_id:
                raise ValueError(f"{self.mode} requires checkpoint_id")
            if live_binding_id:
                raise ValueError(f"{self.mode} must not carry live_binding_id")
        elif self.mode in {"live_binding", "promote_live"}:
            if not live_binding_id:
                raise ValueError(f"{self.mode} requires live_binding_id")
            if checkpoint_id:
                raise ValueError(f"{self.mode} must not carry checkpoint_id")
        else:  # current
            if checkpoint_id or live_binding_id:
                raise ValueError("current must not carry checkpoint/live selectors")
        return self


class AdataSourcePreparation(BaseModel):
    """Host-owned record identifying one applied plan/supplementary source."""

    preparation_id: str = Field(
        default_factory=lambda: f"prep_{uuid.uuid4().hex}",
        min_length=1,
        max_length=240,
    )
    scope_id: str = Field(
        default_factory=lambda: f"prep_scope_{uuid.uuid4().hex}",
        min_length=1,
        max_length=240,
    )
    owner_kind: Literal["plan", "supplementary"] = "plan"
    owner_id: str = Field(min_length=1, max_length=240)
    consumer_plan_ids: List[str] = Field(default_factory=list)
    consumer_supplementary_request_ids: List[str] = Field(default_factory=list)
    selection: AdataSourceSelection = Field(default_factory=AdataSourceSelection)
    selection_signature: str = ""
    application_token: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        min_length=1,
        max_length=240,
    )
    session_lineage_id: str = "main"
    origin_session_lineage_id: Optional[str] = None
    reconcile_from_session_lineage_id: Optional[str] = None
    binding_id: Optional[str] = None
    binding_name: Optional[str] = None
    anchor_name: Optional[str] = None
    binding_identity_status: Literal[
        "exact", "evolved", "missing", "unknown",
    ] = "unknown"
    adata_lineage_id: Optional[str] = None
    loaded_checkpoint_id: Optional[str] = None
    ancestor_checkpoint_id: Optional[str] = None
    status: Literal[
        "pending", "applied_warm", "durable", "pending_reconcile",
        "unresolved", "failed",
    ] = "pending"
    runtime_instance_id: Optional[str] = None
    runtime_revision: Optional[int] = Field(default=None, ge=0)
    lifecycle_generation: Optional[int] = Field(default=None, ge=0)
    runtime_manifest_revision: Optional[int] = Field(default=None, ge=0)
    runtime_pkl_sha256: Optional[str] = None
    runtime_pairing_state_version: Optional[int] = Field(default=None, ge=0)
    runtime_pairing_checkpoint: Optional[str] = None
    registry_saved_name_ok: Optional[bool] = None
    binding_saved_names_ok: Optional[bool] = None
    error_summary: Optional[str] = Field(default=None, max_length=1000)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator(
        "preparation_id",
        "scope_id",
        "owner_id",
        "binding_id",
        "binding_name",
        "anchor_name",
        "adata_lineage_id",
        "loaded_checkpoint_id",
        "ancestor_checkpoint_id",
        "runtime_instance_id",
        "origin_session_lineage_id",
        "reconcile_from_session_lineage_id",
        mode="before",
    )
    @classmethod
    def _strip_ids(cls, value: Any, info) -> Any:
        if value is None:
            return None
        return str(value).strip()

    @model_validator(mode="after")
    def _validate_preparation(self) -> "AdataSourcePreparation":
        if not self.scope_id:
            raise ValueError("scope_id is required")
        if not self.owner_id:
            raise ValueError("owner_id is required")
        if self.selection_signature:
            self.selection_signature = _validate_sha256(
                self.selection_signature, "selection_signature"
            ) or ""
        if self.runtime_pkl_sha256:
            self.runtime_pkl_sha256 = _validate_sha256(
                self.runtime_pkl_sha256, "runtime_pkl_sha256"
            )
        if self.selection.mode == "current" and self.binding_id:
            raise ValueError("current selection must not have a binding id")
        return self


class AdataLiveBindingRecord(BaseModel):
    """Host-side catalog of one registered live working object."""

    binding_id: str = Field(
        default_factory=lambda: f"lb_{uuid.uuid4().hex}",
        min_length=1,
        max_length=240,
    )
    name: str = Field(min_length=1, max_length=240)
    anchor_name: str = Field(min_length=1, max_length=240)
    mode_created: Literal["checkpoint_copy", "live_binding", "promote_live"]
    adata_lineage_id: str = Field(min_length=1, max_length=240)
    created_from_checkpoint_id: Optional[str] = None
    ancestor_checkpoint_id: Optional[str] = None
    preparation_id: str = Field(min_length=1, max_length=240)
    session_lineage_id: str = "main"
    origin_session_lineage_id: Optional[str] = None
    reconcile_from_session_lineage_id: Optional[str] = None
    status: Literal[
        "registered", "pending_reconcile", "stale", "unresolved", "deleted",
    ] = "registered"
    last_runtime_instance_id: Optional[str] = None
    last_runtime_revision: Optional[int] = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator(
        "binding_id",
        "name",
        "anchor_name",
        "adata_lineage_id",
        "created_from_checkpoint_id",
        "ancestor_checkpoint_id",
        "preparation_id",
        "session_lineage_id",
        "origin_session_lineage_id",
        "reconcile_from_session_lineage_id",
        "last_runtime_instance_id",
        mode="before",
    )
    @classmethod
    def _strip_binding_ids(cls, value: Any, info) -> Any:
        if value is None:
            return None
        return str(value).strip()
