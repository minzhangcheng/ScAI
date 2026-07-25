"""Shared Snapshot V2 test fixtures.

Tests should customize ``make_snapshot_v2_manifest()`` rather than
duplicating full schema payloads after the Task 1 atomic schema migration.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeDependency,
    RuntimeDependencyManifest,
    RuntimeManifest,
    RuntimeSkip,
    SnapshotManifestV2,
    validate_system_commit,
)
from backend.tests.split_facade import patch_facade, setattr_facade
from backend.utils.crypto import get_integrity_secret

# Facade whose implementation lives in backend.services.snapshot_v2.service_parts;
# the helpers below rebind names on the facade *and* every part that reads them.
SNAPSHOT_SERVICE = "backend.services.snapshot_v2.service"


def patch_snapshot_service_session_states(states):
    """Patch snapshot_v2.service ``get_session_states`` to return ``states``.

    Context manager; yields the MagicMock standing in for the real lookup.
    """
    return patch_facade(SNAPSHOT_SERVICE, "get_session_states", return_value=states)


def set_snapshot_service_session_states(monkeypatch, resolver):
    """monkeypatch snapshot_v2.service ``get_session_states`` to ``resolver``."""
    setattr_facade(monkeypatch, SNAPSHOT_SERVICE, "get_session_states", resolver)


def patch_snapshot_service_session_lock(**mock_kwargs):
    """Patch snapshot_v2.service ``get_session_lock``, mock.patch keyword style."""
    return patch_facade(SNAPSHOT_SERVICE, "get_session_lock", **mock_kwargs)


def _ref(seed: str = "a", size: int = 1024, media_type: str = "application/json") -> ObjectRef:
    return ObjectRef(sha256=(seed * 64)[:64], size=size, media_type=media_type)


def dill_dependency(name: str = "results") -> RuntimeDependency:
    return RuntimeDependency(
        name=name,
        serializer="dill",
        type_name="builtins.dict",
    )


def dependency_manifest(
    required: dict[str, RuntimeDependency] | dict[str, str] | None = None,
    incidental_skips: list[RuntimeSkip] | None = None,
) -> RuntimeDependencyManifest:
    required = required or {}
    normalized: dict[str, RuntimeDependency] = {}
    for key, value in required.items():
        if isinstance(value, RuntimeDependency):
            normalized[key] = value
        elif isinstance(value, str):
            normalized[key] = RuntimeDependency(
                name=key,
                serializer=value,  # type: ignore[arg-type]
                type_name="unknown",
            )
        else:
            raise TypeError(f"unsupported dependency value for {key!r}: {type(value)}")
    return RuntimeDependencyManifest(
        required=normalized,
        incidental_skips=list(incidental_skips or []),
    )


def complete_runtime(
    *,
    required: dict[str, RuntimeDependency] | dict[str, str] | None = None,
    captured: dict[str, ObjectRef] | None = None,
    residual_names: list[str] | None = None,
    residual_ref: ObjectRef | None = None,
    status: str = "complete",
) -> dict[str, Any]:
    """Build a RuntimeManifest-compatible dict for validation tests."""
    required = required or {}
    captured = captured or {}
    residual_names = list(residual_names or [])
    if residual_names and residual_ref is None:
        residual_ref = _ref("r", size=64, media_type="application/octet-stream")
    return {
        "status": status,
        "dependencies": dependency_manifest(required=required).model_dump(mode="json"),
        "residual_ref": residual_ref.model_dump(mode="json") if residual_ref else None,
        "captured_residual_names": residual_names,
        "adata_refs": {k: v.model_dump(mode="json") for k, v in captured.items()},
    }


def runtime_manifest(
    *,
    status: str = "complete",
    required: dict[str, RuntimeDependency] | dict[str, str] | None = None,
    captured: dict[str, ObjectRef] | None = None,
    residual_names: list[str] | None = None,
    residual_ref: ObjectRef | None = None,
) -> RuntimeManifest:
    payload = complete_runtime(
        required=required,
        captured=captured,
        residual_names=residual_names,
        residual_ref=residual_ref,
        status=status,
    )
    return RuntimeManifest.model_validate(payload)


def minimal_manifest_payload(**overrides: Any) -> dict[str, Any]:
    """Minimal valid SnapshotManifestV2 dict using the frozen design fields."""
    state_ref = _ref("a", size=1024, media_type="application/json")
    notebook_ref = _ref("b", size=2048, media_type="application/x-ipynb+json")
    payload: dict[str, Any] = {
        "format_version": 2,
        "snapshot_id": "snap-001",
        "session_id": "sess-test",
        "name": "test-snapshot",
        "description": "",
        "phase": "manual",
        "created_at": "2026-07-11T12:00:00Z",
        "round_number": 0,
        "parent_id": None,
        "branch_id": "main",
        "branch_label": "main",
        "source_restore_transaction_id": None,
        "analysis_state_ref": state_ref.model_dump(mode="json"),
        "notebook_ref": notebook_ref.model_dump(mode="json"),
        "checkpoint_ref": None,
        "runtime": complete_runtime(required={}, captured={}),
        "workspace": [
            {
                "path": "figures/plot.png",
                "object_hash": ("c" * 64),
                "size": 2048,
                "mode": 0o644,
                "kind": "file",
            },
        ],
        "continuation": {
            "kind": "idle",
            "graph_version": "2.0",
        },
        "logical_size_bytes": 3072,
        "physical_added_bytes": 3072,
        "root_hash": "",
        "objects": {
            state_ref.sha256: state_ref.model_dump(mode="json"),
            notebook_ref.sha256: notebook_ref.model_dump(mode="json"),
            ("c" * 64): {
                "sha256": "c" * 64,
                "size": 2048,
                "media_type": "image/png",
            },
        },
    }
    payload.update(overrides)
    return payload


def make_snapshot_v2_manifest(**overrides: Any) -> SnapshotManifestV2:
    """Authoritative factory for committed SnapshotManifestV2 test instances."""
    payload = minimal_manifest_payload(**overrides)
    return SnapshotManifestV2.model_validate(payload)


def sign_manifest_bytes(manifest_bytes: bytes) -> str:
    """HMAC-SHA256 using the host integrity secret (production signing)."""
    secret = get_integrity_secret()
    key = secret if isinstance(secret, bytes) else secret.encode()
    return hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()


def write_manifest_to_store(
    manifest: SnapshotManifestV2,
    commit_dir: Path,
    integrity_dir: Path,
) -> SnapshotManifestV2:
    """Publish a manifest with root_hash + host-secret HMAC (matches ManifestStore)."""
    from backend.services.snapshot_v2.object_store import canonical_json

    commit_dir.mkdir(parents=True, exist_ok=True)
    integrity_dir.mkdir(parents=True, exist_ok=True)

    raw = canonical_json(manifest)
    root_hash = hashlib.sha256(raw).hexdigest()
    manifest.root_hash = root_hash
    full = manifest.model_dump(mode="json")
    manifest_bytes = json.dumps(
        full, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()

    manifest_path = commit_dir / f"{manifest.snapshot_id}.json"
    manifest_path.write_bytes(manifest_bytes)

    sig_path = integrity_dir / f"{manifest.snapshot_id}.sig"
    sig_path.write_text(sign_manifest_bytes(manifest_bytes))
    return manifest


def verify_manifest_hmac(
    integrity_dir: Path,
    committed_dir: Path,
    manifest: SnapshotManifestV2,
) -> bool:
    """Return True if the on-disk HMAC matches the host integrity secret."""
    sig_path = integrity_dir / f"{manifest.snapshot_id}.sig"
    if not sig_path.exists():
        return False
    manifest_bytes = (committed_dir / f"{manifest.snapshot_id}.json").read_bytes()
    sig = sig_path.read_text().strip()
    expected = sign_manifest_bytes(manifest_bytes)
    return hmac.compare_digest(expected, sig)


__all__ = [
    "SNAPSHOT_SERVICE",
    "complete_runtime",
    "dependency_manifest",
    "dill_dependency",
    "make_snapshot_v2_manifest",
    "minimal_manifest_payload",
    "patch_snapshot_service_session_lock",
    "patch_snapshot_service_session_states",
    "runtime_manifest",
    "set_snapshot_service_session_states",
    "sign_manifest_bytes",
    "validate_system_commit",
    "verify_manifest_hmac",
    "v2_manifest",
    "write_manifest_to_store",
]


def v2_manifest(**kwargs):
    """Legacy-kwargs adapter around make_snapshot_v2_manifest."""
    from backend.services.snapshot_v2.models import runtime_manifest_from_generation

    if "captured_at" in kwargs:
        kwargs["created_at"] = kwargs.pop("captured_at")
    if "capsule" in kwargs and "continuation" not in kwargs:
        kwargs["continuation"] = kwargs.pop("capsule")
    if "branch" in kwargs and "branch_id" not in kwargs:
        kwargs["branch_id"] = kwargs.pop("branch")
    kwargs.pop("generation", None)
    if "runtime_snapshot_capture" in kwargs:
        gen = kwargs.pop("runtime_snapshot_capture")
        if gen is not None and "runtime" not in kwargs:
            kwargs["runtime"] = runtime_manifest_from_generation(gen, status="complete")
    kwargs.pop("runtime_dependencies", None)

    def _dump(v):
        if hasattr(v, "model_dump"):
            return v.model_dump(mode="json")
        if isinstance(v, dict):
            return {k: _dump(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_dump(x) for x in v]
        return v

    for key in (
        "workspace", "objects", "runtime", "continuation",
        "analysis_state_ref", "notebook_ref", "checkpoint_ref",
    ):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = _dump(kwargs[key])
    return make_snapshot_v2_manifest(**kwargs)
