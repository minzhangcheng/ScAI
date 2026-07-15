from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeDependency,
    RuntimeDependencyManifest,
    RuntimeManifest,
)
from backend.services.snapshot_v2.runtime_names import (
    runtime_h5ad_filename,
    runtime_h5ad_path,
    validate_runtime_global_name,
)
from backend.utils.runtime_apply import RuntimeApplyError, atomic_replace_h5ad


def _ref() -> ObjectRef:
    return ObjectRef(
        sha256="0" * 64,
        size=1,
        media_type="application/x-hdf5",
    )


@pytest.mark.parametrize(
    "name",
    [
        "../escaped",
        "nested/name",
        "a.b",
        "",
        "1starts_with_digit",
        "name with space",
        "x" * 129,
    ],
)
def test_validate_runtime_global_name_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        validate_runtime_global_name(name)


def test_runtime_h5ad_path_is_confined_to_root(tmp_path):
    path = runtime_h5ad_path(tmp_path, "adata")
    assert path == tmp_path / "adata.h5ad"
    assert runtime_h5ad_filename("adata_2") == "adata_2.h5ad"


def test_runtime_manifest_rejects_unsafe_adata_ref_key():
    with pytest.raises(ValidationError):
        RuntimeManifest(
            status="complete",
            dependencies=RuntimeDependencyManifest(
                required={
                    "../escaped": RuntimeDependency(
                        name="../escaped",
                        serializer="h5ad",
                        type_name="AnnData",
                    )
                }
            ),
            adata_refs={"../escaped": _ref()},
        )


def test_runtime_dependency_manifest_rejects_mismatched_key_and_name():
    with pytest.raises(ValidationError):
        RuntimeDependencyManifest(
            required={
                "adata": RuntimeDependency(
                    name="other",
                    serializer="h5ad",
                    type_name="AnnData",
                )
            }
        )


def test_runtime_dependency_manifest_rejects_unsafe_required_key():
    with pytest.raises(ValidationError):
        RuntimeDependencyManifest(
            required={
                "../escaped": RuntimeDependency(
                    name="adata",
                    serializer="h5ad",
                    type_name="AnnData",
                )
            }
        )


def test_atomic_replace_h5ad_rejects_path_traversal_name(tmp_path, monkeypatch):
    src = tmp_path / "source.h5ad"
    src.write_bytes(b"not-a-real-h5ad-for-name-validation")

    with pytest.raises(RuntimeApplyError, match="invalid runtime variable name"):
        atomic_replace_h5ad("sid-name-check", str(src), var_name="../escaped")
