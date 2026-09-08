"""Integrity blocks; coverage warns; generations without a manifest still load.

A dump missing a name its own manifest declares is corrupt, and a namespace
missing an unknown subset must never reach a workflow. A dump missing a name the
manifest recorded as skipped is ordinary -- that happens in every real session.
And a dump written before manifests existed declares nothing at all, which is
age rather than corruption: refusing those would make this change undeployable
against every session already on disk.
"""
import hashlib

import pytest

try:
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        runtime_manifest_sig_path,
        write_runtime_manifest,
    )
    from backend.utils.sandbox_parts.runtime_restore import (
        RuntimeIntegrityError,
        plan_restore,
        verify_restored_generation,
    )
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


def _generation(tmp_path, saved, skipped=()):
    pkl = tmp_path / "sid_session.pkl"
    pkl.write_bytes(b"SCAI_ZSTD_DILL_V1\nbytes")
    manifest_path = tmp_path / "sid_session.runtime.json"
    write_runtime_manifest(
        manifest_path,
        build_runtime_manifest(
            revision=2,
            state_revision=2,
            pkl_digest=hashlib.sha256(pkl.read_bytes()).hexdigest(),
            saved_names=list(saved),
            skipped_names=[
                {"name": name, "type": "X", "reason": "serialization_failed"}
                for name in skipped
            ],
            serializer_version="dill-zstd-1",
        ),
    )
    return manifest_path, pkl


def test_a_complete_generation_verifies(tmp_path):
    manifest_path, pkl = _generation(tmp_path, ["adata", "markers"])
    plan = plan_restore(manifest_path, pkl)
    assert plan.legacy is False
    assert plan.declared == {"adata", "markers"}
    verify_restored_generation(plan, {"adata", "markers", "extra"})


def test_a_declared_name_that_did_not_restore_is_corruption(tmp_path):
    manifest_path, pkl = _generation(tmp_path, ["adata", "markers"])
    with pytest.raises(RuntimeIntegrityError, match="markers"):
        verify_restored_generation(plan_restore(manifest_path, pkl), {"adata"})


def test_a_skipped_name_absent_is_not_corruption(tmp_path):
    manifest_path, pkl = _generation(tmp_path, ["adata"], skipped=["ss"])
    verify_restored_generation(plan_restore(manifest_path, pkl), {"adata"})


def test_a_dump_whose_digest_disagrees_is_refused_before_loading(tmp_path):
    manifest_path, pkl = _generation(tmp_path, ["adata"])
    pkl.write_bytes(b"SCAI_ZSTD_DILL_V1\ndifferent")
    with pytest.raises(RuntimeIntegrityError, match="digest"):
        plan_restore(manifest_path, pkl)


def test_a_generation_without_a_manifest_is_legacy_not_corrupt(tmp_path):
    pkl = tmp_path / "sid_session.pkl"
    pkl.write_bytes(b"SCAI_ZSTD_DILL_V1\nbytes")
    plan = plan_restore(tmp_path / "sid_session.runtime.json", pkl)
    assert plan.legacy is True
    assert plan.declared is None
    verify_restored_generation(plan, {"whatever", "it", "held"})


def test_a_present_but_unsigned_manifest_is_corruption_not_legacy(tmp_path):
    """Absent is old. Present-and-broken is tampering, and must not degrade."""
    manifest_path, pkl = _generation(tmp_path, ["adata"])
    runtime_manifest_sig_path(manifest_path).unlink()
    with pytest.raises(RuntimeIntegrityError):
        plan_restore(manifest_path, pkl)


def test_a_stray_signature_without_a_manifest_is_corruption_not_legacy(tmp_path):
    """Half a generation record is evidence of loss, not of age."""
    manifest_path, pkl = _generation(tmp_path, ["adata"])
    manifest_path.unlink()
    with pytest.raises(RuntimeIntegrityError):
        plan_restore(manifest_path, pkl)


def test_the_digest_is_streamed_rather_than_read_into_memory(tmp_path, monkeypatch):
    """Real dumps reach several gigabytes; the backend must not hold one.

    ``Path.read_bytes`` is made to raise so the constraint is enforced by the
    test rather than merely intended: a future edit that reaches for the whole
    file fails here instead of in production.
    """
    manifest_path, pkl = _generation(tmp_path, ["adata"])

    def _refuse(self):
        raise AssertionError(f"read the whole dump into memory: {self}")

    from pathlib import Path as _Path

    original = _Path.read_bytes

    def _guarded(self):
        if str(self).endswith("_session.pkl"):
            _refuse(self)
        return original(self)

    monkeypatch.setattr(_Path, "read_bytes", _guarded)

    plan = plan_restore(manifest_path, pkl)
    assert plan.declared == {"adata"}
