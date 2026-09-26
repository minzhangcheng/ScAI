"""The committed contract for one runtime generation.

The dump's HMAC covers its bytes. It does not cover the names the generation
claims to have saved, the names it admits it skipped, or the state revision it
belongs to -- and those three are exactly what integrity, coverage and
coherence are decided from. So the manifest is signed on its own, validated
strictly on read, and installed by the transaction that installs the dump.

The module under test is imported behind a guard: a module that raises while
being collected makes the runner report an incomplete collection and publish no
evidence at all, which is precisely the state a red-tree run is in.
"""
import json

import pytest

try:
    from backend.utils.runtime_manifest import (
        RuntimeManifestError,
        build_runtime_manifest,
        canonical_manifest_bytes,
        read_runtime_manifest,
        runtime_manifest_sig_path,
        write_runtime_manifest,
    )
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


DIGEST = "a" * 64


def _manifest(**overrides):
    payload = dict(
        revision=7,
        state_revision=7,
        pkl_digest=DIGEST,
        saved_names=["adata", "markers"],
        skipped_names=[
            {"name": "ss", "type": "SingleSampleGSEA", "reason": "serialization_failed"}
        ],
        serializer_version="dill-zstd-1",
    )
    payload.update(overrides)
    return build_runtime_manifest(**payload)


def test_round_trip_through_a_signed_file(tmp_path):
    path = tmp_path / "s_session.runtime.json"
    write_runtime_manifest(path, _manifest())
    assert runtime_manifest_sig_path(path).is_file()
    assert read_runtime_manifest(path) == _manifest()


def test_a_tampered_manifest_is_refused(tmp_path):
    path = tmp_path / "s_session.runtime.json"
    write_runtime_manifest(path, _manifest())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["saved_names"] = ["adata"]
    path.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    with pytest.raises(RuntimeManifestError, match="signature"):
        read_runtime_manifest(path)


def test_a_signed_but_malformed_manifest_is_refused(tmp_path):
    """A valid signature over invalid content is still an invalid generation."""
    from backend.utils.crypto import sign_hmac_sha256

    path = tmp_path / "s_session.runtime.json"
    manifest = _manifest()
    manifest["saved_names"] = ["adata", "adata"]
    payload = canonical_manifest_bytes(manifest)
    path.write_bytes(payload)
    runtime_manifest_sig_path(path).write_text(
        sign_hmac_sha256(payload), encoding="utf-8"
    )
    with pytest.raises(RuntimeManifestError, match="validation"):
        read_runtime_manifest(path)
