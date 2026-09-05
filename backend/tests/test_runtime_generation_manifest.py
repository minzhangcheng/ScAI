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
import hashlib
import json

import pytest

try:
    from backend.utils.runtime_manifest import (
        RUNTIME_MANIFEST_SCHEMA_VERSION,
        RuntimeManifestError,
        build_runtime_manifest,
        canonical_manifest_bytes,
        read_runtime_manifest,
        runtime_manifest_path,
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


def test_manifest_records_the_full_contract():
    manifest = _manifest()
    assert manifest["schema_version"] == RUNTIME_MANIFEST_SCHEMA_VERSION
    assert manifest["revision"] == 7
    assert manifest["state_revision"] == 7
    assert manifest["pkl_sha256"] == DIGEST
    assert manifest["saved_names"] == ["adata", "markers"]
    assert manifest["skipped_names"][0]["reason"] == "serialization_failed"
    assert manifest["serializer_version"] == "dill-zstd-1"


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


def test_a_missing_signature_is_refused(tmp_path):
    path = tmp_path / "s_session.runtime.json"
    write_runtime_manifest(path, _manifest())
    runtime_manifest_sig_path(path).unlink()
    with pytest.raises(RuntimeManifestError, match="unreadable"):
        read_runtime_manifest(path)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pkl_digest": "not-a-digest"}, "sha256"),
        ({"saved_names": ["adata", "adata"]}, "duplicate"),
        ({"saved_names": ["1bad"]}, "identifier"),
        (
            {
                "saved_names": ["adata"],
                "skipped_names": [
                    {"name": "adata", "type": "AnnData", "reason": "serialization_failed"}
                ],
            },
            "both saved and skipped",
        ),
        (
            {"skipped_names": [{"name": "ss", "type": "X", "reason": "made_up"}]},
            "reason",
        ),
    ],
)
def test_invalid_manifests_are_rejected_at_build_time(overrides, message):
    with pytest.raises(ValueError, match=message):
        _manifest(**overrides)


def test_an_unknown_schema_version_is_refused(tmp_path):
    from backend.utils.crypto import sign_hmac_sha256

    path = tmp_path / "s_session.runtime.json"
    manifest = _manifest()
    manifest["schema_version"] = 99
    payload = canonical_manifest_bytes(manifest)
    path.write_bytes(payload)
    runtime_manifest_sig_path(path).write_text(
        sign_hmac_sha256(payload), encoding="utf-8"
    )
    with pytest.raises(RuntimeManifestError, match="schema_version"):
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


def test_canonical_bytes_are_stable():
    once = canonical_manifest_bytes(_manifest())
    twice = canonical_manifest_bytes(json.loads(once.decode("utf-8")))
    assert once == twice


def test_skipped_names_are_ordered_so_the_digest_is_stable():
    manifest = build_runtime_manifest(
        revision=1,
        state_revision=1,
        pkl_digest=DIGEST,
        saved_names=[],
        skipped_names=[
            {"name": "zeta", "type": "X", "reason": "excluded_by_kind"},
            {"name": "alpha", "type": "Y", "reason": "excluded_volatile"},
        ],
        serializer_version="dill-zstd-1",
    )
    assert [entry["name"] for entry in manifest["skipped_names"]] == ["alpha", "zeta"]


def test_manifest_path_sits_beside_the_dump(tmp_path):
    assert runtime_manifest_path("abc", states_dir=tmp_path) == (
        tmp_path / "abc_session.runtime.json"
    )
