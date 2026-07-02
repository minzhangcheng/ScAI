import pytest
from backend.utils import blob_store


def test_inline_rejects_traversal_digest(tmp_path):
    secret = tmp_path / "secret.b64"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    malicious = f"{blob_store.BLOB_REF_PREFIX}../secret"
    with pytest.raises(ValueError):
        blob_store.inline_blobs(malicious, blob_dir)


def test_inline_round_trips_valid_blob(tmp_path):
    blob_dir = tmp_path / "blobs"
    payload = "x" * 2000
    ref = blob_store.externalize_blobs(payload, blob_dir)
    assert ref.startswith(blob_store.BLOB_REF_PREFIX)
    assert blob_store.inline_blobs(ref, blob_dir) == payload


def test_short_hex_digest_rejected(tmp_path):
    malicious = f"{blob_store.BLOB_REF_PREFIX}abc"
    with pytest.raises(ValueError, match="Invalid blob digest"):
        blob_store.inline_blobs(malicious, tmp_path)
