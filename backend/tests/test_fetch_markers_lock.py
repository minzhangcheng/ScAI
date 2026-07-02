import re
import pathlib
import os
import tempfile
import pytest


def test_no_plaintext_marker_url():
    src = pathlib.Path("scripts/fetch_markers.py").read_text()
    assert "http://117.50.127.228" not in src


def test_lock_acquire_no_toctou(tmp_path):
    """Verify _acquire_lock doesn't use unlink-then-open which is racy."""
    src = pathlib.Path("scripts/fetch_markers.py").read_text()
    # Allow os.replace as the atomic alternative, forbid unlink + O_EXCL sequence
    # Check there is no os.unlink followed by O_CREAT|O_EXCL
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "os.unlink" in line and i + 1 < len(lines):
            nxt = lines[i + 1]
            if "O_CREAT" in nxt and "O_EXCL" in nxt:
                assert False, f"TOCTOU at line {i+1}: unlink then O_EXCL"


def test_acquire_lock_stale_reclaim(tmp_path):
    """Integration test: stale lock is reclaimed atomically."""
    from importlib import util
    spec = util.spec_from_file_location(
        "fetch_markers_lock",
        pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lock = tmp_path / ".fetch.lock"
    lock.write_text("stale")
    # Set mtime far in the past
    old = 1_000_000_000  # 2001-09-09
    os.utime(lock, (old, old))

    fd = mod._acquire_lock(lock)
    assert fd is not None, "should reclaim stale lock"
    os.close(fd)
    assert lock.exists()


def test_acquire_lock_active_skips(tmp_path):
    """Fresh lock should be skipped, not reclaimed."""
    from importlib import util
    spec = util.spec_from_file_location(
        "fetch_markers_lock",
        pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lock = tmp_path / ".fetch.lock"
    lock.write_text("fresh")

    fd = mod._acquire_lock(lock)
    assert fd is None, "should skip fresh lock"


def test_cellmarker2_rejects_bad_checksum(monkeypatch):
    """SHA-256 mismatch on CellMarker2 download raises ValueError."""
    from importlib import util
    spec = util.spec_from_file_location(
        "fetch_markers_cksum",
        pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "CELLMARKER2_SHA256", "0" * 64)
    monkeypatch.setattr(mod, "_http_get_bytes", lambda url, verify=True: b"corrupted-bytes")
    with pytest.raises(ValueError, match="checksum"):
        mod._download_cellmarker2()


def _load_fetch_markers():
    from importlib import util
    spec = util.spec_from_file_location(
        "fetch_markers_mod",
        pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cellmarker2_default_is_fail_secure_requires_valid_tls(monkeypatch):
    """With no pinned checksum (default None), the download MUST require a valid
    TLS chain (verify=True) — never accept unverified data over verify=False."""
    mod = _load_fetch_markers()
    assert mod.CELLMARKER2_SHA256 is None  # ships unpinned

    seen = {}
    def _fake_get(url, verify=True):
        seen["verify"] = verify
        return b"any-bytes"
    monkeypatch.setattr(mod, "_http_get_bytes", _fake_get)
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: "PARSED")
    monkeypatch.setattr(mod, "project_to_schema", lambda raw, *a, **k: raw)

    assert mod._download_cellmarker2() == "PARSED"
    assert seen["verify"] is True  # fail-secure: TLS must be verified when unpinned


def test_cellmarker2_pinned_checksum_allows_insecure_transport(monkeypatch):
    """When a SHA-256 is pinned, the raw-IP transport (verify=False) is allowed
    but content is verified against the pinned digest."""
    import hashlib
    mod = _load_fetch_markers()
    payload = b"real-bytes"
    monkeypatch.setattr(mod, "CELLMARKER2_SHA256", hashlib.sha256(payload).hexdigest())

    seen = {}
    def _fake_get(url, verify=True):
        seen["verify"] = verify
        return payload
    monkeypatch.setattr(mod, "_http_get_bytes", _fake_get)
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: "PARSED")
    monkeypatch.setattr(mod, "project_to_schema", lambda raw, *a, **k: raw)

    assert mod._download_cellmarker2() == "PARSED"
    assert seen["verify"] is False  # pinned checksum permits untrusted transport
