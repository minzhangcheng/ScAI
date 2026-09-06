"""Tests for durable file I/O and HMAC primitives."""
import hmac
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.utils.files import atomic_write_bytes, atomic_write_text, atomic_write_json, atomic_copy_file, fsync_directory
from backend.utils.crypto import sign_hmac_sha256, verify_hmac_sha256
from backend.tests.split_facade import split_source


class TestAtomicWrite:
    def test_atomic_write_bytes_creates_file(self, tmp_path):
        target = tmp_path / "state.json"
        atomic_write_bytes(target, b'{"key": "value"}')
        assert target.read_bytes() == b'{"key": "value"}'

    def test_atomic_write_bytes_creates_parents(self, tmp_path):
        target = tmp_path / "sub" / "nested" / "state.json"
        atomic_write_bytes(target, b"data")
        assert target.read_bytes() == b"data"

    def test_atomic_write_bytes_with_mode(self, tmp_path):
        target = tmp_path / "exec.sh"
        atomic_write_bytes(target, b"#!/bin/sh\necho hi", mode=0o755)
        assert target.exists()
        assert os.stat(target).st_mode & 0o777 == 0o755

    def test_atomic_write_bytes_removes_temp_when_replace_fails(self, tmp_path, monkeypatch):
        target = tmp_path / "state.json"
        monkeypatch.setattr(os, "replace", Mock(side_effect=OSError("disk fault")))
        with pytest.raises(OSError, match="disk fault"):
            atomic_write_bytes(target, b"payload")
        assert not target.exists()
        assert list(tmp_path.glob(".*.tmp")) == []

    def test_atomic_write_text(self, tmp_path):
        target = tmp_path / "hello.txt"
        atomic_write_text(target, "Hello World")
        assert target.read_text() == "Hello World"

    def test_atomic_write_json(self, tmp_path):
        target = tmp_path / "data.json"
        atomic_write_json(target, {"a": 1, "b": [2, 3]})
        assert target.read_text() == '{"a": 1, "b": [2, 3]}'

    def test_atomic_copy_file(self, tmp_path):
        src = tmp_path / "source.bin"
        src.write_bytes(b"\x00\x01\x02")
        dst = tmp_path / "dest.bin"
        atomic_copy_file(src, dst)
        assert dst.read_bytes() == b"\x00\x01\x02"

    def test_atomic_copy_file_creates_parent(self, tmp_path):
        src = tmp_path / "source.bin"
        src.write_bytes(b"data")
        dst = tmp_path / "sub" / "dest.bin"
        atomic_copy_file(src, dst)
        assert dst.read_bytes() == b"data"


class TestFsyncDirectory:
    def test_fsync_does_not_raise(self, tmp_path):
        fsync_directory(tmp_path)

    def test_fsync_on_nested_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True, exist_ok=True)
        fsync_directory(nested)


class TestHmacPrimitives:
    def test_sign_and_verify_roundtrip(self):
        sig = sign_hmac_sha256(b"hello", secret=b"mysecret")
        assert verify_hmac_sha256(b"hello", sig, secret=b"mysecret") is True

    def test_verify_rejects_tampered_data(self):
        sig = sign_hmac_sha256(b"hello", secret=b"mysecret")
        assert verify_hmac_sha256(b"world", sig, secret=b"mysecret") is False

    def test_verify_rejects_tampered_signature(self):
        sig = sign_hmac_sha256(b"hello", secret=b"mysecret")
        bad_sig = sig[:-1] + ("0" if sig[-1] == "1" else "1")
        assert verify_hmac_sha256(b"hello", bad_sig, secret=b"mysecret") is False

    def test_verify_uses_constant_time_compare(self, monkeypatch):
        compare = Mock(return_value=True)
        monkeypatch.setattr(hmac, "compare_digest", compare)
        signature = sign_hmac_sha256(b"body", secret=b"secret")
        assert verify_hmac_sha256(b"body", signature, secret=b"secret") is True
        compare.assert_called_once()

    def test_different_secrets_produce_different_signatures(self):
        sig_a = sign_hmac_sha256(b"data", secret=b"secret-a")
        sig_b = sign_hmac_sha256(b"data", secret=b"secret-b")
        assert sig_a != sig_b

    def test_default_secret_is_stable_within_process(self):
        sig_1 = sign_hmac_sha256(b"data")
        sig_2 = sign_hmac_sha256(b"data")
        assert sig_1 == sig_2


class TestHmacDriftGuard:
    def test_no_raw_hmac_new_in_runtime_apply_or_sandbox(self):
        import pathlib
        roots = [
            pathlib.Path("backend/utils/runtime_apply.py"),
        ]
        # sandbox.py is allowlisted ONLY for the streaming helper below.
        offenders = []
        for root in roots:
            files = [root] if root.is_file() else root.rglob("*.py")
            for f in files:
                text = f.read_text()
                if "hmac.new(" in text:
                    offenders.append(str(f))
        assert offenders == [], f"raw hmac.new in integrity modules: {offenders}"

    def test_sandbox_pkl_hmac_uses_shared_helper(self):
        # The raw hmac.new( this caps moved into the parts package, so reading the
        # facade alone would make the cap vacuous.
        text = split_source("backend.utils.sandbox")
        # The non-streaming signer must use the shared helper.
        assert "sign_hmac_sha256" in text
        # The single remaining raw use is the streaming helper (allowlisted).
        assert text.count("hmac.new(") <= 1
