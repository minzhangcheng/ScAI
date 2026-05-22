"""Safety tests for archive import: P0-4.

Verifies that tar.gz import rejects symlinks/hardlinks/device entries, and that
zip import uses Path.is_relative_to() for traversal detection. Without these
checks, an attacker can craft an archive that extracts files outside the
session workspace (symlink to /etc/passwd, ../ traversal in zip).
"""
import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.routes.export_import import _import_tar_gz, _import_zip


def _build_tar_with_member(tmp_path: Path, member_factory) -> Path:
    """Build a tar.gz at tmp_path/evil.tar.gz with a manifest + a single crafted member.

    member_factory(tf) is called to add the malicious member.
    """
    archive = tmp_path / "evil.tar.gz"
    session_id = "sess123"
    manifest = (
        f'{{"session_id": "{session_id}", "version": "1.0", "data_base_dir": "/tmp/foo"}}'
    ).encode("utf-8")

    with tarfile.open(str(archive), mode="w:gz") as tf:
        info = tarfile.TarInfo(name=f"{session_id}/manifest.json")
        info.size = len(manifest)
        tf.addfile(info, io.BytesIO(manifest))
        member_factory(tf, session_id)

    return archive


def test_tar_rejects_symlink_member(tmp_path):
    def add_symlink(tf, session_id):
        info = tarfile.TarInfo(name=f"{session_id}/workspace/{session_id}/evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)

    archive = _build_tar_with_member(tmp_path, add_symlink)

    with pytest.raises(HTTPException) as exc:
        _import_tar_gz(str(archive), user_id="u1")
    assert exc.value.status_code == 400
    assert "link" in str(exc.value.detail).lower()


def test_tar_rejects_hardlink_member(tmp_path):
    def add_hardlink(tf, session_id):
        info = tarfile.TarInfo(name=f"{session_id}/workspace/{session_id}/evil")
        info.type = tarfile.LNKTYPE
        info.linkname = f"{session_id}/manifest.json"
        tf.addfile(info)

    archive = _build_tar_with_member(tmp_path, add_hardlink)

    with pytest.raises(HTTPException) as exc:
        _import_tar_gz(str(archive), user_id="u1")
    assert exc.value.status_code == 400
    assert "link" in str(exc.value.detail).lower()


def test_tar_rejects_device_member(tmp_path):
    def add_device(tf, session_id):
        info = tarfile.TarInfo(name=f"{session_id}/workspace/{session_id}/evil")
        info.type = tarfile.BLKTYPE
        info.devmajor = 1
        info.devminor = 1
        tf.addfile(info)

    archive = _build_tar_with_member(tmp_path, add_device)

    with pytest.raises(HTTPException) as exc:
        _import_tar_gz(str(archive), user_id="u1")
    assert exc.value.status_code == 400
    assert "device" in str(exc.value.detail).lower() or "special" in str(exc.value.detail).lower()


def test_zip_rejects_traversal():
    buf = io.BytesIO()
    session_id = "sess456"
    state_json = b'{"id": "sess456", "name": "x", "user_id": "u1"}'

    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{session_id}/state.json", state_json)
        # Malicious workspace entry with ../ traversal
        zf.writestr(f"{session_id}/workspace/../../../etc/evil.txt", b"pwned")

    with pytest.raises(HTTPException) as exc:
        _import_zip(buf.getvalue(), user_id="u1")
    assert exc.value.status_code == 400
    assert "escape" in str(exc.value.detail).lower() or "traversal" in str(exc.value.detail).lower()
