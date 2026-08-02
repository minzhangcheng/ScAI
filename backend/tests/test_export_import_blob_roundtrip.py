"""Export -> import must preserve externalized blob content losslessly.

Regression test for the bug where <session_dir>/blobs/ and
<session_dir>/snapshots/<snap_id>_blobs/ were never packed into the
tar.gz, so after import `inline_blobs()` left `@scai-blob:<hash>` tokens
in messages instead of restoring the original payload.
"""
import asyncio
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


_ISOLATED_TEST_ENV = "SCAI_EXPORT_BLOB_ROUNDTRIP_TEST"


def _run_isolated_roundtrip(test_name: str, tmp_path: Path) -> None:
    """Run a round-trip in a fresh interpreter with its data root set early.

    These tests exercise modules that intentionally bind path helpers and the
    default configuration at import time. Reloading them inside the shared
    pytest process changes class and singleton identities for later tests.
    """
    env = os.environ.copy()
    env["SCAI_DATA_DIR"] = str(tmp_path / "data")
    env[_ISOLATED_TEST_ENV] = test_name
    node = f"{Path(__file__).resolve()}::{test_name}"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _build_state_with_blob(session_id: str, payload: str):
    """Construct an AnalysisState carrying a long string that will be externalized."""
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id=session_id, name="blob-roundtrip", user_id="tester")
    state.custom_prompts = {"pi": payload}
    return state


def _make_snapshot_blob(session_id: str, payload: str, snap_id: str) -> str:
    """Externalize `payload` into <session>/snapshots/<snap_id>_blobs/ and
    write a tiny JSON state file that references it. Returns the digest."""
    from backend.config import get_session_dir
    from backend.utils.blob_store import externalize_blobs, BLOB_REF_PREFIX

    snap_dir = get_session_dir(session_id) / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    blob_dir = snap_dir / f"{snap_id}_blobs"
    state_obj = {"snapshot_payload": payload}
    externalized = externalize_blobs(state_obj, blob_dir)
    (snap_dir / f"{snap_id}.json").write_text(
        json.dumps(externalized), encoding="utf-8"
    )
    token = externalized["snapshot_payload"]
    assert token.startswith(BLOB_REF_PREFIX)
    return token[len(BLOB_REF_PREFIX):]


def _build_export_tar(session_id: str) -> bytes:
    """Mirror the production export logic."""
    from backend.config import DATA_BASE_DIR, get_session_metadata_path
    from backend.routes.export_import import (
        _add_reproducibility_to_tar,
        _add_session_blobs_to_tar,
        _add_snapshots_to_tar,
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        manifest = json.dumps({
            "session_id": session_id,
            "version": "2.0",
            "data_base_dir": str(DATA_BASE_DIR),
        }).encode("utf-8")
        info = tarfile.TarInfo(name=f"{session_id}/manifest.json")
        info.size = len(manifest)
        tf.addfile(info, io.BytesIO(manifest))

        session_file = get_session_metadata_path(session_id)
        if session_file and session_file.exists():
            tf.add(str(session_file), arcname=f"{session_id}/sessions/{session_id}.json")

        _add_session_blobs_to_tar(tf, session_id)
        _add_reproducibility_to_tar(tf, session_id)
        _add_snapshots_to_tar(tf, session_id)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_export_import_preserves_session_blob(tmp_path):
    test_name = "test_export_import_preserves_session_blob"
    if os.environ.get(_ISOLATED_TEST_ENV) != test_name:
        await asyncio.to_thread(_run_isolated_roundtrip, test_name, tmp_path)
        return

    from backend.config import get_session_dir, get_session_metadata_path
    from backend.routes.export_import import _import_tar_gz
    from backend.services.session_store import async_save_session
    from backend.utils.blob_store import BLOB_REF_PREFIX, MIN_BLOB_LEN

    session_id = "11111111-1111-1111-1111-111111111111"
    payload = "X" * (MIN_BLOB_LEN + 100)
    state = _build_state_with_blob(session_id, payload)
    await async_save_session(session_id, state, force=True)

    # Sanity: the saved session.json should reference, not inline, the payload.
    raw = get_session_metadata_path(session_id).read_text(encoding="utf-8")
    assert payload not in raw
    assert BLOB_REF_PREFIX in raw

    tar_bytes = _build_export_tar(session_id)

    # Wipe the source so import has to restore from the archive.
    import shutil
    shutil.rmtree(get_session_dir(session_id))

    tar_path = tmp_path / "export.tar.gz"
    tar_path.write_bytes(tar_bytes)
    restored = await asyncio.to_thread(_import_tar_gz, str(tar_path), "tester")

    # The restored AnalysisState should carry the original payload, not the token.
    assert restored.custom_prompts["pi"] == payload


@pytest.mark.asyncio
async def test_export_import_preserves_snapshot_blob(tmp_path):
    test_name = "test_export_import_preserves_snapshot_blob"
    if os.environ.get(_ISOLATED_TEST_ENV) != test_name:
        await asyncio.to_thread(_run_isolated_roundtrip, test_name, tmp_path)
        return

    from backend.config import get_session_dir
    from backend.routes.export_import import _import_tar_gz
    from backend.services.session_store import async_save_session
    from backend.utils.blob_store import MIN_BLOB_LEN

    session_id = "22222222-2222-2222-2222-222222222222"
    snap_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    payload = "Y" * (MIN_BLOB_LEN + 50)

    state = _build_state_with_blob(session_id, "short")
    await async_save_session(session_id, state, force=True)
    digest = _make_snapshot_blob(session_id, payload, snap_id)

    tar_bytes = _build_export_tar(session_id)

    import shutil
    shutil.rmtree(get_session_dir(session_id))

    tar_path = tmp_path / "export.tar.gz"
    tar_path.write_bytes(tar_bytes)
    restored = await asyncio.to_thread(_import_tar_gz, str(tar_path), "tester")

    # Snapshot blob file must have been restored to its sibling directory.
    snap_blob = get_session_dir(restored.id) / "snapshots" / f"{snap_id}_blobs" / f"{digest}.b64"
    assert snap_blob.exists(), f"snapshot blob not restored: {snap_blob}"
    assert snap_blob.read_text(encoding="utf-8") == payload
