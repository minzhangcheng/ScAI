"""Test that path boundary checking prevents prefix-matching bypass."""

from pathlib import Path
import os
import tempfile


def test_is_write_allowed_boundary():
    from backend.utils.sandbox import _path_is_within

    # Should match: /data/sessions/abc/x is within /data/sessions/abc
    assert _path_is_within("/data/sessions/abc/x", "/data/sessions/abc")

    # Should NOT match: /data/sessions/abc-evil/x is NOT within /data/sessions/abc
    assert not _path_is_within("/data/sessions/abc-evil/x", "/data/sessions/abc")

    # Should match: exact root
    assert _path_is_within("/data/sessions/abc", "/data/sessions/abc")

    # Symlink resolution
    with tempfile.TemporaryDirectory() as tmp:
        real = os.path.join(tmp, "real")
        link = os.path.join(tmp, "link")
        os.mkdir(real)
        os.symlink(real, link)
        assert _path_is_within(os.path.join(link, "x"), real)


def test_allowed_roots_no_bare_tmp(tmp_path):
    from backend.utils.sandbox import LocalJupyterSandbox
    sb = LocalJupyterSandbox(
        session_id="s-test-h8",
        state_dir=tmp_path / "states",
        working_dir=tmp_path,
    )
    roots = sb._allowed_write_roots()
    assert "/tmp" not in roots
    assert any("/tmp/scai-" in r for r in roots)


def test_numba_cache_is_inside_session_temp_root(tmp_path):
    from backend.utils.sandbox import LocalJupyterSandbox, _path_is_within

    sb = LocalJupyterSandbox(
        session_id="numba-boundary",
        state_dir=tmp_path / "states",
        working_dir=tmp_path,
    )
    cache_path = sb._session_tmp / "numba_cache"
    assert _path_is_within(str(cache_path), str(sb._session_tmp))
