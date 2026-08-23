"""Static and real-kernel proof for session-scoped workspace deletion."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.utils.sandbox import validate_code_safety


def test_computed_and_safe_literal_os_deletions_are_statically_allowed():
    cases = (
        (
            "import os\n"
            "target = _WORKSPACE_DIR / 'tables' / filename\n"
            "os.remove(target)\n"
        ),
        (
            "import os\n"
            "target = _WORKSPACE_DIR / 'figures' / filename\n"
            "os.unlink(target)\n"
        ),
        (
            "import os\n"
            "target = _WORKSPACE_DIR / 'tables' / dirname\n"
            "os.rmdir(target)\n"
        ),
        "import os\nos.remove('tables/tmp.csv')",
        "import os\nos.unlink('./figures/draft.png')",
        "import os\nos.rmdir('annotation_validation/scratch')",
        "import os\nos.remove('/workspace/scai/tables/tmp.csv')",
        "import os\nos.unlink('/tmp/scai-session/scratch.dat')",
        "import os\nos.remove(path='tables/tmp.csv')",
        "import os\nos.unlink(path='./figures/draft.png')",
        "import os\nos.rmdir(path='/tmp/scai-session/scratch')",
        (
            "import os\n"
            "values = ['discard']\n"
            "values.remove('discard')\n"
            "os.remove(_WORKSPACE_DIR / 'tables' / filename)\n"
        ),
    )
    for code in cases:
        assert validate_code_safety(code) == (True, ""), code


def test_unsafe_literal_os_deletions_are_statically_blocked():
    cases = (
        ("import os\nos.remove('/etc/passwd')", "os.remove"),
        (
            "import os\nos.unlink('/data/sessions/other/session.json')",
            "os.unlink",
        ),
        ("import os\nos.rmdir('/data/sessions')", "os.rmdir"),
        ("import os\nos.remove('../outside.txt')", "os.remove"),
        ("import os\nos.remove(path='/etc/passwd')", "os.remove"),
        (
            "import os\nos.unlink(path='/data/sessions/other/session.json')",
            "os.unlink",
        ),
        ("import os\nos.rmdir(path='/data/sessions')", "os.rmdir"),
        ("import os\nos.remove(a, b)", "os.remove"),
        ("import os\nos.remove()", "os.remove"),
        (
            "import os\nos.remove('tables/ok.csv')\nfoo.os.remove('tables/no.csv')",
            "os.remove",
        ),
        ("import os\nfoo.os.remove('tables/no.csv')", "os.remove"),
    )
    for code, expected in cases:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert expected in reason, (code, reason)


def test_recursive_rename_replace_and_move_operations_remain_blocked():
    cases = (
        ("import shutil\nshutil.rmtree(_WORKSPACE_DIR / 'tables')", "rmtree"),
        ("import os\nos.rename(source, target)", "os.rename"),
        ("import os\nos.replace(source, target)", "os.replace"),
        ("import shutil\nshutil.move(source, target)", "move"),
        ("import os\nos.system('rm -rf tables')", "os.system"),
    )
    for code, expected in cases:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert expected in reason, (code, reason)


def test_malformed_source_and_existing_write_boundary_fail_closed():
    malformed = "import os\nos.remove(target)\nif True print('bad')\n"
    safe, reason = validate_code_safety(malformed)
    assert not safe
    assert "os.remove" in reason

    assert validate_code_safety("open('/etc/scai-test', 'r+')")[0] is False
    assert validate_code_safety("open('/etc/hosts', 'rb')") == (True, "")
    assert validate_code_safety("Path('tables/result.txt').open('r+')") == (
        True,
        "",
    )


@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_kernel_deletion_is_session_scoped(tmp_path):
    from backend.utils.sandbox import LocalJupyterSandbox

    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "states"
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside", encoding="utf-8")
    other_workspace = tmp_path / "other-session" / "workspace"
    other_workspace.mkdir(parents=True)
    other_file = other_workspace / "other.txt"
    other_file.write_text("other", encoding="utf-8")
    other_dir = other_workspace / "empty-dir"
    other_dir.mkdir()
    other_path_dir = other_workspace / "path-empty-dir"
    other_path_dir.mkdir()
    other_original_dir = other_workspace / "original-empty-dir"
    other_original_dir.mkdir()
    other_native_dir = other_workspace / "native-empty-dir"
    other_native_dir.mkdir()
    other_private_root = Path("/tmp") / f"scai-s4-other-private-{tmp_path.name}"
    other_private_root.mkdir()
    other_private_file = other_private_root / "other-private.txt"
    other_private_file.write_text("other-private", encoding="utf-8")
    metadata = tmp_path / "session.json"
    metadata.write_text("metadata", encoding="utf-8")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    upload_file = uploads / "input.txt"
    upload_file.write_text("upload", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    report_file = reports / "report.html"
    report_file.write_text("report", encoding="utf-8")
    symlink = workspace / "tables" / "outside-link.txt"
    outside_dirfd = tmp_path / "dir-fd"
    outside_dirfd.mkdir()
    dirfd_file = outside_dirfd / "dirfd.txt"
    dirfd_file.write_text("dir-fd", encoding="utf-8")
    dirfd_dir = outside_dirfd / "dirfd-empty"
    dirfd_dir.mkdir()

    sandbox = LocalJupyterSandbox(
        session_id="s4-live-deletion",
        state_dir=str(state_dir),
        working_dir=str(workspace),
        timeout=120,
    )
    await sandbox.start()
    try:
        symlink.parent.mkdir(parents=True, exist_ok=True)
        symlink.symlink_to(outside_file)
        private_tmp = sandbox._session_tmp
        positive_code = (
            "import os\n"
            "from pathlib import Path\n"
            f"workspace_file = Path({str(workspace / 'tables' / 'computed.txt')!r})\n"
            "workspace_file.write_text('workspace')\n"
            "os.remove(workspace_file)\n"
            f"literal_file = Path({str(workspace / 'figures' / 'literal.txt')!r})\n"
            "literal_file.write_text('literal')\n"
            f"os.unlink({str(workspace / 'figures' / 'literal.txt')!r})\n"
            f"workspace_dir = Path({str(workspace / 'tables' / 'empty-dir')!r})\n"
            "workspace_dir.mkdir()\n"
            "os.rmdir(workspace_dir)\n"
            f"workspace_path_file = Path({str(workspace / 'tables' / 'path-unlink.txt')!r})\n"
            "workspace_path_file.write_text('workspace-path')\n"
            "workspace_path_file.unlink()\n"
            f"workspace_path_dir = Path({str(workspace / 'tables' / 'path-empty-dir')!r})\n"
            "workspace_path_dir.mkdir()\n"
            "workspace_path_dir.rmdir()\n"
            f"private_remove_file = Path({str(private_tmp / 'remove.txt')!r})\n"
            "private_remove_file.write_text('private-remove')\n"
            "os.remove(private_remove_file)\n"
            f"private_unlink_file = Path({str(private_tmp / 'unlink.txt')!r})\n"
            "private_unlink_file.write_text('private-unlink')\n"
            "os.unlink(private_unlink_file)\n"
            f"private_rmdir_dir = Path({str(private_tmp / 'rmdir-dir')!r})\n"
            "private_rmdir_dir.mkdir()\n"
            "os.rmdir(private_rmdir_dir)\n"
            f"private_path_file = Path({str(private_tmp / 'path-unlink.txt')!r})\n"
            "private_path_file.write_text('private-path')\n"
            "private_path_file.unlink()\n"
            f"private_path_dir = Path({str(private_tmp / 'path-rmdir-dir')!r})\n"
            "private_path_dir.mkdir()\n"
            "private_path_dir.rmdir()\n"
            "assert not workspace_file.exists()\n"
            "assert not literal_file.exists()\n"
            "assert not workspace_dir.exists()\n"
            "assert not workspace_path_file.exists()\n"
            "assert not workspace_path_dir.exists()\n"
            "assert not private_remove_file.exists()\n"
            "assert not private_unlink_file.exists()\n"
            "assert not private_rmdir_dir.exists()\n"
            "assert not private_path_file.exists()\n"
            "assert not private_path_dir.exists()\n"
        )
        positive = await sandbox.execute(
            positive_code, save_to_history=False
        )
        assert positive.success, positive.error

        negative_code = (
            "import ctypes\n"
            "import os\n"
            "from pathlib import Path\n"
            f"outside = Path({str(outside_file)!r})\n"
            f"other = Path({str(other_file)!r})\n"
            f"other_empty = Path({str(other_dir)!r})\n"
            f"other_path_empty = Path({str(other_path_dir)!r})\n"
            f"other_original_empty = Path({str(other_original_dir)!r})\n"
            f"other_native_empty = Path({str(other_native_dir)!r})\n"
            f"other_private = Path({str(other_private_file)!r})\n"
            f"metadata_target = Path({str(metadata)!r})\n"
            f"upload_target = Path({str(upload_file)!r})\n"
            f"report_target = Path({str(report_file)!r})\n"
            f"link_target = Path({str(symlink)!r})\n"
            f"host_target = '/etc/hosts'\n"
            "def _must_block(action):\n"
            "    try:\n"
            "        action()\n"
            "    except (PermissionError, OSError):\n"
            "        return\n"
            "    raise AssertionError('outside deletion unexpectedly succeeded')\n"
            "_must_block(lambda: os.remove(outside))\n"
            "_must_block(lambda: os.unlink(other))\n"
            "_must_block(lambda: os.rmdir(other_empty))\n"
            "_must_block(lambda: other_path_empty.rmdir())\n"
            "_must_block(lambda: os.remove(other_private))\n"
            "_must_block(lambda: metadata_target.unlink())\n"
            "_must_block(lambda: upload_target.unlink())\n"
            "_must_block(lambda: report_target.unlink())\n"
            "_must_block(lambda: Path('/etc/hosts').unlink())\n"
            "_must_block(lambda: os.remove(link_target))\n"
            "_must_block(lambda: _ORIGINAL_OS_REMOVE(outside))\n"
            "_must_block(lambda: _ORIGINAL_OS_RMDIR(other_original_empty))\n"
            "_must_block(lambda: os.remove(host_target))\n"
            "fd = os.open(" + repr(str(outside_dirfd)) + ", os.O_RDONLY)\n"
            "try:\n"
            "    _must_block(lambda: os.unlink('dirfd.txt', dir_fd=fd))\n"
            "    _must_block(lambda: os.rmdir('dirfd-empty', dir_fd=fd))\n"
            "finally:\n"
            "    os.close(fd)\n"
            "libc = ctypes.CDLL(None)\n"
            "native_unlink_result = libc.unlink(os.fsencode(str(outside)))\n"
            "native_rmdir_result = libc.rmdir(os.fsencode(str(other_native_empty)))\n"
            "assert native_unlink_result != 0\n"
            "assert native_rmdir_result != 0\n"
            "assert outside.read_text() == 'outside'\n"
            "assert other.read_text() == 'other'\n"
            "assert other_empty.exists()\n"
            "assert other_path_empty.exists()\n"
            "assert other_original_empty.exists()\n"
            "assert other_native_empty.exists()\n"
            "assert other_private.read_text() == 'other-private'\n"
            "assert metadata_target.read_text() == 'metadata'\n"
            "assert upload_target.read_text() == 'upload'\n"
            "assert report_target.read_text() == 'report'\n"
            "assert link_target.is_symlink()\n"
            "assert Path(" + repr(str(dirfd_file)) + ").read_text() == 'dir-fd'\n"
            "assert Path(" + repr(str(dirfd_dir)) + ").is_dir()\n"
        )
        negative = await sandbox.execute(
            negative_code, save_to_history=False, skip_safety_check=True
        )
        assert negative.success, negative.error
    finally:
        await sandbox.stop(save=False)
        other_private_file.unlink(missing_ok=True)
        other_private_root.rmdir()
