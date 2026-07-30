"""P1-11: Sandbox dill restore/save failures must surface (not silently skip)."""
import re

import pytest

import backend.utils.sandbox as sb_mod


def _read_src() -> str:
    with open(sb_mod.__file__) as f:
        return f.read()


def _extract_method(src: str, *names: str) -> str:
    for name in names:
        m = re.search(
            rf"(    (?:async )?def {name}\b.*?)(?=\n    (?:async )?def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        if m:
            return m.group(1)
    raise AssertionError(f"could not locate any of {names}")


def test_restore_failure_raises_not_silent():
    """P1-11: dill restore failure must raise instead of silently continuing."""
    src = _read_src()
    body = _extract_method(src, "_restore_session", "restore_session")
    # Must raise RuntimeError on restore failure rather than only setting a flag and continuing
    assert "raise RuntimeError" in body or "raise Exception" in body, (
        "P1-11: restore failure must raise so the workflow does not continue "
        "with inconsistent kernel state"
    )
    assert "ZstdDecompressor" in body
    assert "_raw.seek(0)" in body, "legacy uncompressed runtime must remain readable"


@pytest.mark.asyncio
async def test_save_session_refuses_to_overwrite_after_a_failed_restore(tmp_path):
    """The pkl on disk is better evidence than a half-restored namespace.

    ``_restore_session`` assigns into ``globals()`` frame by frame, so an aborted
    restore leaves a partial namespace. Dumping that over the live pkl and
    re-signing it destroyed the only good copy and made the *next* start succeed
    with a silently truncated runtime, still reporting the runtime as complete.
    """
    import asyncio
    from unittest.mock import MagicMock

    from backend.utils.sandbox import LocalJupyterSandbox

    sandbox = LocalJupyterSandbox.__new__(LocalJupyterSandbox)
    sandbox.session_id = "restore-failed"
    sandbox._session_restore_failed = True
    sandbox._exec_lock = asyncio.Lock()
    pkl = tmp_path / "restore-failed_session.pkl"
    pkl.write_bytes(b"ORIGINAL")
    sandbox.notebook_state = MagicMock(session_path=pkl)
    # Reaching the kernel at all would be a bug; make it explode if we do.
    sandbox.execute = MagicMock(side_effect=AssertionError("must not reach the kernel"))

    await sandbox.save_session()

    assert pkl.read_bytes() == b"ORIGINAL"


def test_start_tears_down_a_kernel_whose_restore_failed():
    """A half-restored kernel must not stay reachable.

    Nothing downstream re-checks the flag: ``execute()`` guards only on
    ``kc is None`` and the bioinformatics agent only on ``km is None``.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/utils/sandbox_parts/kernel.py")
    start = source.split("async def start(", 1)[1]
    assert "await self._restore_session()" in start
    assert "await self.stop(save=False)" in start, (
        "a failed restore must tear the kernel down, without saving"
    )


def test_save_session_collects_and_reports_skipped_vars():
    """P1-11: save_session must track skipped vars (no silent except: pass) and log them."""
    src = _read_src()
    body = _extract_method(src, "save_session", "_save_session")
    # The save logic involves both Python-side handling AND kernel-side injected code.
    # The injected code (string) MUST collect skipped variable names and print them,
    # and the Python side MUST log them when present.
    assert "skipped" in body.lower(), "P1-11: must track skipped variable names"
    assert "logger." in body, "P1-11: must log skipped vars / report them"
