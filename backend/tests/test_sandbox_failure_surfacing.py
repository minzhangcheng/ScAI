"""P1-11: Sandbox dill restore/save failures must surface (not silently skip)."""

import pytest



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

    # __init__ is cheap -- it starts no kernel -- so call it for real and
    # override only the collaborators this test needs to control. Bypassing
    # it with __new__ made the fixture drift: every attribute __init__ later
    # grew was simply absent, and save_session reads several of them.
    sandbox = LocalJupyterSandbox(
        session_id="restore-failed",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
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
    # Bounded at the next method: the unbounded slice ran on into `restart()`,
    # whose `await self.stop()` is correct there and would fail the rule below.
    start = source.split("async def start(", 1)[1].split("\n    async def ", 1)[0]
    assert "await self._restore_session()" in start

    # The teardown call, whichever of the two it is. `start` used to call
    # `stop(save=False)`; it now calls `_shutdown_kernel_unlocked()`, because
    # the export freeze holds the execution lock and the old path tried to
    # re-acquire it. Both satisfy this guard's actual rule.
    teardown = [
        call for call in ("await self._shutdown_kernel_unlocked()",
                          "await self.stop(save=False)")
        if call in start
    ]
    assert teardown, (
        "a failed restore must tear the kernel down; start() calls neither "
        "_shutdown_kernel_unlocked() nor stop(save=False)"
    )
    assert "await self.stop()" not in start, (
        "a failed restore must not take the saving stop() path"
    )

    # And the helper it uses must be incapable of saving, rather than merely
    # called with save=False -- which is what the previous spelling-match
    # assumed without checking.
    if "await self._shutdown_kernel_unlocked()" in start:
        helper = source.split("async def _shutdown_kernel_unlocked(", 1)[1]
        helper = helper.split("\n    async def ", 1)[0]
        for forbidden in ("save_session", "notebook_state.save", "save=True"):
            assert forbidden not in helper, (
                f"the teardown helper reaches {forbidden}; a half-restored "
                "namespace must never be persisted"
            )
