"""P1-11: Sandbox dill restore/save failures must surface (not silently skip)."""
import inspect

import pytest

from backend.utils.sandbox_parts import runtime_restore, runtime_save


def test_restore_failure_raises_not_silent():
    """P1-11: dill restore failure must raise instead of silently continuing."""
    body = inspect.getsource(runtime_restore.restore_session)
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


def _latched_sandbox(tmp_path, *, baseline, live):
    """A sandbox whose restore failed and which has run one cell since.

    ``baseline`` is what the aborted restore left in the namespace, ``live`` what
    it holds now; both are the *persistable* names, i.e. what save_session would
    actually dump. The kernel is stubbed so the test states its namespace
    directly instead of standing up a Jupyter kernel to produce one.
    """
    import asyncio
    import json
    from unittest.mock import MagicMock

    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    # __init__ is cheap -- it starts no kernel -- so call it for real and
    # override only the collaborators this test needs to control. Bypassing
    # it with __new__ made the fixture drift: every attribute __init__ later
    # grew was simply absent, and save_session reads several of them.
    sandbox = LocalJupyterSandbox(
        session_id="restore-failed-rebuild",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sandbox._session_restore_failed = True
    sandbox._restore_failed_at_revision = 0
    sandbox._restore_failed_globals = set(baseline)
    sandbox.runtime_revision = 1  # a cell HAS run since the failure
    sandbox.runtime_dependency_tracker = None
    sandbox._known_user_globals = set(live)
    sandbox._exec_lock = asyncio.Lock()
    pkl = tmp_path / "restore-failed-rebuild_session.pkl"
    pkl.write_bytes(b"ORIGINAL")
    sandbox.notebook_state = MagicMock(session_path=pkl)

    async def _fake_exec(code, **kwargs):
        if "__PERSISTABLE_GLOBALS__" in code:
            return CodeExecutionResult(
                success=True,
                output="__PERSISTABLE_GLOBALS__:" + json.dumps(sorted(live)),
            )
        if "__USER_GLOBALS__" in code:
            return CodeExecutionResult(
                success=True,
                output="__USER_GLOBALS__:" + json.dumps(sorted(live)),
            )
        pkl.write_bytes(b"REBUILT")
        return CodeExecutionResult(success=True, output="Session saved: 1 variables")

    sandbox._execute_unlocked = _fake_exec
    return sandbox, pkl


@pytest.mark.asyncio
async def test_a_cell_that_built_nothing_does_not_disarm_the_restore_guard(tmp_path):
    """One ``import scanpy as sc`` is not a rebuilt analysis.

    The guard used to require only that ``runtime_revision`` had advanced once
    past the mark taken when the restore failed, and it advances on *any*
    successful cell. A single import therefore disarmed it, and since modules
    are never written to the pkl, the resulting save put a namespace with no
    analysis state in it over the only surviving copy of the pre-crash runtime --
    after which the next resume rehydrated that and reported the runtime
    complete. A module leaves the persistable namespace exactly as it found it,
    which is why the comparison is against contents rather than a cell count.
    """
    baseline = {"In", "Out", "SEED"}
    sandbox, pkl = _latched_sandbox(tmp_path, baseline=baseline, live=baseline)

    await sandbox.save_session()

    assert pkl.read_bytes() == b"ORIGINAL"
    assert sandbox._session_restore_failed is True, (
        "the latch must stay armed until something is actually rebuilt"
    )


@pytest.mark.asyncio
async def test_a_rebuilt_namespace_is_persisted_over_the_previous_pkl(tmp_path):
    """A legacy/manual recovery latch may persist genuinely rebuilt state.

    Normal start/restart discards a partial namespace. This defense-in-depth
    branch covers a restore-failed namespace supplied by an older or manual
    recovery path: once it proves real recomputation, refusing forever would
    make the next resume silently rehydrate the pre-crash pkl instead.
    """
    baseline = {"In", "Out", "SEED"}
    sandbox, pkl = _latched_sandbox(
        tmp_path, baseline=baseline, live=baseline | {"adata"}
    )
    from unittest.mock import AsyncMock

    persistence = {
        "lifecycle_generation": sandbox._expected_lifecycle_generation,
        "current_revision": 0,
        "saved_revision": 0,
    }

    def _update_persistence(status, **updates):
        persistence.update({"status": status, **updates})
        sandbox.notebook_state.runtime_persistence = persistence
        return dict(persistence)

    sandbox.notebook_state.runtime_persistence = persistence
    sandbox.notebook_state.update_runtime_persistence.side_effect = (
        _update_persistence
    )
    sandbox.runtime_persistence_commit_callback = AsyncMock()

    assert await sandbox.save_session() is True

    assert pkl.read_bytes() == b"REBUILT"
    assert sandbox._session_restore_failed is False
    assert sandbox._restore_failed_globals is None
    sandbox.runtime_persistence_commit_callback.assert_awaited_once()
    event = sandbox.runtime_persistence_commit_callback.await_args.args[0]
    assert event["session_id"] == sandbox.session_id
    assert event["current_revision"] == 1
    assert event["saved_revision"] == 1


@pytest.mark.asyncio
async def test_an_unmeasured_baseline_keeps_the_previous_pkl(tmp_path):
    """No baseline means no evidence, and no evidence means keep the pkl."""
    baseline = {"In", "Out", "SEED"}
    sandbox, pkl = _latched_sandbox(
        tmp_path, baseline=baseline, live=baseline | {"adata"}
    )
    sandbox._restore_failed_globals = None  # the probe could not be taken

    await sandbox.save_session()

    assert pkl.read_bytes() == b"ORIGINAL"


def test_restore_failure_records_the_namespace_it_left_behind():
    """The baseline has to be captured where the latch is armed.

    Taken later it would already include whatever the agent rebuilt, and the
    comparison in save_session would compare the namespace against itself.
    """
    body = inspect.getsource(runtime_restore.restore_session)
    assert "_restore_failed_globals" in body, (
        "the failed-restore branch must record the persistable namespace it left"
    )


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


def test_save_session_collects_and_reports_skipped_vars():
    """P1-11: save_session must track skipped vars (no silent except: pass) and log them."""
    body = "\n".join(
        (
            inspect.getsource(runtime_save._build_save_code),
            inspect.getsource(runtime_save._execute_save),
        )
    )
    # The save logic involves both Python-side handling AND kernel-side injected code.
    # The injected code (string) MUST collect skipped variable names and print them,
    # and the Python side MUST log them when present.
    assert "skipped" in body.lower(), "P1-11: must track skipped variable names"
    assert "logger." in body, "P1-11: must log skipped vars / report them"
