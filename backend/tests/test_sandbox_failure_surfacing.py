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

    sandbox = LocalJupyterSandbox.__new__(LocalJupyterSandbox)
    sandbox.session_id = "restore-failed-rebuild"
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

    assert await sandbox.save_session() is True

    assert pkl.read_bytes() == b"REBUILT"
    assert sandbox._session_restore_failed is False
    assert sandbox._restore_failed_globals is None


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
    src = _read_src()
    body = _extract_method(src, "_restore_session", "restore_session")
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
