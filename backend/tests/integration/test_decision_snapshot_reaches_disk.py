"""A decision snapshot taken from a generation a real save committed.

The unit tests build the generation by hand, which proves the writer copies and
verifies what it is given. What they cannot prove is that what a real save
leaves on disk is what the writer expects to find: the dump's streaming zstd
format, the signed generation manifest beside it, and the three paired files
the commit transaction writes under ``states/generation/<revision>/``.

That gap is the one this project keeps finding. ``carried_definitions`` passed
every unit test on every hop while nothing reached disk, because the save never
handed the parsed record on. So the assertions here are on the bundle read back
from disk, compared against the generation read back from disk.

Requires a real kernel: ``SCAI_RUN_KERNEL_TESTS=1``, i.e. the ``kernel``
profile.
"""
from __future__ import annotations

import hashlib
import os

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


@pytest_asyncio.fixture
async def sandbox(session_id, monkeypatch):
    """A sandbox whose directories are the ones ``get_session_*`` resolves to.

    The capture copies ``session.json`` from the session directory and the
    workspace from beside the state directory, so a sandbox pointed at an
    unrelated temporary tree would exercise a layout production never has.
    """
    import shutil

    from backend.config import get_session_dir, get_session_states
    from backend.utils.sandbox import LocalJupyterSandbox

    session_dir = get_session_dir(session_id)
    states = get_session_states(session_id)
    states.mkdir(parents=True, exist_ok=True)
    workspace = session_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (session_dir / "uploads").mkdir(parents=True, exist_ok=True)

    from backend.state import sessions
    from backend.tests.test_round_snapshot_capture import _write_checkpoint_db

    monkeypatch.setitem(sessions, session_id, _state(session_id))
    _write_checkpoint_db(states / "checkpoints.sqlite", session_id, round_number=2)

    sb = LocalJupyterSandbox(
        session_id=session_id, state_dir=str(states), working_dir=str(workspace),
    )
    await sb.start()
    try:
        yield sb
    finally:
        await sb.stop(save=False)
        shutil.rmtree(session_dir, ignore_errors=True)


@pytest.fixture
def snapshot_root(tmp_path, monkeypatch):
    from backend.config import _config

    root = tmp_path / "snapshots"
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", root)
    return root


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed_manifest(sandbox) -> dict:
    from backend.utils.runtime_manifest import read_runtime_manifest, runtime_manifest_path

    return read_runtime_manifest(
        runtime_manifest_path(
            sandbox.session_id, states_dir=str(sandbox.notebook_state.state_dir),
        )
    )


def _workflow(sandbox):
    import types

    return types.SimpleNamespace(sandbox=sandbox, session_id=sandbox.session_id)


def _state(session_id: str, *, round_number: int = 2):
    from backend.models.schemas import AnalysisState

    return AnalysisState(
        id=session_id, name="Kernel capture", user_id="owner-1",
        current_round=round_number, max_rounds=5,
    )


@pytest.mark.asyncio
async def test_a_capture_holds_the_generation_a_real_save_committed(
    sandbox, snapshot_root,
):
    from backend.constants import GATE_ROUND_WAIT
    from backend.services.round_snapshots.layout import (
        MANIFEST_NAME,
        PAIR_CHECKPOINT_RELPATH,
        PAIR_STATE_RELPATH,
        RUNTIME_PKL_RELPATH,
        snapshot_dir,
    )
    from backend.services.round_snapshots.manifest import read_snapshot_manifest
    from backend.services.round_snapshots.writer import capture_decision_snapshot
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    result = await sandbox.execute(
        "import numpy as np\n"
        "counts = np.arange(24).reshape(6, 4)\n"
        "def scale(x):\n    return x / 10\n"
        "scaled = scale(counts)\n",
        skip_safety_check=True,
    )
    assert result.success, result.error
    # A figure the analysis wrote, so the workspace is not empty.
    (sandbox.notebook_state.working_dir / "figure.png").write_bytes(b"PNG bytes")
    assert await sandbox.save_session() is True

    state = _state(sandbox.session_id)
    snapshot = await capture_decision_snapshot(
        _workflow(sandbox), state, gate=GATE_ROUND_WAIT, workflow_state={"current_round": 2},
    )
    assert snapshot is not None, "the capture found nothing to hold"

    committed = _committed_manifest(sandbox)
    bundle = snapshot_dir(sandbox.session_id, snapshot)
    manifest = read_snapshot_manifest(bundle / MANIFEST_NAME)

    assert manifest["runtime_revision"] == committed["revision"]
    assert manifest["pkl_sha256"] == committed["pkl_sha256"]
    assert snapshot == f"r002-round-g{committed['revision']}"

    # Every declared file is present and is what the manifest says it is.
    for relpath, entry in manifest["files"].items():
        target = bundle / relpath
        assert target.is_file(), relpath
        assert _digest(target) == entry["sha256"], relpath

    # And the ones that came from the generation are byte-for-byte the
    # generation's, not merely well-formed.
    states = sandbox.notebook_state.state_dir
    pair = states / GENERATION_PAIR_DIRNAME / str(committed["revision"])
    assert (bundle / RUNTIME_PKL_RELPATH).read_bytes() == (
        states / f"{sandbox.session_id}_session.pkl"
    ).read_bytes()
    assert (bundle / PAIR_STATE_RELPATH).read_bytes() == (pair / "state.json").read_bytes()
    assert _digest(bundle / PAIR_CHECKPOINT_RELPATH) == _digest(pair / "checkpoints.sqlite")

    assert "workspace/figure.png" in manifest["files"]
    assert manifest["coverage"]["saved_names"] >= 2
    assert manifest["capture_seconds"] >= 0
    assert manifest["bytes_total"] > 0


@pytest.mark.asyncio
async def test_the_dump_in_the_bundle_still_reads_as_a_runtime_generation(
    sandbox, snapshot_root,
):
    """A bundle nobody can rehydrate is a bundle nobody can restore.

    Read with the reader the kernel uses -- the format marker, then a streaming
    zstd sequence of dill frames -- because "the bytes match" and "the bytes are
    loadable" are different claims and only the second one is the point.

    Alias frames have to be resolved, not skipped, and the first version of
    this test learned that the hard way: ``answer = 42`` is the *same object*
    as the kernel bootstrap's ``SEED = 42``, because CPython caches small
    integers, so the save de-duplicated it into an alias. A reader that ignored
    three-tuples reported the name as missing from a dump that holds it.
    """
    from backend.constants import GATE_ROUND_WAIT
    from backend.services.round_snapshots.layout import RUNTIME_PKL_RELPATH, snapshot_dir
    from backend.services.round_snapshots.writer import capture_decision_snapshot

    result = await sandbox.execute(
        "answer = 42\nlabels = ['a', 'b', 'c']\n", skip_safety_check=True,
    )
    assert result.success, result.error
    assert await sandbox.save_session() is True

    state = _state(sandbox.session_id)
    snapshot = await capture_decision_snapshot(
        _workflow(sandbox), state, gate=GATE_ROUND_WAIT,
    )
    assert snapshot is not None

    import dill
    import zstandard

    restored = {}
    path = snapshot_dir(sandbox.session_id, snapshot) / RUNTIME_PKL_RELPATH
    with open(path, "rb") as raw:
        assert raw.read(len(b"SCAI_ZSTD_DILL_V1\n")) == b"SCAI_ZSTD_DILL_V1\n"
        with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            while True:
                try:
                    frame = dill.load(stream)
                except EOFError:
                    break
                if (
                    isinstance(frame, tuple)
                    and len(frame) == 3
                    and frame[0] == "__SCAI_ALIAS__"
                ):
                    _, name, target = frame
                    restored[name] = restored[target]
                elif isinstance(frame, tuple) and len(frame) == 2:
                    restored[frame[0]] = frame[1]
    assert "answer" in restored, sorted(restored)
    assert restored["answer"] == 42
    assert restored["labels"] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_decision_reached_after_more_work_commits_before_it_captures(
    sandbox, snapshot_root,
):
    """The freshness rule, against the real save rather than a stand-in.

    This is the annotation decision's shape: cells have run since the last
    commit, so the capture asks for a generation of its own and the bundle
    holds the namespace as it is now, not as it was several steps ago.
    """
    from backend.constants import GATE_ANNOTATION_CONFIRMATION
    from backend.services.round_snapshots.layout import MANIFEST_NAME, snapshot_dir
    from backend.services.round_snapshots.manifest import read_snapshot_manifest
    from backend.services.round_snapshots.writer import capture_decision_snapshot

    assert (await sandbox.execute("early = 1\n", skip_safety_check=True)).success
    assert await sandbox.save_session() is True
    first = _committed_manifest(sandbox)["revision"]

    assert (await sandbox.execute("late = 2\n", skip_safety_check=True)).success

    state = _state(sandbox.session_id, round_number=1)
    snapshot = await capture_decision_snapshot(
        _workflow(sandbox), state, gate=GATE_ANNOTATION_CONFIRMATION,
    )
    assert snapshot is not None

    manifest = read_snapshot_manifest(
        snapshot_dir(sandbox.session_id, snapshot) / MANIFEST_NAME
    )
    assert manifest["runtime_revision"] > first
    assert manifest["runtime_revision"] == _committed_manifest(sandbox)["revision"]
    assert "late" in _committed_manifest(sandbox)["saved_names"]
