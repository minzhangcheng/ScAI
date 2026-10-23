"""Exercise notebook replay from a recorded session in a fresh kernel.

The comparison variables, tolerances, and limitations are documented in
``docs/notebook-replay.md``. This smoke test covers its synthetic session;
representative workflow replay requires separate evidence.

Requires a real kernel through the ``kernel`` test profile.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import pathlib

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]

REPO = pathlib.Path(__file__).resolve().parents[3]
HARNESS = REPO / "scripts" / "scai-notebook-replay"

# Seeded, so a replay that reproduces the run produces the same numbers, and a
# divergence means something rather than meaning numpy.
SESSION_CODE = """
import numpy as np
import anndata as ad

np.random.seed(42)
adata = ad.AnnData(X=np.random.rand(20, 5).astype("float32"))
adata.obs["leiden"] = ["a", "b"] * 10
print("built", adata.shape)
"""


def _harness():
    spec = importlib.util.spec_from_loader(
        "scai_notebook_replay",
        importlib.machinery.SourceFileLoader("scai_notebook_replay", str(HARNESS)),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest_asyncio.fixture
async def replayable_session(session_id, tmp_path):
    """A session with a bootstrap in its notebook and a committed dump."""
    import shutil

    from backend.config import get_session_dir, get_session_states
    from backend.utils.sandbox import LocalJupyterSandbox

    session_dir = get_session_dir(session_id)
    states = get_session_states(session_id)
    states.mkdir(parents=True, exist_ok=True)
    workspace = session_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    sb = LocalJupyterSandbox(
        session_id=session_id, state_dir=str(states), working_dir=str(workspace),
    )
    await sb.start()
    try:
        result = await sb.execute(SESSION_CODE, skip_safety_check=True, declared_outputs=["adata"])
        assert result.success, result.error
        # One recorded failure, so the replay has an expected-to-fail cell --
        # and a cell after it, or "the replay did not stop at the first failure"
        # would have nothing to check.
        failed = await sb.execute("raise ValueError('recorded')", skip_safety_check=True)
        assert failed.success is False
        after = await sb.execute(
            "adata.obs['post_failure'] = 'continued'", skip_safety_check=True,
        )
        assert after.success, after.error
        assert await sb.save_session() is True
        sb.notebook_state.save()
        from backend.models.schemas import AnalysisState, CodeExecution
        state = AnalysisState(id=session_id, user_id="owner", code_history=[CodeExecution(
            code=SESSION_CODE, declared_persistent_outputs=["adata"],
            realized_persistent_outputs=["adata"],
        )])
        (session_dir / "session.json").write_text(state.model_dump_json())
        yield sb, states
    finally:
        await sb.stop(save=False)
        shutil.rmtree(session_dir, ignore_errors=True)


def test_the_harness_reads_a_dump_the_real_save_path_wrote(replayable_session):
    """The committed dump is a zstd stream of dill frames behind a magic header.

    The harness reimplements that reader, so this is the assertion that its
    reimplementation matches the writer — the one thing its unit tests, which
    write their own dumps, structurally cannot check.
    """
    _sb, states = replayable_session
    tool = _harness()

    dumps = sorted(states.rglob("*.pkl"))
    assert len(dumps) == 1, dumps

    namespace = tool.load_committed_namespace(dumps[0])
    assert "adata" in namespace, sorted(namespace)
    assert namespace["adata"].shape == (20, 5)


def test_the_harness_reads_a_notebook_the_real_writer_wrote(replayable_session):
    """Including the bootstrap marker A2-9 added, which is what makes a replay
    run in the environment the session ran in rather than the reader's."""
    _sb, states = replayable_session
    tool = _harness()

    cells = tool.read_cells(states)
    assert cells, "no cells"
    assert cells[0]["bootstrap"] is True
    assert any(c["original_outcome"] == "raised" for c in cells), (
        "the recorded failure is missing; A2-9's gate is what puts it there"
    )
    assert any("np.random.seed(42)" in c["code"] for c in cells)


def test_the_fingerprint_is_taken_from_the_committed_dump(replayable_session, tmp_path):
    """Not from a live kernel. By the time a session is worth replaying its
    kernel is gone, and the dump is what the persistence layer says the run
    ended with."""
    _sb, states = replayable_session
    tool = _harness()

    out = tmp_path / "fingerprint.json"

    class _Args:
        session_dir = str(states)

    _Args.out = str(out)
    assert tool.cmd_fingerprint(_Args()) == 0

    payload = json.loads(out.read_text())
    assert payload["present"] is True
    assert payload["shape"] == [20, 5]
    assert payload["obs_categoricals"]["leiden"] == [("a", 10), ("b", 10)] or \
        payload["obs_categoricals"]["leiden"] == [["a", 10], ["b", 10]]
    assert payload["x_nonzero"] > 0


def test_the_comparison_accepts_a_fingerprint_of_itself(replayable_session, tmp_path):
    """A fingerprint compared against itself must match on every declared
    variable. If it does not, the comparison is wrong in a way no divergence
    test would reveal — it would just look like the run diverged.
    """
    _sb, states = replayable_session
    tool = _harness()

    out = tmp_path / "fingerprint.json"

    class _Args:
        session_dir = str(states)

    _Args.out = str(out)
    tool.cmd_fingerprint(_Args())
    payload = json.loads(out.read_text())

    results = tool.compare(payload, payload)
    diverged = [entry for entry in results if entry["verdict"] == "diverged"]
    assert diverged == [], diverged
    assert any(entry["verdict"] == "match" for entry in results)


def test_the_bootstrap_lets_a_replay_choose_a_clean_workspace(replayable_session):
    """Spec 5.3.4's point, and the replay contract's precondition.

    The bootstrap records the absolute roots of the run it came from and chdirs
    to them. If those are fixed, a replay "into a clean workspace" leaves that
    workspace on cell 1 and writes into the original session's directory -- so
    the contract's central precondition was not achievable by any notebook this
    code produces, and two replays of one session would fight over its files.

    The literal stays as the default, because the notebook must still describe
    the run it recorded. Nothing sets these in production.
    """
    _sb, states = replayable_session
    tool = _harness()

    cells = tool.read_cells(states)
    bootstrap = [c for c in cells if c["bootstrap"]]
    assert bootstrap, "no bootstrap cell"
    joined = "\n".join(c["code"] for c in bootstrap)

    assert "SCAI_REPLAY_WORKSPACE" in joined
    assert "SCAI_REPLAY_SESSION_TMP" in joined
    assert "SCAI_REPLAY_STATE_DIR" in joined
    # Derived, not restated: an overridden workspace the write guard would then
    # refuse to write in is worse than no override at all.
    assert "_ALLOWED_WRITE_ROOTS = [\n    _SCAI_WORKSPACE," in joined


@pytest.mark.asyncio
async def test_a_replay_reproduces_the_run_including_its_bootstrap(
    replayable_session, tmp_path,
):
    """The whole harness, end to end, on a seeded toy session.

    **Not spec §7 Goal 4**, which needs a completed analysis on real input data.
    This asserts the machinery: every cell executes -- cell 0 included -- the
    recorded failure fails again, the replay continues past it, and every
    declared variable matches. A seeded session should reproduce exactly, so a
    divergence here is a defect rather than floating-point drift.

    This test previously asserted the opposite about cell 0. The bootstrap
    imported `scai_kernel_trust` and three compat shims that exist only because
    ScAI's launcher puts `backend/utils/kernel_startup` on the kernel path, so a
    plain kernel raised ModuleNotFoundError and the test pinned that as a known
    limitation. The 2026-09-07 amendment to the replay contract defines "clean
    kernel" to include those modules, the bootstrap now tolerates their absence,
    and the harness puts the directory on the path -- so cell 0 runs, and the
    thing to assert is that it does.
    """
    _sb, states = replayable_session
    tool = _harness()

    fingerprint = tmp_path / "fingerprint.json"
    report = tmp_path / "report.json"

    class _Fp:
        session_dir = str(states)
        out = str(fingerprint)

    assert tool.cmd_fingerprint(_Fp()) == 0

    class _Rp:
        session_dir = str(states)
        workspace = str(tmp_path / "clean")

    _Rp.fingerprint = str(fingerprint)
    _Rp.report = str(report)

    exit_code = tool.cmd_replay(_Rp())

    payload = json.loads(report.read_text())
    outcomes = payload["cell_outcomes"]
    assert len(outcomes) == payload["cells"] > 0

    summary = "\n".join(
        "cell %s (bootstrap=%s): %s -> %s %s"
        % (row["index"], row["bootstrap"], row["original"], row["replayed"],
           row.get("replayed_ename"))
        for row in payload["outcome_divergences"]
    )

    # Cell 0 is the environment. If it does not run, nothing after it means
    # anything, and the notebook is not the complete record A2-9 made it.
    bootstrap_rows = [row for row in outcomes if row["bootstrap"]]
    assert bootstrap_rows, "the notebook has no bootstrap cell"
    assert all(row["replayed"] == "ok" for row in bootstrap_rows), summary

    # It did not stop at the recorded failure.
    raised = [row for row in outcomes if row["original"] == "raised"]
    assert raised, "the session's recorded failure is missing"
    after_failure = [row for row in outcomes if row["index"] > raised[0]["index"]]
    assert after_failure, "the session has no cell after its recorded failure"
    assert all(row["replayed"] == "ok" for row in after_failure), (
        "the replay stopped at the first failure: " + repr(after_failure)
    )

    assert payload["outcome_divergences"] == [], summary

    diverged = [v for v in payload["variables"] if v["verdict"] == "diverged"]
    assert diverged == [], diverged
    assert any(v["verdict"] == "match" for v in payload["variables"])
    assert exit_code == 0

    assert (tmp_path / "clean").exists()


def test_the_bootstrap_tolerates_scais_kernel_modules_being_absent():
    """The conditional guard, asserted on the source it generates.

    Conditional rather than removed: inside ScAI the modules are always there
    and the cell does exactly what it always did, so the recorded cell is still
    what ran. A rewrite would have cost that, which is why this is a guard.
    """
    from backend.utils.sandbox_parts.kernel_bootstrap import (
        _COMMON_BOOTSTRAP_TEMPLATE,
    )

    source = _COMMON_BOOTSTRAP_TEMPLATE
    assert "try:\n    import scai_kernel_trust as _scai_kernel_trust" in source
    assert "if _scai_kernel_trust is not None:" in source
    # The shims are imported when missing, not merely skipped: they patch
    # library defects and can move numbers.
    assert "__import__(_scai_compat_name)" in source
    assert 'sys.modules["scai_pydeseq2_compat"].install()' not in source


def test_the_harness_puts_the_kernel_startup_modules_on_the_path():
    """"Same installation" includes them. Without this the replay runs but is a
    weaker claim than the contract describes, and the harness says so."""
    source = HARNESS.read_text()
    assert 'REPO / "backend" / "utils" / "kernel_startup"' in source
    assert 'environment["PYTHONPATH"]' in source
    assert "weaker claim" in source
