"""The red-tree check has to be able to run the suite it is checking.

A `git archive` export has no `.git`, and the runner resolves HEAD before it
reaches pytest, so the export-based recipe exits without running anything and
silently produces no evidence at all. A detached worktree carries a `.git`
file pointing at the main repository, so `git rev-parse` works there.

The script is read inside each test rather than at import. A module that
raises while being collected makes the runner report an incomplete collection
and publish no evidence at all, which is precisely the state a red-tree run is
in -- so an import-time read would replace every readable failure with the same
opaque gate message.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "scai-red-tree"


def _script() -> str:
    if not SCRIPT_PATH.is_file():
        pytest.fail(f"{SCRIPT_PATH} does not exist")
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_uses_a_worktree_rather_than_an_archive():
    script = _script()
    assert "git worktree add --detach" in script
    assert "git archive" not in script


def test_runs_the_pinned_runner_inside_the_base_tree():
    assert "scripts/scai-test fast" in _script()


def test_a_passing_test_on_the_base_tree_is_reported_as_failure_to_prove():
    assert "proves nothing" in _script()


def test_worktree_lives_outside_tmp():
    """A /tmp export costs four phantom sandbox failures under the runner."""
    script = _script()
    assert "/home/minzhang/scai-red-tree" in script
    assert "/tmp/scai-red-tree" not in script


def test_the_worktree_is_removed_even_when_the_run_fails():
    script = _script()
    assert "trap" in script and "git worktree remove" in script


def test_a_non_zero_exit_alone_is_not_accepted_as_evidence():
    """The runner exits non-zero for usage errors too.

    A parametrised node id containing brackets is read as a glob and rejected
    before pytest starts, so the check would have reported a red tree with no
    test having run.
    """
    script = _script()
    assert "RED-TREE INCONCLUSIVE" in script
    assert "evidence.json" in script
    assert "published no evidence" in script


def test_a_setup_error_is_not_accepted_as_evidence():
    """A count is not a signature.

    A fixture failing on a missing dependency raises failed+errors just as a
    real detection does, and would otherwise be reported as proof.
    """
    script = _script()
    assert "junit.xml" in script
    assert "a detection would" in script
    assert "module under test is missing" in script


def test_the_worktree_path_is_unique_per_invocation():
    """Two concurrent checks must not delete each other's worktree."""
    assert "${BASE_SHA:0:12}.$$" in _script()


def test_named_nodes_must_each_be_the_node_that_failed():
    """An unrelated pre-existing failure in the same file is not evidence."""
    script = _script()
    assert "UNPROVEN" in script
    assert "Every named node has to be the one that failed" in script


def test_the_script_is_executable():
    if not SCRIPT_PATH.is_file():
        pytest.fail(f"{SCRIPT_PATH} does not exist")
    assert SCRIPT_PATH.stat().st_mode & 0o111, "scripts/scai-red-tree is not executable"
