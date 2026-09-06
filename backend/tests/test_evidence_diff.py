"""The evidence differ must not report agreement it cannot see."""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "scai-evidence-diff"


@pytest.fixture(autouse=True)
def _require_the_script_under_test():
    """Fail every test here while the differ does not exist.

    Without this, ``python3 <missing script>`` exits 2 with empty stdout, so
    the tests that assert exit 2 would pass on the unpatched tree for entirely
    the wrong reason. The wording matters: scripts/scai-red-tree accepts a
    setup-phase error as red only when it carries this exact phrase.
    """
    if not SCRIPT.is_file():
        pytest.fail("module under test is missing: %s" % SCRIPT)


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="3" failures="{n}">
<testcase classname="t/test_a.py" name="test_ok"/>
<testcase classname="t/test_a.py" name="test_bad">{body}</testcase>
<testcase classname="t/test_b.py" name="test_other"/>
</testsuite></testsuites>
"""
FAILURE = '<failure message="boom">trace</failure>'


def _run_dir(root, name, body, *, profile="backend-full", complete=True,
             declared_failed=None):
    d = root / name
    d.mkdir(parents=True)
    n = 1 if body else 0
    (d / "junit.xml").write_text(JUNIT.format(n=n, body=body))
    (d / "evidence.json").write_text(json.dumps({
        "profile": profile,
        "counts": {"collected": 3, "passed": 3 - n,
                   "failed": n if declared_failed is None else declared_failed,
                   "errors": 0, "skipped": 0},
        "outcomes": {"complete": complete, "count": 3},
    }))
    return d


def _diff(a, b):
    return subprocess.run(
        ["python3", str(SCRIPT), str(a), str(b)],
        capture_output=True, text=True,
    )


def test_identical_failing_sets_agree(tmp_path):
    a = _run_dir(tmp_path, "a", FAILURE)
    b = _run_dir(tmp_path, "b", FAILURE)
    result = _diff(a, b)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "introduced: none" in result.stdout


def test_a_new_failure_is_reported_and_fails(tmp_path):
    a = _run_dir(tmp_path, "a", "")
    b = _run_dir(tmp_path, "b", FAILURE)
    result = _diff(a, b)
    assert result.returncode == 1
    assert "t/test_a.py::test_bad" in result.stdout


def test_an_empty_junit_is_inconclusive_not_agreement(tmp_path):
    """Zero parsed test cases must never read as 'nothing changed'."""
    a = _run_dir(tmp_path, "a", FAILURE)
    b = _run_dir(tmp_path, "b", FAILURE)
    (b / "junit.xml").write_text(
        '<?xml version="1.0"?><testsuites></testsuites>'
    )
    result = _diff(a, b)
    assert result.returncode == 2
    assert "introduced: none" not in result.stdout


def test_a_missing_junit_is_inconclusive(tmp_path):
    a = _run_dir(tmp_path, "a", FAILURE)
    b = _run_dir(tmp_path, "b", FAILURE)
    (b / "junit.xml").unlink()
    result = _diff(a, b)
    assert result.returncode == 2


def test_a_missing_evidence_file_is_inconclusive(tmp_path):
    a = _run_dir(tmp_path, "a", FAILURE)
    b = _run_dir(tmp_path, "b", FAILURE)
    (b / "evidence.json").unlink()
    result = _diff(a, b)
    assert result.returncode == 2


def test_an_incomplete_collection_is_inconclusive(tmp_path):
    """A run whose collection did not complete measured nothing."""
    a = _run_dir(tmp_path, "a", FAILURE)
    b = _run_dir(tmp_path, "b", FAILURE, complete=False)
    result = _diff(a, b)
    assert result.returncode == 2
    assert "introduced: none" not in result.stdout


def test_two_different_profiles_are_not_comparable(tmp_path):
    a = _run_dir(tmp_path, "a", FAILURE, profile="backend-full")
    b = _run_dir(tmp_path, "b", FAILURE, profile="kernel")
    result = _diff(a, b)
    assert result.returncode == 2


def test_junit_and_counts_must_agree(tmp_path):
    """A truncated junit.xml beside an intact count is the false-green shape."""
    a = _run_dir(tmp_path, "a", FAILURE)
    b = _run_dir(tmp_path, "b", FAILURE, declared_failed=7)
    result = _diff(a, b)
    assert result.returncode == 2
    assert "introduced: none" not in result.stdout
