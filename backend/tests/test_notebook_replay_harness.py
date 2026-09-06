"""The replay harness's judgement, tested without a replay.

The comparison is the part that can be quietly wrong: a tolerance applied to the
wrong variable, a divergence counted in only one direction, or a threshold that
drifted away from the contract document after the document was committed. None
of those show up as an error -- they show up as a replay that passes.

The replay itself needs a real session and is deferred; docs/coding/plain-
persistence-baseline.md records that.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "scai-notebook-replay"
CONTRACT = REPO / "docs" / "coding" / "plain-persistence-replay-contract.md"


def _harness():
    """Load a script with no .py extension the way its shebang implies."""
    spec = importlib.util.spec_from_loader(
        "scai_notebook_replay",
        importlib.machinery.SourceFileLoader("scai_notebook_replay", str(HARNESS)),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _harness()


# --- the contract is the authority; the harness transcribes it ---------------


def test_the_harness_tolerances_still_match_the_committed_contract(harness):
    """A threshold that drifted from the document measures something else.

    The document was committed before the harness existed and says it may not be
    edited in response to a result. This is what stops the harness being edited
    instead.
    """
    table = CONTRACT.read_text()
    declared = {}
    for row in re.findall(r"^\| *\d+ *\|(.+)$", table, re.MULTILINE):
        cells = [c.strip() for c in row.split("|")]
        tolerance = cells[-2] if cells[-1] == "" else cells[-1]
        declared[cells[0]] = tolerance

    stated = " ".join(declared.values())
    assert "rel 1e-6" in stated, "the sum tolerance left the document"
    assert "rel 1e-4" in stated, "the embedding tolerance left the document"

    assert harness.TOLERANCES["x_sum"] == 1e-6
    assert harness.TOLERANCES["obsm_values"] == 1e-4
    # Everything else in the table is exact, and exactness is not a number.
    exact = {k for k, v in harness.TOLERANCES.items() if v is None}
    assert exact == {
        "shape", "obs_names_digest", "var_names_digest", "x_nonzero",
        "obs_categoricals", "obsm_shapes", "highly_variable",
        "persistent_outputs",
    }


def test_the_contract_document_exists_and_forbids_editing_after_a_result(harness):
    text = CONTRACT.read_text()
    assert "may be edited in response to a replay outcome" in text or \
        "may not be edited" in text or "Nothing below may be edited" in text
    assert "Do not adjust this document" in text


# --- the tolerance rule ------------------------------------------------------


def test_the_relative_rule_is_the_one_the_contract_states(harness):
    """abs(a-b) <= rtol * max(1, abs(b)).

    The max(1, ...) matters: without it the rule collapses to exact equality
    near zero, and a sum that legitimately cancels to ~0 would diverge on the
    last bit.
    """
    close = harness._close
    assert close(1.0 + 1e-7, 1.0, 1e-6)
    assert not close(1.0 + 1e-5, 1.0, 1e-6)
    # Near zero the floor keeps the bound absolute rather than vanishing.
    assert close(1e-7, 0.0, 1e-6)
    # Large values scale.
    assert close(1e9 + 500.0, 1e9, 1e-6)
    assert not close(1e9 + 5000.0, 1e9, 1e-6)


def test_nan_equals_nan_but_not_a_number(harness):
    """A NaN where the original had a value is a divergence, not a match.

    Comparing NaN with `==` returns False both ways, so an implementation that
    forgot this would report a divergence for two identical runs -- and one that
    over-corrected would hide a run that started producing NaN.
    """
    close = harness._close
    nan = float("nan")
    assert close(nan, nan, 1e-6)
    assert not close(nan, 1.0, 1e-6)
    assert not close(1.0, nan, 1e-6)


# --- variable comparison -----------------------------------------------------


def test_an_exact_variable_diverges_on_any_difference(harness):
    """A single cell in a different cluster is a real divergence.

    This is the reason the categorical comparison has no tolerance: rounding it
    away would blind the contract to the class of bug it exists for.
    """
    original = {"obs_categoricals": {"leiden": [("0", 500), ("1", 499)]}}
    replayed = {"obs_categoricals": {"leiden": [("0", 499), ("1", 500)]}}
    verdicts = {e["variable"]: e["verdict"] for e in harness.compare(original, replayed)}
    assert verdicts["obs_categoricals"] == "diverged"

    same = harness.compare(original, dict(original))
    assert {e["variable"]: e["verdict"] for e in same}["obs_categoricals"] == "match"


def test_a_sum_within_tolerance_matches_and_outside_it_does_not(harness):
    inside = harness.compare({"x_sum": 1.0e6}, {"x_sum": 1.0e6 + 0.5})
    assert {e["variable"]: e["verdict"] for e in inside}["x_sum"] == "match"

    outside = harness.compare({"x_sum": 1.0e6}, {"x_sum": 1.0e6 + 50.0})
    entry = [e for e in outside if e["variable"] == "x_sum"][0]
    assert entry["verdict"] == "diverged"
    assert entry["original"] == 1.0e6 and entry["replayed"] == 1.0e6 + 50.0


def test_a_variable_absent_from_both_runs_is_skipped_not_passed(harness):
    """"Nothing to compare" is not "matched". A fingerprint that extracted
    nothing would otherwise report a clean pass."""
    verdicts = {e["variable"]: e["verdict"] for e in harness.compare({}, {})}
    assert set(verdicts.values()) == {"absent"}
    assert "match" not in verdicts.values()


def test_a_variable_present_in_only_one_run_diverges(harness):
    result = harness.compare({"shape": [100, 2000]}, {})
    entry = [e for e in result if e["variable"] == "shape"][0]
    assert entry["verdict"] == "diverged"
    assert "one run only" in entry["detail"]


def test_an_embedding_outside_the_loose_bound_still_diverges(harness):
    """1e-4 is the loosest number in the table and is still a bound.

    A divergence larger than it means the seed did not take, which is what
    spec 5.3.5 is about.
    """
    original = {"obsm_values": {"X_umap": {
        "column_sums": [10.0, -4.0], "n_finite": 2000, "abs_mean": 3.5}}}
    drifted = {"obsm_values": {"X_umap": {
        "column_sums": [10.0, -4.0], "n_finite": 2000, "abs_mean": 3.9}}}
    entry = [e for e in harness.compare(original, drifted)
             if e["variable"] == "obsm_values"][0]
    assert entry["verdict"] == "diverged"

    jitter = {"obsm_values": {"X_umap": {
        "column_sums": [10.0 + 1e-5, -4.0], "n_finite": 2000, "abs_mean": 3.5}}}
    entry = [e for e in harness.compare(original, jitter)
             if e["variable"] == "obsm_values"][0]
    assert entry["verdict"] == "match"


# --- outcome divergence ------------------------------------------------------


def _cell(index, original, replayed, *, oname=None, rname=None):
    return {
        "index": index, "id": "c%d" % index, "bootstrap": False,
        "original": original, "replayed": replayed,
        "original_ename": oname, "replayed_ename": rname, "seconds": 0.0,
    }


def test_a_cell_that_now_succeeds_diverged_as_much_as_one_that_now_fails(harness):
    """Both directions.

    A replay that "fixed" a recorded failure did not reproduce the run. It is
    the easier divergence to rationalise away and the one most likely to mean
    the replay ran different code.
    """
    rows = [
        _cell(0, "ok", "ok"),
        _cell(1, "raised", "ok", oname="ValueError"),
        _cell(2, "ok", "raised", rname="NameError"),
    ]
    diverged = harness.outcome_divergences(rows)
    assert [d["index"] for d in diverged] == [1, 2]
    assert all(d["kind"] == "outcome" for d in diverged)


def test_a_cell_that_fails_identically_is_a_match(harness):
    """Cells that raised in the original run are replayed and expected to raise."""
    rows = [_cell(0, "raised", "raised", oname="KeyError", rname="KeyError")]
    assert harness.outcome_divergences(rows) == []


def test_the_same_cell_failing_differently_is_reported(harness):
    """A NameError where there was a KeyError is the replay contract's own
    failure mode: the namespace the cell assumed was not rebuilt."""
    rows = [_cell(0, "raised", "raised", oname="KeyError", rname="NameError")]
    diverged = harness.outcome_divergences(rows)
    assert len(diverged) == 1
    assert diverged[0]["kind"] == "exception-type"


def test_the_harness_does_not_stop_at_the_first_failure(harness):
    """A harness that stopped could not report the second divergence, and the
    original run did not stop either.

    Walked rather than grepped: the string "raised" contains "raise", so a
    substring check on this loop is green for the wrong reason in one direction
    and red for the wrong reason in the other.
    """
    import ast as _ast

    tree = _ast.parse(HARNESS.read_text())
    loops = [
        node for node in _ast.walk(tree)
        if isinstance(node, _ast.For)
        and isinstance(node.target, _ast.Name)
        and node.target.id == "cell"
        and isinstance(node.iter, _ast.Name)
        and node.iter.id == "cells"
    ]
    assert len(loops) == 1, "the execution loop moved"

    stoppers = [
        node for node in _ast.walk(loops[0])
        if isinstance(node, (_ast.Break, _ast.Raise, _ast.Return))
    ]
    assert stoppers == [], [type(n).__name__ for n in stoppers]

    # And nothing inside it swallows an exception either, which would turn a
    # cell that raised into one recorded as ok.
    handlers = [n for n in _ast.walk(loops[0]) if isinstance(n, _ast.ExceptHandler)]
    assert handlers == []


def test_the_harness_refuses_to_guess_which_session_it_was_given(harness):
    """Two notebooks or two dumps under one directory means the caller pointed
    at a parent. Picking the first would silently fingerprint one run and replay
    another."""
    source = HARNESS.read_text()
    assert source.count("name the session directory precisely") == 2
