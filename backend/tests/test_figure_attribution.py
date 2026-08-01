"""Figure → round attribution across its three sources.

Source 1 (supervision records) is authoritative but sparse in practice — the
sandbox only emits its [Auto-saved] marker for figures still open when a cell
ends — so mtime attribution is a normal path, not a rare fallback.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

from backend.services.report.figure_attribution import (
    attribute_figures_to_rounds,
    figures_by_round,
    round_boundaries,
)

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _round(n, minutes):
    return SimpleNamespace(
        round_number=n, phase="innovation", direction="",
        timestamp=_T0 + timedelta(minutes=minutes),
    )


def _state(rounds, pub=None):
    return SimpleNamespace(round_results=rounds, publication_figures=pub or [])


def _png(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    if mtime is not None:
        import os
        os.utime(path, (mtime.timestamp(), mtime.timestamp()))
    return path


def test_boundaries_are_half_open_and_unbounded_at_the_edges():
    bounds = round_boundaries(_state([_round(1, 10), _round(2, 20), _round(3, 30)]))
    assert [b[0] for b in bounds] == [1, 2, 3]
    assert bounds[0][1] == float("-inf")
    assert bounds[-1][2] == float("inf")
    # Round 2 owns what was produced after round 1 finished, up to its own stamp.
    assert bounds[1][1] == (_T0 + timedelta(minutes=10)).timestamp()
    assert bounds[1][2] == (_T0 + timedelta(minutes=20)).timestamp()


def test_supervision_records_win_over_mtime(tmp_path):
    state = _state([_round(1, 10), _round(2, 20)])
    # mtime says round 2, the record says round 1 — the record is authoritative.
    fig = _png(tmp_path / "a.png", _T0 + timedelta(minutes=15))
    records = [{"round_number": 1, "saved_figure_filenames": ["a.png"]}]

    attr = attribute_figures_to_rounds(
        ["a.png"], state=state, records=records, figure_paths={"a.png": fig}
    )
    assert attr == {"a.png": 1}


def test_publication_round_used_when_no_record(tmp_path):
    state = _state(
        [_round(1, 10), _round(2, 20)],
        pub=[SimpleNamespace(png_path="/somewhere/b.png", round_number=2)],
    )
    fig = _png(tmp_path / "b.png", _T0 + timedelta(minutes=5))
    attr = attribute_figures_to_rounds(
        ["b.png"], state=state, records=[], figure_paths={"b.png": fig}
    )
    assert attr == {"b.png": 2}


def test_mtime_covers_figures_no_record_mentions(tmp_path):
    """The real-world case: the supervision log captured nothing at all."""
    state = _state([_round(1, 10), _round(2, 20), _round(3, 30)])
    paths = {
        "early.png": _png(tmp_path / "early.png", _T0 + timedelta(minutes=5)),
        "mid.png": _png(tmp_path / "mid.png", _T0 + timedelta(minutes=15)),
        "late.png": _png(tmp_path / "late.png", _T0 + timedelta(minutes=25)),
        "after.png": _png(tmp_path / "after.png", _T0 + timedelta(minutes=99)),
    }
    attr = attribute_figures_to_rounds(
        list(paths), state=state, records=[], figure_paths=paths
    )
    assert attr == {"early.png": 1, "mid.png": 2, "late.png": 3, "after.png": 3}


def test_unattributable_figures_are_omitted_not_guessed(tmp_path):
    state = _state([_round(1, 10)])
    attr = attribute_figures_to_rounds(
        ["ghost.png"], state=state, records=[], figure_paths={}
    )
    assert attr == {}


def test_records_match_on_basename(tmp_path):
    state = _state([_round(1, 10)])
    records = [{"round_number": 1,
                "saved_figure_filenames": ["/abs/results/figures/c.png"]}]
    attr = attribute_figures_to_rounds(["c.png"], state=state, records=records)
    assert attr == {"c.png": 1}


def test_records_without_a_round_are_ignored():
    records = [{"saved_figure_filenames": ["d.png"]}]
    attr = attribute_figures_to_rounds(["d.png"], state=None, records=records)
    assert attr == {}


def test_figures_by_round_inverts_the_map():
    grouped = figures_by_round({"a.png": 1, "b.png": 2, "c.png": 1})
    assert sorted(grouped[1]) == ["a.png", "c.png"]
    assert grouped[2] == ["b.png"]


def test_no_rounds_yields_no_attribution(tmp_path):
    fig = _png(tmp_path / "e.png", _T0)
    attr = attribute_figures_to_rounds(
        ["e.png"], state=_state([]), records=[], figure_paths={"e.png": fig}
    )
    assert attr == {}
