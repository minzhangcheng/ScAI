"""The `0 figures` completion log is misleading: it counts only inline base64
captures (always ~0), never disk-saved figures. Verify the shared helper that
counts saved figures from an analysis log."""
from types import SimpleNamespace

from backend.agents.bioinformatics_agent import count_saved_figures


def test_counts_saved_figure_filenames_across_steps():
    log = [
        SimpleNamespace(saved_figure_filenames=["a.png", "b.png"]),
        SimpleNamespace(saved_figure_filenames=[]),
        SimpleNamespace(saved_figure_filenames=["c.png"]),
    ]
    assert count_saved_figures(log) == 3


def test_handles_missing_attr_and_dicts():
    log = [
        SimpleNamespace(),
        {"saved_figure_filenames": ["x.png", "y.png"]},
        None,
    ]
    assert count_saved_figures(log) == 2


def test_empty_log_returns_zero():
    assert count_saved_figures([]) == 0
    assert count_saved_figures(None) == 0


def test_counts_publication_figure_tool_and_deduplicates_names():
    log = [
        {"saved_figure_filenames": ["same_name.jpg"], "pub_figure_name": "same_name"},
        SimpleNamespace(pub_figure_name="publication_only", saved_figure_filenames=[]),
    ]
    assert count_saved_figures(log) == 2
