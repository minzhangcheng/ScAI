"""The `0 figures` completion log is misleading: it counts only inline base64
captures (always ~0), never disk-saved figures. Verify the shared helper that
counts saved figures from an analysis log."""
from types import SimpleNamespace

import pytest

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
        {
            "saved_figure_filenames": ["same_name.jpg"],
            "pub_figure_name": "same_name",
            "pub_figure_result": "Figure same_name has been generated and saved.",
        },
        SimpleNamespace(
            pub_figure_name="publication_only",
            pub_figure_result="Figure publication_only has been approved by PI and saved.",
            saved_figure_filenames=[],
        ),
    ]
    assert count_saved_figures(log) == 2


@pytest.mark.parametrize(
    "result",
    [
        None,
        "JPG file was not created: /tmp/missing.jpg",
        "Figure appears to be blank (file size: 120 bytes)",
        "Figure generation failed: ValueError",
        "Figure x has been skipped after 3 failed attempts",
    ],
)
def test_failed_publication_figure_attempt_is_not_counted(result):
    log = [SimpleNamespace(
        pub_figure_name="not_saved",
        pub_figure_result=result,
        saved_figure_filenames=[],
    )]
    assert count_saved_figures(log) == 0
