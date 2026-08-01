"""Tests for the Results brief-gallery figure discovery helper."""
import os
from pathlib import Path

import pytest

from backend.services.report.figure_discovery import (
    discover_workspace_figures,
    infer_figure_type,
)


def _touch(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _make_workspace(monkeypatch, tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "backend.services.report.figure_discovery.get_session_workspace",
        lambda session_id: ws,
    )
    return ws


def test_discovers_figures_dir_and_workspace_root(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "figures" / "umap_cell_types.png")
    _touch(ws / "root_overview.jpg")

    figs = discover_workspace_figures("sess-1")
    names = {f["name"] for f in figs}
    assert names == {"umap_cell_types", "root_overview"}
    by_name = {f["name"]: f for f in figs}
    assert by_name["umap_cell_types"]["source"] == "figures"
    assert by_name["umap_cell_types"]["type"] == "umap"
    assert by_name["root_overview"]["source"] == "workspace_root"


def test_infer_figure_type_rules():
    assert infer_figure_type("hep_umap_final") == "umap"
    assert infer_figure_type("deg_volcano") == "volcano"
    assert infer_figure_type("marker_violin") == "violin"
    assert infer_figure_type("marker_dotplot") == "dotplot"
    assert infer_figure_type("state_heatmap") == "heatmap"
    assert infer_figure_type("composition_barplot") == "barplot"
    assert infer_figure_type("cell_proportion") == "barplot"
    assert infer_figure_type("mystery_plot") == "other"


def test_discovers_nested_output_figures(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "round3_outputs" / "figures" / "volcano_r3.png")
    _touch(ws / "stage2_outputs" / "figures" / "hep_umap_state.png")
    _touch(ws / "custom_outputs" / "figures" / "generic_barplot.png")

    figs = discover_workspace_figures("sess-nested")
    by_name = {f["name"]: f for f in figs}
    assert set(by_name) == {"volcano_r3", "hep_umap_state", "generic_barplot"}
    assert by_name["volcano_r3"]["source"] == "round3_outputs/figures"
    assert by_name["hep_umap_state"]["source"] == "stage2_outputs/figures"
    assert by_name["generic_barplot"]["source"] == "custom_outputs/figures"


def test_ignores_unsupported_extensions(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "figures" / "keep.png")
    _touch(ws / "figures" / "skip.svg")
    _touch(ws / "figures" / "notes.txt")

    names = {f["name"] for f in discover_workspace_figures("sess-ext")}
    assert names == {"keep"}


def test_pairs_same_stem_pdf(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "figures" / "paired.png")
    _touch(ws / "figures" / "paired.pdf")
    _touch(ws / "figures" / "lonely.png")

    by_name = {f["name"]: f for f in discover_workspace_figures("sess-pdf")}
    assert by_name["paired"]["pdf_path"].endswith("paired.pdf")
    assert by_name["lonely"]["pdf_path"] is None


def test_pdf_symlink_outside_workspace_is_dropped(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    _touch(outside / "secret.pdf")
    _touch(ws / "figures" / "img.png")
    (ws / "figures" / "img.pdf").symlink_to(outside / "secret.pdf")

    by_name = {f["name"]: f for f in discover_workspace_figures("sess-pdf-traversal")}
    assert "img" in by_name
    assert by_name["img"]["pdf_path"] is None


def test_dedup_prefers_figures_dir_over_nested(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "figures" / "shared.png", mtime=1000.0)
    _touch(ws / "round2_outputs" / "figures" / "shared.png", mtime=9000.0)

    figs = discover_workspace_figures("sess-dedup")
    shared = [f for f in figs if f["name"] == "shared"]
    assert len(shared) == 1
    assert shared[0]["source"] == "figures"


def test_dedup_newest_wins_within_same_priority(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "a_outputs" / "figures" / "dup.png", mtime=1000.0)
    _touch(ws / "b_outputs" / "figures" / "dup.png", mtime=5000.0)

    figs = discover_workspace_figures("sess-dup-nested")
    dup = [f for f in figs if f["name"] == "dup"]
    assert len(dup) == 1
    assert dup[0]["source"] == "b_outputs/figures"


def test_ordering_by_priority_then_newest_first(monkeypatch, tmp_path):
    """Newest first within a source tier, because callers cap by head-slicing.

    ``_assemble_brief_figures`` truncates this list to DISCOVERY_CAP entries. With
    the oldest first, the cap discarded exactly the figures a late round had just
    produced, so the property under test is the ordering the cap depends on.
    """
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "figures" / "f_old.png", mtime=1000.0)
    _touch(ws / "figures" / "f_new.png", mtime=2000.0)
    _touch(ws / "root_img.png", mtime=500.0)
    _touch(ws / "x_outputs" / "figures" / "nested_img.png", mtime=100.0)

    order = [f["name"] for f in discover_workspace_figures("sess-order")]
    assert order == ["f_new", "f_old", "root_img", "nested_img"]


def test_missing_workspace_returns_empty(monkeypatch, tmp_path):
    missing = tmp_path / "workspace"
    monkeypatch.setattr(
        "backend.services.report.figure_discovery.get_session_workspace",
        lambda session_id: missing,
    )
    assert discover_workspace_figures("sess-missing") == []


def test_agent_chosen_subdirectory_is_discovered(monkeypatch, tmp_path):
    """A directory the agent invented is still analysis output.

    Nothing constrains where generated code saves a figure. A real run put 35 of
    its 59 images in a directory matching none of the three named scan roots, and
    every one of them was excluded from the gallery while the detailed report --
    which resolves recursively -- showed them.
    """
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "round3_results" / "r3_umap.png", mtime=100.0)

    found = {f["name"]: f for f in discover_workspace_figures("sess-deep")}
    assert "r3_umap" in found
    assert found["r3_umap"]["source"] == "round3_results"
    assert found["r3_umap"]["rel_path"] == "round3_results/r3_umap.png"
    assert found["r3_umap"]["diagnostic"] is False


def test_named_roots_win_a_stem_the_recursive_tier_also_sees(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "figures" / "shared.png", mtime=100.0)
    _touch(ws / "nested" / "shared.png", mtime=999.0)

    found = [f for f in discover_workspace_figures("sess-dedup") if f["name"] == "shared"]
    assert len(found) == 1
    assert found[0]["source"] == "figures"


def test_annotation_validation_is_discovered_but_tagged_diagnostic(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "annotation_validation" / "check.png", mtime=100.0)
    _touch(ws / "figures" / "result.png", mtime=100.0)

    ordered = discover_workspace_figures("sess-diag")
    by_name = {f["name"]: f for f in ordered}
    assert by_name["check"]["diagnostic"] is True
    assert by_name["result"]["diagnostic"] is False
    # Diagnostics sort last, so a candidate cap sheds them before results.
    assert [f["name"] for f in ordered] == ["result", "check"]


@pytest.mark.parametrize(
    "rel",
    [
        ".hidden/x.png",
        "__pycache__/x.png",
        "reports/figures/x.png",
        "a/b/c/too_deep.png",
    ],
)
def test_recursive_tier_skips_non_output_paths(monkeypatch, tmp_path, rel):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / rel, mtime=100.0)
    assert discover_workspace_figures("sess-skip") == []
