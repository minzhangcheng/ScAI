"""Placement enforcement: a figure sits in the section owning its round,
appears once, and is never silently dropped."""
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

from backend.services.report.figure_placement import enforce_figure_placement
from backend.services.report.report_outline import build_report_outline

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _round(n, phase, direction=""):
    return SimpleNamespace(
        round_number=n, phase=phase, direction=direction,
        timestamp=_T0 + timedelta(minutes=n * 10),
    )


def _outline():
    return build_report_outline(SimpleNamespace(
        round_results=[
            _round(1, "exploration"),
            _round(2, "innovation", "A"),
            _round(3, "innovation", "B"),
            _round(4, "integration"),
        ],
        publication_figures=[],
    ))


def _sections(outline):
    return [
        {"id": s.id, "title": s.default_title, "content": "narrative.",
         "analysis_type": s.analysis_type, "figures": [], "tables": []}
        for s in outline
    ]


def _figs_in(section):
    return [m.strip() for m in re.findall(r"\[FIGURE:\s*([^\]]+)\]", section["content"])]


def _by_id(sections):
    return {s["id"]: s for s in sections}


def test_cross_round_reference_is_relocated():
    """A round-3 figure cannot illustrate the exploration narrative, whatever
    its filename suggests — this is the ..._qc_overview failure."""
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    by_id["exploration_findings"]["content"] += (
        "\n[FIGURE: r3_qc_overview.png]\n[FIGURE_CAPTION: looks like QC]\n"
    )

    report = enforce_figure_placement(
        sections, outline, {"r3_qc_overview.png": 3}, ["r3_qc_overview.png"]
    )

    assert _figs_in(by_id["exploration_findings"]) == []
    assert _figs_in(by_id["direction_2"]) == ["r3_qc_overview.png"]
    assert report.relocated == [("r3_qc_overview.png", "exploration_findings", "direction_2")]
    # The caption travels with the figure.
    assert "looks like QC" in by_id["direction_2"]["content"]


def test_repeat_references_keep_only_the_first():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    by_id["direction_1"]["content"] += (
        "\n[FIGURE: a.png]\n[FIGURE_CAPTION: one]\n"
        "more text\n[FIGURE: a.png]\n[FIGURE_CAPTION: two]\n"
        "\n[FIGURE: a.png]\n"
    )

    report = enforce_figure_placement(sections, outline, {"a.png": 2}, ["a.png"])

    assert _figs_in(by_id["direction_1"]) == ["a.png"]
    assert len(report.duplicates_removed) == 2


def test_unreferenced_figures_are_appended_to_their_round_section():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)

    report = enforce_figure_placement(
        sections, outline,
        {"x.png": 1, "y.png": 3}, ["x.png", "y.png"],
    )

    assert _figs_in(by_id["exploration_findings"]) == ["x.png"]
    assert _figs_in(by_id["direction_2"]) == ["y.png"]
    assert sorted(report.appended) == [("x.png", "exploration_findings"),
                                       ("y.png", "direction_2")]
    assert report.unplaced == []


def test_appended_figures_follow_production_order():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)

    enforce_figure_placement(
        sections, outline,
        {"late.png": 2, "early.png": 2, "mid.png": 2},
        ["late.png", "early.png", "mid.png"],
        figure_order={"early.png": 1.0, "mid.png": 2.0, "late.png": 3.0},
    )

    assert _figs_in(by_id["direction_1"]) == ["early.png", "mid.png", "late.png"]


def test_correctly_placed_figures_keep_their_position_in_the_prose():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    by_id["direction_1"]["content"] = (
        "Opening claim.\n[FIGURE: a.png]\n[FIGURE_CAPTION: cap]\n"
        "Closing discussion that refers to the figure above."
    )

    enforce_figure_placement(sections, outline, {"a.png": 2}, ["a.png"])

    content = by_id["direction_1"]["content"]
    assert content.index("Opening claim") < content.index("[FIGURE: a.png]")
    assert content.index("[FIGURE: a.png]") < content.index("Closing discussion")


def test_references_to_nonexistent_files_are_dropped():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    by_id["direction_1"]["content"] += "\n[FIGURE: hallucinated.png]\n"

    report = enforce_figure_placement(sections, outline, {}, ["real.png"])

    assert "hallucinated.png" not in by_id["direction_1"]["content"]
    assert report.unknown_removed == [("hallucinated.png", "direction_1")]


def test_unattributable_figure_stays_where_the_model_put_it():
    """With no round to move it to, the model's own judgement is the best
    available signal."""
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    by_id["direction_1"]["content"] += "\n[FIGURE: orphan.png]\n"

    report = enforce_figure_placement(sections, outline, {}, ["orphan.png"])

    assert _figs_in(by_id["direction_1"]) == ["orphan.png"]
    assert report.relocated == []


def test_figures_of_a_round_with_no_section_land_in_the_last_figure_section():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)

    report = enforce_figure_placement(
        sections, outline, {"z.png": 99}, ["z.png"]
    )

    # Round 99 owns no section; the figure is still shown rather than dropped.
    assert report.unplaced == []
    assert any("z.png" in _figs_in(s) for s in sections)


def test_sections_that_own_no_figures_never_receive_any():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    for sid in ("abstract", "methods", "mechanism_candidates", "conclusions"):
        by_id[sid]["content"] += f"\n[FIGURE: f{sid}.png]\n"
    known = [f"f{sid}.png" for sid in
             ("abstract", "methods", "mechanism_candidates", "conclusions")]
    attribution = {name: 2 for name in known}

    enforce_figure_placement(sections, outline, attribution, known)

    for sid in ("abstract", "methods", "mechanism_candidates", "conclusions"):
        assert _figs_in(by_id[sid]) == []
    assert sorted(_figs_in(by_id["direction_1"])) == sorted(known)


def test_empty_section_list_reports_everything_unplaced():
    report = enforce_figure_placement([], _outline(), {"a.png": 1}, ["a.png"])
    assert report.unplaced == ["a.png"]


def test_text_only_integration_section_is_not_reported_figureless():
    outline = _outline()
    sections = _sections(outline)
    report = enforce_figure_placement(
        sections,
        outline,
        {"explore.png": 1, "a.png": 2, "b.png": 3},
        ["explore.png", "a.png", "b.png"],
    )

    assert report.figureless_sections == []
    integration = next(s for s in outline if s.kind == "integration_final")
    assert integration.is_text_only


def test_missing_ordinary_section_is_reported_figureless():
    outline = _outline()
    sections = _sections(outline)
    sections = [
        section for section in sections
        if section["id"] != "direction_1"
    ]

    report = enforce_figure_placement(
        sections,
        outline,
        {"explore.png": 1, "b.png": 3},
        ["explore.png", "b.png"],
    )

    assert "direction_1" in report.figureless_sections
    integration = next(s for s in outline if s.kind == "integration_final")
    assert integration.id not in report.figureless_sections


# ------------------------------------ when the outline does not partition rounds

def _legacy_outline():
    """The single-pass outline: organised by analysis type, not by round."""
    return build_report_outline(SimpleNamespace(
        round_results=[_round(1, "exploration"), _round(2, "exploration")],
        publication_figures=[],
    ))


def test_a_round_claimed_by_several_sections_relocates_nothing():
    """Ownership is a partition or it is not ownership.

    The legacy outline gives data_qc, clustering_annotation,
    differential_expression and key_findings the same rounds tuple. Treating the
    first claimant as the owner swept every figure in the report into quality
    control and left three chapters describing panels that were no longer there.
    """
    outline = _legacy_outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    placed = {
        "data_qc": "qc.png",
        "clustering_annotation": "umap.png",
        "differential_expression": "volcano.png",
        "key_findings": "dotplot.png",
    }
    for sid, fig in placed.items():
        by_id[sid]["content"] += f"\n[FIGURE: {fig}]\n"
    attribution = {"qc.png": 1, "umap.png": 1, "volcano.png": 2, "dotplot.png": 2}

    report = enforce_figure_placement(
        sections, outline, attribution, list(placed.values()))

    assert report.relocated == []
    for sid, fig in placed.items():
        assert _figs_in(by_id[sid]) == [fig]


def test_a_marker_written_with_the_other_extension_keeps_its_caption():
    """Figures are saved as JPG; the report model writes .png.

    The HTML renderer resolves that, but placement runs first -- and deleting
    the marker takes the caption the model wrote with it, then re-appends the
    figure uncaptioned at the end of the section, away from the prose about it.
    """
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    by_id["direction_1"]["content"] = (
        "The clusters separate. [FIGURE: umap.png]\n"
        "[FIGURE_CAPTION: UMAP by cell type.] The panel above shows myeloid cells."
    )

    report = enforce_figure_placement(
        sections, outline, {"umap.jpg": 2}, ["umap.jpg"])

    body = by_id["direction_1"]["content"]
    assert report.unknown_removed == []
    assert _figs_in(by_id["direction_1"]) == ["umap.jpg"]
    assert "[FIGURE_CAPTION: UMAP by cell type.]" in body
    # Still where the model put it, not appended after the closing sentence.
    assert body.index("[FIGURE:") < body.index("The panel above")


def test_registered_description_replaces_model_guessed_caption():
    outline = _outline()
    sections = _sections(outline)
    by_id = _by_id(sections)
    by_id["direction_1"]["content"] += (
        "\n[FIGURE: composition.jpg]\n"
        "[FIGURE_CAPTION: UMAP projection of annotated cells]\n"
    )

    enforce_figure_placement(
        sections,
        outline,
        {"composition.jpg": 2},
        ["composition.jpg"],
        figure_captions={
            "composition.jpg": "Bar plots of annotated cell-type proportions."
        },
    )

    content = by_id["direction_1"]["content"]
    assert "Bar plots of annotated cell-type proportions." in content
    assert "UMAP projection" not in content


def test_a_duplicated_section_id_does_not_capture_the_figures():
    """A repeated heading is appended after the outline sections, not merged.

    Resolving the id last-wins aimed every relocated and appended figure at that
    trailing copy, so the report printed them in a stray chapter after the
    conclusions while the real one stood empty.
    """
    outline = _outline()
    owner = next(s.id for s in outline if 2 in (s.rounds or ()))
    sections = _sections(outline)
    real = _by_id(sections)[owner]
    real["content"] = "The real chapter."
    sections.append({"id": owner, "title": "Stray", "content": "Stray copy.",
                     "analysis_type": "general", "figures": [], "tables": []})

    report = enforce_figure_placement(sections, outline, {"qc.png": 2}, ["qc.png"])

    assert report.appended == [("qc.png", owner)]
    assert _figs_in(real) == ["qc.png"]
    assert _figs_in(sections[-1]) == []
