"""Renderer-level layout linting and the report gate that reads its verdict.

The defect class these cover is the one an LLM vision reviewer names but cannot
localise: a legend anchored outside its axes landing on the neighbouring panel,
a panel letter hanging into the tick labels, tick labels crushed together. The
linter measures them from the renderer, so the assertions here are about
geometry, not about what any model said.
"""
import ast
import json

import pytest

from backend.models.schemas import FigureReviewStatus, PublicationFigure
from backend.services.figure_layout import (
    BLOCKING_KINDS,
    LAYOUT_LINT_SENTINEL,
    LAYOUT_LINT_SRC,
    blocking_violations,
    format_violations_for_prompt,
    lint_ran,
    parse_layout_violations,
    summarize_violations,
)
from backend.services.report.artifacts import (
    eligible_artifacts,
    has_blocking_layout_defects,
    is_report_eligible,
)


# ---------------------------------------------------------------------------
# Sentinel parsing (no matplotlib needed)
# ---------------------------------------------------------------------------

def test_sandbox_source_is_valid_python():
    """It is embedded in an f-string wrapper; a syntax error would surface only
    at figure-generation time, inside the kernel, as an opaque failure."""
    ast.parse(LAYOUT_LINT_SRC)


def test_parse_reads_the_sentinel_line():
    payload = [{"kind": "text_over_text", "severity": "block", "text": "Con"}]
    output = (
        "Saved JPG: /w/figures/a.jpg\n"
        + LAYOUT_LINT_SENTINEL + json.dumps(payload) + "\n"
        "Saved PDF: /w/figures/a.pdf\n"
    )
    assert parse_layout_violations(output) == payload
    assert lint_ran(output) is True


def test_absent_sentinel_is_not_reported_as_clean():
    """A linter that did not run and a figure with no defects both yield an
    empty list, so the two are distinguished by lint_ran, not by the list."""
    output = "Saved JPG: /w/figures/a.jpg"
    assert parse_layout_violations(output) == []
    assert lint_ran(output) is False


def test_unparseable_payload_is_survivable():
    assert parse_layout_violations(LAYOUT_LINT_SENTINEL + "not json") == []
    assert parse_layout_violations(None) == []
    assert parse_layout_violations("") == []


def test_blocking_is_selected_by_severity_not_by_kind():
    """The same kind is blocking or advisory depending on what collided, so the
    filter reads the severity the linter assigned. Comparing its output against
    BLOCKING_KINDS instead would be satisfied by any membership at all."""
    violations = [
        {"kind": "text_over_text", "severity": "block"},
        {"kind": "text_over_text", "severity": "advise"},
        {"kind": "text_over_foreign_axes", "severity": "block"},
        {"kind": "tick_label_collision", "severity": "block"},
        {"kind": "tiny_text", "severity": "advise"},
    ]
    selected = blocking_violations(violations)
    assert len(selected) == 3
    assert all(v["severity"] == "block" for v in selected)
    assert set(BLOCKING_KINDS) == {
        "text_over_text", "text_over_foreign_axes", "tick_label_collision"
    }


def test_prompt_text_carries_coordinates():
    """The point of the linter is handing the redraw exact numbers rather than
    the prose a vision reviewer can already produce."""
    text = format_violations_for_prompt([{
        "kind": "text_over_foreign_axes", "severity": "block",
        "text": "Con FXR/bile", "what": "legend entry",
        "overlap_px2": 312.4, "bbox": [531.5, 208.6, 642.5, 218.6],
        "foreign_axes_bbox": [601.5, 44.0, 900.0, 352.0],
    }])
    assert "BLOCKING" in text
    assert "312.4" in text
    assert "531.5" in text


def test_prompt_labels_a_violation_from_its_severity_not_its_kind():
    """The reviewer is told not to approve anything marked BLOCKING "regardless
    of revision count", while blocking_violations() excludes advisory crowding —
    so a label taken from the kind spends the whole revision budget on a defect
    no redraw can clear, and the figure then exits revision_requested and is
    dropped from every report."""
    crowding = {
        "kind": "text_over_text", "severity": "advise",
        "text": "CD8+ Effector Memory T", "what": "annotation",
        "collides_with": "CD4+ Naive T", "collides_with_what": "annotation",
        "overlap_px2": 210.5, "bbox": [1.0, 2.0, 3.0, 4.0],
    }
    advisory_text = format_violations_for_prompt([crowding])
    assert "BLOCKING" not in advisory_text
    assert "advisory" in advisory_text

    # the same kind, blocking, must still read as urgent...
    covered = dict(crowding, severity="block")
    assert "BLOCKING" in format_violations_for_prompt([covered])
    # ...and the two must not come out labelled the same way
    assert format_violations_for_prompt([covered]) != advisory_text

    # a figure-level artist over a panel is measured but advisory
    figure_level = {
        "kind": "text_over_foreign_axes", "severity": "advise",
        "text": "Con", "what": "figure legend entry", "overlap_px2": 300.0,
        "bbox": [1.0, 2.0, 3.0, 4.0], "foreign_axes_bbox": [0.0, 0.0, 9.0, 9.0],
    }
    assert "BLOCKING" not in format_violations_for_prompt([figure_level])


def test_summary_counts_by_kind():
    summary = summarize_violations([
        {"kind": "text_over_text"}, {"kind": "text_over_text"},
        {"kind": "tiny_text"},
    ])
    assert "text_over_text=2" in summary
    assert "tiny_text=1" in summary
    assert summarize_violations([]) == "no layout violations"


# ---------------------------------------------------------------------------
# Report eligibility
# ---------------------------------------------------------------------------

def _figure(name, status=FigureReviewStatus.APPROVED, **kwargs):
    return PublicationFigure(
        name=name, description="", figure_type="other",
        png_path=f"{name}.jpg", pdf_path=f"{name}.pdf",
        review_status=status, **kwargs,
    )


def test_blocking_layout_defect_keeps_a_figure_out_of_reports():
    """accepted_with_reservations is reachable both by a reviewer signing off
    and by the revision loop giving up, and says nothing about legibility."""
    figure = _figure(
        "covered", FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
        layout_violations=[{"kind": "text_over_text", "severity": "block"}],
    )
    assert has_blocking_layout_defects(figure) is True
    assert is_report_eligible(figure) is False
    assert eligible_artifacts([figure]) == []


def test_advisory_layout_defect_does_not_block():
    figure = _figure(
        "small_text", layout_violations=[{"kind": "tiny_text", "severity": "advise"}],
    )
    assert is_report_eligible(figure) is True


def test_figure_without_layout_data_stays_eligible():
    """Sessions restored from before the linter existed carry no violations;
    absent measurement must not be read as a failed measurement."""
    assert is_report_eligible(_figure("legacy")) is True


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture(scope="module")
def lint():
    namespace = {}
    exec(compile(LAYOUT_LINT_SRC, "<layout_lint>", "exec"), namespace, namespace)
    return namespace["_scai_lint_figure_layout"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_legend_anchored_outside_its_axes_lands_on_the_neighbour(lint):
    """bbox_to_anchor moves a legend out of its own axes without reserving any
    space for it, so it is drawn over whatever sits to the right."""
    fig = plt.figure(figsize=(10, 4), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.36)
    ax = fig.add_subplot(gs[0, 0])
    for label in ("Con FXR/bile output", "Con NRF2 adaptation", "DKO FXR/bile output"):
        ax.plot([1, 2, 3], np.random.rand(3), label=label)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ax2 = fig.add_subplot(gs[0, 1])
    genes = ["Slc10a1", "Slco1a1", "Nr0b2", "Cyp7a1", "Abcb11", "Abcc2", "Abcc3"]
    ax2.imshow(np.random.rand(len(genes), 2), aspect="auto")
    ax2.set_yticks(range(len(genes)))
    ax2.set_yticklabels(genes)

    kinds = {v["kind"] for v in lint(fig)}
    assert "text_over_foreign_axes" in kinds
    assert "text_over_text" in kinds


def _two_panels_with_a_figure_legend(layout=None, **legend_kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=100, layout=layout)
    for a in axes:
        a.plot([1, 2, 3], [1, 2, 3], label="Con")
        a.plot([1, 2, 3], [3, 2, 1], label="DKO")
        a.set_xlabel("Zonation bin")
        a.set_ylabel("Module score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **legend_kwargs)
    return fig, axes


def test_a_figure_level_legend_over_a_panel_is_not_a_spill(lint):
    """Handing a legend that has outgrown its axes to the figure is the remedy
    the plotting instructions recommend, and fig.legend's default location is
    inside the figure rectangle — i.e. on top of a panel, since unlike
    'outside ...' it reserves no space. Judged as a spill out of a foreign
    panel it blocks the figure out of every report, and no redraw clears it,
    so the recommended fix would be permanently unpublishable."""
    fig, _ = _two_panels_with_a_figure_legend(layout="constrained")
    hits = lint(fig)
    assert [v for v in hits if v["kind"] == "text_over_foreign_axes"], (
        "the overlap is still measured and reported"
    )
    assert not blocking_violations(hits)

    # ...and the placement that does reserve space comes out clean
    clean, _ = _two_panels_with_a_figure_legend(
        layout="constrained", loc="outside right upper"
    )
    assert lint(clean) == []


def test_a_figure_legend_covering_panel_text_still_blocks(lint):
    """Exempting figure-level artists from the foreign-axes check must not
    blind the gate. What makes text unreadable is other text drawn on top of
    it, and that is judged by the text-over-text check, which does not care
    which artist owns which."""
    fig, axes = _two_panels_with_a_figure_legend(
        loc="center", bbox_to_anchor=(0.29, 0.5), fontsize=11
    )
    for y, name in ((0.55, "Periportal program"), (0.42, "Pericentral program")):
        axes[0].text(0.5, y, name, transform=axes[0].transAxes, ha="center",
                     fontsize=11)

    hits = lint(fig)
    assert blocking_violations(hits)
    assert {v["kind"] for v in blocking_violations(hits)} == {"text_over_text"}


def test_panel_letter_hanging_from_the_top_hits_the_tick_labels(lint):
    """On a short panel the 0.08 axes-fraction offset is smaller than the glyph,
    so va='top' drops the letter back into the first tick rows. The same helper
    with va='bottom' does not, which is what isolates va as the cause."""
    def build(va):
        fig = plt.figure(figsize=(6, 1.2), dpi=100)
        ax = fig.add_subplot(111)
        rows = ["r7_nmf8_p1", "r7_nmf8_p4", "r7_nmf8_p7", "r7_nmf8_p3"]
        ax.imshow(np.random.rand(len(rows), 2), aspect="auto")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.text(-0.12, 1.08, "E", transform=ax.transAxes, fontsize=14,
                fontweight="bold", va=va, ha="right")
        return fig

    assert blocking_violations(lint(build("top")))
    assert not blocking_violations(lint(build("bottom")))


def test_annotation_escaping_the_axes_reaches_the_suptitle(lint):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    fig.suptitle("FXR/bile-acid suppression and NRF2 adaptation in DKO hepatocytes")
    ax = fig.add_subplot(111)
    ax.boxplot([np.random.rand(20), np.random.rand(20) + 0.5])
    ax.text(0.5, 1.12, "p<1e-4", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, clip_on=False)

    assert "text_over_text" in {v["kind"] for v in lint(fig)}


def test_too_many_rows_for_the_panel_height(lint):
    """Judged by overlap along the tick axis, past a fraction of the label's own
    extent: a text box carries ascent/descent leading its glyphs do not fill, so
    stacked labels' boxes touch before anything is actually unreadable."""
    fig = plt.figure(figsize=(8, 2.4), dpi=100)
    ax = fig.add_subplot(111)
    genes = [f"Gene{i:02d}" for i in range(36)]
    ax.imshow(np.random.rand(len(genes), 6), aspect="auto")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8)

    hits = [v for v in lint(fig) if v["kind"] == "tick_label_collision"]
    assert hits
    assert hits[0]["n_labels"] == 36
    assert hits[0]["n_collisions"] > 0
    assert hits[0]["severity"] == "block"


def test_rotated_tick_labels_are_not_a_collision(lint):
    """Rotating a crowded categorical axis is what one DOES about it, and is the
    remedy this very violation recommends. A rotated label's axis-aligned box is
    far wider than its glyph run, so judging the projection would report
    parallel diagonal strips as colliding and the recommended fix would never
    clear the violation."""
    names = ["CD8+ Effector Memory T", "CD4+ Naive T", "CD14+ Monocyte",
             "B cell", "NK cell", "Dendritic cell", "Megakaryocyte", "pDC"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.bar(range(len(names)), np.random.rand(len(names)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")

    assert not [v for v in lint(fig) if v["kind"] == "tick_label_collision"]

    # the same labels horizontal genuinely do collide
    ax.set_xticklabels(names, rotation=0)
    assert [v for v in lint(fig) if v["kind"] == "tick_label_collision"]


def test_axis_off_container_is_not_a_panel(lint):
    """An axes turned off paints nothing, so nothing can be drawn "over" it.
    scanpy builds legend columns that way. The legend here sits squarely on the
    blank holder's rectangle without nesting inside it, so only the axison test
    can clear it."""
    fig = plt.figure(figsize=(9, 4), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    for label in ("Con FXR/bile output", "DKO FXR/bile output", "Con NRF2"):
        ax.plot([1, 2, 3], np.random.rand(3), label=label)
    # anchored out of its own axes and straight onto the blank holder
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    holder = fig.add_subplot(gs[0, 1])
    holder.set_axis_off()

    assert not blocking_violations(lint(fig))

    # ...and the identical layout with a real panel there IS a spill
    holder.set_axis_on()
    holder.imshow(np.random.rand(6, 3), aspect="auto")
    assert blocking_violations(lint(fig))


def test_nested_axes_do_not_spill_into_their_parent(lint):
    """An inset colorbar lives inside its parent's rectangle by design, and its
    tick labels necessarily land on the parent's plotting area.

    Built through axes_grid1's inset_axes, which registers the child on the
    figure. ``Axes.inset_axes`` puts it in ``ax.child_axes`` instead, where
    ``fig.get_axes()`` never sees it — so that spelling cannot exercise this at
    all, and a test written with it passes no matter what the grouping does.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    im = ax.imshow(np.random.rand(20, 20))
    cax = inset_axes(ax, width="4%", height="40%", loc="upper right", borderpad=1)
    fig.colorbar(im, cax=cax)
    ax.set_title("Spatial expression")

    assert cax in fig.get_axes(), "this spelling must register the child axes"
    assert not blocking_violations(lint(fig))


def test_polar_projection_is_not_judged_as_a_tick_run(lint):
    """Theta labels follow a circle, so projecting them onto one axis puts
    non-neighbours next to each other and every polar figure would be blocked."""
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection="polar")
    theta = np.linspace(0, 2 * np.pi, 9)[:-1]
    ax.bar(theta, np.random.rand(8), width=0.6)

    assert not [v for v in lint(fig) if v["kind"] == "tick_label_collision"]


def test_annotations_crowding_inside_one_axes_only_advise(lint):
    """legend_loc='on data' puts cluster names at cluster centroids; they touch
    whenever two clusters are close. That is data density, not a layout fault,
    and no redraw reliably fixes it."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.scatter(np.random.rand(50), np.random.rand(50))
    for label, (x, y) in zip(
        ["CD8+ Effector Memory T", "CD4+ Naive T", "B cell"],
        [(0.5, 0.5), (0.52, 0.5), (0.5, 0.52)],
    ):
        ax.text(x, y, label, ha="center", va="center", fontsize=12)

    hits = [v for v in lint(fig) if v["kind"] == "text_over_text"]
    assert hits, "the collision is still reported"
    assert all(v["severity"] == "advise" for v in hits)
    assert not blocking_violations(hits)


def test_a_blocking_collision_survives_the_per_kind_cap(lint):
    """Each check caps its report at _MAX_PER_KIND with a head slice, and
    text_over_text collects both severities in artist-insertion order. One
    crowded panel produces dozens of advisory hits before any other panel is
    reached, so unsorted the cap fills with them and carries every genuinely
    covered text off the report — and off the gate that reads it, which is how
    an unreadable figure ships badged Approved."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
    axes = axes.ravel()
    # panel 0, collected first: centroid labels on top of each other, advisory
    axes[0].scatter(rng.random(60), rng.random(60), s=5)
    for k in range(15):
        axes[0].text(0.45 + 0.004 * k, 0.45 + 0.004 * k, f"Cluster {k:02d}",
                     ha="center", va="center", fontsize=11)
    # panels 1-3: the legend is drawn over an annotation — genuinely unreadable
    for a in axes[1:]:
        a.plot([1, 2, 3], [1, 2, 3], label="Series alpha")
        a.legend(loc="upper left", fontsize=10)
        a.text(0.16, 0.955, "annotated peak", transform=a.transAxes,
               ha="left", va="top", fontsize=10)

    hits = [v for v in lint(fig) if v["kind"] == "text_over_text"]
    assert len(hits) == 8, "the cap must engage, or this proves nothing"
    assert any(v["severity"] == "advise" for v in hits), (
        "the advisory crowding must be competing for the same cap"
    )
    assert blocking_violations(hits), "the covered text must survive it"
    severities = [v["severity"] for v in hits]
    assert severities == sorted(severities, key=lambda s: s != "block")


def test_measuring_does_not_move_what_savefig_writes(lint, tmp_path):
    """A layout engine re-runs on every draw and is not idempotent, so the
    linter's own draw would otherwise shift the delivered image relative to the
    coordinates it just reported."""
    def build():
        # tight layout plus a colorbar is the shape that re-solves on each draw
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=100, layout="tight")
        for a in axes:
            im = a.imshow(np.random.rand(12, 12))
            a.set_xlabel("Zonation bin")
            a.set_ylabel("Median module score")
            a.set_xticks(range(12))
            a.set_xticklabels([f"bin {i}" for i in range(12)], rotation=45,
                              ha="right")
            fig.colorbar(im, ax=a)
        fig.suptitle("Two panels with colorbars")
        return fig

    def geometry(fig):
        # read the positions savefig left behind; drawing again would re-solve
        return [tuple(round(v, 2) for v in a.get_position().bounds)
                for a in fig.get_axes()]

    np.random.seed(0)
    plain = build()
    plain.savefig(tmp_path / "plain.png", dpi=100, bbox_inches="tight")
    after_plain = geometry(plain)
    plt.close(plain)

    np.random.seed(0)
    measured = build()
    lint(measured)
    reported = geometry(measured)
    measured.savefig(tmp_path / "linted.png", dpi=100, bbox_inches="tight")
    after_save = geometry(measured)
    plt.close(measured)

    # the geometry the linter reported is the geometry that got saved...
    assert reported == after_save
    # ...and it is the geometry an unlinted save would have produced
    assert reported == after_plain


def test_unmeasurable_figure_is_not_reported_as_clean(lint):
    """Returning [] would print the clean sentinel, and an unmeasurable figure
    would read downstream as a measured-clean one."""
    class _Broken:
        def draw(self, *a, **k):
            raise RuntimeError("no renderer here")

        def get_renderer(self):
            raise RuntimeError("no renderer here")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([1, 2], [1, 2])
    object.__setattr__(fig, "canvas", _Broken())
    assert lint(fig) is None


def test_a_well_laid_out_figure_reports_nothing(lint):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100, layout="constrained")
    axes[0].plot([1, 2, 3], [1, 4, 9], label="Con")
    axes[0].plot([1, 2, 3], [2, 3, 5], label="DKO")
    axes[0].legend(loc="upper left")
    axes[0].set_xlabel("Zonation bin")
    axes[0].set_ylabel("Median module score")
    axes[1].bar(["A", "B", "C"], [3, 5, 2])
    axes[1].set_xlabel("Module class")
    axes[1].set_ylabel("Effect")

    assert lint(fig) == []


def test_twin_axes_share_a_rectangle_without_spilling(lint):
    """twinx() axes occupy the same rectangle by construction, so one's tick
    labels are not 'spilling into' the other."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100, layout="constrained")
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_ylabel("left axis")
    ax2 = ax.twinx()
    ax2.plot([1, 2, 3], [9, 4, 1])
    ax2.set_ylabel("right axis")

    assert not blocking_violations(lint(fig))


def test_dense_annotation_matrix_skips_the_sibling_comparison(lint):
    """A value matrix packed tightly enough that its labels touch is still a
    heatmap, not a defect, and comparing its cells pairwise is quadratic. The
    skip must actually engage: this grid is built so the annotations DO overlap,
    so a test that merely spaces them out could not fail for the reason it
    names."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    matrix = np.random.rand(12, 12)
    ax.imshow(matrix, aspect="auto")
    for i in range(12):
        for j in range(12):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    entries = [t for t in ax.texts if t.get_text().strip()]
    assert len(entries) > 60, "the dense-annotation threshold must be crossed"
    assert not blocking_violations(lint(fig))


def test_out_of_view_tick_labels_are_not_collected(lint):
    """get_*ticklabels() also returns ticks beyond the view limits; they sit off
    the canvas but still report get_visible() is True, so unfiltered they become
    phantom neighbours in the tick-run check. The filter tests for disjointness
    from the axes rather than for the box centre, because ha='right' and
    rotation displace a box off its tick by half its width — and a label dropped
    by the filter is invisible to every other check too, which would hide real
    spills."""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_ylim(0, 5)
    assert lint(fig) == []

    # a label displaced off its tick by ha='right' is still in view and must
    # survive the filter, so a genuine spill onto the neighbour is still seen
    fig2 = plt.figure(figsize=(9, 3), dpi=100)
    gs = fig2.add_gridspec(1, 2, wspace=0.02)
    left = fig2.add_subplot(gs[0, 0])
    left.imshow(np.random.rand(4, 4), aspect="auto")
    left.yaxis.tick_right()
    left.set_yticks(range(4))
    left.set_yticklabels(["A very long row label here"] * 4, ha="right")
    right = fig2.add_subplot(gs[0, 1])
    right.imshow(np.random.rand(4, 4), aspect="auto")
    assert blocking_violations(lint(fig2))


# ---------------------------------------------------------------------------
# Wiring into the sandbox wrapper
# ---------------------------------------------------------------------------

def test_publication_wrapper_runs_the_linter_before_saving():
    from backend.agents.bioinformatics.kernel_templates import (
        _build_publication_figure_wrapper,
    )

    source = _build_publication_figure_wrapper(
        "pass", "/w/figures", "/w/figures/a.jpg", "/w/figures/a.pdf"
    )
    ast.parse(source)
    # Match the CALL, not the `def` that LAYOUT_LINT_SRC embeds higher up: a
    # bare `"_scai_emit_layout_lint(fig)" in source` is satisfied by the
    # definition alone, so deleting the call would leave this test green.
    call = "\n                _scai_emit_layout_lint(fig)\n"
    assert call in source
    definition = "\ndef _scai_emit_layout_lint(fig):"
    assert definition in source
    assert source.index(call) > source.index(definition)
    # ...and before the save, or a blocking defect would be measured on a file
    # that has already been written and announced
    assert source.index(call) < source.index("Saved JPG")
    # matplotlib's own layout warnings are captured rather than left to the
    # kernel's global warnings.filterwarnings('ignore')
    assert "catch_warnings" in source
