"""A stale file stem must not withhold an artifact whose own record is current.

Regression cover for the defect that halved report.html. ``invalidated_artifact_stems``
and ``invalidate_current_artifacts`` are two staleness mechanisms over the same event;
only the second is per-record and round-scoped. Before 0.38.2 the first had no round
bound at all, so one representation-mutating cell late in a session blacklisted every
earlier artifact, and 0.37.9 had just extended that blacklist to the publication-figure
tier of the report collector. Replaying one real 8-round session gave 10 figures before
and 5 after.
"""
import hashlib
from pathlib import Path

from backend.models.schemas import FigureReviewStatus, PublicationFigure
from backend.services.report.artifacts import (
    eligible_artifacts,
    excluded_artifact_stems,
    gallery_artifacts,
    is_diagnostic,
    is_report_eligible,
)
from backend.services.report.figure_placement import enforce_figure_placement
from backend.services.report.figures import FiguresMixin
from backend.services.report.report_outline import OutlineSection


def _figure(name, status=FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS, **kwargs):
    return PublicationFigure(
        name=name,
        description="",
        figure_type=kwargs.pop("figure_type", "other"),
        png_path=kwargs.pop("png_path", f"{name}.jpg"),
        pdf_path=kwargs.pop("pdf_path", f"{name}.pdf"),
        review_status=status,
        **kwargs,
    )


# --- excluded_artifact_stems -------------------------------------------------


def test_stale_stem_does_not_withhold_a_figure_whose_record_is_current():
    kept = _figure("atlas_overview")
    stems = excluded_artifact_stems(
        [kept], eligible_artifacts([kept]),
        invalidated_stems=["atlas_overview"],
    )
    assert "atlas_overview" not in stems


def test_stale_stem_still_withholds_a_superseded_figure():
    superseded = _figure("panel", is_current=False)
    stems = excluded_artifact_stems(
        [superseded], eligible_artifacts([superseded]),
        invalidated_stems=["panel"],
    )
    assert "panel" in stems


def test_stale_stem_still_withholds_an_unregistered_workspace_file():
    """Nothing here can tell a genuinely stale file from one named by mistake."""
    registered = _figure("registered")
    stems = excluded_artifact_stems(
        [registered], eligible_artifacts([registered]),
        invalidated_stems=["registered", "loose_savefig_output"],
    )
    assert stems == {"loose_savefig_output"}


def test_review_status_still_withholds_independently_of_the_stem_set():
    rejected = _figure("rejected", FigureReviewStatus.REVISION_REQUESTED)
    stems = excluded_artifact_stems(
        [rejected], eligible_artifacts([rejected]), invalidated_stems=["rejected"],
    )
    assert "rejected" in stems


def test_gallery_keeps_a_revision_requested_figure_the_stem_set_named():
    """The gallery's weaker predicate must not be re-tightened by the stem set."""
    revised = _figure("revised", FigureReviewStatus.REVISION_REQUESTED)
    stems = excluded_artifact_stems(
        [revised], gallery_artifacts([revised]), invalidated_stems=["revised"],
    )
    assert stems == set()


def test_no_invalidated_stems_is_the_plain_eligibility_difference():
    good, bad = _figure("good"), _figure("bad", FigureReviewStatus.REVISION_REQUESTED)
    assert excluded_artifact_stems([good, bad], eligible_artifacts([good, bad])) == {"bad"}


def test_empty_inputs_are_tolerated():
    assert excluded_artifact_stems(None, None) == set()
    assert excluded_artifact_stems([], [], invalidated_stems=None) == set()


def test_the_real_session_shape_keeps_all_five_reclaimed_figures():
    """The exact configuration observed in session c31c0b42.

    Five figures were current, layout-clean and accepted, yet their stems sat in a
    20-entry blacklist grown by a numpy-only ``np.log1p`` call misread as an
    expression-representation change three rounds earlier.
    """
    reclaimed = [
        "atlas_lineage_condition_library",
        "composition_library_effects",
        "bulk_nnls_deconvolution",
        "integrated_mechanism_evidence_snrna",
        "state_resolved_remodeling_heatmap",
    ]
    superseded = _figure("mesenchymal_vascular_states", is_current=False)
    figures = [_figure(name) for name in reclaimed] + [superseded]
    stems = excluded_artifact_stems(
        figures, eligible_artifacts(figures),
        invalidated_stems=reclaimed + ["mesenchymal_vascular_states", "mouse_liver_umap_descriptive"],
    )
    assert not stems & set(reclaimed)
    assert stems == {"mesenchymal_vascular_states", "mouse_liver_umap_descriptive"}


def test_workspace_scan_reclaims_the_stale_stem_too(tmp_path):
    """End-to-end through the fallback generator's scanner, not just the predicate."""
    workspace = tmp_path / "workspace"
    figures_dir = workspace / "figures"
    report_figures = tmp_path / "report" / "figures"
    figures_dir.mkdir(parents=True)
    report_figures.mkdir(parents=True)
    (figures_dir / "kept.jpg").write_bytes(b"kept")
    (figures_dir / "superseded.jpg").write_bytes(b"superseded")

    scanner = FiguresMixin()
    scanner.workspace_dir = workspace
    scanner.figures_dir = report_figures
    found = scanner._scan_existing_figures(
        publication_figures=[
            _figure("kept", png_path=str(figures_dir / "kept.jpg")),
            _figure("superseded", png_path=str(figures_dir / "superseded.jpg"),
                    is_current=False),
        ],
        invalidated_stems={"kept", "superseded"},
    )
    assert {item["filename"] for item in found} == {"kept.jpg"}


# --- diagnostic figures ------------------------------------------------------


def test_a_platform_diagnostic_is_reportable_without_a_review():
    panel = _figure("exploration_qc_diagnostics", FigureReviewStatus.PENDING,
                    figure_type="qc", diagnostic=True)
    assert is_diagnostic(panel)
    assert is_report_eligible(panel)


def test_an_undiagnosed_pending_figure_is_still_withheld():
    assert not is_report_eligible(_figure("ordinary", FigureReviewStatus.PENDING))


def test_a_diagnostic_is_not_exempt_from_the_layout_gate():
    """Legibility is measured by the renderer, so it is the one question an
    unreviewed figure can still be asked."""
    covered = _figure(
        "exploration_qc_diagnostics", FigureReviewStatus.PENDING, diagnostic=True,
        layout_violations=[{"kind": "text_over_text", "severity": "block"}],
    )
    assert not is_report_eligible(covered)


def test_an_advisory_layout_violation_does_not_withhold_a_diagnostic():
    """Advisory violations describe density, not unreadability."""
    crowded = _figure(
        "exploration_qc_diagnostics", FigureReviewStatus.PENDING, diagnostic=True,
        layout_violations=[{"kind": "text_over_text", "severity": "advise"}],
    )
    assert is_report_eligible(crowded)


def test_a_superseded_diagnostic_is_withheld():
    panel = _figure("exploration_qc_diagnostics", FigureReviewStatus.PENDING,
                    diagnostic=True, is_current=False)
    assert not is_report_eligible(panel)


def test_a_same_round_representation_change_still_withdraws_a_diagnostic():
    """Why the panel is re-rendered after Stage 2, not only after Stage 1.

    ``diagnostic`` waives the review-status check and nothing else. A Stage-2 cell
    that mutates the expression representation invalidates every round-1 artifact,
    the panel included, and the escape sits downstream of ``is_current`` -- so a
    panel rendered only at the end of Stage 1 would vanish from every surface.
    """
    from backend.services.report.artifacts import invalidate_current_artifacts

    panel = _figure("exploration_qc_diagnostics", FigureReviewStatus.PENDING,
                    diagnostic=True, round_number=1)
    assert is_report_eligible(panel)
    invalidate_current_artifacts([panel], round_number=1, reason="representation changed")
    assert not is_report_eligible(panel)
    assert not panel.is_current


def test_diagnostic_defaults_false_for_records_written_before_the_field():
    assert not is_diagnostic(_figure("legacy"))
    assert not is_diagnostic(object())


# --- figureless figure-owning sections ---------------------------------------


def _outline():
    return [
        OutlineSection("abstract", "abstract", "Abstract"),
        OutlineSection("methods", "methods", "Methods"),
        OutlineSection("exploration_findings", "exploration", "Phase 1", (1,)),
        OutlineSection("direction_1", "direction", "Round 2", (2,)),
    ]


def test_a_figure_owning_section_with_no_figure_is_recorded():
    sections = [
        {"id": "abstract", "content": "summary"},
        {"id": "methods", "content": "procedure"},
        {"id": "exploration_findings", "content": "prose over numbers, no panel"},
        {"id": "direction_1", "content": "result\n[FIGURE: panel.jpg]\n"},
    ]
    report = enforce_figure_placement(
        sections, _outline(), {"panel.jpg": 2}, ["panel.jpg"],
    )
    assert report.figureless_sections == ["exploration_findings"]
    assert "figureless=1" in report.summary()


def test_methods_owns_no_rounds_and_is_never_reported_as_figureless():
    sections = [
        {"id": "methods", "content": "procedure only"},
        {"id": "exploration_findings", "content": "[FIGURE: panel.jpg]"},
    ]
    report = enforce_figure_placement(
        sections, _outline(), {"panel.jpg": 1}, ["panel.jpg"],
    )
    assert report.figureless_sections == []


def test_an_appended_figure_clears_the_figureless_flag():
    """The unreferenced-figure pass runs first, so a section the model forgot to
    illustrate is not reported as figureless once placement repairs it."""
    sections = [
        {"id": "exploration_findings", "content": "no marker at all"},
        {"id": "direction_1", "content": "[FIGURE: b.jpg]"},
    ]
    report = enforce_figure_placement(
        sections, _outline(), {"a.jpg": 1, "b.jpg": 2}, ["a.jpg", "b.jpg"],
    )
    assert ("a.jpg", "exploration_findings") in report.appended
    assert report.figureless_sections == []


# --- content-hash deduplication ----------------------------------------------


def _collect(tmp_path, files, publication):
    """Run the live collector over a synthetic workspace."""
    from backend.agents.supervisor import report_html

    workspace = tmp_path / "workspace"
    figures_dir = workspace / "figures"
    figures_dir.mkdir(parents=True)
    for rel, payload in files.items():
        (workspace / rel).write_bytes(payload)

    mixin_cls = next(
        cls for cls in vars(report_html).values()
        if isinstance(cls, type) and hasattr(cls, "_collect_figure_paths")
    )

    class _State:
        publication_figures = publication
        invalidated_artifact_stems: list = []

    return mixin_cls._collect_figure_paths(
        mixin_cls.__new__(mixin_cls), [], workspace, state=_State(),
    )


def test_one_file_reached_under_two_names_is_embedded_once(tmp_path):
    same = b"\xff\xd8identical-file-bytes"
    figure_map = _collect(
        tmp_path,
        {
            "figures/registered_panel.jpg": same,
            "loose_savefig_copy.jpg": same,
            "unrelated.jpg": b"\xff\xd8something-else",
        },
        [_figure("registered_panel", png_path="registered_panel.jpg")],
    )
    digests = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in figure_map.values()]
    assert len(digests) == len(set(digests)), "one file was embedded twice"
    # The registered copy survives: it carries the round number and review status.
    assert "registered_panel.jpg" in figure_map
    assert "loose_savefig_copy.jpg" not in figure_map
    assert "unrelated.jpg" in figure_map


def test_the_registered_copy_wins_whichever_order_it_is_reached_in(tmp_path):
    same = b"\xff\xd8identical-file-bytes"
    figure_map = _collect(
        tmp_path,
        {"aaa_loose.jpg": same, "figures/zzz_registered.jpg": same},
        [_figure("zzz_registered", png_path="zzz_registered.jpg")],
    )
    assert list(figure_map) == ["zzz_registered.jpg"]


def test_one_drawing_re_encoded_twice_is_a_known_limit(tmp_path):
    """The neighbouring case that looks identical and is deliberately not caught.

    Session c31c0b42 saved a panel as PNG and re-rendered it as JPG; the report
    embedded both. Their bytes differ, so an exact hash cannot see it, and deciding
    two *pictures* are the same needs a perceptual comparison with a false-positive
    profile this check does not have. Pinned so the limit is stated rather than
    discovered.
    """
    figure_map = _collect(
        tmp_path,
        {
            "figures/same_drawing.jpg": b"\xff\xd8jpeg-encoding-of-one-picture",
            "same_drawing_source.png": b"\x89PNGpng-encoding-of-one-picture",
        },
        [_figure("same_drawing", png_path="same_drawing.jpg")],
    )
    assert set(figure_map) == {"same_drawing.jpg", "same_drawing_source.png"}


def test_an_unreadable_figure_does_not_abort_collection(tmp_path):
    figure_map = _collect(
        tmp_path, {"figures/good.jpg": b"\xff\xd8fine"},
        [_figure("good", png_path="good.jpg")],
    )
    assert "good.jpg" in figure_map
