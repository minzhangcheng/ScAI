"""Tests for the Results brief-gallery figure discovery helper."""
import base64
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.services.report.figure_discovery import (
    backfill_legacy_publication_integrity,
    discover_figures_in_workspace,
    discover_workspace_figures,
    ensure_legacy_artifact_inventory,
    infer_figure_type,
    prepare_discovered_figures,
    publication_figure_metadata,
    resolve_state_figure_candidates,
)
from backend.models.schemas import AnalysisState, FigureReviewStatus, PublicationFigure
from backend.services.report.artifacts import (
    ARTIFACT_REGISTRY_VERSION,
    archival_render_available,
    is_report_eligible,
    resolve_verified_archival_artifact,
)
from backend.tests.figure_image_helpers import valid_pdf_bytes


def _touch(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _image_bytes(noise: float = 100) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise((256, 256), noise).convert("RGB").save(buffer, format="PNG")
    payload = buffer.getvalue()
    assert len(payload) > 2000
    return payload


def _render(path: Path, payload: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload or _image_bytes())
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


def test_restricted_supplementary_never_shadows_same_stem_normal_output(
    monkeypatch, tmp_path,
):
    """A quarantined repair image cannot consume a legitimate result stem.

    The named ``*_outputs/figures`` scan has higher priority than the recursive
    custom-directory tier.  Without the restriction-first comparison below, a
    newer/higher-priority automatic-supplement file won de-duplication and was
    then removed by archival filtering, taking the normal image with it.
    """
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(
        ws / "supplementary_outputs" / "figures" / "shared_result.png",
        mtime=9000.0,
    )
    _touch(ws / "analysis_results" / "shared_result.png", mtime=1000.0)

    figs = discover_workspace_figures("sess-restricted-shadow")
    shared = [f for f in figs if f["name"] == "shared_result"]

    assert len(shared) == 1
    assert shared[0]["source"] == "analysis_results"
    assert shared[0]["restricted_supplementary"] is False


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


def test_nested_agent_output_is_admitted(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / "figures" / "round3" / "de" / "volcano.png", mtime=100.0)

    found = {f["name"]: f for f in discover_workspace_figures("sess-depth3")}
    assert "volcano" in found
    assert found["volcano"]["rel_path"] == "figures/round3/de/volcano.png"
    assert found["volcano"]["source"] == "figures/round3/de"


def test_recursive_discovery_has_no_depth_limit(monkeypatch, tmp_path):
    ws = _make_workspace(monkeypatch, tmp_path)
    deep = ws.joinpath(*[f"level-{index}" for index in range(12)], "deep.png")
    _touch(deep, mtime=100.0)

    found = {f["name"]: f for f in discover_workspace_figures("sess-unbounded")}
    assert found["deep"]["rel_path"] == deep.relative_to(ws).as_posix()


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


def _publication_figure(name, *, status="approved", figure_type="barplot",
                        description="authoritative description", **overrides):
    png_path = overrides.pop("png_path", f"/workspace/figures/{name}.png")
    render_sha256 = overrides.pop("render_sha256", None)
    if render_sha256 is None and Path(png_path).is_file():
        render_sha256 = hashlib.sha256(Path(png_path).read_bytes()).hexdigest()
    return SimpleNamespace(
        name=name,
        png_path=png_path,
        png_base64=overrides.pop("png_base64", None),
        render_sha256=render_sha256,
        preview_sha256=overrides.pop("preview_sha256", None),
        pdf_sha256=overrides.pop("pdf_sha256", "2" * 64),
        csv_path=None,
        figure_type=figure_type,
        description=description,
        interpretation="",
        diagnostic=False,
        round_number=1,
        review_status=status,
        is_current=True,
        invalidated_at=None,
        layout_violations=[],
        **overrides,
    )


def test_registered_metadata_overrides_deceptive_filename_inference(tmp_path):
    path = _render(tmp_path / "cell_state_umap.png")
    raw = [{
        "name": "cell_state_umap", "type": "umap", "path": str(path),
        "pdf_path": None, "mtime": 1.0, "source": "figures",
        "rel_path": "figures/cell_state_umap.png", "diagnostic": False,
    }]
    registered = _publication_figure(
        "cell_state_umap", figure_type="barplot",
        description="Bar plots of annotated cell-type proportions.",
        png_path=str(path),
    )

    prepared = prepare_discovered_figures(raw, [registered])

    assert prepared[0]["type"] == "barplot"
    assert prepared[0]["description"].startswith("Bar plots")
    assert prepared[0]["is_publication"] is True


def test_unrenderable_registered_stem_cannot_reenter_as_workspace_figure(tmp_path):
    candidate = _render(tmp_path / "ghost.png")
    registered = _publication_figure(
        "ghost", png_path=str(tmp_path / "missing.png"),
    )
    raw = [{
        "name": "ghost", "type": "other", "path": str(candidate),
        "pdf_path": None, "mtime": 1.0, "source": "figures",
        "rel_path": "figures/ghost.png", "diagnostic": False,
    }]

    assert prepare_discovered_figures(raw, [registered]) == []


def test_same_stem_overwrite_must_match_registered_render_hash(tmp_path):
    original = _image_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    path = _render(tmp_path / "bound.png", original)
    registered = _publication_figure(
        "bound",
        png_path=str(path),
        png_base64=base64.b64encode(original).decode(),
        render_sha256=original_hash,
        preview_sha256=original_hash,
    )
    # A later generic save uses the reviewed stem for different bytes.
    replacement = _image_bytes(noise=20)
    assert hashlib.sha256(replacement).hexdigest() != original_hash
    path.write_bytes(replacement)
    raw = [{
        "name": "bound", "type": "other", "path": str(path),
        "pdf_path": None, "mtime": 2.0, "source": "figures",
        "rel_path": "figures/bound.png", "diagnostic": False,
    }]

    assert prepare_discovered_figures(raw, [registered]) == []


def test_registered_metadata_retains_plan_provenance(tmp_path):
    path = _render(tmp_path / "planned_panel.png")
    registered = _publication_figure("planned_panel", png_path=str(path))
    registered.plan_step_id = 6
    registered.plan_id = "plan-v2"
    registered.purpose = "result"
    registered.pi_degraded_acceptance = True
    registered.pi_degraded_acceptance_reason = "PI retained the best candidate."

    metadata = publication_figure_metadata([registered])["planned_panel"]

    assert metadata["plan_step_id"] == 6
    assert metadata["plan_id"] == "plan-v2"
    assert metadata["purpose"] == "result"
    assert metadata["pi_degraded_acceptance"] is True
    assert metadata["pi_degraded_acceptance_reason"] == (
        "PI retained the best candidate."
    )
    assert metadata["reservation_notice"]["pi_degraded_acceptance"] is True
    assert "PI retained the best candidate" in " ".join(
        metadata["reservation_notice"]["details"]
    )


def test_annotation_diagnostic_exception_is_context_only(tmp_path):
    path = _render(tmp_path / "annotation_validation" / "marker_check.png")
    raw = [{
        "name": "marker_check", "type": "other", "path": str(path),
        "pdf_path": None, "mtime": 1.0, "source": "annotation_validation",
        "rel_path": "annotation_validation/marker_check.png", "diagnostic": True,
    }]

    assert prepare_discovered_figures(raw, []) == []
    assert [f["name"] for f in prepare_discovered_figures(
        raw, [], allow_unregistered_diagnostics=True,
    )] == ["marker_check"]


def test_unregistered_report_candidate_requires_complete_image_decode(tmp_path):
    corrupt = tmp_path / "looks_like_png.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 3000)
    tiny = tmp_path / "one_pixel.png"
    Image.new("RGB", (1, 1), "white").save(tiny)
    valid = _render(tmp_path / "valid.png")
    raw = [
        {
            "name": "looks_like_png", "type": "other", "path": str(corrupt),
            "pdf_path": None, "mtime": 1.0, "source": "figures",
            "rel_path": "figures/looks_like_png.png", "diagnostic": False,
        },
        {
            "name": "one_pixel", "type": "other", "path": str(tiny),
            "pdf_path": None, "mtime": 1.5, "source": "figures",
            "rel_path": "figures/one_pixel.png", "diagnostic": False,
        },
        {
            "name": "valid", "type": "other", "path": str(valid),
            "pdf_path": None, "mtime": 2.0, "source": "figures",
            "rel_path": "figures/valid.png", "diagnostic": False,
        },
    ]

    assert [item["name"] for item in prepare_discovered_figures(raw, [])] == [
        "valid"
    ]


def test_strict_registry_excludes_unregistered_images_without_mtime_guessing(tmp_path):
    path = _render(tmp_path / "strict.png")
    os.utime(path, (1.0, 1.0))
    raw = [{
        "name": "strict", "type": "other", "path": str(path),
        "pdf_path": None, "mtime": 1.0, "source": "figures",
        "rel_path": "figures/strict.png", "diagnostic": False,
    }]

    assert prepare_discovered_figures(
        raw,
        [],
        registry_version=ARTIFACT_REGISTRY_VERSION,
    ) == []


def test_missing_registry_version_creates_explicit_legacy_inventory(tmp_path):
    workspace = tmp_path / "workspace"
    path = _render(workspace / "figures" / "legacy.png")
    os.utime(path, (1.0, 1.0))
    state = AnalysisState(id="legacy-session", user_id="u")

    inventory = ensure_legacy_artifact_inventory(state, workspace)
    candidates = prepare_discovered_figures(
        discover_figures_in_workspace(workspace),
        state.publication_figures,
        registry_version=state.artifact_registry_version,
        legacy_artifacts=inventory,
    )

    assert state.artifact_registry_version == ARTIFACT_REGISTRY_VERSION
    assert len(inventory) == 1
    assert inventory[0].provenance_kind == "legacy_unregistered"
    assert candidates[0]["is_legacy"] is True
    assert candidates[0]["legacy_artifact_id"] == inventory[0].id
    assert candidates[0]["provenance_kind"] == "legacy_unregistered"


def test_legacy_publication_backfill_trusts_only_record_preview(tmp_path):
    workspace = tmp_path / "workspace"
    image = _render(workspace / "figures" / "reviewed.png")
    pdf = image.with_suffix(".pdf")
    pdf.write_bytes(valid_pdf_bytes())
    preview = base64.b64encode(image.read_bytes()).decode()
    figure = PublicationFigure(
        name="reviewed",
        description="Historical reviewed figure",
        figure_type="other",
        png_path=str(image),
        pdf_path=str(pdf),
        png_base64=preview,
        review_status=FigureReviewStatus.APPROVED,
    )
    state = AnalysisState(
        id="legacy-publication-session",
        user_id="u",
        publication_figures=[figure],
    )

    assert backfill_legacy_publication_integrity(state, workspace) == 1
    ensure_legacy_artifact_inventory(state, workspace)

    assert figure.render_sha256 == hashlib.sha256(image.read_bytes()).hexdigest()
    assert figure.preview_sha256 == hashlib.sha256(image.read_bytes()).hexdigest()
    assert figure.pdf_sha256 is None
    assert not is_report_eligible(figure)
    assert archival_render_available(figure)


def test_legacy_publication_backfill_rejects_replaced_raster_and_pdf(tmp_path):
    workspace = tmp_path / "workspace"
    image = _render(
        workspace / "figures" / "reviewed.png",
        _image_bytes(noise=25),
    )
    reviewed_preview = image.read_bytes()
    pdf = image.with_suffix(".pdf")
    pdf.write_bytes(valid_pdf_bytes())
    figure = PublicationFigure(
        name="reviewed",
        description="Historical reviewed figure",
        figure_type="other",
        png_path=str(image),
        pdf_path=str(pdf),
        png_base64=base64.b64encode(reviewed_preview).decode(),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = AnalysisState(
        id="legacy-replaced-publication",
        user_id="u",
        publication_figures=[figure],
    )

    image.write_bytes(_image_bytes(noise=180))
    pdf.write_bytes(valid_pdf_bytes() + b"replacement")

    assert backfill_legacy_publication_integrity(state, workspace) == 1
    assert figure.preview_sha256 == hashlib.sha256(reviewed_preview).hexdigest()
    assert figure.render_sha256 is None
    assert figure.pdf_sha256 is None
    assert not is_report_eligible(figure)
    inventory = ensure_legacy_artifact_inventory(state, workspace)
    assert all(item.name != "reviewed" for item in inventory)

    resolved = resolve_verified_archival_artifact(
        figure,
        workspace,
        kind="image",
    )
    assert resolved is not None
    assert resolved[0] == reviewed_preview
    assert resolved[0] != image.read_bytes()


def test_legacy_publication_backfill_does_not_trust_paths_without_preview(tmp_path):
    workspace = tmp_path / "workspace"
    image = _render(workspace / "figures" / "path-only.png")
    pdf = image.with_suffix(".pdf")
    pdf.write_bytes(valid_pdf_bytes())
    figure = PublicationFigure(
        name="path-only",
        description="No frozen record bytes",
        figure_type="other",
        png_path=str(image),
        pdf_path=str(pdf),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = AnalysisState(
        id="legacy-path-only-publication",
        user_id="u",
        publication_figures=[figure],
    )

    assert backfill_legacy_publication_integrity(state, workspace) == 0
    assert figure.render_sha256 is None
    assert figure.preview_sha256 is None
    assert figure.pdf_sha256 is None
    assert not is_report_eligible(figure)


def test_strict_registry_never_adopts_missing_publication_hashes(tmp_path):
    workspace = tmp_path / "workspace"
    image = _render(workspace / "figures" / "unbound.png")
    figure = PublicationFigure(
        name="unbound",
        description="Strict record without provenance",
        figure_type="other",
        png_path=str(image),
        pdf_path=str(image.with_suffix(".pdf")),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = AnalysisState(
        id="strict-publication-session",
        user_id="u",
        publication_figures=[figure],
        artifact_registry_version=ARTIFACT_REGISTRY_VERSION,
    )

    assert backfill_legacy_publication_integrity(state, workspace) == 0
    assert figure.render_sha256 is None
    assert not is_report_eligible(figure)


def test_legacy_inventory_is_frozen_after_first_migration_scan(tmp_path):
    workspace = tmp_path / "workspace"
    _render(workspace / "figures" / "present_at_migration.png")
    state = AnalysisState(id="legacy-session", user_id="u")

    first = ensure_legacy_artifact_inventory(state, workspace)
    _render(workspace / "figures" / "appeared_later.png")
    second = ensure_legacy_artifact_inventory(state, workspace)

    assert state.artifact_registry_version == ARTIFACT_REGISTRY_VERSION
    assert [item.name for item in first] == ["present_at_migration"]
    assert [item.name for item in second] == ["present_at_migration"]


@pytest.mark.parametrize("pdf_existed_at_freeze", [False, True])
def test_frozen_legacy_image_cannot_adopt_an_added_or_replaced_pdf(
    tmp_path, pdf_existed_at_freeze,
):
    workspace = tmp_path / "workspace"
    image = _render(workspace / "figures" / "legacy.png")
    pdf = image.with_suffix(".pdf")
    if pdf_existed_at_freeze:
        pdf.write_bytes(valid_pdf_bytes(color=(10, 20, 30)))
    state = AnalysisState(id="legacy-session", user_id="u")
    ensure_legacy_artifact_inventory(state, workspace)

    pdf.write_bytes(valid_pdf_bytes(color=(200, 30, 20)))
    candidates = resolve_state_figure_candidates(
        state, workspace, report_only=False
    )

    assert len(candidates) == 1
    assert candidates[0].get("pdf_path") in (None, "")
    assert candidates[0].get("pdf_sha256") is None


def test_frozen_legacy_pair_keeps_its_exact_pdf_path_and_hash(tmp_path):
    workspace = tmp_path / "workspace"
    image = _render(workspace / "figures" / "legacy.png")
    pdf = image.with_suffix(".pdf")
    pdf_payload = valid_pdf_bytes(color=(10, 20, 30))
    pdf.write_bytes(pdf_payload)
    state = AnalysisState(id="legacy-session", user_id="u")
    inventory = ensure_legacy_artifact_inventory(state, workspace)

    candidates = resolve_state_figure_candidates(
        state, workspace, report_only=False
    )

    assert len(inventory) == 1
    assert inventory[0].pdf_relative_path == "figures/legacy.pdf"
    assert inventory[0].pdf_sha256 == hashlib.sha256(pdf_payload).hexdigest()
    assert candidates[0]["pdf_path"] == str(pdf)
    assert candidates[0]["pdf_sha256"] == inventory[0].pdf_sha256


def test_frozen_legacy_identity_keeps_exact_path_when_same_bytes_are_copied(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    frozen_path = _render(
        workspace / "round1_outputs" / "figures" / "shared.png"
    )
    state = AnalysisState(id="legacy-session", user_id="u")
    ensure_legacy_artifact_inventory(state, workspace)

    # A later higher-priority disk entry must not borrow the frozen record merely
    # because it has the same stem and bytes.
    copied_path = _render(
        workspace / "figures" / "shared.png", frozen_path.read_bytes()
    )
    candidates = resolve_state_figure_candidates(
        state, workspace, report_only=False
    )

    assert copied_path != frozen_path
    assert len(candidates) == 1
    assert Path(candidates[0]["path"]) == frozen_path
    assert candidates[0]["legacy_artifact_id"] == state.legacy_artifacts[0].id


def test_ineligible_registered_validation_figure_cannot_reenter_from_disk():
    raw = [{
        "name": "marker_check", "type": "other", "path": "/w/annotation_validation/marker_check.png",
        "pdf_path": None, "mtime": 1.0, "source": "annotation_validation",
        "rel_path": "annotation_validation/marker_check.png", "diagnostic": True,
    }]
    rejected = _publication_figure("marker_check", status="revision_requested")

    assert prepare_discovered_figures(
        raw, [rejected], allow_unregistered_diagnostics=True,
    ) == []


@pytest.mark.parametrize(
    "rel",
    [
        ".hidden/x.png",
        "__pycache__/x.png",
        "reports/figures/x.png",
    ],
)
def test_recursive_tier_skips_non_output_paths(monkeypatch, tmp_path, rel):
    ws = _make_workspace(monkeypatch, tmp_path)
    _touch(ws / rel, mtime=100.0)
    assert discover_workspace_figures("sess-skip") == []
