import base64
import hashlib
from datetime import datetime
from pathlib import Path

from backend.models.schemas import FigureReviewStatus, PublicationFigure
from backend.services.report.artifacts import (
    archival_render_available,
    eligible_artifacts,
    gallery_artifacts,
    has_frozen_publication_pair,
    is_gallery_eligible,
    is_report_eligible,
    materialize_archival_preview,
    resolve_archival_render,
    resolve_verified_archival_artifact,
)
from backend.services.report.figures import FiguresMixin
from backend.tests.figure_image_helpers import valid_image_bytes, write_valid_image


def _figure(name, status, **kwargs):
    png_path = kwargs.pop("png_path", f"{name}.jpg")
    pdf_path = kwargs.pop("pdf_path", f"{name}.pdf")
    png_base64 = kwargs.get("png_base64")
    candidate = Path(png_path)
    if "render_sha256" not in kwargs:
        kwargs["render_sha256"] = (
            hashlib.sha256(candidate.read_bytes()).hexdigest()
            if candidate.is_file() else "1" * 64
        )
    if "pdf_sha256" not in kwargs:
        kwargs["pdf_sha256"] = "2" * 64
    if "preview_sha256" not in kwargs and png_base64:
        kwargs["preview_sha256"] = hashlib.sha256(
            base64.b64decode(png_base64)
        ).hexdigest()
    return PublicationFigure(
        name=name, description="", figure_type="other",
        png_path=png_path, pdf_path=pdf_path,
        review_status=status, **kwargs,
    )


def test_artifact_facade_keeps_shared_and_selection_imports_nominal():
    from backend.services.report import artifacts

    for name in (
        "verified_artifact_bytes",
        "resolve_archival_render",
        "render_is_intact",
        "compact_figure_event_payload",
        "published_head_artifacts",
        "invalidate_current_artifacts",
    ):
        exported = getattr(artifacts, name)
        assert exported.__module__ == "backend.services.report.artifacts"


def test_only_current_approved_artifacts_are_reportable():
    approved = _figure("approved", FigureReviewStatus.APPROVED)
    reserved = _figure("reserved", FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS)
    pending = _figure("pending", FigureReviewStatus.PENDING)
    revision = _figure("revision", FigureReviewStatus.REVISION_REQUESTED)
    stale = _figure("stale", FigureReviewStatus.APPROVED, is_current=False)

    assert is_report_eligible(approved)
    assert is_report_eligible(reserved)
    assert [f.name for f in eligible_artifacts([approved, reserved, pending, revision, stale])] == [
        "approved", "reserved"
    ]


def test_technically_valid_unavailable_review_is_reportable_but_not_approved():
    unavailable = _figure(
        "unavailable",
        FigureReviewStatus.PENDING,
        review_unavailable=True,
    )
    blocked = _figure(
        "blocked-unavailable",
        FigureReviewStatus.PENDING,
        review_unavailable=True,
        layout_violations=[{"kind": "text_over_text", "severity": "block"}],
    )

    assert is_report_eligible(unavailable) is True
    assert is_report_eligible(blocked) is False


def test_only_explicit_pi_degraded_acceptance_waives_unresolved_observations():
    observations = {
        "layout_violations": [{"kind": "text_over_text", "severity": "block"}],
        "unresolved_must_fix": ["Increase the legend spacing"],
    }
    ordinary = _figure(
        "ordinary-reserved",
        FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
        **observations,
    )
    degraded = _figure(
        "pi-degraded",
        FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
        pi_degraded_acceptance=True,
        pi_degraded_acceptance_reason="PI selected the best exhausted candidate.",
        **observations,
    )

    assert is_report_eligible(ordinary) is False
    assert is_report_eligible(degraded) is True


def test_valid_preview_digest_survives_malformed_render_digest():
    payload = valid_image_bytes("JPEG")
    figure = _figure(
        "preview-fallback",
        FigureReviewStatus.APPROVED,
        png_path="/missing/preview-fallback.jpg",
        png_base64=base64.b64encode(payload).decode(),
        render_sha256="malformed-but-nonempty",
        preview_sha256=hashlib.sha256(payload).hexdigest(),
        pdf_sha256="2" * 64,
    )

    assert has_frozen_publication_pair(figure) is True
    assert archival_render_available(figure) is True
    assert is_report_eligible(figure) is True


def test_gallery_keeps_report_blocked_renders_as_badged_output():
    pending = _figure("pending", FigureReviewStatus.PENDING)
    rejected = _figure("rejected", FigureReviewStatus.REVISION_REQUESTED)
    layout_blocked = _figure(
        "layout-blocked",
        FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
        layout_violations=[{"kind": "text_over_text", "severity": "block"}],
    )
    failed = _figure(
        "failed", FigureReviewStatus.PENDING, producer_status="failed",
    )
    invalidated = _figure(
        "invalidated", FigureReviewStatus.APPROVED, is_current=False,
    )

    assert is_gallery_eligible(pending)
    assert [item.name for item in gallery_artifacts([
        pending, rejected, layout_blocked, failed, invalidated,
    ])] == ["pending", "rejected", "layout-blocked"]
    assert eligible_artifacts([
        pending, rejected, layout_blocked, failed, invalidated,
    ]) == []


def test_hashless_legacy_publication_review_is_not_report_eligible():
    legacy = PublicationFigure(
        name="legacy",
        description="Review predates integrity binding",
        figure_type="other",
        png_path="legacy.jpg",
        pdf_path="legacy.pdf",
        review_status=FigureReviewStatus.APPROVED,
    )

    assert is_report_eligible(legacy) is False
    assert eligible_artifacts([legacy]) == []


def test_workspace_scan_cannot_reintroduce_ineligible_publication_figure(tmp_path):
    workspace = tmp_path / "workspace"
    figures = workspace / "figures"
    report_figures = tmp_path / "report" / "figures"
    figures.mkdir(parents=True)
    report_figures.mkdir(parents=True)
    write_valid_image(figures / "approved.jpg", min_bytes=2200)
    write_valid_image(figures / "revision.jpg", color=(90, 30, 20), min_bytes=2200)

    scanner = FiguresMixin()
    scanner.workspace_dir = workspace
    scanner.figures_dir = report_figures
    found = scanner._scan_existing_figures(publication_figures=[
        _figure("approved", FigureReviewStatus.APPROVED,
                png_path=str(figures / "approved.jpg")),
        _figure("revision", FigureReviewStatus.REVISION_REQUESTED,
                png_path=str(figures / "revision.jpg")),
    ])
    assert {item["filename"] for item in found} == {"approved.jpg"}


def test_recursive_scan_admits_only_registered_eligible_annotation_diagnostic(tmp_path):
    workspace = tmp_path / "workspace"
    validation = workspace / "annotation_validation"
    report_figures = tmp_path / "report" / "figures"
    validation.mkdir(parents=True)
    report_figures.mkdir(parents=True)
    eligible_path = validation / "eligible.jpg"
    loose_path = validation / "loose.jpg"
    write_valid_image(eligible_path, min_bytes=2200)
    write_valid_image(loose_path, color=(90, 30, 20), min_bytes=2200)

    scanner = FiguresMixin()
    scanner.workspace_dir = workspace
    scanner.figures_dir = report_figures
    figure = _figure(
        "eligible", FigureReviewStatus.APPROVED,
        png_path=str(eligible_path),
    )
    figure.figure_type = "barplot"
    figure.description = "Authoritative cell-type proportions."
    figure.plan_step_id = 4
    figure.plan_id = "plan-v2"
    figure.purpose = "result"

    found = scanner._scan_existing_figures(publication_figures=[figure])

    assert {item["filename"] for item in found} == {"eligible.jpg"}
    assert found[0]["inferred_type"] == "barplot"
    assert found[0]["description"] == "Authoritative cell-type proportions."
    assert found[0]["plan_step_id"] == 4
    assert found[0]["plan_id"] == "plan-v2"
    assert found[0]["purpose"] == "result"


# ------------------------------- the diagnostic waiver rests on a measurement

def _unverified():
    """The marker the renderer stores when its in-kernel layout lint crashed."""
    return {"kind": "lint_skipped", "severity": "advise",
            "detail": "overlap checking was skipped"}


def test_a_diagnostic_is_waived_past_review_only_once_its_layout_was_measured():
    """The waiver's whole justification is that the lint still ran.

    A platform-rendered diagnostic has no reviewer, so the layout gate is the
    only thing that looks at it. Recording an unmeasured layout as an empty
    violation list made "measured, nothing wrong" and "never measured" the same
    record, and the figure most likely to crash the lint is the one most likely
    to need it.
    """
    measured = _figure("qc_measured", FigureReviewStatus.PENDING, diagnostic=True)
    unmeasured = _figure("qc_unmeasured", FigureReviewStatus.PENDING,
                         diagnostic=True, layout_violations=[_unverified()])

    assert is_report_eligible(measured) is True
    assert is_report_eligible(unmeasured) is False


def test_an_unmeasured_layout_does_not_withhold_a_reviewed_figure():
    """A human looked at this one, so the missing measurement is not the only check."""
    reviewed = _figure("umap", FigureReviewStatus.APPROVED,
                       layout_violations=[_unverified()])
    assert is_report_eligible(reviewed) is True


def test_archival_render_requires_file_or_valid_record_preview(tmp_path):
    missing = _figure(
        "missing", FigureReviewStatus.APPROVED,
        png_path=str(tmp_path / "old" / "missing.png"),
    )
    assert archival_render_available(missing) is False

    payload = valid_image_bytes("PNG")
    assert len(payload) < 2000, "a compact valid preview must not inherit the JPG floor"
    missing.png_base64 = base64.b64encode(payload).decode()
    missing.preview_sha256 = hashlib.sha256(payload).hexdigest()
    assert archival_render_available(missing) is True


def test_record_preview_is_materialized_only_inside_workspace(tmp_path):
    payload = valid_image_bytes("JPEG")
    figure = _figure(
        "restored", FigureReviewStatus.APPROVED,
        png_path="/untrusted/old/location/restored.png",
        png_base64=base64.b64encode(payload).decode(),
    )

    recovered = materialize_archival_preview(figure, tmp_path)

    assert recovered == tmp_path / "figures" / "restored.jpg"
    assert recovered.read_bytes() == payload


def test_verified_preview_recovery_reuses_record_bound_bytes(monkeypatch, tmp_path):
    from backend.services.report import artifacts

    payload = valid_image_bytes("JPEG")
    figure = _figure(
        "single-read",
        FigureReviewStatus.APPROVED,
        png_path=str(tmp_path / "missing" / "single-read.jpg"),
        png_base64=base64.b64encode(payload).decode(),
        render_sha256=None,
    )
    verified_paths = []
    original_verify = artifacts.verified_artifact_bytes

    def counted_verify(path, *args, **kwargs):
        verified_paths.append(Path(path))
        return original_verify(path, *args, **kwargs)

    monkeypatch.setattr(artifacts, "verified_artifact_bytes", counted_verify)

    resolved = resolve_verified_archival_artifact(
        figure,
        tmp_path,
        kind="image",
    )

    assert resolved is not None
    returned, recovered = resolved
    assert returned == payload
    assert recovered.read_bytes() == payload
    # The destination was probed once before creation and never re-read after
    # the already-verified preview bytes were written.
    assert verified_paths.count(recovered) == 1


def test_materialized_preview_restores_record_creation_mtime(tmp_path):
    payload = valid_image_bytes("PNG")
    created_at = datetime(2025, 1, 2, 3, 4, 5)
    figure = _figure(
        "round_owned", FigureReviewStatus.APPROVED,
        png_path="/old/workspace/round_owned.png",
        png_base64=base64.b64encode(payload).decode(),
        created_at=created_at,
    )

    recovered = materialize_archival_preview(figure, tmp_path)

    assert recovered is not None
    assert abs(recovered.stat().st_mtime - created_at.timestamp()) < 1.0


def test_magic_bytes_without_a_decodable_image_are_rejected(tmp_path):
    corrupt = b"\x89PNG\r\n\x1a\n" + b"x" * 2200
    figure = _figure(
        "corrupt", FigureReviewStatus.APPROVED,
        png_path=str(tmp_path / "missing.png"),
        png_base64=base64.b64encode(corrupt).decode(),
    )

    assert archival_render_available(figure) is False
    assert materialize_archival_preview(figure, tmp_path) is None


def test_decodable_one_pixel_placeholder_is_not_an_archival_preview(tmp_path):
    from io import BytesIO
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(buffer, "PNG")
    figure = _figure(
        "placeholder", FigureReviewStatus.APPROVED,
        png_path=str(tmp_path / "missing.png"),
        png_base64=base64.b64encode(buffer.getvalue()).decode(),
    )

    assert archival_render_available(figure) is False
    assert materialize_archival_preview(figure, tmp_path) is None


def test_record_hash_rejects_a_later_same_name_overwrite(tmp_path):
    workspace = tmp_path / "workspace"
    figures = workspace / "figures"
    figures.mkdir(parents=True)
    path = figures / "reviewed.jpg"
    original = write_valid_image(path, min_bytes=2200)
    preview = valid_image_bytes("JPEG")
    figure = _figure(
        "reviewed", FigureReviewStatus.APPROVED,
        png_path=str(path),
        png_base64=base64.b64encode(preview).decode(),
        render_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=hashlib.sha256(preview).hexdigest(),
    )

    write_valid_image(path, color=(220, 10, 10), min_bytes=2200)
    resolved = resolve_archival_render(figure, workspace)

    assert resolved is not None
    assert resolved.read_bytes() == preview
    assert resolved != path


def test_hash_mismatch_without_record_preview_is_not_available(tmp_path):
    path = tmp_path / "reviewed.jpg"
    original = write_valid_image(path, min_bytes=2200)
    figure = _figure(
        "reviewed", FigureReviewStatus.APPROVED,
        png_path=str(path),
        render_sha256=hashlib.sha256(original).hexdigest(),
    )
    write_valid_image(path, color=(220, 10, 10), min_bytes=2200)

    assert archival_render_available(figure) is False


def test_materialized_record_preview_survives_stripped_base64(tmp_path):
    workspace = tmp_path / "workspace"
    figures = workspace / "figures"
    figures.mkdir(parents=True)
    path = figures / "reviewed.jpg"
    original = write_valid_image(path, min_bytes=2200)
    preview = valid_image_bytes("JPEG")
    preview_digest = hashlib.sha256(preview).hexdigest()
    figure = _figure(
        "reviewed", FigureReviewStatus.APPROVED,
        png_path=str(path),
        png_base64="[image]",
        render_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=preview_digest,
    )
    write_valid_image(path, color=(220, 10, 10), min_bytes=2200)
    preview_path = figures / f"reviewed_record_preview_{preview_digest[:16]}.jpg"
    preview_path.write_bytes(preview)

    assert archival_render_available(figure) is True
    resolved = resolve_verified_archival_artifact(
        figure,
        workspace,
        kind="image",
    )
    assert resolved == (preview, preview_path)
