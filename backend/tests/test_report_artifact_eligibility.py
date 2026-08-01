from backend.models.schemas import FigureReviewStatus, PublicationFigure
from backend.services.report.artifacts import eligible_artifacts, is_report_eligible
from backend.services.report.figures import FiguresMixin


def _figure(name, status, **kwargs):
    png_path = kwargs.pop("png_path", f"{name}.jpg")
    pdf_path = kwargs.pop("pdf_path", f"{name}.pdf")
    return PublicationFigure(
        name=name, description="", figure_type="other",
        png_path=png_path, pdf_path=pdf_path,
        review_status=status, **kwargs,
    )


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


def test_workspace_scan_cannot_reintroduce_ineligible_publication_figure(tmp_path):
    workspace = tmp_path / "workspace"
    figures = workspace / "figures"
    report_figures = tmp_path / "report" / "figures"
    figures.mkdir(parents=True)
    report_figures.mkdir(parents=True)
    (figures / "approved.jpg").write_bytes(b"approved")
    (figures / "revision.jpg").write_bytes(b"revision")

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
