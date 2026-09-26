
import pytest

from backend.services.report.paths import (
    _sanitize_branch_id,
)


def test_branch_components_do_not_traverse_or_collide():
    components = {
        branch_id: _sanitize_branch_id(branch_id)
        for branch_id in (".", "..", ".hidden", "branch/A", "branch_A")
    }
    assert len(set(components.values())) == len(components)
    assert all(component not in {"", ".", ".."} for component in components.values())
    assert all("/" not in component and "\\" not in component for component in components.values())


from backend.models.schemas import (
    FigureReviewStatus,
    PublicationFigure,
)


def test_branch_clone_archives_full_pair_before_same_name_redraw(tmp_path):
    import hashlib
    from pathlib import Path

    from backend.services.report.artifacts import (
        resolve_archival_pdf,
        resolve_archival_render,
    )
    from backend.services.report.lineage import clone_publication_figures_for_branch
    from backend.tests.figure_image_helpers import valid_pdf_bytes, write_valid_image

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    mutable_jpg = figures_dir / "shared_result.jpg"
    mutable_pdf = figures_dir / "shared_result.pdf"
    original_jpg = write_valid_image(
        mutable_jpg, color=(30, 80, 120), min_bytes=3000
    )
    original_pdf = valid_pdf_bytes(color=(30, 80, 120))
    mutable_pdf.write_bytes(original_pdf)
    original = PublicationFigure(
        id="fig-main",
        name="shared_result",
        description="shared ancestor",
        figure_type="heatmap",
        png_path=str(mutable_jpg),
        pdf_path=str(mutable_pdf),
        render_sha256=hashlib.sha256(original_jpg).hexdigest(),
        pdf_sha256=hashlib.sha256(original_pdf).hexdigest(),
        lineage_id="main",
        review_status=FigureReviewStatus.APPROVED,
    )

    cloned = clone_publication_figures_for_branch(
        [original],
        branch_id="branch-a",
        source_lineage_id="main",
        workspace_path=tmp_path,
    )

    source, inherited = cloned
    assert source.png_path == inherited.png_path
    assert source.pdf_path == inherited.pdf_path
    assert Path(source.png_path).is_relative_to(figures_dir / ".lineage")
    assert Path(source.pdf_path).is_relative_to(figures_dir / ".lineage")
    assert Path(source.png_path).read_bytes() == original_jpg
    assert Path(source.pdf_path).read_bytes() == original_pdf

    cloned = clone_publication_figures_for_branch(
        cloned,
        branch_id="branch-a",
        source_lineage_id="main",
        workspace_path=tmp_path,
    )
    assert len(cloned) == 2
    source, inherited = cloned
    assert inherited.png_path == source.png_path
    assert inherited.pdf_path == source.pdf_path

    # The restored branch is free to publish the same logical filename. Neither
    # inherited record may fall back to the mutable top-level pair afterward.
    write_valid_image(mutable_jpg, color=(150, 20, 20), min_bytes=3000)
    mutable_pdf.write_bytes(valid_pdf_bytes(color=(150, 20, 20)))
    for record in (source, inherited):
        assert resolve_archival_render(record, tmp_path).read_bytes() == original_jpg
        assert resolve_archival_pdf(record, tmp_path).read_bytes() == original_pdf


def _archivable_figure(
    workspace, record_id, *, name=None, lineage="main", logical_id=None, color=(30, 60, 90),
):
    import hashlib

    from backend.models.schemas import FigureReviewStatus, PublicationFigure
    from backend.tests.figure_image_helpers import valid_pdf_bytes, write_valid_image

    figure_name = name or record_id
    jpg = workspace / "figures" / f"{figure_name}.jpg"
    pdf = jpg.with_suffix(".pdf")
    jpg.parent.mkdir(parents=True, exist_ok=True)
    image_payload = write_valid_image(jpg, color=color, min_bytes=2200)
    pdf_payload = valid_pdf_bytes(color=color)
    pdf.write_bytes(pdf_payload)
    return PublicationFigure(
        id=record_id,
        name=figure_name,
        description=figure_name,
        figure_type="heatmap",
        png_path=str(jpg),
        pdf_path=str(pdf),
        render_sha256=hashlib.sha256(image_payload).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf_payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        lineage_id=lineage,
        logical_figure_id=logical_id or f"logical-{record_id}",
    )


def test_lineage_archive_rejects_symlinked_root_without_external_writes(tmp_path):
    from backend.services.report.lineage import clone_publication_figures_for_branch

    source = _archivable_figure(tmp_path, "source")
    outside = tmp_path / "outside-lineage"
    outside.mkdir()
    (tmp_path / "figures" / ".lineage").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(RuntimeError):
        clone_publication_figures_for_branch(
            [source], branch_id="branch-a", workspace_path=tmp_path
        )

    assert list(outside.iterdir()) == []


def test_second_source_archive_failure_leaves_all_records_unchanged(
    tmp_path, monkeypatch,
):
    from backend.services.report import lineage

    first = _archivable_figure(tmp_path, "first", color=(20, 40, 60))
    second = _archivable_figure(tmp_path, "second", color=(200, 40, 20))
    first_before = first.model_dump()
    second_before = second.model_dump()
    real_archive = lineage._archive_inherited_figure
    calls = 0

    def fail_second(figure, workspace_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second source archive failed")
        return real_archive(figure, workspace_path)

    monkeypatch.setattr(lineage, "_archive_inherited_figure", fail_second)
    with pytest.raises(OSError):
        lineage.clone_publication_figures_for_branch(
            [first, second], branch_id="branch-a", workspace_path=tmp_path
        )

    assert first.model_dump() == first_before
    assert second.model_dump() == second_before
