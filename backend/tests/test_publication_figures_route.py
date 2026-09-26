import base64
import hashlib
from pathlib import Path

import pytest
from fastapi import HTTPException

import backend.config as backend_config
import backend.routes.reports_parts.publication as publication_routes
from backend.services.report import artifacts
from backend.models.schemas import (
    AnalysisState,
    FigureReviewStatus,
    PublicationFigure,
)
from backend.routes.reports_parts.publication import (
    download_publication_figure_pdf,
    download_publication_figure_png,
)
from backend.tests.figure_image_helpers import (
    valid_image_bytes,
    valid_pdf_bytes,
    write_valid_image,
)


def _download_state(figure: PublicationFigure) -> AnalysisState:
    state = AnalysisState(id="session-download", name="Downloads", user_id="user-1")
    state.publication_figures = [figure]
    return state


def _configure_download_paths(monkeypatch, tmp_path):
    workspace = tmp_path / "sessions" / "session-download" / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(
        publication_routes, "get_session_workspace", lambda _session_id: workspace
    )
    monkeypatch.setattr(publication_routes, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(publication_routes, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(backend_config, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(backend_config, "REPORTS_DIR", tmp_path)
    return workspace


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "pdf"])
async def test_registered_download_rejects_other_session_before_file_read(
    tmp_path, monkeypatch, kind,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    foreign = (
        tmp_path
        / "sessions"
        / "other-session"
        / "workspace"
        / "figures"
        / ("foreign.pdf" if kind == "pdf" else "foreign.jpg")
    )
    foreign.parent.mkdir(parents=True)
    if kind == "pdf":
        payload = valid_pdf_bytes()
        foreign.write_bytes(payload)
        figure = PublicationFigure(
            id="foreign",
            name="foreign",
            description="",
            figure_type="other",
            png_path="",
            pdf_path=str(foreign),
            pdf_sha256=hashlib.sha256(payload).hexdigest(),
            render_sha256="1" * 64,
            review_status=FigureReviewStatus.APPROVED,
        )
        downloader = download_publication_figure_pdf
    else:
        payload = write_valid_image(foreign, min_bytes=2200)
        figure = PublicationFigure(
            id="foreign",
            name="foreign",
            description="",
            figure_type="other",
            png_path=str(foreign),
            pdf_path="",
            render_sha256=hashlib.sha256(payload).hexdigest(),
            review_status=FigureReviewStatus.APPROVED,
        )
        downloader = download_publication_figure_png

    state = _download_state(figure)
    state.artifact_registry_version = 1
    reads = []
    original_read = artifacts._read_regular_file_nofollow

    def record_read(path):
        reads.append(Path(path))
        return original_read(path)

    monkeypatch.setattr(artifacts, "_read_regular_file_nofollow", record_read)

    with pytest.raises(HTTPException) as exc:
        await downloader("session-download", figure.id, state=state)

    assert exc.value.status_code == 404
    assert foreign not in reads
    assert all(path.is_relative_to(workspace) for path in reads)


@pytest.mark.asyncio
async def test_png_download_recovers_hash_bound_preview_after_same_name_overwrite(
    tmp_path, monkeypatch
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "result.jpg"
    path.parent.mkdir()
    original = write_valid_image(path, min_bytes=2200)
    preview = valid_image_bytes("JPEG")
    figure = PublicationFigure(
        name="result",
        description="Reviewed result",
        figure_type="other",
        png_path=str(path),
        pdf_path="",
        png_base64=base64.b64encode(preview).decode(),
        render_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=hashlib.sha256(preview).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
    )
    write_valid_image(path, color=(220, 10, 10), min_bytes=2200)

    response = await download_publication_figure_png(
        "session-download", figure.id, state=_download_state(figure)
    )

    resolved = Path(response.path)
    assert resolved.read_bytes() == preview
    assert resolved != path


@pytest.mark.asyncio
async def test_pdf_download_rejects_valid_same_name_bytes_with_wrong_digest(
    tmp_path, monkeypatch
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "result.pdf"
    path.parent.mkdir()
    original = valid_pdf_bytes()
    path.write_bytes(original)
    figure = PublicationFigure(
        name="result",
        description="Reviewed result",
        figure_type="other",
        png_path="",
        pdf_path=str(path),
        pdf_sha256=hashlib.sha256(original).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
    )
    path.write_bytes(valid_pdf_bytes(color=(220, 10, 10)))

    with pytest.raises(HTTPException) as exc:
        await download_publication_figure_pdf(
            "session-download", figure.id, state=_download_state(figure)
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_branch_and_source_downloads_keep_archived_pair_after_same_name_redraw(
    tmp_path, monkeypatch,
):
    from backend.services.report.lineage import clone_publication_figures_for_branch

    workspace = _configure_download_paths(monkeypatch, tmp_path)
    jpg = workspace / "figures" / "shared.jpg"
    pdf = jpg.with_suffix(".pdf")
    jpg.parent.mkdir()
    source_jpg = write_valid_image(jpg, min_bytes=2200)
    source_pdf = valid_pdf_bytes()
    pdf.write_bytes(source_pdf)
    source = PublicationFigure(
        id="source-id",
        name="shared",
        description="Inherited source",
        figure_type="heatmap",
        png_path=str(jpg),
        pdf_path=str(pdf),
        render_sha256=hashlib.sha256(source_jpg).hexdigest(),
        pdf_sha256=hashlib.sha256(source_pdf).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        lineage_id="main",
    )
    figures = clone_publication_figures_for_branch(
        [source],
        branch_id="branch-a",
        source_lineage_id="main",
        workspace_path=workspace,
    )
    inherited = figures[1]
    write_valid_image(jpg, color=(220, 20, 20), min_bytes=2200)
    pdf.write_bytes(valid_pdf_bytes(color=(220, 20, 20)))
    state = AnalysisState(
        id="session-download",
        user_id="user-1",
        publication_figures=figures,
        artifact_registry_version=1,
    )

    state.restore_branch_id = "branch-a"
    source_png = await download_publication_figure_png(
        "session-download", source.id, state=state, branch_id="main"
    )
    source_pdf_response = await download_publication_figure_pdf(
        "session-download", source.id, state=state, branch_id="main"
    )
    assert Path(source_png.path).read_bytes() == source_jpg
    assert Path(source_pdf_response.path).read_bytes() == source_pdf

    branch_png = await download_publication_figure_png(
        "session-download", inherited.id, state=state, branch_id="branch-a"
    )
    branch_pdf = await download_publication_figure_pdf(
        "session-download", inherited.id, state=state, branch_id="branch-a"
    )
    assert Path(branch_png.path).read_bytes() == source_jpg
    assert Path(branch_pdf.path).read_bytes() == source_pdf
