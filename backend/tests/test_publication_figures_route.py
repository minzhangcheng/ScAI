from types import SimpleNamespace
import base64
import hashlib
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

import backend.config as backend_config
import backend.routes.reports_parts.publication as publication_routes
from backend.services.report import artifacts
from backend.models.schemas import AnalysisState, FigureReviewStatus, PublicationFigure
from backend.routes.reports import get_publication_figures
from backend.routes.reports_parts.publication import (
    download_publication_figure_pdf,
    download_publication_figure_png,
)
from backend.tests.split_facade import patch_facade
from backend.tests.figure_image_helpers import (
    valid_image_bytes,
    valid_pdf_bytes,
    write_valid_image,
)


@pytest.mark.asyncio
async def test_publication_figures_include_plan_provenance(tmp_path):
    state = AnalysisState(id="session-figures", name="Figures", user_id="user-1")
    png_path = tmp_path / "planned_result.png"
    write_valid_image(png_path, min_bytes=2200)
    state.publication_figures = [
        PublicationFigure(
            name="planned_result",
            description="Result requested by the approved plan",
            figure_type="custom_result",
            png_path=str(png_path),
            pdf_path="/tmp/planned_result.pdf",
            plan_step_id=6,
            plan_id="plan-v2",
            purpose="result",
            review_status=FigureReviewStatus.APPROVED,
        )
    ]
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))

    async def fake_owner(session_id: str, user_id: str):
        assert session_id == "session-figures"
        assert user_id == "user-1"
        return state

    with patch_facade(
        "backend.routes.reports",
        "require_session_ownership",
        side_effect=fake_owner,
    ):
        response = await get_publication_figures("session-figures", request)

    assert response["figures"][0]["plan_step_id"] == 6
    assert response["figures"][0]["plan_id"] == "plan-v2"
    assert response["figures"][0]["purpose"] == "result"


@pytest.mark.asyncio
async def test_publication_figures_exposes_only_the_reportable_logical_head(tmp_path):
    old_path = tmp_path / "old.jpg"
    head_path = tmp_path / "head.jpg"
    write_valid_image(old_path, min_bytes=2200)
    write_valid_image(head_path, color=(20, 90, 140), min_bytes=2200)
    old = PublicationFigure(
        id="old",
        name="result_v1",
        description="Old revision",
        figure_type="heatmap",
        png_path=str(old_path),
        pdf_path="",
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id="result",
        is_published_head=False,
    )
    head = PublicationFigure(
        id="head",
        name="result_final",
        description="Published revision",
        figure_type="heatmap",
        png_path=str(head_path),
        pdf_path="",
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id="result",
        revision_of_id="old",
        supersedes_id="old",
        is_published_head=True,
    )
    pending = PublicationFigure(
        id="pending",
        name="result_redraw",
        description="Pending redraw",
        figure_type="heatmap",
        png_path=str(head_path),
        pdf_path="",
        review_status=FigureReviewStatus.PENDING,
        logical_figure_id="result",
        revision_of_id="head",
        is_published_head=False,
    )
    state = AnalysisState(
        id="session-figures",
        name="Figures",
        user_id="user-1",
        publication_figures=[old, head, pending],
    )
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))

    async def fake_owner(**_kwargs):
        return state

    with patch_facade(
        "backend.routes.reports",
        "require_session_ownership",
        side_effect=fake_owner,
    ):
        response = await get_publication_figures("session-figures", request)

    assert [figure["id"] for figure in response["figures"]] == ["head"]
    assert response["figures"][0]["logical_figure_id"] == "result"
    assert response["figures"][0]["supersedes_id"] == "old"


@pytest.mark.asyncio
async def test_publication_figures_hide_record_without_physical_render():
    state = AnalysisState(id="session-figures", name="Figures", user_id="user-1")
    state.publication_figures = [
        PublicationFigure(
            name="missing",
            description="No source file or record preview",
            figure_type="other",
            png_path="/definitely/missing/result.png",
            pdf_path="",
            review_status=FigureReviewStatus.APPROVED,
        )
    ]
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))

    async def fake_owner(**_kwargs):
        return state

    with patch_facade(
        "backend.routes.reports",
        "require_session_ownership",
        side_effect=fake_owner,
    ):
        response = await get_publication_figures("session-figures", request)

    assert response["figures"] == []
    assert response["total"] == 0
    assert response["approved"] == 0


@pytest.mark.asyncio
async def test_publication_figures_hide_invalidated_records():
    state = AnalysisState(id="session-figures", name="Figures", user_id="user-1")
    state.publication_figures = [
        PublicationFigure(
            name="stale",
            description="superseded",
            figure_type="other",
            png_path="/tmp/stale.jpg",
            pdf_path="/tmp/stale.pdf",
            is_current=False,
            invalidated_at=datetime.now(),
        )
    ]
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))

    async def fake_owner(**_kwargs):
        return state

    with patch_facade(
        "backend.routes.reports",
        "require_session_ownership",
        side_effect=fake_owner,
    ):
        response = await get_publication_figures("session-figures", request)

    assert response["figures"] == []
    assert response["total"] == 0


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


def test_verified_artifact_reader_has_an_exact_size_boundary(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(artifacts, "MAX_VERIFIED_ARTIFACT_BYTES", 4096)
    boundary = tmp_path / "boundary.bin"
    oversized = tmp_path / "oversized.bin"
    boundary_payload = b"a" * 4096
    boundary.write_bytes(boundary_payload)
    oversized.write_bytes(boundary_payload + b"b")

    assert artifacts._read_regular_file_nofollow(boundary) == boundary_payload
    assert artifacts._read_regular_file_nofollow(oversized) is None


@pytest.mark.parametrize(
    ("pixel_limit", "accepted"),
    [(256, True), (255, False)],
)
def test_verified_image_reader_enforces_pixel_boundary(
    tmp_path, monkeypatch, pixel_limit, accepted,
):
    image = tmp_path / "bounded.png"
    payload = valid_image_bytes("PNG")
    image.write_bytes(payload)
    monkeypatch.setattr(artifacts, "MAX_VERIFIED_IMAGE_PIXELS", pixel_limit)
    record = {"render_sha256": hashlib.sha256(payload).hexdigest()}

    verified = artifacts.verified_artifact_bytes(
        image, record, kind="image"
    )

    assert (verified == payload) is accepted


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
async def test_pdf_download_accepts_structurally_valid_legacy_record_without_digest(
    tmp_path, monkeypatch
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "legacy.pdf"
    path.parent.mkdir()
    path.write_bytes(valid_pdf_bytes())
    figure = PublicationFigure(
        name="legacy",
        description="Legacy result",
        figure_type="other",
        png_path="",
        pdf_path=str(path),
        review_status=FigureReviewStatus.APPROVED,
    )

    response = await download_publication_figure_pdf(
        "session-download", figure.id, state=_download_state(figure)
    )
    assert Path(response.path) == path


@pytest.mark.asyncio
async def test_pdf_download_rejects_invalid_legacy_file(tmp_path, monkeypatch):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "broken.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.4\n" + b"broken" * 200)
    figure = PublicationFigure(
        name="broken",
        description="Broken result",
        figure_type="other",
        png_path="",
        pdf_path=str(path),
        review_status=FigureReviewStatus.APPROVED,
    )

    with pytest.raises(HTTPException) as exc:
        await download_publication_figure_pdf(
            "session-download", figure.id, state=_download_state(figure)
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_invalidated_registered_figure_cannot_fall_back_to_report_message(
    tmp_path, monkeypatch
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "stale.jpg"
    path.parent.mkdir()
    write_valid_image(path, min_bytes=2200)
    figure = PublicationFigure(
        name="stale",
        description="superseded",
        figure_type="other",
        png_path=str(path),
        pdf_path="",
        is_current=False,
        invalidated_at=datetime.now(),
    )
    state = _download_state(figure)
    state.realtime_messages = [{
        "type": "integrated_report",
        "content": {
            "figures": [{
                "id": figure.id,
                "name": figure.name,
                "png_path": str(path),
                "is_publication": True,
            }]
        },
    }]

    with pytest.raises(HTTPException) as exc:
        await download_publication_figure_png(
            "session-download", figure.id, state=state
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "downloader",
    [download_publication_figure_png, download_publication_figure_pdf],
)
async def test_failed_redraw_keeps_old_logical_head_downloadable(
    tmp_path, monkeypatch, downloader
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    jpg = workspace / "figures" / "result.jpg"
    pdf = workspace / "figures" / "result.pdf"
    jpg.parent.mkdir()
    jpg_payload = write_valid_image(jpg, min_bytes=2200)
    pdf_payload = valid_pdf_bytes()
    pdf.write_bytes(pdf_payload)
    old = PublicationFigure(
        id="old-head",
        name="result",
        description="Last good render",
        figure_type="heatmap",
        png_path=str(jpg),
        pdf_path=str(pdf),
        render_sha256=hashlib.sha256(jpg_payload).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf_payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id="logical-result",
        producer_status="published",
        is_published_head=True,
    )
    failed = PublicationFigure(
        id="failed-redraw",
        name="result",
        description="Crashed redraw",
        figure_type="heatmap",
        png_path="",
        pdf_path="",
        review_status=FigureReviewStatus.PENDING,
        logical_figure_id="logical-result",
        revision_of_id=old.id,
        producer_status="failed",
        is_published_head=False,
    )
    state = AnalysisState(
        id="session-download",
        user_id="user-1",
        publication_figures=[old, failed],
    )

    response = await downloader("session-download", old.id, state=state)
    assert Path(response.path).exists()
    with pytest.raises(HTTPException) as exc:
        await downloader("session-download", failed.id, state=state)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "downloader",
    [download_publication_figure_png, download_publication_figure_pdf],
)
async def test_successful_supersede_makes_old_registry_id_404(
    tmp_path, monkeypatch, downloader
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    archive_jpg = workspace / "figures" / ".revisions" / "old" / "result.jpg"
    archive_pdf = archive_jpg.with_suffix(".pdf")
    archive_jpg.parent.mkdir(parents=True, exist_ok=True)
    old_jpg_payload = write_valid_image(archive_jpg, min_bytes=2200)
    old_pdf_payload = valid_pdf_bytes(color=(80, 20, 20))
    archive_pdf.write_bytes(old_pdf_payload)
    head_jpg = workspace / "figures" / "result.jpg"
    head_pdf = workspace / "figures" / "result.pdf"
    head_payload = write_valid_image(
        head_jpg, color=(20, 80, 20), min_bytes=2200
    )
    head_pdf_payload = valid_pdf_bytes(color=(20, 80, 20))
    head_pdf.write_bytes(head_pdf_payload)
    old = PublicationFigure(
        id="old",
        name="result",
        description="Superseded",
        figure_type="heatmap",
        png_path=str(archive_jpg),
        pdf_path=str(archive_pdf),
        render_sha256=hashlib.sha256(old_jpg_payload).hexdigest(),
        pdf_sha256=hashlib.sha256(old_pdf_payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id="logical-result",
        producer_status="published",
        is_published_head=False,
    )
    head = PublicationFigure(
        id="new-head",
        name="result",
        description="Current",
        figure_type="heatmap",
        png_path=str(head_jpg),
        pdf_path=str(head_pdf),
        render_sha256=hashlib.sha256(head_payload).hexdigest(),
        pdf_sha256=hashlib.sha256(head_pdf_payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id="logical-result",
        revision_of_id=old.id,
        supersedes_id=old.id,
        producer_status="published",
        is_published_head=True,
    )
    state = AnalysisState(
        id="session-download",
        user_id="user-1",
        publication_figures=[old, head],
    )

    with pytest.raises(HTTPException) as exc:
        await downloader("session-download", old.id, state=state)
    assert exc.value.status_code == 404
    response = await downloader("session-download", head.id, state=state)
    assert Path(response.path).exists()


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

    source_png = await download_publication_figure_png(
        "session-download", source.id, state=state
    )
    source_pdf_response = await download_publication_figure_pdf(
        "session-download", source.id, state=state
    )
    assert Path(source_png.path).read_bytes() == source_jpg
    assert Path(source_pdf_response.path).read_bytes() == source_pdf

    state.restore_branch_id = "branch-a"
    branch_png = await download_publication_figure_png(
        "session-download", inherited.id, state=state
    )
    branch_pdf = await download_publication_figure_pdf(
        "session-download", inherited.id, state=state
    )
    assert Path(branch_png.path).read_bytes() == source_jpg
    assert Path(branch_pdf.path).read_bytes() == source_pdf


@pytest.mark.asyncio
async def test_png_download_uses_frozen_legacy_id_not_realtime_message_id(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "workspace_only.jpg"
    path.parent.mkdir()
    payload = write_valid_image(path, min_bytes=2200)
    state = AnalysisState(id="session-download", name="Downloads", user_id="user-1")
    state.realtime_messages = [{
        "type": "integrated_report",
        "content": {
            "figures": [{
                "id": "workspace-id",
                "name": "workspace_only",
                "png_path": str(path),
                "render_sha256": hashlib.sha256(payload).hexdigest(),
                "is_publication": False,
            }]
        },
    }]

    with pytest.raises(HTTPException) as exc:
        await download_publication_figure_png(
            "session-download", "workspace-id", state=state,
        )
    assert exc.value.status_code == 404

    assert state.artifact_registry_version == 1
    assert len(state.legacy_artifacts) == 1
    frozen_id = state.legacy_artifacts[0].id
    response = await download_publication_figure_png(
        "session-download", frozen_id, state=state,
    )

    assert Path(response.path) == path

    write_valid_image(path, color=(220, 20, 20), min_bytes=2200)
    with pytest.raises(HTTPException) as changed:
        await download_publication_figure_png(
            "session-download", frozen_id, state=state,
        )
    assert changed.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_workspace,include_digest", [
    (False, False),
    (True, True),
])
async def test_workspace_figure_png_fallback_rejects_unbound_or_foreign_entry(
    tmp_path, monkeypatch, outside_workspace, include_digest,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    base = tmp_path / "another-session" if outside_workspace else workspace
    path = base / "figures" / "workspace_only.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = write_valid_image(path, min_bytes=2200)
    figure = {
        "id": "workspace-id",
        "name": "workspace_only",
        "png_path": str(path),
        "is_publication": False,
    }
    if include_digest:
        figure["render_sha256"] = hashlib.sha256(payload).hexdigest()
    state = AnalysisState(id="session-download", name="Downloads", user_id="user-1")
    state.realtime_messages = [{
        "type": "integrated_report",
        "content": {"figures": [figure]},
    }]

    with pytest.raises(HTTPException) as exc:
        await download_publication_figure_png(
            "session-download", "workspace-id", state=state,
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_pdf_download_uses_frozen_legacy_path_and_digest(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    image_path = workspace / "figures" / "workspace_only.jpg"
    image_path.parent.mkdir()
    image_payload = write_valid_image(image_path, min_bytes=2200)
    pdf_path = image_path.with_suffix(".pdf")
    pdf_path.write_bytes(valid_pdf_bytes())
    state = AnalysisState(id="session-download", name="Downloads", user_id="user-1")
    state.realtime_messages = [{
        "type": "integrated_report",
        "content": {
            "figures": [{
                "id": "workspace-id",
                "name": "workspace_only",
                "png_path": str(image_path),
                "pdf_path": str(pdf_path),
                "render_sha256": hashlib.sha256(image_payload).hexdigest(),
                "is_publication": False,
            }]
        },
    }]

    with pytest.raises(HTTPException) as exc:
        await download_publication_figure_pdf(
            "session-download", "workspace-id", state=state,
        )

    assert exc.value.status_code == 404
    assert len(state.legacy_artifacts) == 1
    frozen_id = state.legacy_artifacts[0].id
    response = await download_publication_figure_pdf(
        "session-download", frozen_id, state=state,
    )
    assert Path(response.path) == pdf_path
    assert response.body == pdf_path.read_bytes()

    pdf_path.write_bytes(valid_pdf_bytes(color=(220, 20, 20)))
    assert response.body != pdf_path.read_bytes()
    with pytest.raises(HTTPException) as changed:
        await download_publication_figure_pdf(
            "session-download", frozen_id, state=state,
        )
    assert changed.value.status_code == 404
