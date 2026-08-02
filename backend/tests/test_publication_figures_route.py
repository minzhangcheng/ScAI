from types import SimpleNamespace
import base64
import hashlib
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

import backend.config as backend_config
import backend.routes.reports_parts.publication as publication_routes
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
async def test_workspace_figure_png_fallback_requires_current_workspace_and_digest(
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

    response = await download_publication_figure_png(
        "session-download", "workspace-id", state=state,
    )

    assert Path(response.path) == path


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
async def test_workspace_figure_pdf_fallback_requires_pdf_digest(
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
    assert "integrity metadata" in exc.value.detail
