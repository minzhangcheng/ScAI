from types import SimpleNamespace
import asyncio
import base64
import hashlib
import threading
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
    png_payload = write_valid_image(png_path, min_bytes=2200)
    state.publication_figures = [
        PublicationFigure(
            name="planned_result",
            description="Result requested by the approved plan",
            figure_type="custom_result",
            png_path=str(png_path),
            pdf_path="/tmp/planned_result.pdf",
            render_sha256=hashlib.sha256(png_payload).hexdigest(),
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
async def test_publication_figures_exposes_all_current_completed_renders(tmp_path):
    old_path = tmp_path / "old.jpg"
    head_path = tmp_path / "head.jpg"
    old_payload = write_valid_image(old_path, min_bytes=2200)
    head_payload = write_valid_image(
        head_path, color=(20, 90, 140), min_bytes=2200,
    )
    old = PublicationFigure(
        id="old",
        name="result_v1",
        description="Old revision",
        figure_type="heatmap",
        png_path=str(old_path),
        pdf_path="",
        render_sha256=hashlib.sha256(old_payload).hexdigest(),
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
        render_sha256=hashlib.sha256(head_payload).hexdigest(),
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
        render_sha256=hashlib.sha256(head_payload).hexdigest(),
        review_status=FigureReviewStatus.PENDING,
        layout_violations=[{"kind": "text_over_text", "severity": "block"}],
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

    assert [figure["id"] for figure in response["figures"]] == [
        "old", "head", "pending",
    ]
    assert response["figures"][1]["logical_figure_id"] == "result"
    assert response["figures"][1]["supersedes_id"] == "old"
    assert response["figures"][2]["review_status"] == "pending"
    assert response["figures"][2]["layout_blocking"] is True


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
async def test_pdf_download_rejects_hashless_legacy_publication_record(
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

    with pytest.raises(HTTPException) as exc:
        await download_publication_figure_pdf(
            "session-download", figure.id, state=_download_state(figure)
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_png_download_rejects_hashless_record_before_and_after_mutation(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "legacy.jpg"
    path.parent.mkdir()
    write_valid_image(path, min_bytes=2200)
    figure = PublicationFigure(
        name="legacy",
        description="Legacy reviewed result",
        figure_type="other",
        png_path=str(path),
        pdf_path="",
        review_status=FigureReviewStatus.APPROVED,
    )
    state = _download_state(figure)
    state.artifact_registry_version = 1

    with pytest.raises(HTTPException) as original:
        await download_publication_figure_png(
            "session-download", figure.id, state=state,
        )
    assert original.value.status_code == 404

    write_valid_image(path, color=(220, 20, 20), min_bytes=2200)
    with pytest.raises(HTTPException) as replaced:
        await download_publication_figure_png(
            "session-download", figure.id, state=state,
        )
    assert replaced.value.status_code == 404


@pytest.mark.asyncio
async def test_png_download_keeps_event_loop_responsive_during_image_validation(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "large-validation.jpg"
    path.parent.mkdir()
    payload = write_valid_image(path, min_bytes=2200)
    figure = PublicationFigure(
        name="large-validation",
        description="Bound render",
        figure_type="other",
        png_path=str(path),
        pdf_path="",
        render_sha256=hashlib.sha256(payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = _download_state(figure)
    state.artifact_registry_version = 1
    entered = threading.Event()
    release = threading.Event()
    real_resolve = publication_routes.resolve_verified_archival_artifact

    def slow_resolve(*args, **kwargs):
        entered.set()
        release.wait(timeout=1.5)
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(
        publication_routes, "resolve_verified_archival_artifact", slow_resolve
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    task = asyncio.create_task(download_publication_figure_png(
        "session-download", figure.id, state=state,
    ))
    while not entered.is_set() and loop.time() - started < 0.5:
        await asyncio.sleep(0.005)
    responsive = entered.is_set() and loop.time() - started < 0.5
    from backend.services.session_lifecycle import get_lifecycle_lock
    lifecycle_lock = get_lifecycle_lock("session-download")
    await asyncio.wait_for(lifecycle_lock.acquire(), timeout=0.2)
    lifecycle_lock.release()
    release.set()
    response = await task

    assert responsive
    assert response.body == payload


@pytest.mark.asyncio
async def test_pdf_download_verification_runs_in_owned_worker(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "worker-validated.pdf"
    path.parent.mkdir()
    payload = valid_pdf_bytes()
    path.write_bytes(payload)
    figure = PublicationFigure(
        name="worker-validated",
        description="Bound PDF",
        figure_type="other",
        png_path="",
        pdf_path=str(path),
        pdf_sha256=hashlib.sha256(payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = _download_state(figure)
    state.artifact_registry_version = 1
    event_loop_thread = threading.get_ident()
    verifier_threads = set()
    real_resolve = publication_routes.resolve_verified_archival_artifact

    def observed_resolve(*args, **kwargs):
        verifier_threads.add(threading.get_ident())
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(
        publication_routes,
        "resolve_verified_archival_artifact",
        observed_resolve,
    )
    response = await download_publication_figure_pdf(
        "session-download", figure.id, state=state,
    )

    assert response.body == payload
    assert verifier_threads and event_loop_thread not in verifier_threads


@pytest.mark.asyncio
async def test_png_download_reads_and_decodes_the_selected_file_once(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "single-pass.jpg"
    path.parent.mkdir()
    payload = write_valid_image(path, min_bytes=2200)
    figure = PublicationFigure(
        name="single-pass",
        description="Bound render",
        figure_type="other",
        png_path=str(path),
        pdf_path="",
        render_sha256=hashlib.sha256(payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = _download_state(figure)
    state.artifact_registry_version = 1
    reads = 0
    real_read = artifacts._read_regular_file_nofollow

    def observed_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(artifacts, "_read_regular_file_nofollow", observed_read)

    response = await download_publication_figure_png(
        "session-download", figure.id, state=state,
    )

    assert response.body == payload
    assert reads == 1


@pytest.mark.asyncio
async def test_concurrent_legacy_downloads_coalesce_inventory_by_generation(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "legacy-output.jpg"
    path.parent.mkdir()
    payload = write_valid_image(path, min_bytes=2200)

    def old_state():
        state = AnalysisState(
            id="session-download",
            name="Downloads",
            user_id="user-1",
            lifecycle_generation=0,
        )
        state.realtime_messages = [{
            "type": "integrated_report",
            "content": {"figures": [{
                "id": "old-workspace-id",
                "name": "legacy-output",
                "png_path": str(path),
            }]},
        }]
        return state

    first = old_state()
    second = old_state()
    calls = 0
    worker_threads = set()
    event_loop_thread = threading.get_ident()
    real_scan = publication_routes.ensure_legacy_artifact_inventory

    def counted_scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        worker_threads.add(threading.get_ident())
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(
        publication_routes, "ensure_legacy_artifact_inventory", counted_scan,
    )
    with publication_routes._legacy_inventory_cache_guard:
        publication_routes._legacy_inventory_cache.clear()

    responses = await asyncio.gather(*[
        download_publication_figure_png(
            "session-download",
            "old-workspace-id",
            state=first if index % 2 == 0 else second,
        )
        for index in range(30)
    ])

    assert calls == 1
    assert worker_threads and event_loop_thread not in worker_threads
    assert all(response.body == payload for response in responses)
    assert first.artifact_registry_version == 1
    assert second.artifact_registry_version == 1


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
async def test_successful_supersede_keeps_old_current_render_browsable(
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

    old_response = await downloader("session-download", old.id, state=state)
    head_response = await downloader("session-download", head.id, state=state)
    assert Path(old_response.path).exists()
    assert Path(head_response.path).exists()


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
async def test_png_download_accepts_pre_registry_message_id_via_frozen_inventory(
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

    legacy_alias = await download_publication_figure_png(
        "session-download", "workspace-id", state=state,
    )
    assert Path(legacy_alias.path) == path

    assert state.artifact_registry_version == 1
    assert len(state.legacy_artifacts) == 1
    frozen_id = state.legacy_artifacts[0].id
    response = await download_publication_figure_png(
        "session-download", frozen_id, state=state,
    )

    assert Path(response.path) == path

    write_valid_image(path, color=(220, 20, 20), min_bytes=2200)
    with pytest.raises(HTTPException) as changed_alias:
        await download_publication_figure_png(
            "session-download", "workspace-id", state=state,
        )
    assert changed_alias.value.status_code == 404
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
async def test_workspace_figure_png_alias_accepts_inventory_and_rejects_foreign_entry(
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
    }
    if include_digest:
        figure["render_sha256"] = hashlib.sha256(payload).hexdigest()
        figure["is_publication"] = False
    state = AnalysisState(id="session-download", name="Downloads", user_id="user-1")
    state.realtime_messages = [{
        "type": "integrated_report",
        "content": {"figures": [figure]},
    }]

    if outside_workspace:
        with pytest.raises(HTTPException) as exc:
            await download_publication_figure_png(
                "session-download", "workspace-id", state=state,
            )
        assert exc.value.status_code == 404
    else:
        response = await download_publication_figure_png(
            "session-download", "workspace-id", state=state,
        )
        assert Path(response.path) == path
        write_valid_image(path, color=(220, 20, 20), min_bytes=2200)
        with pytest.raises(HTTPException) as changed:
            await download_publication_figure_png(
                "session-download", "workspace-id", state=state,
            )
        assert changed.value.status_code == 404


@pytest.mark.asyncio
async def test_workspace_figure_alias_rejects_message_digest_mismatch(
    tmp_path, monkeypatch,
):
    workspace = _configure_download_paths(monkeypatch, tmp_path)
    path = workspace / "figures" / "workspace_only.jpg"
    path.parent.mkdir(parents=True)
    write_valid_image(path, min_bytes=2200)
    state = AnalysisState(id="session-download", name="Downloads", user_id="user-1")
    state.realtime_messages = [{
        "type": "integrated_report",
        "content": {"figures": [{
            "id": "workspace-id",
            "name": "workspace_only",
            "png_path": str(path),
            "render_sha256": "0" * 64,
            "is_publication": False,
        }]},
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

    alias_response = await download_publication_figure_pdf(
        "session-download", "workspace-id", state=state,
    )
    assert Path(alias_response.path) == pdf_path
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
