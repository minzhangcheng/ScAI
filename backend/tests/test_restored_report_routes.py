import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.models.schemas import (
    AnalysisState,
    FigureReviewStatus,
    PublicationFigure,
    PublicationTable,
)
from backend.routes.reports import (
    get_annotation_report_html,
    get_exploration_report_html,
    get_integration_report_html,
    get_publication_figures,
    get_publication_tables,
)
from backend.tests.figure_image_helpers import write_valid_image
from backend.tests.split_facade import patch_facade


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "path_field"),
    [
        (get_exploration_report_html, "exploration_report_html_path"),
        (get_annotation_report_html, "annotation_report_html_path"),
        (get_integration_report_html, "integration_report_html_path"),
    ],
)
async def test_transient_report_html_accepts_only_active_lineage(
    tmp_path, handler, path_field
):
    html_path = tmp_path / f"{path_field}.html"
    html_path.write_text("<html>report</html>", encoding="utf-8")
    state = AnalysisState(id="session-report", user_id="user-1")
    state.restore_branch_id = "restore/a"
    setattr(state, path_field, str(html_path))
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))

    async def fake_owner(session_id: str, user_id: str):
        return state

    with patch_facade(
        "backend.routes.reports",
        "require_session_ownership",
        side_effect=fake_owner,
    ), patch_facade(
        "backend.routes.reports", "get_session_dir", return_value=tmp_path
    ):
        response = await handler("session-report", request, branch_id="restore/a")
        assert os.fspath(response.path) == os.fspath(html_path)

        response = await handler("session-report", request)
        assert os.fspath(response.path) == os.fspath(html_path)

        with pytest.raises(HTTPException) as exc:
            await handler("session-report", request, branch_id="main")
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_publication_lists_use_requested_or_active_lineage(tmp_path):
    main_path = tmp_path / "main.png"
    branch_path = tmp_path / "branch.png"
    main_payload = write_valid_image(main_path, min_bytes=2200)
    branch_payload = write_valid_image(branch_path, min_bytes=2200)
    state = AnalysisState(
        id="session-report",
        user_id="user-1",
        restore_branch_id="restore/a",
        publication_figures=[
            PublicationFigure(
                id="main-figure", name="main", description="main",
                figure_type="other", png_path=str(main_path), pdf_path="",
                render_sha256=hashlib.sha256(main_payload).hexdigest(),
                review_status=FigureReviewStatus.APPROVED, lineage_id="main",
            ),
            PublicationFigure(
                id="branch-figure", name="branch", description="branch",
                figure_type="other", png_path=str(branch_path), pdf_path="",
                render_sha256=hashlib.sha256(branch_payload).hexdigest(),
                review_status=FigureReviewStatus.APPROVED, lineage_id="restore/a",
            ),
        ],
        publication_tables=[
            PublicationTable(
                id="main-table", name="main", description="main",
                csv_path="/tmp/main.csv", lineage_id="main",
            ),
            PublicationTable(
                id="branch-table", name="branch", description="branch",
                csv_path="/tmp/branch.csv", lineage_id="restore/a",
            ),
        ],
    )
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))

    async def fake_owner(session_id: str, user_id: str):
        return state

    with patch_facade(
        "backend.routes.reports",
        "require_session_ownership",
        side_effect=fake_owner,
    ):
        active_figures = await get_publication_figures("session-report", request)
        active_tables = await get_publication_tables("session-report", request)
        main_figures = await get_publication_figures(
            "session-report", request, branch_id="main"
        )
        main_tables = await get_publication_tables(
            "session-report", request, branch_id="main"
        )

    assert [item["id"] for item in active_figures["figures"]] == ["branch-figure"]
    assert [item["id"] for item in active_tables["tables"]] == ["branch-table"]
    assert [item["id"] for item in main_figures["figures"]] == ["main-figure"]
    assert [item["id"] for item in main_tables["tables"]] == ["main-table"]
