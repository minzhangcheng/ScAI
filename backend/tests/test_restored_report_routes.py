import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.models.schemas import (
    AnalysisState,
)
from backend.routes.reports import (
    get_annotation_report_html,
    get_exploration_report_html,
    get_integration_report_html,
)
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
