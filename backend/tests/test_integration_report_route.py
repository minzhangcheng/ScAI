import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.responses import FileResponse

from backend.models.schemas import AnalysisState
from backend.routes.reports import get_integration_report_html
from backend.tests.split_facade import patch_facade


@pytest.mark.asyncio
async def test_integration_report_html_served_when_path_exists(tmp_path):
    state = AnalysisState(id="sess1", name="T", user_id="u1")
    html_path = tmp_path / "integration_report.html"
    html_path.write_text("<html>integration</html>", encoding="utf-8")
    state.integration_report_html_path = str(html_path)
    request = SimpleNamespace(state=SimpleNamespace(user_id=None))

    async def fake_owner(session_id: str, user_id: str):
        assert session_id == "sess1"
        assert user_id is None
        return state

    # Bind on every module of the reports split, not just the facade: the handler
    # resolves these names from the reports_parts module it happens to live in.
    with patch_facade("backend.routes.reports", "require_session_ownership", side_effect=fake_owner), \
         patch_facade("backend.routes.reports", "get_session_dir", return_value=tmp_path):
        response = await get_integration_report_html("sess1", request)

    assert isinstance(response, FileResponse)
    assert os.fspath(response.path) == os.fspath(html_path)
    assert response.media_type == "text/html"


@pytest.mark.asyncio
async def test_integration_report_html_404_when_no_path():
    state = AnalysisState(id="sess2", name="T", user_id="u1")
    state.integration_report_html_path = None
    request = SimpleNamespace(state=SimpleNamespace(user_id=None))

    async def fake_owner(session_id: str, user_id: str):
        assert session_id == "sess2"
        return state

    with patch_facade("backend.routes.reports", "require_session_ownership", side_effect=fake_owner), \
         patch_facade("backend.routes.reports", "get_session_dir", return_value=Path("/tmp")):
        with pytest.raises(HTTPException) as exc:
            await get_integration_report_html("sess2", request)

    assert exc.value.status_code == 404
