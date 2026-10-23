"""Readonly accounts may package existing reports without gaining write access."""
import io
import re
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import config
from backend.middleware.auth import AuthMiddleware
from backend.models.schemas import AnalysisState
from backend.routes.reports_parts import downloads
from backend.services import session_store
from backend.services.report import sections
from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
from backend.utils import auth


@pytest.fixture
def readonly_downloads(tmp_path, monkeypatch):
    user = SimpleNamespace(id="reader", status="readonly", role="user")
    monkeypatch.setattr(auth, "_db", SimpleNamespace(
        get_user_by_id=AsyncMock(return_value=user),
    ))
    state = AnalysisState(user_id=user.id, name="Saved analysis")
    monkeypatch.setattr(session_store, "ensure_session_loaded", AsyncMock(return_value=state))
    session_dir = tmp_path / "session"
    report_dir = session_dir / "reports"
    workspace = session_dir / "workspace"
    for relative, content in {
        "reports/report.html": b"<html>Saved report</html>",
        "reports/figures/plot.svg": b"<svg/>",
        "reports/tables/results.csv": b"gene,value\nexample,1\n",
        f"workspace/{REPRODUCIBILITY_DIR_NAME}/analysis.py": b"print('saved')\n",
        "session.json": state.model_dump_json().encode(),
    }.items():
        path = session_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(downloads, "resolve_published_report_dir", lambda *_: report_dir)
    monkeypatch.setattr(downloads, "UPLOAD_TEMP_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "get_session_workspace", lambda _: workspace)
    monkeypatch.setattr(sections, "get_session_workspace", lambda _: workspace)
    monkeypatch.setattr(config.default_config, "enable_reproducibility", True)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(downloads.router)
    headers = {"Authorization": f"Bearer {auth.create_access_token(user.id, 'reader')}"}
    with TestClient(app, headers=headers) as client:
        yield client, state, user, session_dir


@pytest.mark.parametrize("method,endpoint,member", [
    ("POST", "download-all-figures", "plot.svg"),
    ("POST", "download-all-tables", "results.csv"),
    ("POST", "download-reproducibility", "analysis.py"),
    ("POST", "download-full", "report.html"),
    ("GET", "download-reproducibility", "analysis.py"),
])
def test_readonly_report_bundles_preserve_results_and_ownership(readonly_downloads, method, endpoint, member):
    client, state, _, session_dir = readonly_downloads
    state_before = state.model_dump_json()
    files_before = {p.relative_to(session_dir): p.read_bytes()
                    for p in session_dir.rglob("*") if p.is_file()}
    url = f"/api/session/{state.id}/report/{endpoint}"
    response = client.request(method, url, params={"branch_id": "main"})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert member in archive.namelist()
        assert archive.read(member)
    assert state.model_dump_json() == state_before
    assert {p.relative_to(session_dir): p.read_bytes()
            for p in session_dir.rglob("*") if p.is_file()} == files_before
    assert not list(downloads.UPLOAD_TEMP_DIR.iterdir())
    state.user_id = "another-owner"
    assert client.request(method, url).status_code == 404
    assert not list(downloads.UPLOAD_TEMP_DIR.iterdir())


def test_reproducibility_html_link_downloads_without_auth_header(readonly_downloads):
    client, state, _, _ = readonly_downloads
    html = sections.SectionsMixin()._build_reproducibility_section_html(state)
    link = re.search(r'<a href="([^"]+)" download', html)
    assert link is not None
    # A browser follows the report's anchor without an Authorization header.
    client.headers.pop("Authorization")
    response = client.get(link.group(1))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read("analysis.py") == b"print('saved')\n"
    assert not list(downloads.UPLOAD_TEMP_DIR.iterdir())


@pytest.mark.parametrize("method,tail", [
    ("POST", "generate-detailed-report"),
    ("POST", "report/partial/approve"),
    ("POST", "report/download-full/extra"),
    ("POST", "report/download-full-extra"),
    ("POST", "nested/report/download-full"),
    ("POST", "export/prepare"),
    ("PATCH", "report/download-full"),
    ("DELETE", "report/download-all-figures"),
    ("DELETE", ""),
])
def test_readonly_download_exception_does_not_allow_writes(readonly_downloads, method, tail):
    client, state, _, _ = readonly_downloads
    response = client.request(method, f"/api/session/{state.id}/{tail}".rstrip("/"))
    assert response.status_code == 403
    assert response.json()["detail"] == "Readonly account cannot perform write operations"
    assert not downloads.UPLOAD_TEMP_DIR.exists()


@pytest.mark.parametrize("status", ["pending", "deleting"])
def test_report_download_exception_does_not_bypass_account_revocation(readonly_downloads, status):
    client, state, user, _ = readonly_downloads
    user.status = status
    response = client.post(f"/api/session/{state.id}/report/download-full")
    assert response.status_code == 403
    assert not downloads.UPLOAD_TEMP_DIR.exists()
