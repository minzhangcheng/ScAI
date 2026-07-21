"""Tests for API key masking in literature and agent-server GET endpoints."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend import utils
from backend.utils.auth import create_access_token, hash_password
from backend.utils.db import AppDB
from backend.utils.secret_redaction import mask_api_key


@pytest_asyncio.fixture
async def client_and_token(tmp_path):
    """Wire up an isolated AppDB into the auth module so the middleware
    finds users in the test database, not the real one."""
    old_db = utils.auth._db

    db_path = tmp_path / "test_masking.db"
    new_db = AppDB(db_path)
    await new_db.init()
    utils.auth._db = new_db

    pw = hash_password("x")
    uid = await new_db.create_user(
        "tuser", pw, email="t@x.io", role="user", status="active"
    )
    token = create_access_token(uid, "tuser")

    yield uid, {"Authorization": f"Bearer {token}"}

    utils.auth._db = old_db


@pytest.mark.asyncio
async def test_get_literature_config_masks_key(client_and_token):
    uid, headers = client_and_token
    from backend.state import sessions
    from backend.models.schemas import AnalysisState
    state = AnalysisState(session_id="test-mask-lit", user_id=uid)
    state.literature_config["deep_research_api_key"] = "sk-secret"
    sessions["test-mask-lit"] = state

    from backend.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get("/api/session/test-mask-lit/literature-config",
                        headers=headers)
    assert r.status_code == 200, r.text
    assert "sk-secret" not in r.text


@pytest.mark.asyncio
async def test_get_agent_server_config_masks_key(client_and_token):
    uid, headers = client_and_token
    from backend.state import sessions
    from backend.models.schemas import AnalysisState
    state = AnalysisState(session_id="test-mask-agt", user_id=uid)
    state.agent_server_configs["pi"] = {"llm_base_url": "http://x", "api_key": "sk-secret"}
    sessions["test-mask-agt"] = state

    from backend.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get("/api/session/test-mask-agt/agent-server-config",
                        headers=headers)
    assert r.status_code == 200, r.text
    assert "sk-secret" not in r.text
    assert r.json()["pi"]["api_key"] == mask_api_key("sk-secret")
