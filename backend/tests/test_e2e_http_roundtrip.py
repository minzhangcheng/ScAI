"""HTTP-level end-to-end: export a mid-analysis session, import it, continue.

Everything goes through the real routes, middleware and auth — not through the
internal helpers the other checks call directly.
"""
import asyncio
import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from backend.routes.auth import _login_rate_limiter
from backend.utils import auth as auth_module
from backend.utils.db import AppDB


@pytest.fixture(autouse=True)
def _reset_limiter():
    _login_rate_limiter._events.clear()
    yield
    _login_rate_limiter._events.clear()


@pytest.fixture
def client(tmp_path):
    old_db = auth_module._db
    new_db = AppDB(tmp_path / "e2e.db")
    loop = asyncio.new_event_loop()
    loop.run_until_complete(new_db.init())
    loop.close()
    auth_module._db = new_db
    from backend.main import app
    c = TestClient(app)
    yield c
    auth_module._db = old_db


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _token(client, username="boss", password="adminpass123"):
    client.post("/api/auth/setup/admin", json={"username": username, "password": password})
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _populate(session_id, *, live_pkl: bool):
    """Give the session the artefacts a real mid-analysis run would have."""
    from backend.config import get_session_states, get_session_workspace, get_session_uploads
    from backend.utils.runtime_apply import _live_pkl

    states = get_session_states(session_id)
    states.mkdir(parents=True, exist_ok=True)
    (states / f"{session_id}.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "sc.pp.pca(adata)"}]}),
        encoding="utf-8",
    )
    (states / f"{session_id}_bioinfo_checkpoint.json").write_text(
        json.dumps({"step": 7}), encoding="utf-8"
    )
    (states / "checkpoints.sqlite").write_bytes(b"SQLITE-CHECKPOINT")
    if live_pkl:
        _live_pkl(session_id).write_bytes(b"DILL-KERNEL-DUMP")

    ws = get_session_workspace(session_id)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "umap.png").write_bytes(b"\x89PNG-figure")

    up = get_session_uploads(session_id)
    up.mkdir(parents=True, exist_ok=True)
    (up / "matrix.h5ad").write_bytes(b"H5AD-DATA")


def _advance(session_id, round_=3):
    """Put the session where a mid-analysis export would find it."""
    from backend.models.schemas import TaskStatus
    from backend.state import sessions

    state = sessions[session_id]
    state.current_round = round_
    state.status = TaskStatus.RUNNING
    return state


def _export_download(client, token, session_id, *, include_runtime):
    q = "?include_runtime=true" if include_runtime else ""
    prep = client.post(f"/api/session/{session_id}/export/prepare{q}", headers=_auth(token))
    assert prep.status_code == 200, prep.text
    body = prep.json()
    dl = client.get(body["download_url"], headers=_auth(token))
    assert dl.status_code == 200, dl.text
    return dl.content


def _import_archive(client, token, blob, *, trust):
    headers = {
        **_auth(token),
        "X-Filename": "export_test.tar.gz",
        "Content-Type": "application/octet-stream",
    }
    if trust:
        headers["X-Trust"] = "true"
    r = client.post("/api/session/import", content=blob, headers=headers)
    assert r.status_code == 200, r.text
    import_id = r.json()["data"]["import_id"]

    for _ in range(120):
        s = client.get(f"/api/session/import/{import_id}", headers=_auth(token))
        assert s.status_code == 200, s.text
        job = s.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert job["status"] == "completed", f"import did not complete: {job}"
    return job["session_id"]


@pytest.mark.parametrize(
    "include_runtime, trust, keep_source",
    [
        (True, True, False),    # cross-instance migration, runtime carried
        (True, True, True),     # same instance: the id collides, a new one is allocated
        (False, False, False),  # the default export: no runtime at all
        (True, False, False),   # runtime exported, not trusted on the way in
    ],
)
def test_export_import_continue(client, include_runtime, trust, keep_source):
    from backend.config import get_session_dir, get_session_states
    from backend.models.schemas import TaskStatus
    from backend.state import sessions
    from backend.utils.runtime_apply import _live_pkl

    token = _token(client)

    started = client.post(
        "/api/session/start",
        json={"instruction": "cluster the cells"},
        headers=_auth(token),
    )
    assert started.status_code == 200, started.text
    src_id = started.json()["session_id"]

    _populate(src_id, live_pkl=True)
    _advance(src_id)
    client.post(f"/api/session/{src_id}/save", headers=_auth(token))

    blob = _export_download(client, token, src_id, include_runtime=include_runtime)
    assert blob[:2] == b"\x1f\x8b", "export is not gzip"

    if not keep_source:
        client.delete(f"/api/session/{src_id}", headers=_auth(token))
        sessions.pop(src_id, None)

    new_id = _import_archive(client, token, blob, trust=trust)
    if keep_source:
        assert new_id != src_id, "a colliding id must not be reused"

    # --- everything the archive carried is readable through the API ---
    loaded = client.get(f"/api/session/{new_id}/load", headers=_auth(token))
    assert loaded.status_code == 200, loaded.text

    states = get_session_states(new_id)
    assert (states / f"{new_id}.ipynb").exists(), "analysis notebook lost"
    assert (states / f"{new_id}_bioinfo_checkpoint.json").exists(), "bioinfo checkpoint lost"
    assert (get_session_dir(new_id) / "workspace" / "umap.png").exists() or \
           any(get_session_dir(new_id).rglob("umap.png")), "figure lost"
    assert any(get_session_dir(new_id).rglob("matrix.h5ad")), "uploaded data lost"

    # --- the session must not claim a run nobody owns ---
    state = sessions[new_id]
    assert state.status != TaskStatus.RUNNING, "imported session claims to be running"

    # --- the claim about the kernel must match the disk ---
    kernel_present = _live_pkl(new_id).exists()
    assert kernel_present == (include_runtime and trust), (
        f"kernel presence {kernel_present} does not match "
        f"include_runtime={include_runtime} trust={trust}"
    )
    if kernel_present:
        assert state.runtime_restore_status == "complete"
    else:
        assert state.runtime_restore_status == "unavailable"

    # --- and it must be able to move forward ---
    run = client.post(f"/api/session/{new_id}/run", headers=_auth(token))
    assert run.status_code == 200, run.text
    assert run.json()["success"] is True

    after = sessions[new_id]
    assert after.status == TaskStatus.RUNNING
    if not kernel_present:
        assert after.runtime_recompute_pending is True, (
            "a kernel-less re-run must tell the agent to reload from disk"
        )
        assert after.restore_start_intent == "explicit_start", (
            "without a kernel the run must be routed through restore_entry"
        )
    assert after.runtime_restore_status == "complete"
