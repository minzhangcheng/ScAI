"""HTTP-level end-to-end: export a mid-analysis session, import it, continue.

Everything goes through the real routes, middleware and auth — not through the
internal helpers the other checks call directly.
"""
import asyncio
import json
import shutil
import time

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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
    asyncio.run(_create_checkpoint_fixture(states / "checkpoints.sqlite"))
    if live_pkl:
        _live_pkl(session_id).write_bytes(b"DILL-KERNEL-DUMP")

    ws = get_session_workspace(session_id)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "umap.png").write_bytes(b"\x89PNG-figure")

    up = get_session_uploads(session_id)
    up.mkdir(parents=True, exist_ok=True)
    (up / "matrix.h5ad").write_bytes(b"H5AD-DATA")


async def _create_checkpoint_fixture(path):
    """Create the smallest saver-owned checkpoint database for E2E export."""
    connection = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(connection)
    await saver.setup()
    await connection.close()


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


@pytest.fixture
def snapshot_v2(request, monkeypatch):
    """Pin the Snapshot V2 operator gate for the duration of a round trip.

    The gate is a process-scoped module global, and the repo conftest turns it
    *on* for the whole suite. An ordinary deployment leaves it off, and both
    ``/run`` and ``/resume`` call ``normalize_disabled_snapshot_state`` only on
    that path — so the configuration this round trip is supposed to certify was
    the one configuration it never ran under.
    """
    from backend.config import _config

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", request.param)
    return request.param


@pytest.mark.parametrize("snapshot_v2", [False, True], indirect=True)
@pytest.mark.parametrize(
    "include_runtime, trust, keep_source",
    [
        (True, True, False),    # cross-instance migration, runtime carried
        (True, True, True),     # same instance: the id collides, a new one is allocated
        (False, False, False),  # the default export: no runtime at all
        (True, False, False),   # runtime exported, not trusted on the way in
        (False, True, False),   # trust is harmless without exported runtime
        (False, False, True),   # colliding source without runtime gets a new ID
    ],
)
def test_export_import_continue(
    client, snapshot_v2, include_runtime, trust, keep_source
):
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

    stale_generation = None
    if not keep_source:
        stale_generation = sessions[src_id].model_copy(deep=True)
        client.delete(f"/api/session/{src_id}", headers=_auth(token))
        sessions.pop(src_id, None)

    new_id = _import_archive(client, token, blob, trust=trust)
    if keep_source:
        assert new_id != src_id, "a colliding id must not be reused"
    else:
        from backend.services.session_lifecycle import (
            SessionGoneError,
            assert_generation_writable,
            read_lifecycle_record,
        )

        assert new_id == src_id, (
            "after deletion the archive ID should be rebuilt as a new local "
            "generation, not permanently exiled by the old tombstone"
        )
        lifecycle = read_lifecycle_record(new_id)
        assert lifecycle["status"] == "active"
        assert lifecycle["generation"] == 2
        assert lifecycle["deleted_generations"] == [1]
        with pytest.raises(SessionGoneError):
            assert_generation_writable(new_id, stale_generation)
        # A cleanup invocation retained from generation 1 must re-read the
        # lifecycle record and leave the newly active generation 2 untouched.
        from backend.services.session_deletion import _delete_tombstoned_files

        stale_sweep = asyncio.run(_delete_tombstoned_files(new_id))
        assert stale_sweep == {
            "deleted_sessions": 0,
            "deleted_state_files": 0,
        }
        assert get_session_dir(new_id).is_dir()

    # --- everything the archive carried is readable through the API ---
    loaded = client.get(f"/api/session/{new_id}/load", headers=_auth(token))
    assert loaded.status_code == 200, loaded.text

    states = get_session_states(new_id)
    assert (states / f"{new_id}.ipynb").exists(), "analysis notebook lost"
    # The bioinformatics checkpoint is kernel-bound: it tells the node to resume
    # its autonomous loop, which is only meaningful if the values that loop
    # produced arrived with it. Without the kernel it must be gone, or the node
    # resumes the old conversation and never sees the kernel-is-empty notice.
    assert (states / f"{new_id}_bioinfo_checkpoint.json").exists() == (
        include_runtime and trust
    ), "the bioinfo checkpoint must travel with its kernel and only with it"
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

    # --- ...and "moving forward" has to mean the graph is actually entered ---
    # POST /run only records intent and hands back a WebSocket URL; the handler
    # on the other end is what dispatches. Asserting the status /run itself just
    # wrote says nothing about whether anything would execute, so check the
    # branch that handler takes and where that branch enters the graph.
    from backend.routes.websockets import should_resume_workflow
    from backend.services.snapshot_v2.restore_entry import build_restore_entry_plan

    assert should_resume_workflow(after) is True, (
        "a session that has already run must continue, not start over"
    )
    if not kernel_present:
        plan = build_restore_entry_plan(after)
        assert plan.target_node == "initialize_round", (
            "a kernel-less continuation must re-enter the round loop; anything "
            "else holds without executing"
        )
        assert plan.preserve_round_on_initialize is True, (
            "re-entry must redo the current round from the imported data, not "
            "advance past it and lose it"
        )


@pytest.mark.parametrize("snapshot_v2", [False, True], indirect=True)
def test_the_resume_button_moves_a_kernel_without_its_checkpoint(client, snapshot_v2):
    """Re-import a runtime-carrying archive onto the instance that wrote it.

    The id collides, so a new one is allocated and the checkpoint is dropped —
    it is keyed by the archive's thread id. The kernel survives, which leaves
    ``runtime_restore_status`` at ``complete``, and the frontend only offers
    Re-run while the runtime is *un*available. So Resume is the sole control
    here, and it never sets a start intent of its own: unless the import armed
    one, ``resume()`` finds no checkpoint and no continuation target and holds
    without executing.

    Every other case in this file continues through ``POST /run``, which arms
    the intent itself and therefore cannot tell whether the import did.
    """
    from backend.config import get_session_states
    from backend.models.schemas import TaskStatus
    from backend.routes.websockets import should_resume_workflow
    from backend.services.snapshot_v2.restore_entry import build_restore_entry_plan
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

    blob = _export_download(client, token, src_id, include_runtime=True)
    new_id = _import_archive(client, token, blob, trust=True)
    assert new_id != src_id, "a colliding id must not be reused"

    state = sessions[new_id]
    assert _live_pkl(new_id).exists(), "the trusted kernel should have landed"
    assert not (get_session_states(new_id) / "checkpoints.sqlite").exists(), (
        "a checkpoint keyed by the archive's thread id must not be reused"
    )
    assert state.runtime_restore_status == "complete", (
        "the kernel is here, so the UI will offer Resume and not Re-run"
    )

    resumed = client.post(f"/api/session/{new_id}/resume", headers=_auth(token))
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["success"] is True

    after = sessions[new_id]
    assert after.status == TaskStatus.RUNNING
    assert should_resume_workflow(after) is True
    plan = build_restore_entry_plan(after)
    assert plan.target_node == "initialize_round", (
        "Resume must re-enter the round loop; without an armed start intent "
        "resume() holds and the button is a silent no-op"
    )
    assert plan.preserve_round_on_initialize is True


@pytest.mark.parametrize("snapshot_v2", [False, True], indirect=True)
def test_a_session_exported_before_its_first_round_still_starts(client, snapshot_v2):
    """Export during the window between Start and the first round completing.

    ``current_round`` is only incremented once PI planning returns, but the
    status is RUNNING from the moment Start is pressed, so this window is wide
    open in practice. Both import fixups bail out at round 0 — no runtime mark,
    no restore intent — which leaves the frontend rendering neither Start (the
    status is no longer PENDING) nor Re-run (the runtime never got marked
    unavailable). Resume is the only control left, so it has to work.
    """
    from backend.models.schemas import TaskStatus
    from backend.routes.websockets import should_resume_workflow
    from backend.state import sessions

    token = _token(client)
    started = client.post(
        "/api/session/start",
        json={"instruction": "cluster the cells"},
        headers=_auth(token),
    )
    assert started.status_code == 200, started.text
    src_id = started.json()["session_id"]

    _populate(src_id, live_pkl=False)
    _advance(src_id, round_=0)
    client.post(f"/api/session/{src_id}/save", headers=_auth(token))

    blob = _export_download(client, token, src_id, include_runtime=False)
    client.delete(f"/api/session/{src_id}", headers=_auth(token))
    sessions.pop(src_id, None)

    new_id = _import_archive(client, token, blob, trust=False)
    state = sessions[new_id]
    assert state.status == TaskStatus.PAUSED, "nothing here owns that run"
    assert state.current_round == 0

    resumed = client.post(f"/api/session/{new_id}/resume", headers=_auth(token))
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["data"].get("runtime_unavailable") is not True, (
        "a session that never computed anything has no runtime to miss"
    )

    after = sessions[new_id]
    assert after.status == TaskStatus.RUNNING
    assert should_resume_workflow(after) is False, (
        "round 0 must start a fresh run; resume() refuses to invent one and "
        "would hold without executing anything"
    )
