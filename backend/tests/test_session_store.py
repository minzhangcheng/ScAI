"""Tests for session_store: save/load round-trip + list discovery."""
import asyncio
import base64
import json
import os
import time

import pytest
import pytest_asyncio

os.environ.setdefault("SCAI_DATA_DIR", "/tmp/scai_test_data")

from backend.services.session_store import (
    save_session_to_disk, load_session_from_disk,
    async_save_session, ensure_session_loaded, list_saved_sessions,
)
from backend.models.schemas import (
    AnalysisState,
    FigureReviewStatus,
    PublicationFigure,
)
from backend.models.schemas import TaskStatus
from backend.state import last_save_times, sessions
from backend.config import ensure_session_dirs, get_session_workspace
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes


def _make_state(session_id: str, user_id: str = "u-1") -> AnalysisState:
    return AnalysisState(id=session_id, user_id=user_id, name="t")


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    sid = "sess-roundtrip"
    ensure_session_dirs(sid)
    state = _make_state(sid)
    save_session_to_disk(sid, state)
    loaded = load_session_from_disk(sid)
    assert loaded is not None
    assert loaded.id == sid
    assert loaded.user_id == "u-1"


def test_load_rejects_empty_user_id(tmp_path, monkeypatch):
    """Simulate a legacy session file with empty user_id (written before constraint was added)."""
    import tempfile
    from backend.config import SESSION_DIR, get_session_metadata_path
    sid = "sess-no-user"
    ensure_session_dirs(sid)
    # Write a raw JSON file directly (bypassing AnalysisState validation)
    session_file = get_session_metadata_path(sid)
    with open(session_file, "w") as f:
        json.dump({"id": sid, "user_id": "", "name": "t"}, f)
    assert load_session_from_disk(sid) is None


@pytest.mark.asyncio
async def test_ensure_session_loaded_caches():
    sid = "sess-cache"
    ensure_session_dirs(sid)
    save_session_to_disk(sid, _make_state(sid))
    sessions.pop(sid, None)
    first = await ensure_session_loaded(sid)
    assert first is not None
    assert sid in sessions
    second = await ensure_session_loaded(sid)
    assert second is first


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["llm_failure", "agent_disabled", "marker_db_missing"]
)
async def test_load_clears_a_gate_only_a_live_workflow_could_release(reason):
    """A restart leaves no workflow, so these gates can never be answered.

    All three are resolved solely by ``POST /{id}/llm-failure-action``, which
    raises ``404 Workflow not found`` when ``workflows[session_id]`` is absent —
    and the WebSocket re-opens the dialog on attach, so the session presents
    buttons that cannot work. ``agent_disabled`` was missing from this cleanup and
    was therefore the one gate that never self-healed.
    """
    from backend.models.schemas import TaskStatus

    sid = f"sess-gate-{reason}"
    ensure_session_dirs(sid)
    state = _make_state(sid)
    state.status = TaskStatus.WAITING_INPUT
    state.wait_input_reason = reason
    state.llm_failure_info = {"node_name": "bioinformatics", "generation": 2}
    state.agent_disabled_info = {"agent": "pathology", "generation": 2}
    save_session_to_disk(sid, state)
    sessions.pop(sid, None)

    loaded = await ensure_session_loaded(sid)

    assert loaded is not None
    assert loaded.wait_input_reason is None
    assert loaded.llm_failure_info is None
    assert loaded.agent_disabled_info is None
    assert loaded.status == TaskStatus.PAUSED


def test_list_saved_sessions_sees_new_structure():
    sid = "sess-listed"
    ensure_session_dirs(sid)
    save_session_to_disk(sid, _make_state(sid))
    found = [s for s in list_saved_sessions() if s["id"] == sid]
    assert found and found[0]["user_id"] == "u-1"


@pytest.mark.asyncio
async def test_paused_state_bypasses_recent_save_throttle():
    sid = "sess-paused-force-save"
    ensure_session_dirs(sid)
    state = _make_state(sid)
    state.status = TaskStatus.RUNNING
    save_session_to_disk(sid, state)
    last_save_times[sid] = time.monotonic()

    state.status = TaskStatus.PAUSED
    await async_save_session(sid, state)

    loaded = load_session_from_disk(sid)
    assert loaded is not None
    assert loaded.status == TaskStatus.PAUSED


@pytest.mark.asyncio
async def test_loading_legacy_session_persists_publication_hash_backfill():
    sid = "sess-legacy-publication-backfill"
    ensure_session_dirs(sid)
    workspace = get_session_workspace(sid)
    figure_path = workspace / "figures" / "historical.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    image = valid_image_bytes("PNG", min_bytes=2200)
    figure_path.write_bytes(image)
    pdf_path = figure_path.with_suffix(".pdf")
    pdf_path.write_bytes(valid_pdf_bytes())
    state = _make_state(sid)
    state.publication_figures = [
        PublicationFigure(
            name="historical",
            description="Reviewed before integrity tracking",
            figure_type="other",
            png_path=str(figure_path),
            pdf_path=str(pdf_path),
            png_base64=base64.b64encode(image).decode(),
            review_status=FigureReviewStatus.APPROVED,
        )
    ]
    state.artifact_registry_version = None
    save_session_to_disk(sid, state)
    sessions.pop(sid, None)

    loaded = await ensure_session_loaded(sid)

    figure = loaded.publication_figures[0]
    assert figure.render_sha256
    assert figure.preview_sha256
    assert figure.pdf_sha256 is None
    sessions.pop(sid, None)
    persisted = load_session_from_disk(sid)
    assert persisted is not None
    persisted_figure = persisted.publication_figures[0]
    assert persisted_figure.render_sha256 == figure.render_sha256
    assert persisted_figure.preview_sha256 == figure.preview_sha256
    assert persisted_figure.pdf_sha256 is None
