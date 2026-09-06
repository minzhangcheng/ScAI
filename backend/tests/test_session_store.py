"""Tests for session_store: save/load round-trip + list discovery."""
import asyncio
import base64
import copy
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
    WebSocketMessage,
)
from backend.models.schemas import TaskStatus
from backend.constants import STRIPPED_FIGURE_PLACEHOLDER, WaitInputReason
from backend.state import last_save_times, sessions
from backend.config import ensure_session_dirs, get_session_states, get_session_workspace
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes
from backend.utils.sandbox import NotebookState


def _make_state(session_id: str, user_id: str = "u-1") -> AnalysisState:
    return AnalysisState(id=session_id, user_id=user_id, name="t")


def test_durable_save_strips_nested_binary_before_blob_write_without_mutation(tmp_path, monkeypatch):
    """Session JSON and blob files receive a copy-only durable payload."""
    sid = "durable-binary-session"
    ensure_session_dirs(sid)
    image = "iVBORw0KGgo" + ("A" * 4096)
    state = _make_state(sid)
    state.realtime_messages = [
        WebSocketMessage(
            type="graphic_abstract",
            content={
                "image_base64": image,
                "nested": {
                    "png_base64": image,
                    "raw_valid": "utf8-bytes".encode("utf-8"),
                    "raw_invalid": b"\xff\xfe",
                },
            },
        )
    ]
    original = copy.deepcopy(state.model_dump(mode="python"))

    save_session_to_disk(sid, state)

    session_file = get_session_states(sid).parent / "session.json"
    raw_session = session_file.read_text(encoding="utf-8")
    persisted = json.loads(raw_session)
    persisted_content = persisted["realtime_messages"][0]["content"]
    blob_dir = session_file.parent / "blobs"
    blob_payloads = [p.read_bytes() for p in blob_dir.glob("*.b64")] if blob_dir.exists() else []
    assert image not in raw_session
    assert persisted_content["nested"]["raw_valid"] == STRIPPED_FIGURE_PLACEHOLDER
    assert persisted_content["nested"]["raw_invalid"] == STRIPPED_FIGURE_PLACEHOLDER
    assert "utf8-bytes" not in raw_session
    assert all(image.encode("utf-8") not in payload for payload in blob_payloads)
    assert state.model_dump(mode="python") == original


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    sid = "sess-roundtrip"
    ensure_session_dirs(sid)
    state = _make_state(sid)
    save_session_to_disk(sid, state)
    loaded = load_session_from_disk(sid)
    assert loaded is not None
    assert loaded.id == sid
    assert loaded.user_id == "u-1"


def test_session_projection_uses_newer_notebook_runtime_metadata():
    sid = "sess-runtime-projection"
    ensure_session_dirs(sid)
    states = get_session_states(sid)
    notebook = NotebookState(sid, str(states))
    notebook.update_runtime_persistence(
        "current",
        current_revision=122,
        saved_revision=122,
        commit_source="regular_save",
    )
    notebook.update_runtime_persistence(
        "live_dirty",
        current_revision=123,
        saved_revision=122,
    )

    state = _make_state(sid)
    state.runtime_persistence.update({
        "status": "live_dirty",
        "current_revision": 95,
        "saved_revision": 92,
    })
    original_projection = dict(state.runtime_persistence)

    save_session_to_disk(sid, state)

    persisted = json.loads((states.parent / "session.json").read_text())
    projection = persisted["runtime_persistence"]
    assert projection["source"] == "notebook_state"
    assert projection["projection_status"] == "current"
    assert projection["status"] == "live_dirty"
    assert projection["current_revision"] == 123
    assert projection["saved_revision"] == 122
    assert projection["last_attempt_status"] == "succeeded"
    assert projection["observed_at"]
    assert state.runtime_persistence == original_projection

    loaded = load_session_from_disk(sid)
    assert loaded is not None
    assert loaded.runtime_persistence["current_revision"] == 123
    assert loaded.runtime_persistence["saved_revision"] == 122


def test_runtime_projection_fails_closed_on_lifecycle_mismatch():
    sid = "sess-runtime-projection-generation"
    ensure_session_dirs(sid)
    states = get_session_states(sid)
    notebook = NotebookState(sid, str(states), lifecycle_generation=4)
    notebook.update_runtime_persistence(
        "current",
        current_revision=8,
        saved_revision=8,
        commit_source="regular_save",
    )

    state = _make_state(sid)
    state.lifecycle_generation = 5
    state.runtime_persistence["current_revision"] = 3
    state.runtime_persistence["saved_revision"] = 3

    save_session_to_disk(sid, state)

    persisted = json.loads((states.parent / "session.json").read_text())
    projection = persisted["runtime_persistence"]
    assert projection["current_revision"] == 3
    assert projection["saved_revision"] == 3
    assert projection["projection_status"] == "lifecycle_mismatch"
    assert "generation 4" in projection["projection_error"]

    # The mismatch is recorded and disclosed, not held at a dialog: spec 5.1
    # replaced the runtime-persistence HITL gate with a persistent
    # non-blocking warning, and a commit belonging to another incarnation is
    # not something a user action could resolve.
    loaded = load_session_from_disk(sid)
    assert loaded is not None
    assert loaded.status != TaskStatus.WAITING_INPUT
    assert loaded.runtime_persistence["projection_status"] == "lifecycle_mismatch"
    assert loaded.runtime_persistence["status"] == "failed"
    assert "generation 4" in loaded.runtime_persistence["projection_error"]


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
