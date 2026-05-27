"""Snapshot blob externalization must run off the event loop.

create_snapshot calls externalize_blobs + json.dumps + file write; all of
that blocking work must happen in a worker thread so the event loop stays
responsive for WebSocket pushes and LLM streaming.
"""
import asyncio
import json
import threading

import pytest

from backend.models.schemas import AnalysisState, PublicationFigure
from backend.utils.blob_store import MIN_BLOB_LEN, BLOB_REF_PREFIX
from backend.utils.snapshot import SnapshotManager


def _make_state_with_big_field():
    """Minimal AnalysisState with one large base64 field that triggers blob externalization."""
    state = AnalysisState(user_id="u")
    state.publication_figures.append(
        PublicationFigure(
            name="f1", description="d", figure_type="umap",
            png_path="", pdf_path="", png_base64="A" * (MIN_BLOB_LEN + 100),
        )
    )
    return state


@pytest.mark.asyncio
async def test_snapshot_blob_io_offloaded_to_worker_thread(tmp_path, monkeypatch):
    """externalize_blobs during create_snapshot must run on a worker thread."""
    import backend.utils.snapshot as snap_mod

    sid = "33333333-3333-3333-3333-333333333333"
    session_dir = tmp_path / sid
    session_dir.mkdir()
    states_dir = session_dir / "states"
    states_dir.mkdir()

    # Monkey-patch path helpers on the snapshot module (where they are imported)
    monkeypatch.setattr(snap_mod, "get_session_dir", lambda s: session_dir)
    monkeypatch.setattr(snap_mod, "get_session_states", lambda s: states_dir)

    main_thread = threading.get_ident()
    recorded = {}
    real_externalize_fn = snap_mod.externalize_blobs

    def spy(obj, blob_dir):
        recorded["thread"] = threading.get_ident()
        return real_externalize_fn(obj, blob_dir)

    monkeypatch.setattr(snap_mod, "externalize_blobs", spy)

    mgr = SnapshotManager(sid)
    state = _make_state_with_big_field()

    snapshot = await mgr.create_snapshot(state, name="test", phase="manual")

    assert "thread" in recorded, "externalize_blobs was never called"
    assert recorded["thread"] != main_thread, (
        f"blob I/O ran on the event loop thread "
        f"(main={main_thread}, worker={recorded['thread']})"
    )
    assert snapshot.size_bytes > 0

    # Verify the JSON file was actually written
    state_path = session_dir / "snapshots" / snapshot.state_file
    assert state_path.exists(), "snapshot JSON file not written"
    raw = state_path.read_text(encoding="utf-8")
    assert BLOB_REF_PREFIX in raw, "blob reference not found in snapshot JSON"

    # Verify blob file exists
    blob_dir = session_dir / "snapshots" / f"{snapshot.id}_blobs"
    assert blob_dir.exists(), "blob directory not created"
    b64_files = list(blob_dir.glob("*.b64"))
    assert len(b64_files) == 1, f"expected 1 blob file, got {len(b64_files)}"