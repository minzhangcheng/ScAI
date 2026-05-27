"""Tests for base64 blob externalization used by session persistence."""
import json
import pytest
from backend.utils.blob_store import externalize_blobs, inline_blobs, BLOB_REF_PREFIX, MIN_BLOB_LEN
from backend.models.schemas import AnalysisState, PublicationFigure


def _long_b64(n=MIN_BLOB_LEN + 100):
    # Valid-ish base64 payload longer than the threshold.
    return "A" * n


def test_roundtrip_preserves_nested_structure(tmp_path):
    big = _long_b64()
    obj = {
        "small": "keep me",
        "figs": [{"png_base64": big}, {"png_base64": big}],  # duplicate -> dedup
        "nested": {"image_base64": big, "n": 3, "flag": True, "none": None},
    }
    blob_dir = tmp_path / "blobs"
    externalized = externalize_blobs(obj, blob_dir)

    # Long strings are replaced by references; short ones untouched.
    assert externalized["small"] == "keep me"
    assert externalized["figs"][0]["png_base64"].startswith(BLOB_REF_PREFIX)
    assert externalized["nested"]["n"] == 3
    # Dedup: identical content -> identical reference -> one file.
    assert externalized["figs"][0]["png_base64"] == externalized["figs"][1]["png_base64"]
    assert len(list(blob_dir.glob("*.b64"))) == 1  # all three big strings share one hash

    inlined = inline_blobs(externalized, blob_dir)
    assert inlined == obj


def test_short_strings_not_externalized(tmp_path):
    obj = {"a": "x" * (MIN_BLOB_LEN - 1)}
    out = externalize_blobs(obj, tmp_path / "blobs")
    assert out["a"] == obj["a"]
    assert not (tmp_path / "blobs").exists()


def test_inline_tolerates_missing_blob_file(tmp_path):
    blob_dir = tmp_path / "blobs"
    obj = {"k": f"{BLOB_REF_PREFIX}deadbeef"}  # reference with no file
    # Missing blob must not crash load; reference is left as-is.
    assert inline_blobs(obj, blob_dir) == {"k": f"{BLOB_REF_PREFIX}deadbeef"}


def test_session_roundtrip_strips_base64_from_json(tmp_path, monkeypatch):
    import backend.services.session_store as ss

    sid = "11111111-1111-1111-1111-111111111111"
    # Route persistence at tmp_path
    session_file = tmp_path / sid / "session.json"
    monkeypatch.setattr(ss, "get_session_metadata_path", lambda s: session_file)
    monkeypatch.setattr(ss, "resolve_session_metadata_path", lambda s: session_file)

    big = "Z" * 5000
    state = AnalysisState(user_id="u")
    state.publication_figures.append(
        PublicationFigure(name="f1", description="d", figure_type="umap",
                          png_path="", pdf_path="", png_base64=big)
    )

    ss.save_session_to_disk(sid, state)

    raw = session_file.read_text(encoding="utf-8")
    assert big not in raw                       # base64 no longer inline
    assert "@scai-blob:" in raw                 # reference present
    assert (session_file.parent / "blobs").exists()

    loaded = ss.load_session_from_disk(sid)
    assert loaded is not None
    assert loaded.publication_figures[0].png_base64 == big   # lossless rehydrate


def test_async_save_offloads_blob_io_to_worker_thread(tmp_path, monkeypatch):
    """externalize_blobs (which writes .b64 files) must run in a worker thread,
    not on the event loop, during async_save_session."""
    import asyncio
    import threading
    import backend.services.session_store as ss
    from backend.models.schemas import AnalysisState, PublicationFigure

    sid = "22222222-2222-2222-2222-222222222222"
    session_file = tmp_path / sid / "session.json"
    monkeypatch.setattr(ss, "get_session_metadata_path", lambda s: session_file)
    monkeypatch.setattr(ss, "resolve_session_metadata_path", lambda s: session_file)
    ss.last_save_times.pop(sid, None)

    main_thread = threading.get_ident()
    recorded = {}
    real_externalize = ss.externalize_blobs

    def spy(obj, blob_dir):
        recorded["thread"] = threading.get_ident()
        return real_externalize(obj, blob_dir)

    monkeypatch.setattr(ss, "externalize_blobs", spy)

    state = AnalysisState(user_id="u")
    state.publication_figures.append(
        PublicationFigure(name="f1", description="d", figure_type="umap",
                          png_path="", pdf_path="", png_base64="Z" * 5000)
    )

    asyncio.run(ss.async_save_session(sid, state, force=True))

    assert "thread" in recorded, "externalize_blobs was never called"
    assert recorded["thread"] != main_thread, "blob I/O ran on the event loop thread"
    assert session_file.exists()


def test_blob_write_is_atomic_on_crash(tmp_path, monkeypatch):
    """If os.replace fails (simulating a crash mid-write), no corrupt final file
    should be left behind — the old direct-write path would leave a half-written
    .b64 file."""
    import backend.utils.blob_store as bs

    blob_dir = tmp_path / "blobs"
    big = "A" * (bs.MIN_BLOB_LEN + 10)
    digest = bs._hash(big)

    # Make os.replace raise — simulates crash during rename
    def boom(*a, **k):
        raise OSError("crash during rename")
    monkeypatch.setattr(bs.os, "replace", boom)

    with pytest.raises(OSError):
        bs.externalize_blobs({"x": big}, blob_dir)

    # No corrupt final file should exist
    assert not (blob_dir / f"{digest}.b64").exists()

    # After undo, a normal write should succeed
    monkeypatch.undo()
    out = bs.externalize_blobs({"x": big}, blob_dir)
    assert out["x"].startswith(bs.BLOB_REF_PREFIX)
    assert (blob_dir / f"{digest}.b64").read_text() == big
