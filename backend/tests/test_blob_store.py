"""Tests for base64 blob externalization used by session persistence."""
import json
import pytest
from backend.utils.blob_store import externalize_blobs, inline_blobs, BLOB_REF_PREFIX, MIN_BLOB_LEN
from backend.models.schemas import AnalysisState, PublicationFigure, WebSocketMessage
from backend.constants import STRIPPED_FIGURE_PLACEHOLDER


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
    digest = "d" * 64  # valid SHA-256 hex length
    obj = {"k": f"{BLOB_REF_PREFIX}{digest}"}  # reference with no file
    # Missing blob must not crash load; reference is left as-is.
    assert inline_blobs(obj, blob_dir) == {"k": f"{BLOB_REF_PREFIX}{digest}"}


def test_session_roundtrip_strips_base64_from_json(tmp_path, monkeypatch):
    import backend.services.session_store as ss

    sid = "11111111-1111-1111-1111-111111111111"
    # Route persistence at tmp_path
    session_file = tmp_path / sid / "session.json"
    monkeypatch.setattr(ss, "get_session_metadata_path", lambda s: session_file)
    monkeypatch.setattr(ss, "resolve_session_metadata_path", lambda s: session_file)

    big = "Z" * 5000
    state = AnalysisState(id=sid, user_id="u")
    state.publication_figures.append(
        PublicationFigure(name="f1", description="d", figure_type="umap",
                          png_path="", pdf_path="", png_base64=big)
    )

    ss.save_session_to_disk(sid, state)

    raw = session_file.read_text(encoding="utf-8")
    assert big not in raw                       # base64 no longer inline

    # A figure's preview no longer reaches the blob store at all: the durable
    # dump replaces it with STRIPPED_FIGURE_PLACEHOLDER and records a
    # preview_sha256 beside it, so there is nothing left for externalize_blobs
    # to write. The "@scai-blob:" handle survives only as a legacy form that
    # inline_blobs still reads back -- see _strip_legacy_blob_references. The
    # externalize/inline round trip itself is covered by the other tests in
    # this file, which drive blob_store directly. preview_sha256 is not
    # asserted here: it is only computed for a decodable image, and this
    # fixture's payload is filler chosen for its length.
    loaded = ss.load_session_from_disk(sid)
    assert loaded is not None
    figure = loaded.publication_figures[0]
    assert figure.png_base64 == STRIPPED_FIGURE_PLACEHOLDER


def test_session_roundtrip_restores_realtime_text_blobs(tmp_path, monkeypatch):
    import backend.services.session_store as ss

    sid = "round-summary-text-session"
    session_file = tmp_path / sid / "session.json"
    monkeypatch.setattr(ss, "get_session_metadata_path", lambda s: session_file)
    monkeypatch.setattr(ss, "resolve_session_metadata_path", lambda s: session_file)

    summary = "Round summary text. " * 80
    integration_summary = "Integration report text. " * 100
    question = json.dumps({
        "question": "Question text. " * 100,
        "context": "Context text. " * 100,
        "options": [],
    })
    response = "PI response text. " * 100
    code = "print('analysis')\n" * 100
    state = AnalysisState(id=sid, user_id="u")
    state.realtime_messages = [
        WebSocketMessage(
            type="round_summary",
            content={
                "round": 4,
                "phase": "integration",
                "summary": summary,
                "integration_report_summary": integration_summary,
            },
        ),
        WebSocketMessage(type="pi_question", content=question),
        WebSocketMessage(type="pi_response", content=response),
        WebSocketMessage(type="code", content=code),
        WebSocketMessage(type="figure", content="iVBORw0KGgo" + ("A" * 2048)),
    ]

    ss.save_session_to_disk(sid, state)

    persisted_messages = json.loads(session_file.read_text(encoding="utf-8"))[
        "realtime_messages"
    ]
    persisted_summary = persisted_messages[0]["content"]
    assert persisted_summary["summary"].startswith(BLOB_REF_PREFIX)
    assert persisted_summary["integration_report_summary"].startswith(BLOB_REF_PREFIX)
    assert persisted_messages[1]["content"].startswith(BLOB_REF_PREFIX)
    assert persisted_messages[2]["content"].startswith(BLOB_REF_PREFIX)
    assert persisted_messages[3]["content"].startswith(BLOB_REF_PREFIX)
    assert persisted_messages[4]["content"] == STRIPPED_FIGURE_PLACEHOLDER

    loaded = ss.load_session_from_disk(sid)
    assert loaded is not None
    loaded_summary = loaded.realtime_messages[0].content
    assert loaded_summary["summary"] == summary
    assert loaded_summary["integration_report_summary"] == integration_summary
    assert loaded.realtime_messages[1].content == question
    assert loaded.realtime_messages[2].content == response
    assert loaded.realtime_messages[3].content == code
    assert loaded.realtime_messages[4].content == STRIPPED_FIGURE_PLACEHOLDER


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

    state = AnalysisState(id=sid, user_id="u")
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
