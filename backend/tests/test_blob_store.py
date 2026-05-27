"""Tests for base64 blob externalization used by session persistence."""
import json
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
