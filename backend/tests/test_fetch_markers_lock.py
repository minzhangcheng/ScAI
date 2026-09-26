import re
import pathlib
import os


def test_marker_sources_are_https_named_hosts():
    """CellMarker used to be fetched from a bare IP, which cannot be authenticated.

    That is why the download had to choose between pinning a checksum and
    refusing to run -- and unpinned, it refused, so the supplementary source was
    never once available. Its own domain presents a valid TLS chain; going back
    to an IP, or to plain http, resurrects the whole dilemma.

    Asserts the constants' values rather than grepping the file, so the comments
    can still record the history that motivated this.
    """
    mod = _load_fetch_markers()
    for name in ("PANGLAODB_URL", "CELLMARKER_URL"):
        url = getattr(mod, name)
        assert url.startswith("https://"), f"{name} must use TLS: {url}"
        host = url.split("/")[2]
        assert not re.fullmatch(r"[\d.]+", host), (
            f"{name} points at raw IP {host}; TLS cannot be verified against it"
        )


def test_acquire_lock_stale_reclaim(tmp_path):
    """Integration test: stale lock is reclaimed atomically."""
    from importlib import util
    spec = util.spec_from_file_location(
        "fetch_markers_lock",
        pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lock = tmp_path / ".fetch.lock"
    lock.write_text("stale")
    # Set mtime far in the past
    old = 1_000_000_000  # 2001-09-09
    os.utime(lock, (old, old))

    fd = mod._acquire_lock(lock)
    assert fd is not None, "should reclaim stale lock"
    os.close(fd)
    assert lock.exists()


def test_acquire_lock_active_skips(tmp_path):
    """Fresh lock should be skipped, not reclaimed."""
    from importlib import util
    spec = util.spec_from_file_location(
        "fetch_markers_lock",
        pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lock = tmp_path / ".fetch.lock"
    lock.write_text("fresh")

    fd = mod._acquire_lock(lock)
    assert fd is None, "should skip fresh lock"


def _load_fetch_markers():
    from importlib import util
    spec = util.spec_from_file_location(
        "fetch_markers_mod",
        pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cellmarker_download_always_verifies_tls(monkeypatch):
    """verify=True is unconditional now that the source is an authenticated domain.

    A marker table steers cell-type annotation, so accepting one over an
    unauthenticated transport would let a network attacker choose cell labels.
    There is no availability argument left for relaxing this: the domain works.
    """
    mod = _load_fetch_markers()

    seen = {}

    def _fake_get(url, verify=True):
        seen["url"] = url
        seen["verify"] = verify
        return b"any-bytes"

    monkeypatch.setattr(mod, "_http_get_bytes", _fake_get)
    monkeypatch.setattr(mod.pd, "read_csv", lambda *_a, **_k: "PARSED")
    monkeypatch.setattr(mod, "project_to_schema", lambda raw, *a, **k: raw)

    assert mod._download_cellmarker() == "PARSED"
    assert seen["verify"] is True
    assert seen["url"].startswith("https://bio-bigdata.hrbmu.edu.cn/")


def test_cellmarker_reads_the_zip_wrapped_tsv_with_narrow_columns(monkeypatch):
    """CellMarker 3.0 ships a 148 MB TSV inside a zip and 22 columns.

    Reading it whole, or as .xlsx like the 2.0 path did, either fails outright or
    balloons memory in the backend process that fetches it.
    """
    mod = _load_fetch_markers()
    seen = {}

    monkeypatch.setattr(mod, "_http_get_bytes", lambda url, verify=True: b"zipbytes")

    def _fake_read_csv(buf, **kw):
        seen.update(kw)
        return "PARSED"

    monkeypatch.setattr(mod.pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr(mod, "project_to_schema", lambda raw, *a, **k: raw)

    mod._download_cellmarker()
    assert seen["compression"] == "zip"
    assert seen["sep"] == "\t"
    assert seen["usecols"] == mod.CELLMARKER_COLS
    assert set(mod.CELLMARKER_COLS) == {"species", "cell_name", "symbol", "tissue_type"}


def test_cellmarker_projection_shape(monkeypatch):
    """The upstream column names changed in 3.0 (Symbol -> symbol)."""
    import pandas as pd

    mod = _load_fetch_markers()
    raw = pd.DataFrame({
        "species": ["Mouse", "Human", "Mouse"],
        "cell_name": ["Kupffer cell", "Kupffer cell", ""],
        "symbol": ["Clec4f", "CLEC4F", "Foo"],
        "tissue_type": ["Liver", "Liver", "Liver"],
    })
    monkeypatch.setattr(mod, "_http_get_bytes", lambda url, verify=True: b"zipbytes")
    monkeypatch.setattr(mod.pd, "read_csv", lambda *_a, **_k: raw)

    out = mod._download_cellmarker()
    assert list(out.columns) == mod.SCHEMA_COLS
    assert set(out["species"]) == {"mouse", "human"}
    assert set(out["source"]) == {"CellMarker"}
    assert (out["tissue"] == "Liver").all()
    assert "Foo" not in set(out["gene_symbol"]), "rows with no cell type must drop"
