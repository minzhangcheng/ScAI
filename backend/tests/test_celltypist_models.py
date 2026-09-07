"""The CellTypist cache is host-fetched runtime data, exposed read-only to kernels.

CellTypist fixes its model directory from ``CELLTYPIST_FOLDER`` at import time
and creates it, so a kernel can only import the package when that folder lies
inside a write root. The host keeps one cache on the data volume and links it
into a per-session view; these tests pin the location rule, the view's
semantics (links refreshed, session-owned files kept, junk excluded) and the
fetch script's environment contract.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from backend.services import celltypist_models as ctm

_REPO = Path(__file__).resolve().parent.parent.parent


def _load_fetch_script():
    """scripts/ is not a package; load by path the way the CLI runs it."""
    path = _REPO / "scripts" / "fetch_celltypist_models.py"
    spec = importlib.util.spec_from_file_location("fetch_celltypist_models_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fill_cache(root: Path, names) -> Path:
    models_dir = root / "data" / "models"
    models_dir.mkdir(parents=True)
    for name in names:
        (models_dir / name).write_bytes(b"x")
    return models_dir


def test_cache_root_is_under_the_data_volume(monkeypatch, tmp_path):
    import backend.config

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path)
    assert ctm.runtime_cache_root() == tmp_path / "celltypist"
    assert ctm.cached_models_dir() == tmp_path / "celltypist" / "data" / "models"


def test_cache_root_is_outside_the_code_tree():
    resolved = ctm.runtime_cache_root().resolve()
    assert not resolved.is_relative_to(_REPO), (
        f"celltypist models would live inside the repo at {resolved}"
    )


def test_fetch_script_honours_scai_data_dir(monkeypatch, tmp_path):
    """The script cannot import backend, so it duplicates the SCAI_DATA_DIR rule.

    Same pin as test_marker_runtime_location: the backend resolves
    ``$SCAI_DATA_DIR`` and appends the cache name, and the script must land on
    exactly that directory or the kernels look where nothing was fetched.
    """
    monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path))
    script = _load_fetch_script()
    assert script.default_dest_dir() == tmp_path.resolve() / "celltypist"


def test_fetch_script_fallback_matches_the_backend_fallback(monkeypatch):
    """With SCAI_DATA_DIR unset both sides must land on <repo>/../data/celltypist."""
    monkeypatch.delenv("SCAI_DATA_DIR", raising=False)
    script = _load_fetch_script()
    assert script.default_dest_dir() == (_REPO.parent / "data").resolve() / "celltypist"


def test_cached_model_files_exposes_models_and_index_only(monkeypatch, tmp_path):
    import backend.config

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path)
    models_dir = _fill_cache(
        tmp_path / "celltypist",
        ["B_model.pkl", "A_model.pkl", "models.json", "NOTICE", ".fetch.lock"],
    )
    (models_dir / "subdir").mkdir()

    names = [p.name for p in ctm.cached_model_files()]
    assert names == ["A_model.pkl", "B_model.pkl", "models.json"]
    assert ctm.cached_model_names() == ["A_model.pkl", "B_model.pkl"]


def test_cached_model_files_is_empty_without_a_cache(monkeypatch, tmp_path):
    import backend.config

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path / "absent")
    assert ctm.cached_model_files() == []
    assert ctm.cached_model_names() == []


def test_link_cache_into_builds_a_symlink_view(monkeypatch, tmp_path):
    import backend.config

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path / "data")
    models_dir = _fill_cache(tmp_path / "data" / "celltypist", ["M.pkl", "models.json"])
    session = tmp_path / "session" / "celltypist"

    linked = ctm.link_cache_into(session)

    view = session / "data" / "models"
    assert linked == ["M.pkl", "models.json"]
    assert (view / "M.pkl").is_symlink()
    assert (view / "M.pkl").resolve() == (models_dir / "M.pkl").resolve()
    assert (view / "M.pkl").read_bytes() == b"x"
    assert (view / "models.json").is_symlink()


def test_link_cache_into_refreshes_links_and_keeps_session_files(monkeypatch, tmp_path):
    """A stale link goes; a model the session downloaded itself shadows the cache."""
    import backend.config

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path / "data")
    models_dir = _fill_cache(tmp_path / "data" / "celltypist", ["Old.pkl", "Shared.pkl"])
    session = tmp_path / "session" / "celltypist"
    ctm.link_cache_into(session)
    view = session / "data" / "models"
    (models_dir / "Old.pkl").unlink()
    (view / "Shared.pkl").unlink()
    (view / "Shared.pkl").write_bytes(b"session copy")
    (view / "Own.pkl").write_bytes(b"downloaded in-session")
    (models_dir / "New.pkl").write_bytes(b"x")

    linked = ctm.link_cache_into(session)

    assert linked == ["New.pkl"]
    assert not (view / "Old.pkl").exists() and not (view / "Old.pkl").is_symlink()
    assert (view / "New.pkl").is_symlink()
    assert not (view / "Shared.pkl").is_symlink()
    assert (view / "Shared.pkl").read_bytes() == b"session copy"
    assert (view / "Own.pkl").read_bytes() == b"downloaded in-session"


def test_link_cache_into_without_a_cache_still_prepares_the_view(monkeypatch, tmp_path):
    import backend.config

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path / "absent")
    session = tmp_path / "session" / "celltypist"

    assert ctm.link_cache_into(session) == []
    assert (session / "data" / "models").is_dir()


class _FakeCelltypist(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """A celltypist stand-in that is IMPORTED, not pre-seeded.

    The real package fixes its model directory from CELLTYPIST_FOLDER while its
    module body runs, so the only faithful check is what the variable held at
    the moment the import executed. A module planted in sys.modules never runs
    a body and would let the export move after the import unnoticed.
    """

    def __init__(self, dest: Path, download=None):
        self.dest = dest
        self.seen = {}
        self.download = download

    def find_spec(self, name, path=None, target=None):
        if name in ("celltypist", "celltypist.models"):
            return importlib.util.spec_from_loader(
                name, self, is_package=(name == "celltypist")
            )
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.seen.setdefault("env_at_import", os.environ.get("CELLTYPIST_FOLDER"))
        if module.__name__ == "celltypist.models":
            module.models_path = str(self.dest / "data" / "models")
            module.download_models = self.download or (
                lambda force_update=False, model=None: None
            )

    def install(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "celltypist", raising=False)
        monkeypatch.delitem(sys.modules, "celltypist.models", raising=False)
        monkeypatch.setattr(sys, "meta_path", [self] + sys.meta_path)


def test_fetch_script_exports_the_folder_before_importing_celltypist(monkeypatch, tmp_path):
    """celltypist reads CELLTYPIST_FOLDER at import, so the export must come first."""
    dest = tmp_path / "celltypist"
    seen = {}

    def download_models(force_update=False, model=None):
        seen["env_at_download"] = os.environ.get("CELLTYPIST_FOLDER")
        seen["force_update"] = force_update
        seen["model"] = model
        models_dir = dest / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "Fetched.pkl").write_bytes(b"x")

    fake = _FakeCelltypist(dest, download_models)
    fake.install(monkeypatch)
    monkeypatch.delenv("CELLTYPIST_FOLDER", raising=False)

    script = _load_fetch_script()
    ok = script.fetch(dest, models=["Fetched.pkl"], force=True)

    assert ok is True
    assert fake.seen == {"env_at_import": str(dest)}
    assert seen == {
        "env_at_download": str(dest),
        "force_update": True,
        "model": ["Fetched.pkl"],
    }
    assert (dest / "NOTICE").read_text() == script.NOTICE_TEXT
    assert not (dest / ".fetch.lock").exists()


def test_fetch_script_reports_an_empty_download_as_failure(monkeypatch, tmp_path):
    dest = tmp_path / "celltypist"
    _FakeCelltypist(dest).install(monkeypatch)
    monkeypatch.delenv("CELLTYPIST_FOLDER", raising=False)

    script = _load_fetch_script()
    assert script.fetch(dest) is False
    assert not (dest / ".fetch.lock").exists()


def test_fetch_script_yields_to_a_running_fetch(monkeypatch, tmp_path):
    monkeypatch.delenv("CELLTYPIST_FOLDER", raising=False)
    script = _load_fetch_script()
    dest = tmp_path / "celltypist"
    dest.mkdir()
    (dest / ".fetch.lock").write_text("")

    assert script.fetch(dest) is False
    assert (dest / ".fetch.lock").exists()
