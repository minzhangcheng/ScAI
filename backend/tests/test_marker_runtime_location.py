"""The marker tables are runtime cache, not shipped content.

They used to live in ``backend/data/markers/``. Two properties of that location
combined badly: ``COPY . .`` in the Dockerfile captured whatever table the build
host happened to have, and the availability check was a bare
``Path.exists()`` — so nothing ever re-downloaded it. A table fetched on
2026-07-18 by a projection that dropped 90.6% of PanglaoDB therefore kept
serving cluster annotation from ~9% of the database across every later release
that fixed the projection.

These tests pin the fix: the tables resolve under the data volume, no copy can
ride along in the code tree or the image, and a table from an older projection
reads as absent so it gets replaced.
"""
import importlib.util
from pathlib import Path

import pytest

from backend.services import marker_priors

_REPO = Path(__file__).resolve().parent.parent.parent


def _load_fetch_markers():
    """scripts/ is not a package; load by path the way marker_priors does."""
    path = _REPO / "scripts" / "fetch_markers.py"
    spec = importlib.util.spec_from_file_location("fetch_markers_location", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fm = _load_fetch_markers()


def test_runtime_dir_is_under_the_data_volume():
    from backend.config import DATA_BASE_DIR

    assert marker_priors.runtime_markers_dir() == Path(DATA_BASE_DIR) / "markers"


def test_runtime_dir_is_outside_the_code_tree():
    """The whole point: a release must not be able to carry a table with it."""
    resolved = marker_priors.runtime_markers_dir().resolve()
    assert not resolved.is_relative_to(_REPO), (
        f"marker tables would live inside the repo at {resolved}"
    )


def test_runtime_dir_follows_scai_data_dir(monkeypatch, tmp_path):
    import backend.config

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path)
    assert marker_priors.runtime_markers_dir() == tmp_path / "markers"


def test_fetch_script_honours_scai_data_dir(monkeypatch, tmp_path):
    """The script cannot import backend, so it duplicates the SCAI_DATA_DIR rule.

    Duplicated rules drift; this and the fallback test below are the pin. If they
    fail, the backend looks in one directory while the fetch writes to another
    and DB priors are permanently unavailable with no error anywhere.
    """
    monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path))
    assert fm.default_dest_dir() == tmp_path.resolve() / "markers"


def test_fetch_script_fallback_matches_the_backend_fallback(monkeypatch):
    """With SCAI_DATA_DIR unset both sides must land on <repo>/../data/markers.

    backend.config spells it ``backend/config/../../../data``; the script spells
    it ``scripts/../../data``. Both mean the repo's sibling ``data`` directory.
    """
    monkeypatch.delenv("SCAI_DATA_DIR", raising=False)
    assert fm.default_dest_dir() == (_REPO.parent / "data").resolve() / "markers"


def test_no_marker_table_is_committed_or_copyable_into_the_image():
    strays = [p for p in _REPO.rglob("panglaodb.tsv") if ".git" not in p.parts]
    strays += [p for p in _REPO.rglob("cellmarker.tsv") if ".git" not in p.parts]
    assert not strays, (
        f"marker tables found inside the code tree: {strays}. They belong under "
        "$SCAI_DATA_DIR/markers; a copy here gets baked into the container image."
    )


def test_backend_data_is_excluded_from_the_docker_build_context():
    """Belt to the braces above: even a stray file must not reach the image.

    Note the trailing-slash patterns are per-directory: ``.dockerignore``'s
    existing ``data/`` entry matches only a top-level ``data/`` and never
    ``backend/data/``, which is why these have to be spelled out.
    """
    text = (_REPO / ".dockerignore").read_text(encoding="utf-8")
    assert "backend/data/markers/" in text
    assert "backend/data/gene_sets/" in text


def test_gene_set_cache_is_also_outside_the_code_tree():
    """Same class of bug: a downloaded GMT under backend/data/ ships in the image."""
    from backend.config import DATA_BASE_DIR
    from backend.services.gene_set_resolver import runtime_gene_sets_dir

    assert runtime_gene_sets_dir() == Path(DATA_BASE_DIR) / "gene_sets"
    assert not runtime_gene_sets_dir().resolve().is_relative_to(_REPO)


def test_gene_set_fetch_defaults_to_the_runtime_cache(monkeypatch, tmp_path):
    """load_collection's ``cache_dir=None`` path must land on the data volume.

    It is the only caller of the default, so if it kept resolving into the code
    tree the move would be cosmetic.
    """
    import backend.config
    from backend.services import gene_set_fetch

    monkeypatch.setattr(backend.config, "DATA_BASE_DIR", tmp_path)
    seen = {}

    def _fake_cache_path(cache_dir, *a, **k):
        seen["dir"] = Path(cache_dir)
        return tmp_path / "absent.gmt"          # miss -> would go on to fetch

    monkeypatch.setattr(gene_set_fetch, "cache_path", _fake_cache_path)
    monkeypatch.setattr(gene_set_fetch, "_get_text",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no network")))

    with pytest.raises(Exception):
        gene_set_fetch.load_collection("2024.1.Hs", "c2.cp.reactome", "human")

    assert seen["dir"] == tmp_path / "gene_sets"
    assert (tmp_path / "gene_sets").is_dir(), "the runtime cache dir must be created"


def test_no_gene_set_cache_is_committed():
    strays = [p for p in _REPO.rglob("*.symbols.gmt") if ".git" not in p.parts]
    assert not strays, f"cached GMTs found inside the code tree: {strays}"


def test_schema_revisions_are_bumped_together():
    assert fm.SCHEMA_REVISION == marker_priors.MARKER_SCHEMA_REVISION, (
        "fetch_markers stamps SCHEMA_REVISION and marker_priors validates it; "
        "bumping only one silently disables DB priors (writer and reader disagree)"
    )


def _write_current(tmp_path, name):
    fm.atomic_write_tsv(
        __import__("pandas").DataFrame([{c: "" for c in fm.SCHEMA_COLS}]),
        tmp_path / name, source_version="vX")


def test_fetch_skips_the_network_when_every_table_is_already_current(tmp_path, monkeypatch):
    def _explode():
        raise AssertionError("must not download when the tables are current")

    monkeypatch.setattr(fm, "_download_panglaodb", _explode)
    monkeypatch.setattr(fm, "_download_cellmarker", _explode)
    for name in fm.EXPECTED_TABLES:
        _write_current(tmp_path, name)

    assert fm.fetch_and_write(tmp_path) is True


def test_fetch_retries_a_supplementary_source_that_failed_last_time(tmp_path, monkeypatch):
    """A current primary must not make the supplementary source unfetchable.

    The short-circuit used to be keyed on panglaodb.tsv alone, so the first run
    that downloaded PanglaoDB and lost CellMarker stamped the primary current
    and every later call returned True without a request. MarkerDB.usable() is
    an any() over the two names, so nothing else ever noticed: the deployment
    ran on PanglaoDB-only priors forever.
    """
    import pandas as pd

    _write_current(tmp_path, "panglaodb.tsv")
    assert not (tmp_path / "cellmarker.tsv").exists()

    primary_calls, supplementary_calls = [], []
    monkeypatch.setattr(fm, "_download_panglaodb",
                        lambda: primary_calls.append(1) or pd.DataFrame(
                            [{c: "" for c in fm.SCHEMA_COLS}]))
    monkeypatch.setattr(fm, "_download_cellmarker",
                        lambda: supplementary_calls.append(1) or pd.DataFrame(
                            [{c: "" for c in fm.SCHEMA_COLS}]))

    assert fm.fetch_and_write(tmp_path) is True
    assert supplementary_calls, "the missing supplementary table must be re-fetched"
    assert fm.is_current_revision(tmp_path / "cellmarker.tsv")
    assert not primary_calls, (
        "the primary is already current; re-fetching it is wasted bandwidth"
    )
    assert (tmp_path / "LICENSE").exists(), (
        "a supplementary-only pass still has to leave the attribution in place"
    )


def test_fetch_keeps_returning_true_when_the_supplementary_source_stays_down(
    tmp_path, monkeypatch,
):
    """Retrying is not the same as blocking: CellMarker remains best-effort."""
    import pandas as pd

    _write_current(tmp_path, "panglaodb.tsv")
    monkeypatch.setattr(fm, "_download_panglaodb",
                        lambda: pd.DataFrame([{c: "" for c in fm.SCHEMA_COLS}]))
    monkeypatch.setattr(fm, "_download_cellmarker",
                        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))

    assert fm.fetch_and_write(tmp_path) is True
    assert not (tmp_path / "cellmarker.tsv").exists()


def test_fetch_redownloads_a_stale_revision(tmp_path, monkeypatch):
    """The failure mode this whole change exists to prevent."""
    import pandas as pd

    (tmp_path / "panglaodb.tsv").write_text(
        "# SOURCE_VERSION=PanglaoDB_27_Mar_2020 DOWNLOAD_DATE=2026-07-18\n"
        "source\tspecies\tcell_type\tgene_symbol\ttissue\tevidence\tref\n"
        "PanglaoDB\tmouse\tAcinar cells\tTRY4\t\t\t\n"
    )
    calls = []

    def _fake_panglaodb():
        calls.append(1)
        return pd.DataFrame([{
            "source": "PanglaoDB", "species": "mouse", "cell_type": "Kupffer cells",
            "gene_symbol": "CLEC4F", "tissue": "Liver", "evidence": "", "ref": "",
        }])

    monkeypatch.setattr(fm, "_download_panglaodb", _fake_panglaodb)
    monkeypatch.setattr(fm, "_download_cellmarker",
                        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))

    assert fm.fetch_and_write(tmp_path) is True
    assert calls, "a stale-revision table must trigger a re-download"

    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.usable() is True
    assert {r["gene_symbol"] for r in db.markers_for("mouse")} == {"CLEC4F"}


def test_fetch_force_redownloads_a_current_table(tmp_path, monkeypatch):
    """HITL retry must be able to replace a file that merely looks current."""
    import pandas as pd

    fm.atomic_write_tsv(
        pd.DataFrame([{c: "" for c in fm.SCHEMA_COLS}]),
        tmp_path / "panglaodb.tsv", source_version="vX")
    calls = []
    monkeypatch.setattr(fm, "_download_panglaodb",
                        lambda: calls.append(1) or pd.DataFrame(
                            [{c: "" for c in fm.SCHEMA_COLS}]))
    monkeypatch.setattr(fm, "_download_cellmarker",
                        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))

    assert fm.fetch_and_write(tmp_path, force=True) is True
    assert calls


def test_fetch_writes_the_data_licence_next_to_the_tables(tmp_path, monkeypatch):
    """Attribution has to travel with the data now that it leaves the repo."""
    import pandas as pd

    monkeypatch.setattr(fm, "_download_panglaodb",
                        lambda: pd.DataFrame([{c: "" for c in fm.SCHEMA_COLS}]))
    monkeypatch.setattr(fm, "_download_cellmarker",
                        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))

    assert fm.fetch_and_write(tmp_path) is True
    licence = (tmp_path / "LICENSE").read_text()
    assert "CC BY-NC 4.0" in licence
    assert "CellMarker" in licence


def test_panglaodb_download_sends_an_explicit_user_agent():
    """panglaodb.se 403s UAs it reads as scrapers, and pandas would send urllib's.

    Guarded at the source level because the failure is silent from the backend's
    side: the fetch just returns False and priors stay unavailable.
    """
    import inspect

    assert "User-Agent" in inspect.getsource(fm._http_get_bytes)
    src = inspect.getsource(fm._download_panglaodb)
    assert "_http_get_bytes" in src, (
        "_download_panglaodb must fetch bytes itself, not hand the URL to pandas"
    )
    assert "ScAI" in fm.USER_AGENT


# ------------------------------------------- recovering a half-fetched database

def test_a_usable_but_incomplete_database_is_topped_up_once(tmp_path, monkeypatch):
    """Nothing else ever asks for the missing source.

    ``usable()`` is an any() over the tables, so a supplementary source whose
    download failed leaves the gate satisfied: the HITL callback never fires,
    trigger_fetch is never called, and the deployment annotates on the primary
    source alone for the life of the data volume.
    """
    _write_current(tmp_path, "panglaodb.tsv")
    db = marker_priors.MarkerDB(tmp_path)
    assert db.usable() is True
    assert db.missing_tables() == ["cellmarker.tsv"]

    calls = []
    monkeypatch.setattr(marker_priors, "trigger_fetch",
                        lambda *a, **k: calls.append(1) or True)
    monkeypatch.setattr(marker_priors, "_topped_up", False)

    assert marker_priors.top_up_missing_tables(db) is True
    assert len(calls) == 1
    # Once per process: a source that stays unreachable must not be retried on
    # every annotation, and must never escalate to a prompt.
    assert marker_priors.top_up_missing_tables(db) is False
    assert len(calls) == 1


def test_a_complete_database_is_never_topped_up(tmp_path, monkeypatch):
    for name in fm.EXPECTED_TABLES:
        _write_current(tmp_path, name)
    db = marker_priors.MarkerDB(tmp_path)
    assert db.missing_tables() == []

    def _explode(*a, **k):
        raise AssertionError("must not fetch when every table is current")

    monkeypatch.setattr(marker_priors, "trigger_fetch", _explode)
    monkeypatch.setattr(marker_priors, "_topped_up", False)
    assert marker_priors.top_up_missing_tables(db) is False


def test_an_unusable_database_is_left_to_the_gate(tmp_path, monkeypatch):
    """With nothing on disk the ordinary gate runs, and it may prompt."""
    db = marker_priors.MarkerDB(tmp_path)
    assert db.usable() is False

    def _explode(*a, **k):
        raise AssertionError("the top-up must not pre-empt the marker DB gate")

    monkeypatch.setattr(marker_priors, "trigger_fetch", _explode)
    monkeypatch.setattr(marker_priors, "_topped_up", False)
    assert marker_priors.top_up_missing_tables(db) is False
