import textwrap
from pathlib import Path
import pandas as pd
import pytest
from backend.services import marker_priors


def _write_tsv(p: Path, rows: str, version="v1"):
    p.write_text(
        f"# SOURCE_VERSION={version} DOWNLOAD_DATE=2026-06-24\n"
        "source\tspecies\tcell_type\tgene_symbol\ttissue\tevidence\tref\n" + textwrap.dedent(rows)
    )


def test_load_skips_version_header_and_filters_species(tmp_path):
    _write_tsv(tmp_path / "panglaodb.tsv", """\
        PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t
        PanglaoDB\tmouse\tpDC\tIrf8\tImmune\t\t
        PanglaoDB\thuman\tpDC\tIRF8\tImmune\t\t
    """)
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    mouse = db.markers_for("mouse")
    assert ("pDC", "Siglech") in {(r["cell_type"], r["gene_symbol"]) for r in mouse}
    assert all(r["species"] == "mouse" for r in mouse)
    assert db.source_version("panglaodb") == "v1"


def test_merge_dedup_per_hit_source(tmp_path):
    _write_tsv(tmp_path / "panglaodb.tsv", "PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t\n")
    _write_tsv(tmp_path / "cellmarker2.tsv", "CellMarker2\tmouse\tpDC\tSiglech\tImmune\t\t\n", version="2.0")
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    hit = next(r for r in db.markers_for("mouse") if r["gene_symbol"] == "Siglech")
    assert hit["source"] == "PanglaoDB+CellMarker2"


def test_is_species_supported_normalizes(tmp_path):
    _write_tsv(tmp_path / "panglaodb.tsv", "PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t\n")
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.is_species_supported("mouse") is True
    assert db.is_species_supported("Mouse") is True            # 大小写归一
    assert db.is_species_supported("Mus musculus") is True      # 学名归一
    assert db.is_species_supported("rat") is False
    assert db.markers_for("Mus musculus")  # 归一后能取到


def test_present_false_when_no_tsv(tmp_path):
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.present() is False


def test_write_workspace_input_creates_tables_and_writes(tmp_path):
    _write_tsv(tmp_path / "panglaodb.tsv", """\
        PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t
        PanglaoDB\tmouse\tpDC\tIrf8\tImmune\t\t
    """)
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    ws = tmp_path / "workspace"
    out = db.write_workspace_input("mouse", ws)
    assert out == ws / "tables" / "marker_priors_input.tsv"
    assert out.exists()
    written = pd.read_csv(out, sep="\t")
    assert set(["cell_type", "gene_symbol", "source"]).issubset(written.columns)
    assert "Siglech" in set(written["gene_symbol"])


def test_write_workspace_input_unsupported_species_returns_none(tmp_path):
    _write_tsv(tmp_path / "panglaodb.tsv", "PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t\n")
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.write_workspace_input("rat", tmp_path / "ws") is None
