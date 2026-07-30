import textwrap
from pathlib import Path
import pandas as pd
import pytest
from backend.services import marker_priors


def _write_tsv(p: Path, rows: str, version="v1", revision=None):
    revision = marker_priors.MARKER_SCHEMA_REVISION if revision is None else revision
    p.write_text(
        f"# SOURCE_VERSION={version} SCHEMA_REVISION={revision} DOWNLOAD_DATE=2026-06-24\n"
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
    _write_tsv(tmp_path / "cellmarker.tsv", "CellMarker\tmouse\tpDC\tSiglech\tImmune\t\t\n", version="3.0")
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    hit = next(r for r in db.markers_for("mouse") if r["gene_symbol"] == "Siglech")
    assert hit["source"] == "PanglaoDB+CellMarker"


def test_is_species_supported_normalizes(tmp_path):
    _write_tsv(tmp_path / "panglaodb.tsv", "PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t\n")
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.is_species_supported("mouse") is True
    assert db.is_species_supported("Mouse") is True            # 大小写归一
    assert db.is_species_supported("Mus musculus") is True      # 学名归一
    assert db.is_species_supported("rat") is False
    assert db.markers_for("Mus musculus")  # 归一后能取到


def test_usable_false_when_no_tsv(tmp_path):
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.usable() is False


def test_usable_true_for_current_revision(tmp_path):
    _write_tsv(tmp_path / "panglaodb.tsv", "PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t\n")
    assert marker_priors.MarkerDB(markers_dir=tmp_path).usable() is True


def test_stale_revision_reads_as_absent_and_is_not_loaded(tmp_path):
    """A table from an older projection must not be served or counted as present.

    Revision 1 tables hold ~9% of PanglaoDB (dual-species rows dropped) and are
    tissue-blind. Treating them as available is what made that state permanent
    on any deployment that had fetched once.
    """
    _write_tsv(tmp_path / "panglaodb.tsv",
               "PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t\n", revision=1)
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.usable() is False
    assert db.markers_for("mouse") == []
    assert db.is_species_supported("mouse") is False


def test_header_without_revision_reads_as_stale(tmp_path):
    """Pre-revision headers carried only SOURCE_VERSION; they are stale too."""
    (tmp_path / "panglaodb.tsv").write_text(
        "# SOURCE_VERSION=PanglaoDB_27_Mar_2020 DOWNLOAD_DATE=2026-07-18\n"
        "source\tspecies\tcell_type\tgene_symbol\ttissue\tevidence\tref\n"
        "PanglaoDB\tmouse\tpDC\tSiglech\t\t\t\n"
    )
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.usable() is False
    assert db.markers_for("mouse") == []


def test_current_table_still_loads_when_other_source_is_stale(tmp_path):
    """One stale supplementary table must not blind the loader to a good main one."""
    _write_tsv(tmp_path / "panglaodb.tsv", "PanglaoDB\tmouse\tpDC\tSiglech\tImmune\t\t\n")
    _write_tsv(tmp_path / "cellmarker.tsv",
               "CellMarker\tmouse\tpDC\tIrf8\tImmune\t\t\n", revision=1)
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    assert db.usable() is True
    genes = {r["gene_symbol"] for r in db.markers_for("mouse")}
    assert genes == {"Siglech"}, "stale cellmarker rows must not be merged in"


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


def test_write_workspace_input_carries_tissue(tmp_path):
    """The ORA kernel labels each prior with the organ, so tissue must reach it."""
    _write_tsv(tmp_path / "panglaodb.tsv", """\
        PanglaoDB\tmouse\tKupffer cells\tClec4f\tLiver\t\t
        PanglaoDB\tmouse\tKupffer cells\tAdgre1\tLiver\t\t
    """)
    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    out = db.write_workspace_input("mouse", tmp_path / "ws")
    written = pd.read_csv(out, sep="\t")
    assert list(written.columns) == ["cell_type", "gene_symbol", "source", "tissue"]
    assert set(written["tissue"]) == {"Liver"}


def test_dedup_unions_tissues_instead_of_keeping_the_first(tmp_path):
    """The kernel asks "is this cell type annotated in my tissue?".

    The dedup used to aggregate tissue with "first", so one arbitrary organ
    survived per (cell type, gene) pair. CellMarker annotates the same marker
    across many tissues, so the answer came back "no" for cell types that are in
    the tissue -- visible as CD8 T cells reporting a single organ.
    """
    _write_tsv(tmp_path / "panglaodb.tsv", "PanglaoDB\tmouse\tMacrophage\tMrc1\tLiver\t\t\n")
    _write_tsv(tmp_path / "cellmarker.tsv", """\
        CellMarker\tmouse\tMacrophage\tMrc1\tSpleen\t\t
        CellMarker\tmouse\tMacrophage\tMrc1\tLung\t\t
        CellMarker\tmouse\tMacrophage\tMrc1\tLiver\t\t
    """, version="3.0")

    db = marker_priors.MarkerDB(markers_dir=tmp_path)
    hit = next(r for r in db.markers_for("mouse") if r["gene_symbol"] == "Mrc1")
    tissues = {t.strip() for t in hit["tissue"].split(marker_priors.TISSUE_SEP)}
    assert tissues == {"Liver", "Spleen", "Lung"}, "all organs must survive dedup"
    assert hit["source"] == "PanglaoDB+CellMarker"


def test_join_tissues_deduplicates_and_flattens_nested_separators():
    join = marker_priors._join_tissues
    assert join(["Liver", "Liver", ""]) == "Liver"
    assert join(["Liver;Spleen", "Spleen", "Lung"]) == "Liver;Spleen;Lung"
    assert join(["", None, "  "]) == ""
