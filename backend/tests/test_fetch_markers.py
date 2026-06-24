import pandas as pd
from pathlib import Path
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "fetch_markers", Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py")
fetch_markers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_markers)


def test_project_to_schema_keeps_required_cols_and_species():
    raw = pd.DataFrame({
        "official gene symbol": ["Siglech", "IRF8", "Foo"],
        "cell type": ["pDC", "pDC", "x"],
        "species": ["Mm", "Hs", "Rn"],  # 鼠/人/鼠科外
    })
    out = fetch_markers.project_to_schema(raw, source="PanglaoDB",
                                          gene_col="official gene symbol",
                                          celltype_col="cell type", species_col="species")
    assert list(out.columns) == fetch_markers.SCHEMA_COLS
    assert set(out["species"]) <= {"mouse", "human"}  # rat 被丢
    assert "Siglech" in set(out["gene_symbol"])


def test_atomic_write_tsv_has_version_header(tmp_path):
    df = pd.DataFrame([{c: "" for c in fetch_markers.SCHEMA_COLS}])
    dest = tmp_path / "panglaodb.tsv"
    fetch_markers.atomic_write_tsv(df, dest, source_version="vX")
    head = dest.read_text().splitlines()[0]
    assert head.startswith("# SOURCE_VERSION=vX")
    assert not (tmp_path / "panglaodb.tsv.tmp").exists()  # tmp 已 rename 掉
