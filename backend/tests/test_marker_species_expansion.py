"""PanglaoDB dual-species rows must expand, not vanish.

PanglaoDB encodes a marker valid in both organisms as the single species value
``"Mm Hs"`` -- 7507 of its 8286 rows. The original ``.map(_SPECIES_MAP)`` had no
key for that value, produced NaN, and the subsequent ``isin(("mouse","human"))``
filter dropped every one of them, leaving 746 rows (402 mouse-only + 344
human-only) and 120 cell types. That is the only non-LLM signal feeding cluster
annotation, so it was running on ~9% of its intended content.

These tests are offline: they exercise the projection against a synthetic frame
shaped like the upstream file.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent


def _load_fetch_markers():
    """scripts/ is not a package; load by path the way marker_priors does."""
    path = _REPO / "scripts" / "fetch_markers.py"
    spec = importlib.util.spec_from_file_location("fetch_markers_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fm = _load_fetch_markers()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Mm Hs", ["mouse", "human"]),   # the 90.6% case that used to vanish
        ("Mm", ["mouse"]),
        ("Hs", ["human"]),
        ("Mus musculus", ["mouse"]),     # multi-word key must not be tokenised
        ("Homo sapiens", ["human"]),
        ("Mouse", ["mouse"]),            # CellMarker capitalisation
        ("HUMAN", ["human"]),
        ("Mm, Hs", ["mouse", "human"]),  # comma-separated variant
        ("4", []),                       # upstream junk (3 real rows)
        ("", []),
    ],
)
def test_map_species_tokens(raw, expected):
    assert fm.map_species_tokens(raw) == expected


def test_dual_species_row_becomes_two_rows():
    raw = pd.DataFrame(
        {
            "official gene symbol": ["CTRB1"],
            "cell type": ["Acinar cells"],
            "species": ["Mm Hs"],
            "organ": ["Pancreas"],
        }
    )
    out = fm.project_to_schema(
        raw, "PanglaoDB", gene_col="official gene symbol",
        celltype_col="cell type", species_col="species", tissue_col="organ",
    )
    assert len(out) == 2, "a dual-species marker must yield one row per species"
    assert set(out["species"]) == {"mouse", "human"}
    assert set(out["gene_symbol"]) == {"CTRB1"}
    assert set(out["tissue"]) == {"Pancreas"}


def test_unrecognised_species_dropped_without_killing_valid_rows():
    raw = pd.DataFrame(
        {
            "official gene symbol": ["A", "B", "C"],
            "cell type": ["T", "T", "T"],
            "species": ["4", "Mm Hs", "Hs"],
            "organ": ["Blood", "Blood", "Blood"],
        }
    )
    out = fm.project_to_schema(
        raw, "PanglaoDB", gene_col="official gene symbol",
        celltype_col="cell type", species_col="species", tissue_col="organ",
    )
    assert set(out["gene_symbol"]) == {"B", "C"}
    assert len(out) == 3  # B -> mouse + human, C -> human


def test_tissue_column_is_populated():
    """organ was never passed, so every row was written tissue-blind."""
    raw = pd.DataFrame(
        {
            "official gene symbol": ["G1"],
            "cell type": ["Hepatocytes"],
            "species": ["Mm Hs"],
            "organ": ["Liver"],
        }
    )
    out = fm.project_to_schema(
        raw, "PanglaoDB", gene_col="official gene symbol",
        celltype_col="cell type", species_col="species", tissue_col="organ",
    )
    assert (out["tissue"] == "Liver").all()


def test_panglaodb_downloader_requests_the_organ_column():
    """Guard the call site: dropping tissue_col silently re-blinds the table."""
    import inspect

    src = inspect.getsource(fm._download_panglaodb)
    assert 'tissue_col="organ"' in src, (
        "_download_panglaodb must pass the organ column as tissue"
    )


def test_schema_is_unchanged():
    """marker_priors.REQUIRED_COLS reads these exact columns."""
    assert fm.SCHEMA_COLS == [
        "source", "species", "cell_type", "gene_symbol", "tissue", "evidence", "ref"
    ]
