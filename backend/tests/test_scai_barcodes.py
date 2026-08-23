import importlib
import sys
from pathlib import Path

import pytest


def test_canonicalize_cross_file_barcode_forms():
    from backend.utils.kernel_startup.scai_barcodes import (
        canonicalize_barcode,
        canonicalize_barcodes,
    )

    values = [
        "sample-01:acgtacgtnn-1",
        "sample_2_ttggaaccnn-2",
        "ccccggggaax",
    ]

    assert canonicalize_barcode(values[0]) == "ACGTACGTNN"
    assert canonicalize_barcode(values[1]) == "TTGGAACCNN"
    assert canonicalize_barcode(values[2]) == "CCCCGGGGAA"
    assert canonicalize_barcodes(values) == [
        "ACGTACGTNN",
        "TTGGAACCNN",
        "CCCCGGGGAA",
    ]


def test_canonicalize_barcodes_rejects_lossy_collision():
    from backend.utils.kernel_startup.scai_barcodes import (
        barcode_index,
        canonicalize_barcodes,
    )

    values = [
        "sample-a:ACGTACGT-1",
        "sample-b_ACGTACGTx",
    ]

    with pytest.raises(ValueError, match="collision"):
        canonicalize_barcodes(values)
    with pytest.raises(ValueError, match="collision"):
        barcode_index(values)

    same_raw = ["sample-a:ACGTACGT-1", "sample-a:ACGTACGT-1"]
    assert canonicalize_barcodes(same_raw) == ["ACGTACGT", "ACGTACGT"]
    assert barcode_index(same_raw) == {"ACGTACGT": 1}


def test_canonicalize_barcode_rejects_arbitrary_identifiers():
    from backend.utils.kernel_startup.scai_barcodes import canonicalize_barcode

    with pytest.raises(ValueError):
        canonicalize_barcode("sample_00001234")


def test_barcode_index_preserves_positions():
    from backend.utils.kernel_startup.scai_barcodes import barcode_index

    values = [
        "sample_AAAAAAAA-1",
        "sample_CCCCCCCC-1",
        "GGGGGGGGx",
    ]

    assert barcode_index(values) == {
        "AAAAAAAA": 0,
        "CCCCCCCC": 1,
        "GGGGGGGG": 2,
    }


def test_kernel_startup_path_supports_top_level_scai_barcodes_import():
    startup_dir = Path(__file__).resolve().parents[1] / "utils" / "kernel_startup"
    sys.path.insert(0, str(startup_dir))
    sys.modules.pop("scai_barcodes", None)
    try:
        module = importlib.import_module("scai_barcodes")
    finally:
        sys.path.remove(str(startup_dir))

    assert Path(module.__file__).resolve().parent == startup_dir.resolve()
    assert callable(module.canonicalize_barcodes)
    assert callable(module.barcode_index)
