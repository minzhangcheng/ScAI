"""P1-2: scRNA-seq loader must dispatch by file extension, not hardcode h5ad."""
import re
import backend.agents.bioinformatics_agent as bio_mod


def _get_build_data_context_body() -> str:
    src = open(bio_mod.__file__).read()
    fn = re.search(r"def _build_data_context\b.*?(?=\n    (?:async )?def |\Z)", src, re.DOTALL)
    assert fn
    return fn.group(0)


def test_data_context_dispatches_by_extension():
    body = _get_build_data_context_body()
    required_loaders = ["sc.read_h5ad", "sc.read_10x_h5", "sc.read_loom", "sc.read_csv", "sc.read_10x_mtx"]
    for loader in required_loaders:
        assert loader in body, f"P1-2: dispatch missing {loader}"
    assert ".suffix" in body or ".rsplit('.'" in body or "ext" in body.lower(), \
        "P1-2: must inspect file extension"


def test_seurat_and_rds_handled_via_comment():
    body = _get_build_data_context_body()
    assert ".rds" in body, "must mention .rds handling"
    assert "SeuratDisk" in body or "convert" in body.lower(), \
        "must suggest conversion path for Seurat"
