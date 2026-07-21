from backend.agents.bioinformatics_agent import workspace_h5ad_write_guard


def test_h5ad_guard_blocks_third_checkpoint(tmp_path):
    (tmp_path / "first.h5ad").write_bytes(b"one")
    (tmp_path / "second.h5ad").write_bytes(b"two")

    reason = workspace_h5ad_write_guard(
        "adata.write_h5ad(_WORKSPACE_DIR / 'third.h5ad')", tmp_path
    )
    assert reason is not None
    assert "At most 2" in reason


def test_h5ad_guard_allows_overwrite_and_non_h5ad_writes(tmp_path):
    (tmp_path / "first.h5ad").write_bytes(b"one")
    (tmp_path / "second.h5ad").write_bytes(b"two")

    assert workspace_h5ad_write_guard(
        "adata.write('first.h5ad')", tmp_path
    ) is None
    assert workspace_h5ad_write_guard(
        "report.write('summary.txt')", tmp_path
    ) is None


def test_h5ad_guard_counts_multiple_new_targets_in_one_cell(tmp_path):
    (tmp_path / "first.h5ad").write_bytes(b"one")
    code = "adata.write('second.h5ad')\nadata.write('third.h5ad')"
    assert workspace_h5ad_write_guard(code, tmp_path) is not None
