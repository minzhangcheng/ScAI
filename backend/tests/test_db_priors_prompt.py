from backend.agents.pi_agent import _format_database_priors_block


def test_block_includes_overlap_and_padj():
    priors = {"8": [
        {"cell_type": "pDC", "p_adj": 3.4e-4, "overlap": ["IRF8", "SIGLECH", "TCF4"]},
        {"cell_type": "hepatocyte", "p_adj": 0.31, "overlap": ["ALB", "APOA1"]},
    ]}
    block = _format_database_priors_block(priors)
    assert "DATABASE PRIORS" in block
    assert "Cluster 8" in block and "pDC" in block
    assert "padj=3.4e-04" in block or "padj=0.00034" in block
    assert "IRF8" in block


def test_empty_priors_yield_empty_block():
    assert _format_database_priors_block({}) == ""
    assert _format_database_priors_block(None) == ""
