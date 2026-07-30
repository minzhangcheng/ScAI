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


def test_block_flags_hits_from_other_organs():
    """Adding CellMarker took the candidate pool from 178 cell types to ~1000.

    Many are fine-grained entries from unrelated organs, and a small gene set
    scores well on two shared genes -- in a real mouse-liver run "Brush cell",
    "Early enterocyte precursor cell" and "Goblet cell" all hit the same two Cyp
    genes at the same p-value. The tissue note is what lets the reader discount
    them; no automatic filter is applied, because organ-name matching on free
    text would silently drop correct hits.
    """
    priors = {"0": [
        {"cell_type": "Brush cell", "p_adj": 6.2e-7, "overlap": ["CYP2C29", "CYP3A11"],
         "tissue_match": False, "n_tissues": 2},
        {"cell_type": "Hepatocytes", "p_adj": 6.9e-3, "overlap": ["CYP2E1", "NR1I3"],
         "tissue_match": True, "n_tissues": 1},
    ]}
    block = _format_database_priors_block(priors)
    assert "tissue: other organs only (2)" in block
    assert "tissue: matches" in block
    assert "other organs only" in block  # the caveat that makes the note actionable


def test_block_reports_breadth_when_there_is_no_tissue_to_match_against():
    """A cell type annotated across dozens of organs carries no tissue evidence."""
    priors = {"1": [
        {"cell_type": "Macrophage", "p_adj": 1e-30, "overlap": ["ADGRE1", "MRC1"],
         "tissue_match": None, "n_tissues": 47},
    ]}
    assert "annotated in 47 organs" in _format_database_priors_block(priors)


def _cluster_lines(block):
    """Only the per-cluster lines; the preamble legitimately says "tissue:"."""
    return [ln for ln in block.splitlines() if ln.startswith("Cluster ")]


def test_block_omits_the_tissue_clause_when_there_is_no_tissue_data():
    """Revision-2 tables and the toy fixtures carry no tissue; no dangling text."""
    for extra in ({}, {"tissue_match": None, "n_tissues": 0}):
        priors = {"3": [dict({"cell_type": "pDC", "p_adj": 0.01,
                              "overlap": ["IRF8", "TCF4"]}, **extra)]}
        lines = _cluster_lines(_format_database_priors_block(priors))
        assert lines and "tissue" not in lines[0]
        assert "(overlap: IRF8, TCF4)" in lines[0]


def test_block_folds_in_synonyms_that_fired_on_the_same_genes():
    """The kernel collapses them into one slot; the prompt must still show them."""
    priors = {"0": [
        {"cell_type": "Brush cell", "p_adj": 6.2e-7, "overlap": ["CYP2C29", "CYP3A11"],
         "also_matched": ["Early enterocyte precursor cell", "Goblet cell", "Tuft cell"],
         "tissue_match": False, "n_tissues": 2},
    ]}
    block = _format_database_priors_block(priors)
    assert "Brush cell / Early enterocyte precursor cell / Goblet cell" in block
    assert "+1 more" in block
