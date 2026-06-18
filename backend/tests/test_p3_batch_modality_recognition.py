import backend.workflow.helpers as helpers_mod
import backend.services.report.analysis_log as log_mod


def test_stage1_keywords_exclude_bulk_terms():
    """Bulk DEG terms belong to Stage 2, not Stage 1 (Stage 1 = QC + clustering + annotation only)."""
    assert "deseq" not in helpers_mod.STAGE1_KEYWORDS
    assert "pydeseq2" not in helpers_mod.STAGE1_KEYWORDS
    assert "volcano" not in helpers_mod.STAGE1_KEYWORDS
    assert "bulk" not in helpers_mod.STAGE1_KEYWORDS
    assert "vst" not in helpers_mod.STAGE1_KEYWORDS
    assert "tmm" not in helpers_mod.STAGE1_KEYWORDS


def test_stage1_keyword_check_ignores_re_prefix():
    """P3-1: 're-cluster' must NOT be classified as stage1."""
    assert hasattr(helpers_mod, "_has_stage1_keyword")
    check = helpers_mod._has_stage1_keyword
    assert check("re-cluster for stage2") is False
    assert check("cluster cells") is True
    assert check("perform clustering") is True
    assert check("re-annotate clusters") is True  # 'cluster' alone is still present after 're-annotate'? No — "re-annotate clusters" has 'clusters' not preceded by 're-'
    assert check("re-annotate") is False
    assert check("run pydeseq2 DEG analysis") is False  # P3-24: no longer a stage1 keyword (stays for Stage 2)


def test_analysis_log_regex_includes_non_scanpy_tools():
    """P3-15: key-code recognition must cover pydeseq2/liana/gseapy."""
    src = open(log_mod.__file__).read()
    for tool in ("pydeseq2", "liana", "gseapy", "pyscenic", "decoupler", "scvelo"):
        assert tool in src, f"P3-15: analysis_log must recognize {tool}"
