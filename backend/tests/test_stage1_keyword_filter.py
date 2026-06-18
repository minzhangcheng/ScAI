"""bulk DEG 任务不得被判定为 stage1（stage1 仅 QC+聚类+注释）。"""
from backend.workflow.helpers import _has_stage1_keyword


def test_qc_and_clustering_are_stage1():
    assert _has_stage1_keyword("Run QC and leiden clustering")
    assert _has_stage1_keyword("Cell type annotation of clusters")
    assert _has_stage1_keyword("Normalize and select HVGs")


def test_bulk_deg_is_not_stage1():
    assert not _has_stage1_keyword("Run pydeseq2 differential expression on bulk RNA-seq")
    assert not _has_stage1_keyword("Generate a volcano plot for bulk DEG")
    assert not _has_stage1_keyword("DESeq2 condition-level analysis")


def test_re_prefixed_still_not_stage1():
    assert not _has_stage1_keyword("re-cluster the T cells")
