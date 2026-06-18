"""stage2 must get downstream-only instruction (DEG/GO/cellchat), not full exploration pipeline."""
from backend.workflow.agents import build_phase_instruction


def test_stage1_forbids_deg():
    txt = build_phase_instruction(planning_stage="stage1", analysis_phase="exploration",
                                  has_bulk=False, has_scrna=True, direction="")
    assert "QC" in txt
    assert "Do NOT run condition-level differential expression" in txt


def test_stage2_is_downstream_only():
    txt = build_phase_instruction(planning_stage="stage2", analysis_phase="exploration",
                                  has_bulk=False, has_scrna=True, direction="")
    assert "STAGE 2" in txt
    assert "Do NOT re-run QC, clustering, or cell type annotation" in txt
    assert "differential expression" in txt.lower()


def test_innovation_branch_unaffected():
    txt = build_phase_instruction(planning_stage="full", analysis_phase="innovation",
                                  has_bulk=False, has_scrna=True, direction="T cell exhaustion")
    assert "INNOVATION PHASE" in txt
    assert "T cell exhaustion" in txt


from backend.workflow.agents import _build_stage2_instruction


def test_stage2_builder_bulk_and_scrna_mandates_cross_validation():
    txt = _build_stage2_instruction(has_bulk=True, has_scrna=True)
    assert "STAGE 2" in txt
    assert "MANDATORY" in txt
    assert "deconvolution" in txt.lower()


def test_stage2_builder_bulk_only_has_no_cross_validation():
    txt = _build_stage2_instruction(has_bulk=True, has_scrna=False)
    assert "Cross-validation/deconvolution not applicable" in txt


def test_stage2_builder_scrna_only_has_no_bulk_section():
    txt = _build_stage2_instruction(has_bulk=False, has_scrna=True)
    assert "Bulk RNA-seq DEG" not in txt
