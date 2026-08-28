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


from backend.workflow.agents import (
    _build_exploration_instruction, _build_stage2_instruction,
)


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


def test_unreplicated_mixed_stage2_uses_bulk_inference_and_descriptive_scrna():
    txt = _build_stage2_instruction(
        has_bulk=True, has_scrna=True, scrna_replicated=False,
    )
    lowered = txt.lower()
    assert "replicated bulk inference" in lowered
    assert "descriptive single-cell" in lowered
    assert "do not manufacture scrna replication" in lowered
    assert "cross-validate against scrna pseudobulk" not in lowered


def test_unreplicated_exploration_does_not_mandate_single_cell_deg_figures():
    txt = _build_exploration_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=False, bulk_replicated=True,
    )
    assert "For EACH cell type group's DEG results" not in txt
    assert "single-cell figures must show descriptive localization" in txt


def test_unreplicated_stage1_forbids_condition_confounded_correction():
    txt = build_phase_instruction(
        planning_stage="stage1", analysis_phase="exploration",
        has_bulk=True, has_scrna=True, direction="", scrna_replicated=False,
    )
    assert "UNREPLICATED SINGLE-CELL CONDITIONS" in txt
    assert "do NOT use Harmony" in txt


def test_unresolved_bulk_does_not_receive_inferential_stage2_tasks():
    txt = _build_stage2_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=False, bulk_replicated=False,
    ).lower()
    assert "unreplicated or unresolved mixed design" in txt
    assert "replicated bulk inference" not in txt
    assert "condition-level p/fdr" in txt


def test_replicated_scrna_does_not_license_unresolved_bulk():
    txt = _build_stage2_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=True, bulk_replicated=False,
    ).lower()
    assert "unresolved bulk design" in txt
    assert "replicated single-cell pseudobulk" in txt
    assert "bulk deg significance" in txt
