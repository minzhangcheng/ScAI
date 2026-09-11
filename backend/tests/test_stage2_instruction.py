"""stage2 must get downstream-only instruction (DEG/GO/cellchat), not full exploration pipeline."""
from backend.workflow.agents import build_phase_instruction


def test_stage1_forbids_deg():
    txt = build_phase_instruction(planning_stage="stage1", analysis_phase="exploration",
                                  has_bulk=False, has_scrna=True, direction="")
    assert "QC" in txt
    assert "Do NOT run condition-level differential expression" in txt
    # Doublet detection and HVG selection are batch-aware where the design says so.
    assert "batch_key" in txt
    assert "live canonical variable `adata`" in txt
    assert "Include `adata` in `persistent_outputs`" in txt


def test_stage2_is_downstream_only():
    txt = build_phase_instruction(planning_stage="stage2", analysis_phase="exploration",
                                  has_bulk=False, has_scrna=True, direction="")
    assert "STAGE 2" in txt
    assert "Do NOT re-run QC, clustering, or cell type annotation" in txt
    assert "differential expression" in txt.lower()
    # The two concrete downstream recipes are offered as options, not defaults.
    assert "get_pseudobulk" in txt
    assert "by_sample" in txt
    assert "df_to_lr" in txt


def test_stage2_uses_live_sealed_adata_not_a_guessed_file():
    txt = build_phase_instruction(
        planning_stage="stage2", analysis_phase="exploration",
        has_bulk=False, has_scrna=True, direction="",
    )
    assert "live, sealed canonical" in txt
    assert "Do NOT reload or rebind `adata`" in txt
    assert "work = adata[mask].copy()" in txt
    assert "adata_annotated.h5ad" not in txt


def test_innovation_branch_unaffected():
    txt = build_phase_instruction(planning_stage="full", analysis_phase="innovation",
                                  has_bulk=False, has_scrna=True, direction="T cell exhaustion")
    assert "INNOVATION PHASE" in txt
    assert "T cell exhaustion" in txt


from backend.workflow.agents import (
    _build_exploration_instruction, _build_stage2_instruction,
)


def test_stage2_builder_leaves_mixed_bulk_method_to_agents():
    txt = _build_stage2_instruction(has_bulk=True, has_scrna=True)
    assert "STAGE 2" in txt
    assert "Bulk scientific ownership" in txt
    assert "PI" in txt and "Bioinformatics" in txt
    assert "approved plan" in txt
    assert "MANDATORY" not in txt
    bulk_section = txt.split("Bulk scientific ownership", 1)[1]
    assert "pydeseq2" not in bulk_section.lower()
    assert "nnls" not in txt.lower()


def test_stage2_builder_bulk_only_uses_the_same_neutral_contract():
    txt = _build_stage2_instruction(has_bulk=True, has_scrna=False)
    assert "Bulk scientific ownership" in txt
    assert "PI" in txt and "Bioinformatics" in txt
    bulk_section = txt.split("Bulk scientific ownership", 1)[1]
    assert "pydeseq2" not in bulk_section.lower()
    assert "fallback" not in txt.lower()


def test_stage2_builder_scrna_only_has_no_bulk_section():
    txt = _build_stage2_instruction(has_bulk=False, has_scrna=True)
    assert "Bulk RNA-seq DEG" not in txt


def test_measured_replication_does_not_override_pi_stage2_plan():
    unreplicated = _build_stage2_instruction(
        has_bulk=True, has_scrna=True, scrna_replicated=False,
    )
    replicated = _build_stage2_instruction(
        has_bulk=True, has_scrna=True, scrna_replicated=True,
    )
    assert unreplicated == replicated
    lowered = unreplicated.lower()
    assert "pi-approved plan" in lowered
    assert "claim level chosen by the pi" in lowered
    assert "pydeseq2" not in lowered.split("bulk scientific ownership", 1)[1]


def test_measured_replication_does_not_override_pi_exploration_plan():
    unreplicated = _build_exploration_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=False, bulk_replicated=True,
    )
    replicated = _build_exploration_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=True, bulk_replicated=True,
    )
    assert unreplicated == replicated
    assert "PI-approved plan" in unreplicated
    assert "single-cell figures must show descriptive localization" not in unreplicated


def test_measured_replication_does_not_install_a_stage1_design_override():
    unreplicated = build_phase_instruction(
        planning_stage="stage1", analysis_phase="exploration",
        has_bulk=True, has_scrna=True, direction="", scrna_replicated=False,
    )
    replicated = build_phase_instruction(
        planning_stage="stage1", analysis_phase="exploration",
        has_bulk=True, has_scrna=True, direction="", scrna_replicated=True,
    )
    assert unreplicated == replicated
    assert "DESIGN OVERRIDE" not in unreplicated
    assert "do NOT use Harmony" not in unreplicated


def test_bulk_replication_flag_does_not_select_a_stage2_bulk_method():
    unresolved = _build_stage2_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=False, bulk_replicated=False,
    )
    replicated = _build_stage2_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=False, bulk_replicated=True,
    )
    assert unresolved == replicated
    assert "method and implementation details" in unresolved


def test_replicated_scrna_does_not_select_a_scientific_method():
    txt = _build_stage2_instruction(
        has_bulk=True, has_scrna=True,
        scrna_replicated=True, bulk_replicated=False,
    ).lower()
    assert "pi-approved plan" in txt
    assert "bulk scientific ownership" in txt
    assert "pydeseq2" not in txt.split("bulk scientific ownership", 1)[1]
    assert "bulk deg significance" not in txt
