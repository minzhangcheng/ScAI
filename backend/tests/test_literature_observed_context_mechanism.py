from backend.models.schemas import AnalysisState, DataDrivenMechanismProgram


def test_observed_context_includes_program_fields():
    from backend.workflow.core import SingleCellAnalysisWorkflow
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "t"

    st = AnalysisState(user_id="u1")
    st.data_driven_mechanism_programs = [DataDrivenMechanismProgram(
        label="iron-redox-lipid hepatocyte program",
        cell_types=["hepatocyte"],
        triggering_observations=["GPX4 down, ACSL4 up"],
        supporting_genes=["Gpx4", "Acsl4"],
        supporting_pathways=["glutathione metabolism"],
        bulk_support="mixed",
        alternative_explanations=["generic stress"],
        missing_evidence=["no lipid peroxidation assay"],
    )]
    ctx = wf._build_observed_context_for_literature({"analysis_state": st})
    assert "iron-redox-lipid hepatocyte program" in ctx
    assert "Gpx4" in ctx
    assert "glutathione metabolism" in ctx
    assert "generic stress" in ctx
    assert "no lipid peroxidation assay" in ctx
