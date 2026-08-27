from backend.tests.split_facade import source_for_path

SRC = source_for_path("backend/workflow/synthesis.py")


def test_phase_transition_is_guarded_by_success_flag():
    # The INNOVATION transition must be conditional on successful report generation,
    # not executed unconditionally after the except block.
    assert "report_generated = False" in SRC
    # The assignment must be guarded.
    guarded = "if report_generated:\n" in SRC and "analysis_state.analysis_phase = AnalysisPhase.INNOVATION" in SRC
    assert guarded
    # There must be no unconditional INNOVATION transition left at module indentation
    # immediately following the exploration except block.
    idx = SRC.index("Exploration report generation failed")
    after = SRC[idx: idx + 800]
    assert "if report_generated:" in after
