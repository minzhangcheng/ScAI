import pytest
from backend.workflow.core import SingleCellAnalysisWorkflow


def test_generation_is_per_node():
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf._llm_failure_generations = {}

    g1 = wf._next_generation("pi_planning")
    g2 = wf._next_generation("bioinformatics_analysis")

    assert wf._generation_for("pi_planning") == g1
    assert wf._generation_for("bioinformatics_analysis") == g2
